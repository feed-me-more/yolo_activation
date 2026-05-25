"""
streaming/receiver.py  –  UDP receiver for YOLO split inference demo.

Receives metadata-free packets grouped only by frame id, reconstructs the
split activation, runs Stage 1 + Stage 2 alignment and the YOLO tail, and
returns detections plus timing stats back to the sender over UDP.
"""
from __future__ import annotations

import json
import socket
import struct
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


HEADER_FMT = "!I"
HEADER_SIZE = struct.calcsize(HEADER_FMT)

LOCAL_RECV_PORT = 5006            # Receiving port
LOCAL_SEND_PORT = 5005            # Sending port
DEST_PORT = 5005                  # Receiver's port
dest_addr = "192.168.0.163"       # Receiver's IP address
# send_addr = "192.168.0.157"     # Sender's IP address


def run_receiver(
    host: str,
    port: int,
    num_packets: int,
    out_dir: str,
    device: str,
    corpus_size: int,
    use_bitmask: bool,
    use_randperm: bool,
    seed: int,
    frame_timeout_ms: int,
):
    from alignment.faiss_index import build_index
    from alignment.hungarian import assign_positions
    from corpus.build import build_corpus
    from corpus.transform import apply_transform
    from models.yolo import YOLOv8Split, postprocess_yolo_predictions
    from utils.packet import PacketTransform, normalise_chunks

    print(f"\n{'═'*72}")
    print("  YOLO Split Inference  –  RECEIVER")
    print(f"  Listening on     : {host}:{port}")
    print(f"  Frame timeout    : {frame_timeout_ms} ms")
    print(f"  Corpus size      : {corpus_size}")
    print(f"{'═'*72}\n")

    dev = torch.device(device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    model = YOLOv8Split(num_packets=num_packets, device=str(dev))
    model.dims.print()

    transform = PacketTransform(
        d=model.dims.activation_d,
        P=num_packets,
        seed=seed,
        use_bitmask=use_bitmask,
        use_randperm=use_randperm,
    )

    corpus_raw, _ = build_corpus(
        model_name="yolo",
        num_packets=num_packets,
        device=str(dev),
        corpus_size=min(corpus_size, 96),
        out_dir=out_dir,
        batch_size=4,
        hf_token="",
        seed=seed,
    )
    _, corpus_transformed = apply_transform(
        model_name="yolo",
        num_packets=num_packets,
        out_dir=out_dir,
        use_bitmask=use_bitmask,
        use_randperm=use_randperm,
        seed=seed,
    )
    index = build_index(corpus_transformed, device=str(dev))

    # Receiving socket
    sock_rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_rx.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
    sock_rx.bind((host, port))
    sock_rx.settimeout(0.01)

    sock_tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    buf: dict[int, dict] = defaultdict(lambda: {"chunks": [], "t_first": None, "sender": None})
    timeout_s = frame_timeout_ms / 1000.0

    print(f"  {'frame':>7}  {'recv':>9}  {'align_ms':>9}  {'decode_ms':>9}  {'total_ms':>9}  {'loss%':>7}")
    print(f"  {'─'*66}")
    print("  Receiver ready. Start the sender now.\n")

    while True:
        try:
            data, sender_addr = sock_rx.recvfrom(65535)
        except socket.timeout:
            _flush_stale(
                buf=buf,
                now=time.perf_counter(),
                timeout_s=timeout_s,
                model=model,
                transform=transform,
                index=index,
                corpus_raw=corpus_raw,
                P=num_packets,
                d_c=transform.d_c,
                sock_tx=sock_tx,
            )
            continue

        if len(data) < HEADER_SIZE:
            continue

        frame_id = struct.unpack(HEADER_FMT, data[:HEADER_SIZE])[0]
        payload = np.frombuffer(data[HEADER_SIZE:], dtype=np.float16).astype(np.float32, copy=False)

        entry = buf[frame_id]
        if entry["t_first"] is None:
            entry["t_first"] = time.perf_counter()
        entry["sender"] = sender_addr
        entry["chunks"].append(payload)

        if len(entry["chunks"]) >= num_packets:
            _process_frame(
                frame_id=frame_id,
                entry=entry,
                model=model,
                transform=transform,
                index=index,
                corpus_raw=corpus_raw,
                P=num_packets,
                d_c=transform.d_c,
                sock_tx=sock_tx,
            )
            del buf[frame_id]


def _flush_stale(buf, now, timeout_s, **kwargs):
    stale = [fid for fid, entry in buf.items() if entry["t_first"] is not None and (now - entry["t_first"]) > timeout_s]
    for fid in stale:
        if buf[fid]["chunks"]:
            _process_frame(frame_id=fid, entry=buf[fid], **kwargs)
        del buf[fid]


def _process_frame(
    frame_id: int,
    entry: dict,
    model,
    transform,
    index,
    corpus_raw: np.ndarray,
    P: int,
    d_c: int,
    sock_tx,
):
    from alignment.hungarian import assign_positions
    from models.yolo import postprocess_yolo_predictions
    from utils.packet import normalise_chunks

    t_start = time.perf_counter()
    raw_received = np.stack(entry["chunks"], axis=0).astype(np.float32, copy=False)
    n_recv = int(raw_received.shape[0])

    t_align = time.perf_counter()
    received_align = normalise_chunks(raw_received)
    if n_recv > 0:
        m_star, benefit, _ = index.retrieve(received_align)
        pred_pos = assign_positions(benefit)
    else:
        m_star = 0
        pred_pos = np.array([], dtype=np.int32)
    align_ms = (time.perf_counter() - t_align) * 1e3

    corpus_entry = _corpus_entry_chunks(corpus_raw[m_star], transform, P, d_c)
    activation = _reconstruct_activation(raw_received, pred_pos, corpus_entry, transform, P, d_c)

    t_dec = time.perf_counter()
    pred = model.decoder(activation)
    detections = postprocess_yolo_predictions(pred)
    decode_ms = (time.perf_counter() - t_dec) * 1e3

    total_ms = (time.perf_counter() - t_start) * 1e3
    packet_loss_pct = 100.0 * (1.0 - n_recv / max(P, 1))

    print(
        f"  {frame_id:>7}  {n_recv:>4}/{P:<4}  {align_ms:>9.2f}  "
        f"{decode_ms:>9.2f}  {total_ms:>9.2f}  {packet_loss_pct:>6.1f}"
    )

    sender_addr = entry.get("sender")
    if sender_addr is not None:
        result = {
            "frame_id": int(frame_id),
            "received_packets": n_recv,
            "num_packets": int(P),
            "packet_loss_pct": float(packet_loss_pct),
            "align_ms": float(align_ms),
            "decode_ms": float(decode_ms),
            "server_ms": float(total_ms),
            "detections": detections[:50].tolist(),
            "class_names": {int(k): v for k, v in model._model.names.items()},
        }
        sock_tx.sendto(json.dumps(result).encode("utf-8"), (sender_addr[0], DEST_PORT))


def _corpus_entry_chunks(corpus_row: np.ndarray, transform, P: int, d_c: int) -> np.ndarray:
    row = np.asarray(corpus_row, dtype=np.float32)
    if getattr(transform, "d_pad", transform.d) > transform.d:
        row = np.pad(row, (0, transform.d_pad - transform.d))
    return row[transform.perm].reshape(P, d_c)


def _reconstruct_activation(
    raw_received: np.ndarray,
    pred_pos: np.ndarray,
    corpus_entry: np.ndarray,
    transform,
    P: int,
    d_c: int,
) -> np.ndarray:
    full_chunks = np.zeros((P, d_c), dtype=np.float32)
    full_chunks[:] = corpus_entry

    for i, j in enumerate(pred_pos):
        if 0 <= j < P:
            full_chunks[j] = transform.decode_chunk(raw_received[i], int(j))

    full_perm = full_chunks.reshape(-1)
    full_raw = np.empty_like(full_perm)
    full_raw[transform.perm] = full_perm
    return full_raw[:transform.d]
