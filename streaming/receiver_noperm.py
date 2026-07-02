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

from temp_plot import plot_data_rate, plot_latency

HEADER_FMT = "!I I d"
HEADER_SIZE = struct.calcsize(HEADER_FMT)

LOCAL_RECV_PORT = 5006            # Receiving port
LOCAL_SEND_PORT = 5005            # Sending port
DEST_PORT = 5005                  # Receiver's port
# dest_addr = "192.168.0.163"       # Receiver's IP address
dest_addr = "10.51.1.167"

# dest_addr = "192.168.0.136"
# LOCAL_RECV_PORT = 5003        # Receiving PORT
# LOCAL_SEND_PORT = 5004        # Sending PORT
# DEST_PORT = 5002			  # Receiver's PORT
# send_addr = "192.168.0.157"     # Sender's IP address
incomplete_pkt = 0
missed_pkt = 0
latency_accumulator = {}



def run_receiver_noperm(
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

    port = LOCAL_RECV_PORT

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

    # Receiving socket
    sock_rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_rx.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8 * 1024 * 1024)
    print(f"  Binding to {host}:{port}...")
    sock_rx.bind((host, port))
    sock_rx.settimeout(0.1)

    sock_tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    buf: dict[int, dict] = defaultdict(lambda: {"chunks": [None]*num_packets, "t_first": None, "sender": None, "latency_ms": [None]*num_packets, "received": 0, "chunk_shape": 0, "last_pkt_time": 0.0})
    # timeout_s = frame_timeout_ms / 1000.0
    timeout_s = 0.2

    print(f"  {'frame':>7}  {'recv':>9}  {'processing_ms':>9}  {'decode_ms':>9}  {'server_ms':>9} {'sent_ms':>9} {'avg_ltncy':>9} {'data_rate':>7}  {'loss%':>6}")
    print(f"  {'─'*66}")
    # print("  Receiver ready. Start the sender now.\n")

#     f"  {frame_id:>7}  {n_recv:>4}/{P:<4}  {decode_ms:>9.2f}  {processing_ms:>9.2f}  "
# f"{server_ms:>9.2f}  {sent_ms:>9.2f} {avg_latency:>9.2f} {data_rate[frame_id]:>7.3f} {packet_loss_pct:>6.1f}"

    pkt_start = {}
    pkt_last = {}
    data_rate = {}
    frm_strt2end = {}
    global incomplete_pkt, missed_pkt, latency_accumulator

    once = True

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
                P=num_packets,
                d_c=transform.d_c,
                sock_tx=sock_tx,
                frm_strt2end=frm_strt2end,
                data_rate=data_rate
            )
            continue

        if len(data) < HEADER_SIZE:
            continue

        frame_id, idx, timestamp = struct.unpack(HEADER_FMT, data[:HEADER_SIZE])
        latency = (time.time() - timestamp) * 1e3
        payload = np.frombuffer(data[HEADER_SIZE:], dtype=np.float16).astype(np.float32, copy=False)

        if payload is not None and payload.size > 0:
            pkt_last[frame_id] = time.time()

        if frame_id == 4294967295:
            print(f"Frame ID: {frame_id}, Index: {idx}, Payload Shape: {payload.shape}\n")
            break

        entry = buf[frame_id]

        if entry["t_first"] is None:
            entry["t_first"] = time.perf_counter()
            pkt_start[frame_id] = timestamp
            # print(f"  Frame {frame_id}: First packet received at {timestamp:.6f}")
            entry["chunk_shape"] = payload.shape

        entry["chunks"][idx] = payload
        entry["sender"] = sender_addr        
        entry["latency_ms"][idx] = latency
        entry["received"] = sum(c is not None for c in entry["chunks"])
        entry["last_pkt_time"] = timestamp

        frm_strt2end[frame_id] = pkt_last[frame_id] - pkt_start[frame_id]

        if entry["received"] >= num_packets:
        
            data_rate[frame_id] = (entry["received"]*entry["chunk_shape"][0]*2)/(frm_strt2end[frame_id])/(1024*1024)
            # print(f"Frame {frame_id}: \nData rate:         {data_rate[frame_id]:.3f} Mbytes/sec")

            _process_frame(
                frame_id=frame_id,
                entry=entry,
                data_rate=data_rate,
                model=model,
                transform=transform,
                P=num_packets,
                d_c=transform.d_c,
                sock_tx=sock_tx,
            )
            # print(f"Latencies for frame {frame_id}: {entry['latency_ms']}")

            latency_accumulator[frame_id] = entry['latency_ms']

            # avg_latency = np.mean([x for x in entry['latency_ms'] if x is not None])
            
            # print(f"Avg pkt latency:    {avg_latency:.3f} ms")
            # latency_accumulator.append(round(float(avg_latency), 4))
            # print("--"*60)

            del buf[frame_id]   

    
    print(f"Total missed packets    : {missed_pkt}")
    print(f"Total incomplete packets: {incomplete_pkt}")
    # print(f"Average packet latency  : {latency_accumulator}")
    print(f"Avg Data rate per frame : {np.mean([data_rate[fid] for fid in data_rate]):.2f} Mbytes/sec")
    print(f"Max Data rate per frame : {np.max([data_rate[fid] for fid in data_rate]):.2f} Mbytes/sec")

    time_taken = pkt_last[next(reversed(data_rate))] - next(iter(pkt_start.values()))
    total_data_bytes_received = (len(pkt_start)*transform.P-missed_pkt)*transform.d_c * 2 

    print(f"Time taken: {time_taken:.3f} seconds, Data received for {len(pkt_start)} frames: {total_data_bytes_received/(1024*1024):.3f} Mbytes at rate of {total_data_bytes_received/(time_taken*1024*1024):.3f} Mbytes/sec")
    
    print(f"Avg overall data rate   : {(total_data_bytes_received/time_taken)/(1024*1024):.3f} Mbytes/sec")

    print("Missing in pkt_last of len(pkt_last)  : ", [i for i in range(302) if i not in pkt_last])
    print("Missing in pkt_start of len(pkt_start): ", [i for i in range(302) if i not in pkt_start])
    print("Missing in data of len(data_rate)     : ", [i for i in range(302) if i not in data_rate])

    # print(f"last a packet received at {pkt_last[next(reversed(data_rate))]:.6f})")

    plot_data_rate(data_rate, "data_rate_scatter")
    plot_latency(latency_accumulator, "latency_scatter")

    all_lats = np.array([
    lat
    for frame_lats in latency_accumulator.values()
    for lat in frame_lats
    if lat is not None])

    print(f"Overall average packet latency: {np.mean(all_lats):.3f} ms")
    print(f"Overall max packet latency    : {np.max(all_lats):.3f} ms")
    print(f"Overall min packet latency    : {np.min(all_lats):.3f} ms")
    print(f"Overall 95th percentile latency: {np.percentile(all_lats, 95):.3f} ms")

def _flush_stale(buf, now, timeout_s, frm_strt2end, data_rate, **kwargs):

    global latency_accumulator
    
    # If start time exists then check if timeout elapsed and mark as stale
    stale = [fid for fid, entry in buf.items() if entry["t_first"] is not None and (now - entry["t_first"]) > timeout_s]
    for fid in stale:
        if buf[fid]["chunks"]:
            # print(f"  Flushing stale frame {fid} with {buf[fid]['received']} packets received after timeout.")

            data_rate[fid] = (buf[fid]['received']*buf[fid]["chunk_shape"][0]*2)/(frm_strt2end[fid])/(1024*1024)
            # print(f"Frame {fid}: \nData rate:         {data_rate[fid]:.3f} Mbytes/sec")

            _process_frame(frame_id=fid, entry=buf[fid], data_rate=data_rate, **kwargs)
        
        latency_accumulator[fid] = buf[fid]['latency_ms']
        # avg_latency = np.mean([x for x in buf[fid]['latency_ms'] if x is not None])
        # print(f"Avg pkt latency:    {avg_latency:.2f} ms")
        # print("--"*60)
        del buf[fid]

def _process_frame(
    frame_id: int,
    entry: dict,
    data_rate: dict,
    model,
    transform,
    P: int,
    d_c: int,
    sock_tx,
):

    from models.yolo import postprocess_yolo_predictions

    t_start = time.perf_counter()

    raw_received = np.stack([
        chunk if chunk is not None else np.zeros(entry["chunk_shape"], dtype=np.float32)
        for chunk in entry["chunks"]
    ], axis=0)

    # print(f"  Frame {frame_id}: Received {entry['received']}/{P} packets, shape {raw_received.shape}")
    n_recv = entry['received']

    activation = raw_received.reshape(-1)[:transform.d]
    # print(f"  Reconstructed activation shape: {activation.shape} for frame {frame_id}")

    t_dec = time.perf_counter()
    pred = model.decoder(activation)
    detections = postprocess_yolo_predictions(pred)
    decode_ms = (time.perf_counter() - t_dec) * 1e3

    processing_ms = (time.perf_counter() - t_start) * 1e3
    server_ms = (time.perf_counter() - entry["t_first"]) * 1e3
    packet_loss_pct = 100.0 * (1.0 - n_recv / max(P, 1))

    global incomplete_pkt, missed_pkt
    if n_recv != P:
        incomplete_pkt +=1 
        missed_pkt += (P - n_recv)

    # print(
    #     f"  {frame_id:>7}  {n_recv:>4}/{P:<4}  {processing_ms:>9.2f}  "
    #     f"{decode_ms:>9.2f}  {server_ms:>9.2f}  {packet_loss_pct:>6.1f}"
    # )

    sender_addr = entry.get("sender")
    if sender_addr is not None:
        result = {
            "frame_id": int(frame_id),
            "received_packets": n_recv,
            "num_packets": int(P),
            "packet_loss_pct": float(packet_loss_pct),
            # "align_ms": float(align_ms),
            "decode_ms": float(decode_ms),
            "processing_ms": float(processing_ms),
            "server_ms": float(server_ms),
            "detections": detections[:50].tolist(),
            "class_names": {int(k): v for k, v in model._model.names.items()},
        }
        
        sock_tx.sendto(json.dumps(result).encode("utf-8"), (sender_addr[0], DEST_PORT))

        sent_ms = (time.perf_counter() - entry["t_first"]) * 1e3

        avg_latency = np.mean([x for x in entry['latency_ms'] if x is not None])

        print(
        f"  {frame_id:>7}  {n_recv:>4}/{P:<4}  {decode_ms:>9.2f}  {processing_ms:>9.2f}  "
        f"{server_ms:>9.2f}  {sent_ms:>9.2f} {avg_latency:>9.3f} {data_rate[frame_id]:>7.3f} {packet_loss_pct:>6.1f}"
    )
