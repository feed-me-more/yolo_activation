"""
streaming/sender.py  –  UDP sender for YOLO split inference demo.

Runs YOLO encoder on video/webcam frames, packetises the split activation,
sends packets to the receiver, waits for a decoded detection response, and
displays/saves the annotated stream.

No positional metadata is sent per packet: only the frame id is attached.
Packet loss/reordering simulation is enabled only for localhost-style runs.
"""
from __future__ import annotations

import json
import random
import socket
import struct
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision import transforms


HEADER_FMT = "!I"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
LOCAL_HOSTS = {"127.0.0.1", "localhost", "0.0.0.0"}
UDP_MAX_PAYLOAD = 65507


LOCAL_RECV_PORT = 5005        # Receiving PORT
LOCAL_SEND_PORT = 5006        # Sending PORT
DEST_PORT = 5006			  # Receiver's PORT
dest_addr = "192.168.0.157"  # Receiver's IP address (laptop)


def run_sender(
    source: str,
    host: str,
    port: int,
    num_packets: int,
    udp_loss: float,
    seed: int,
    use_bitmask: bool,
    use_randperm: bool,
    out_dir: str,
    device: str,
    frame_timeout_ms: int,
    max_frames: int,
    show: bool,
    save_video: bool,
):
    from models.yolo import YOLOv8Split
    from utils.packet import PacketTransform

    is_local_demo = host in LOCAL_HOSTS
    effective_loss = udp_loss if is_local_demo else 0.0
    if udp_loss > 0 and not is_local_demo:
        print("  [sender] Non-local UDP target detected, disabling simulated packet loss.")

    print(f"\n{'═'*72}")
    print("  YOLO Split Inference  –  SENDER")
    print(f"  Target           : {host}:{port}")
    print(f"  Source           : {source}")
    print(f"  Local demo       : {is_local_demo}")
    print(f"  Simulated loss   : {effective_loss:.0%}")
    print(f"  Bitmask          : {use_bitmask}")
    print(f"  RandPerm         : {use_randperm}")
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
    transport_bytes_per_packet = transform.d_c * np.dtype(np.float16).itemsize + HEADER_SIZE
    if transport_bytes_per_packet > UDP_MAX_PAYLOAD:
        raise ValueError(
            f"UDP payload too large for P={num_packets}: {transport_bytes_per_packet} bytes per datagram. "
            f"Increase --num-packets so each packet fits under {UDP_MAX_PAYLOAD} bytes."
        )

    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ])

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8 * 1024 * 1024)
    # sock.bind(("0.0.0.0", port + 1))
    sock.bind(("", LOCAL_SEND_PORT))
    sock.settimeout(max(1.0, frame_timeout_ms / 1000.0 * 8.0))

    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    out_path = None
    writer = None
    if save_video:
        out_dir_path = Path(out_dir)
        out_dir_path.mkdir(parents=True, exist_ok=True)
        stem = Path(source).stem if not source.isdigit() else "webcam"
        out_path = out_dir_path / f"{stem}_stream_sender_annotated.mp4"
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 25.0
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (640, 640),
        )
        print(f"  [sender] Saving annotated video to {out_path}")

    rng = random.Random(seed)
    frame_id = 0
    sent_total = 0
    recv_total = 0
    timeout_total = 0
    e2e_ms_total = 0.0
    processed = 0

    if show:
        print("  Press Q to quit.\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if max_frames > 0 and frame_id >= max_frames:
            break

        display_frame = cv2.resize(frame, (640, 640))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = preprocess(rgb)

        t0 = time.perf_counter()
        act = model.encoder(tensor.unsqueeze(0))
        enc_ms = (time.perf_counter() - t0) * 1e3

        packets_raw = transform.encode(act)
        send_order = list(range(num_packets))
        if is_local_demo:
            rng.shuffle(send_order)

        t_tx = time.perf_counter()
        n_sent = 0
        for idx in send_order:
            if effective_loss > 0 and rng.random() < effective_loss:
                continue
            payload = packets_raw[idx].astype(np.float16, copy=False).tobytes()
            header = struct.pack(HEADER_FMT, frame_id)
            print(f"Sending packet {idx+1}/{num_packets} for frame {frame_id} (payload {len(payload)} bytes) to {host}:{port}")
            sock.sendto(header + payload, (host, port))
            n_sent += 1
            if is_local_demo:
                print("Local demo, so sleep")
                time.sleep(0.0002)
        tx_ms = (time.perf_counter() - t_tx) * 1e3
        wait_start = time.perf_counter()

        result = None
        while True:
            try:
                data, _ = sock.recvfrom(65535)
                print(f"Received response for frame {frame_id} ({len(data)} bytes)")
            except socket.timeout:
                print(f"  Timeout waiting for response for frame {frame_id} after {frame_timeout_ms} ms.")
                break

            try:
                msg = json.loads(data.decode("utf-8"))
                print(f"  Received message: {msg}")
            except Exception:
                continue
            if int(msg.get("frame_id", -1)) != frame_id:
                continue
            result = msg
            break

        e2e_ms = (time.perf_counter() - wait_start) * 1e3 + enc_ms + tx_ms
        sent_total += n_sent
        processed += 1

        if result is None:
            timeout_total += 1
            annotated = display_frame.copy()
            cv2.putText(
                annotated,
                f"frame {frame_id} | timeout | sent {n_sent}/{num_packets}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
            )
            print(
                f"  frame={frame_id:05d}  enc={enc_ms:6.1f}ms  tx={tx_ms:6.1f}ms  "
                f"e2e={e2e_ms:7.1f}ms  sent={n_sent:>3}/{num_packets}  timeout"
            )
        else:
            n_recv = int(result.get("received_packets", 0))
            recv_total += n_recv
            align_ms = float(result.get("align_ms", 0.0))
            decode_ms = float(result.get("decode_ms", 0.0))
            server_ms = float(result.get("server_ms", 0.0))
            e2e_ms_total += e2e_ms

            annotated = _draw_detections(
                display_frame.copy(),
                result.get("detections", []),
                class_names=result.get("class_names", {}),
            )
            overlay = (
                f"frame {frame_id} | sent {n_sent}/{num_packets} | recv {n_recv}/{num_packets} "
                f"| loss {result.get('packet_loss_pct', 0.0):.1f}%"
            )
            cv2.putText(annotated, overlay, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(
                annotated,
                f"enc {enc_ms:.1f}ms tx {tx_ms:.1f}ms align {align_ms:.1f}ms dec {decode_ms:.1f}ms e2e {e2e_ms:.1f}ms",
                (10, 56),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2,
            )
            print(
                f"  frame={frame_id:05d}  enc={enc_ms:6.1f}ms  tx={tx_ms:6.1f}ms  "
                f"align={align_ms:6.1f}ms  dec={decode_ms:6.1f}ms  "
                f"e2e={e2e_ms:7.1f}ms  recv={n_recv:>3}/{num_packets}"
            )

        if writer is not None:
            writer.write(annotated)

        if show:
            cv2.imshow("YOLO Split Sender", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        frame_id += 1

    cap.release()
    if writer is not None:
        writer.release()
    if show:
        cv2.destroyAllWindows()
    sock.close()
    model.offload()

    avg_loss_pct = 100.0 * (1.0 - (recv_total / max(processed * num_packets, 1)))
    avg_e2e = e2e_ms_total / max(processed - timeout_total, 1)
    print(f"\n{'─'*72}")
    print("  Sender Summary")
    print(f"  Frames processed : {processed}")
    print(f"  Timeouts         : {timeout_total}")
    print(f"  Avg recv loss    : {avg_loss_pct:.2f}%")
    print(f"  Avg end-to-end   : {avg_e2e:.2f} ms  (successful frames)")
    if out_path is not None:
        print(f"  Saved video      : {out_path}")
    print(f"{'─'*72}")


def _draw_detections(frame: np.ndarray, detections, class_names) -> np.ndarray:
    names = {int(k): v for k, v in class_names.items()} if isinstance(class_names, dict) else {}
    for det in detections:
        if len(det) < 6:
            continue
        x1, y1, x2, y2, conf, cls_id = det
        cls_id = int(cls_id)
        label = names.get(cls_id, str(cls_id))
        pt1 = (int(round(x1)), int(round(y1)))
        pt2 = (int(round(x2)), int(round(y2)))
        cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{label} {conf:.2f}",
            (pt1[0], max(15, pt1[1] - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
    return frame
