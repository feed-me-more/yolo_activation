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
import pickle
from torchvision import transforms
from compare_accuracy import compare_detections, average

HEADER_FMT = "!I I d"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
LOCAL_HOSTS = {"127.0.0.1", "localhost", "0.0.0.0"}
UDP_MAX_PAYLOAD = 65507


LOCAL_RECV_PORT = 5005        # Receiving PORT
LOCAL_SEND_PORT = 5006        # Sending PORT
DEST_PORT = 5006			  # Receiver's PORT
# dest_addr = "192.168.0.157"  # Receiver's IP address (laptop)
dest_addr = "10.51.25.143"

# Local
# dest_addr = "192.168.0.136"
# LOCAL_RECV_PORT = 5002        # Receiving PORT
# LOCAL_SEND_PORT = 5001        # Sending PORT
# DEST_PORT = 5003			  # Receiver's PORT


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

    port = DEST_PORT
    host = dest_addr

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

    sock_rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_rx.bind(("", LOCAL_RECV_PORT))
    sock_rx.settimeout(0.8)

    start_time = time.perf_counter()

    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    max_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    out_path = "with_corpus_ooo_frame.mp4"
    writer = None
    save_video = True
    if save_video:
        out_dir_path = Path(out_dir)
        out_dir_path.mkdir(parents=True, exist_ok=True)
        stem = Path(source).stem if not source.isdigit() else "webcam"
        out_path = out_dir_path / f"{stem}_stream_sender_annotated.mp4"
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 30.0
        writer = cv2.VideoWriter(
            str(out_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (640, 360),
        )
        print(f"  [sender] Saving annotated video to {out_path}")

    rng = random.Random(seed)
    frame_id = 1
    sent_total = 0
    recv_total = 0
    timeout_total = 0
    e2e_ms_total = 0.0
    server_total = 0.0
    align_total = 0.0
    decode_total = 0.0
    processed = 0
    out_of_order_frame = 0
    seen_out_of_order = set()
    detections_by_frame = {}
    process_frame_total = 0.0

    if show:
        print("  Press Q to quit.\n")

    while True:
        ok, frame = cap.read()
        if not ok or (max_frames > 0 and frame_id > max_frames):
            header = struct.pack(HEADER_FMT, 4294967295, 0, time.time())
            sock.sendto(header, (host, port))
            break
        
        frame_taken = time.perf_counter()

        display_frame = cv2.resize(frame, (640, 640))
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        tensor = preprocess(rgb)        # Converts RGB -> PIL -> Tensor (C,H,W) in [0,1] (normalizes float tensor)

        t0 = time.perf_counter()
        act = model.encoder(tensor.unsqueeze(0))        # Encodes the preprocessed image into split activations, flattens all feature maps into a single vector of shape (1, d)
        enc_ms = (time.perf_counter() - t0) * 1e3

        packets_raw = transform.encode(act)             # Maps the activation vector into P packets of shape (d_c,) each, using the configured bitmask and/or random permutation
        send_order = list(range(num_packets))
        # print(f"Send order for frame {frame_id}: {send_order}")
        if is_local_demo:
            rng.shuffle(send_order)

        t_tx = time.perf_counter()
        n_sent = 0
        for idx in send_order:
            if effective_loss > 0 and rng.random() < effective_loss:
                continue
            payload = packets_raw[idx].astype(np.float16, copy=False).tobytes()
            header = struct.pack(HEADER_FMT, frame_id, idx, time.time())
            sock.sendto(header + payload, (host, port))
            n_sent += 1
            if is_local_demo:
                print("Local demo, so sleep")
                time.sleep(0.0002)

        print(f"Sent for frame {frame_id}")

        tx_ms = (time.perf_counter() - t_tx) * 1e3
        wait_start = time.perf_counter()

        result = None
        while True:
            try:
                data, _ = sock_rx.recvfrom(65535)
            except socket.timeout:
                print(f"Timeout waiting for response for frame {frame_id} after {sock_rx.gettimeout()*1e3:.2f} ms.")
                break

            try:
                msg = json.loads(data.decode("utf-8"))
            except Exception:
                continue
            if int(msg.get("frame_id", -1)) != frame_id and int(msg.get("frame_id", -1)) not in seen_out_of_order:
                out_of_order_frame +=1
                seen_out_of_order.add(int(msg.get("frame_id", -1)))
                print(f"Received response for frame {msg.get('frame_id')} while waiting for frame {frame_id}, storing.")
                continue
            result = msg
            one_frame_stop_time = time.perf_counter() - wait_start
            print(f"Received response for frame {frame_id} after {one_frame_stop_time*1e3:.2f} ms.")
            
            break

        e2e_ms = (time.perf_counter() - wait_start) * 1e3 + enc_ms + tx_ms

        frame_processed_time = time.perf_counter() - frame_taken
        print(f"Frame {frame_id} processed in {frame_processed_time*1e3:.2f} ms.")

        sent_total += n_sent
        processed += 1

        if result is None:
            timeout_total += 1
            annotated = display_frame.copy()
            annotated = cv2.resize(annotated, (640, 360))
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
            processing_ms = float(result.get("processing_ms", 0.0))
            server_ms = float(result.get("server_ms", 0.0))
            e2e_ms_total += e2e_ms
            server_total += server_ms
            align_total += align_ms
            decode_total += decode_ms
            process_frame_total += processing_ms

            annotated, recv_detections = _draw_detections(
                display_frame.copy(),
                result.get("detections", []),
                class_names=result.get("class_names", {}),
            )

            detections_by_frame[frame_id] = np.array(recv_detections, dtype=np.float32) if recv_detections else np.zeros((0, 6), dtype=np.float32)

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
                f"  frame={frame_id:05d}  enc={enc_ms:6.2f}ms  tx={tx_ms:6.2f}ms  "
                f"align={align_ms:6.2f}ms  dec={decode_ms:6.2f}ms  "
                f"e2e={e2e_ms:7.2f}ms  server={server_ms:6.2f}ms  process_frame={processing_ms:6.2f}ms  recv={n_recv:>3}/{num_packets}"
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
    avg_decode = decode_total / max(processed - timeout_total, 1)
    avg_align = align_total / max(processed - timeout_total, 1)
    avg_server = server_total / max(processed - timeout_total, 1)
    avg_process_frame = process_frame_total / max(processed - timeout_total, 1)
    total_time = time.perf_counter() - start_time
    
    result_metric = []

    with open("reference_dets.pkl", "rb") as f:
        reference_dets = pickle.load(f)

    for frame_id, dets in detections_by_frame.items():
        ref_detections_by_frame = reference_dets.get(frame_id, np.zeros((0, 6), dtype=np.float32))
        metrics = compare_detections(dets, ref_detections_by_frame)
        result_metric.append({"frame_id": frame_id, **metrics})

    average_metrics = average(result_metric)

    print(f"\n{'─'*72}")
    print("  Sender Summary")
    print(f"  Frames processed : {processed}")
    print(f"  Timeouts         : {timeout_total}")
    print(f"Out of order frames: {out_of_order_frame}")
    print(f"  Avg recv loss    : {avg_loss_pct:.2f}%")
    print(f"  Avg align time   : {avg_align:.2f} ms")
    print(f"  Avg decode time  : {avg_decode:.2f} ms")
    print(f"  Avg server time  : {avg_server:.2f} ms")
    print(f"  Avg process frame: {avg_process_frame:.2f} ms")
    print(f"  Avg end-to-end   : {avg_e2e:.2f} ms  (successful frames)")
    print(f"    Total time     : {total_time:.2f}")
    if out_path is not None:
        print(f"  Saved video      : {out_path}")
    print(f"Average metrics over {len(result_metric)} frames: {average_metrics}")
    print(f"{'─'*72}")


def _draw_detections(frame: np.ndarray, detections, class_names) -> np.ndarray:
    names = {int(k): v for k, v in class_names.items()} if isinstance(class_names, dict) else {}
    recv_detections = []
    for det in detections:
        if len(det) < 6:
            continue
        x1, y1, x2, y2, conf, cls_id = det
        recv_detections.append(det)
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
    frame = cv2.resize(frame, (640, 360))
    return frame, recv_detections
