"""
config.py  –  single dataclass that drives every script.
All paths, model names, and experiment knobs live here.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple


# ─────────────────────────────────────────────
# Central config
# ─────────────────────────────────────────────

@dataclass
class Config:
    # ── paths ─────────────────────────────────
    out_dir: str = "outputs"          # root for all saved files

    # ── model ─────────────────────────────────
    model_name: str = "mobilenet_v2"  # mobilenet_v2 | vit_b16 | yolo

    # ── chunking ──────────────────────────────
    num_packets: int = 32             # P  (configurable directly)
    mtu_bytes: int = 1024             # printed in summary; does NOT drive P

    # ── corpus ────────────────────────────────
    corpus_size: int = 5000           # M training samples
    query_size: int = 256             # validation samples per experiment

    # ── transforms ────────────────────────────
    use_bitmask: bool = True          # sign-masking on/off
    use_randperm: bool = True         # global permutation on/off

    # ── experiment ────────────────────────────
    loss_rates: Tuple[float, ...] = (0.0, 0.2, 0.4)
    num_random_trials: int = 10       # random baseline repetitions

    # ── FAISS ─────────────────────────────────
    faiss_nprobe: int = 8             # only relevant for IVF index

    # ── misc ──────────────────────────────────
    seed: int = 42
    batch_size: int = 32
    num_workers: int = 0
    device: str = "auto"              # auto | cpu | cuda

    # ── HuggingFace ───────────────────────────
    hf_token: str = field(default_factory=lambda: os.environ.get("HF_TOKEN", ""))

    # ── UDP streaming (YOLO) ──────────────────
    
    # LOCAL_RECV_PORT = 5005        # Receiving PORT
    # LOCAL_SEND_PORT = 5006        # Sending PORT
    # DEST_PORT = 5006			  # Receiver's PORT
    # dest_addr = "192.168.0.157"   # Receiver's IP address (laptop)

    udp_host: str = "192.168.0.157"   # Receiver's IP address (laptop)
    udp_port: int = 5006              # Receiver's PORT
    udp_loss_rate: float = 0.0        # simulated loss on sender side
    video_source: str = "0"           # "0" = webcam, or path to video file
    frame_timeout_ms: int = 120
    max_frames: int = 0
    stream_show: bool = True
    stream_save_video: bool = False

    def resolve_device(self) -> str:
        import torch
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    def model_out_dir(self) -> Path:
        return Path(self.out_dir) / self.model_name

    def corpus_path(self) -> Path:
        return self.model_out_dir() / "corpus_raw.npy"

    def corpus_labels_path(self) -> Path:
        return self.model_out_dir() / "corpus_labels.npy"

    def transform_path(self) -> Path:
        tag = f"bm{int(self.use_bitmask)}_rp{int(self.use_randperm)}"
        return self.model_out_dir() / f"transform_{tag}.npz"

    def transformed_corpus_path(self) -> Path:
        tag = f"bm{int(self.use_bitmask)}_rp{int(self.use_randperm)}"
        return self.model_out_dir() / f"corpus_{tag}.npy"


# ─────────────────────────────────────────────
# Argument parser  (used by every CLI script)
# ─────────────────────────────────────────────

def make_parser(description: str):
    import argparse

    d=Config()

    default_loss_str = ",".join(str(x) for x in d.loss_rates)

    p = argparse.ArgumentParser(description=description)
    p.add_argument("--model",        default=d.model_name,
                   choices=["mobilenet_v2", "vit_b16", "yolo"],
                   help="Which split model to use")
    p.add_argument("--out-dir",      default=d.out_dir,
                   help="Root output directory")
    p.add_argument("--num-packets",  type=int,   default=d.num_packets,
                   help="Number of packets P to split activation into")
    p.add_argument("--corpus-size",  type=int,   default=d.corpus_size)
    p.add_argument("--query-size",   type=int,   default=d.query_size)
    p.add_argument("--loss-rates",   default=default_loss_str,
                   help="Comma-separated packet loss rates, e.g. 0.0,0.2,0.4,0.6")
    p.add_argument("--bitmask",      action="store_true",  default=d.use_bitmask)
    p.add_argument("--no-bitmask",   dest="bitmask", action="store_false")
    p.add_argument("--randperm",     action="store_true",  default=d.use_randperm)
    p.add_argument("--no-randperm",  dest="randperm", action="store_false")
    p.add_argument("--seed",         type=int,   default=d.seed)
    p.add_argument("--batch-size",   type=int,   default=d.batch_size)
    p.add_argument("--device",       default=d.device)
    p.add_argument("--hf-token",     default=d.hf_token,
                   help="HuggingFace token (or set HF_TOKEN env var)")
    # UDP / streaming args
    p.add_argument("--udp-host",     default=d.udp_host)
    p.add_argument("--udp-port",     type=int,   default=d.udp_port)
    p.add_argument("--udp-loss",     type=float, default=d.udp_loss_rate)
    p.add_argument("--source",       default=d.video_source,
                   help="Video source: '0' for webcam or path to video file")
    p.add_argument("--frame-timeout-ms", type=int, default=d.frame_timeout_ms,
                   help="Receiver frame assembly timeout in milliseconds")
    p.add_argument("--max-frames",   type=int, default=d.max_frames,
                   help="Process at most this many frames in streaming mode; 0 means unlimited")
    p.add_argument("--save-video",   action="store_true", default=d.stream_save_video,
                   help="Save annotated sender video to disk")
    p.add_argument("--no-show",      dest="show", action="store_false", default=d.stream_show,
                   help="Disable OpenCV display windows")
    return p


def config_from_args(args) -> Config:
    loss_rates = tuple(float(x) for x in args.loss_rates.split(",") if x.strip())
    hf = args.hf_token or os.environ.get("HF_TOKEN", "")
    return Config(
        out_dir=args.out_dir,
        model_name=args.model,
        num_packets=args.num_packets,
        corpus_size=args.corpus_size,
        query_size=args.query_size,
        loss_rates=loss_rates,
        use_bitmask=args.bitmask,
        use_randperm=args.randperm,
        seed=args.seed,
        batch_size=args.batch_size,
        device=args.device,
        hf_token=hf,
        udp_host=args.udp_host,
        udp_port=args.udp_port,
        udp_loss_rate=args.udp_loss,
        video_source=args.source,
        frame_timeout_ms=args.frame_timeout_ms,
        max_frames=args.max_frames,
        stream_show=args.show,
        stream_save_video=args.save_video,
    )
