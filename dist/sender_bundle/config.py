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
    num_packets: int = 16             # P  (configurable directly)
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
    udp_host: str = "127.0.0.1"
    udp_port: int = 9999
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

    p = argparse.ArgumentParser(description=description)
    p.add_argument("--model",        default="mobilenet_v2",
                   choices=["mobilenet_v2", "vit_b16", "yolo"],
                   help="Which split model to use")
    p.add_argument("--out-dir",      default="outputs",
                   help="Root output directory")
    p.add_argument("--num-packets",  type=int,   default=16,
                   help="Number of packets P to split activation into")
    p.add_argument("--corpus-size",  type=int,   default=5000)
    p.add_argument("--query-size",   type=int,   default=256)
    p.add_argument("--loss-rates",   default="0.0,0.2,0.4",
                   help="Comma-separated packet loss rates, e.g. 0.0,0.2,0.4,0.6")
    p.add_argument("--bitmask",      action="store_true",  default=True)
    p.add_argument("--no-bitmask",   dest="bitmask", action="store_false")
    p.add_argument("--randperm",     action="store_true",  default=True)
    p.add_argument("--no-randperm",  dest="randperm", action="store_false")
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--batch-size",   type=int,   default=32)
    p.add_argument("--device",       default="auto")
    p.add_argument("--hf-token",     default="",
                   help="HuggingFace token (or set HF_TOKEN env var)")
    # UDP / streaming args
    p.add_argument("--udp-host",     default="127.0.0.1")
    p.add_argument("--udp-port",     type=int,   default=9999)
    p.add_argument("--udp-loss",     type=float, default=0.0)
    p.add_argument("--source",       default="0",
                   help="Video source: '0' for webcam or path to video file")
    p.add_argument("--frame-timeout-ms", type=int, default=120,
                   help="Receiver frame assembly timeout in milliseconds")
    p.add_argument("--max-frames",   type=int, default=0,
                   help="Process at most this many frames in streaming mode; 0 means unlimited")
    p.add_argument("--save-video",   action="store_true", default=False,
                   help="Save annotated sender video to disk")
    p.add_argument("--no-show",      dest="show", action="store_false", default=True,
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
