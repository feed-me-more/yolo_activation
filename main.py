"""
main.py  –  unified CLI for all split inference experiments.

Usage examples
──────────────

# Build corpus (MobileNetV2, 5000 samples)
python main.py build-corpus --model mobilenet_v2 --corpus-size 5000 --hf-token $HF_TOKEN

# Apply transform (bitmask ON, randperm ON)
python main.py transform --model mobilenet_v2

# Apply transform with only bitmask (no randperm)
python main.py transform --model mobilenet_v2 --no-randperm

# Run alignment quality test
python main.py align --model mobilenet_v2 --loss-rates 0.0,0.2,0.4,0.6

# Run inference accuracy test
python main.py infer --model mobilenet_v2 --loss-rates 0.0,0.2,0.4

# Run both tests together
python main.py run --model mobilenet_v2 --loss-rates 0.0,0.2,0.4

# UDP streaming sender (YOLO)
python main.py stream-send --source 0 --udp-host 127.0.0.1 --udp-port 9999

# UDP streaming receiver (YOLO, run in separate terminal)
python main.py stream-recv --udp-port 9999
"""
from __future__ import annotations

import sys
import os


def _banner():
    print("""
╔══════════════════════════════════════════════════╗
║   Split Inference  –  Metadata-Free Alignment    ║
╚══════════════════════════════════════════════════╝
""")


def main():
    import argparse
    from config import make_parser, config_from_args

    # ── Top-level command parser ───────────────
    top = argparse.ArgumentParser(
        description="Split inference with metadata-free alignment",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    sub = top.add_subparsers(dest="command", required=True)

    # ── Sub-commands share common args ────────
    for name, help_str in [
        ("build-corpus", "Build raw activation corpus from training data"),
        ("transform",    "Apply bitmask / randperm to existing corpus"),
        ("align",        "Test 1: alignment quality (position recovery)"),
        ("infer",        "Test 2: inference accuracy under packet loss"),
        ("run",          "Run both align and infer tests"),
        ("stream-send",  "YOLO UDP sender (encode + stream)"),
        ("stream-recv",  "YOLO UDP receiver (align + decode)"),
    ]:
        p = sub.add_parser(name, help=help_str, parents=[make_parser(help_str)],
                           add_help=False)

    args = top.parse_args()
    cfg  = config_from_args(args)
    # cfg = Config()

    _banner()
    _print_config(cfg)
    _setup_seed(cfg.seed)

    dev = cfg.resolve_device()
    print(f"  Device : {dev}\n")

    cmd = args.command

    # ─────────────────────────────────────────
    if cmd == "build-corpus":
        from corpus.build import build_corpus
        build_corpus(
            model_name   = cfg.model_name,
            num_packets  = cfg.num_packets,
            device       = dev,
            corpus_size  = cfg.corpus_size,
            out_dir      = cfg.out_dir,
            batch_size   = cfg.batch_size,
            hf_token     = cfg.hf_token,
            seed         = cfg.seed,
        )

    # ─────────────────────────────────────────
    elif cmd == "transform":
        from corpus.transform import apply_transform
        apply_transform(
            model_name   = cfg.model_name,
            num_packets  = cfg.num_packets,
            out_dir      = cfg.out_dir,
            use_bitmask  = cfg.use_bitmask,
            use_randperm = cfg.use_randperm,
            seed         = cfg.seed,
        )

    # ─────────────────────────────────────────
    elif cmd == "align":
        from experiments.alignment_test import run_alignment_test
        run_alignment_test(
            model_name   = cfg.model_name,
            num_packets  = cfg.num_packets,
            device       = dev,
            corpus_size  = cfg.corpus_size,
            query_size   = cfg.query_size,
            loss_rates   = cfg.loss_rates,
            out_dir      = cfg.out_dir,
            use_bitmask  = cfg.use_bitmask,
            use_randperm = cfg.use_randperm,
            seed         = cfg.seed,
            batch_size   = cfg.batch_size,
            hf_token     = cfg.hf_token,
        )

    # ─────────────────────────────────────────
    elif cmd == "infer":
        from experiments.inference_test import run_inference_test
        run_inference_test(
            model_name         = cfg.model_name,
            num_packets        = cfg.num_packets,
            device             = dev,
            corpus_size        = cfg.corpus_size,
            query_size         = cfg.query_size,
            loss_rates         = cfg.loss_rates,
            out_dir            = cfg.out_dir,
            use_bitmask        = cfg.use_bitmask,
            use_randperm       = cfg.use_randperm,
            seed               = cfg.seed,
            batch_size         = cfg.batch_size,
            hf_token           = cfg.hf_token,
            num_random_trials  = cfg.num_random_trials,
        )

    # ─────────────────────────────────────────
    elif cmd == "run":
        from experiments.alignment_test import run_alignment_test
        from experiments.inference_test import run_inference_test

        print("━" * 60)
        print("  STEP 1 / 2  –  Alignment test")
        print("━" * 60)
        run_alignment_test(
            model_name   = cfg.model_name,
            num_packets  = cfg.num_packets,
            device       = dev,
            corpus_size  = cfg.corpus_size,
            query_size   = cfg.query_size,
            loss_rates   = cfg.loss_rates,
            out_dir      = cfg.out_dir,
            use_bitmask  = cfg.use_bitmask,
            use_randperm = cfg.use_randperm,
            seed         = cfg.seed,
            batch_size   = cfg.batch_size,
            hf_token     = cfg.hf_token,
        )

        print("\n" + "━" * 60)
        print("  STEP 2 / 2  –  Inference test")
        print("━" * 60)
        run_inference_test(
            model_name         = cfg.model_name,
            num_packets        = cfg.num_packets,
            device             = dev,
            corpus_size        = cfg.corpus_size,
            query_size         = cfg.query_size,
            loss_rates         = cfg.loss_rates,
            out_dir            = cfg.out_dir,
            use_bitmask        = cfg.use_bitmask,
            use_randperm       = cfg.use_randperm,
            seed               = cfg.seed,
            batch_size         = cfg.batch_size,
            hf_token           = cfg.hf_token,
            num_random_trials  = cfg.num_random_trials,
        )

    # ─────────────────────────────────────────
    elif cmd == "stream-send":
        from streaming.sender import run_sender
        run_sender(
            source       = cfg.video_source,
            host         = cfg.udp_host,
            port         = cfg.udp_port,
            num_packets  = cfg.num_packets,
            udp_loss     = cfg.udp_loss_rate,
            seed         = cfg.seed,
            use_bitmask  = cfg.use_bitmask,
            use_randperm = cfg.use_randperm,
            out_dir      = cfg.out_dir,
            device       = dev,
            frame_timeout_ms = cfg.frame_timeout_ms,
            max_frames   = cfg.max_frames,
            show         = cfg.stream_show,
            save_video   = cfg.stream_save_video,
        )

    # ─────────────────────────────────────────
    elif cmd == "stream-recv":
        from streaming.receiver import run_receiver
        run_receiver(
            host         = "0.0.0.0",
            port         = cfg.udp_port,
            num_packets  = cfg.num_packets,
            out_dir      = cfg.out_dir,
            device       = dev,
            corpus_size  = cfg.corpus_size,
            use_bitmask  = cfg.use_bitmask,
            use_randperm = cfg.use_randperm,
            seed         = cfg.seed,
            frame_timeout_ms = cfg.frame_timeout_ms,
        )


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _print_config(cfg) -> None:
    print("  Configuration")
    print(f"  {'─'*40}")
    print(f"  model        : {cfg.model_name}")
    print(f"  num_packets  : {cfg.num_packets}   (P)")
    print(f"  corpus_size  : {cfg.corpus_size}   (M)")
    print(f"  query_size   : {cfg.query_size}")
    print(f"  loss_rates   : {cfg.loss_rates}")
    print(f"  use_bitmask  : {cfg.use_bitmask}")
    print(f"  use_randperm : {cfg.use_randperm}")
    print(f"  out_dir      : {cfg.out_dir}")
    print(f"  seed         : {cfg.seed}")
    print(f"  {'─'*40}\n")


def _setup_seed(seed: int) -> None:
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
