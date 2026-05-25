"""
experiments/alignment_test.py  –  Test 1: alignment quality.

Measures purely how well the algorithm recovers chunk positions,
independent of downstream model accuracy.

Metrics per loss rate:
  - all_positions_correct_pct  : fraction of queries where ALL positions recovered
  - position_fraction_pct      : average per-chunk position accuracy
  - mean_alignment_ms          : mean retrieval + assignment latency
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch

from models import load_model
from corpus.build import build_corpus
from corpus.transform import apply_transform
from alignment.faiss_index import build_index
from alignment.hungarian import assign_positions, position_accuracy
from utils.packet import PacketTransform, activate_to_packets, simulate_loss, normalise_chunks
from utils.plotting import plot_alignment_bars, print_summary_table


# helper alias so both spellings work
def activate_to_packets(act, transform):
    from utils.packet import activation_to_packets
    return activation_to_packets(act, transform)


def run_alignment_test(
    model_name: str,
    num_packets: int,
    device: str,
    corpus_size: int,
    query_size: int,
    loss_rates: tuple,
    out_dir: str,
    use_bitmask: bool,
    use_randperm: bool,
    seed: int,
    batch_size: int,
    hf_token: str,
) -> List[Dict[str, Any]]:

    out_path = Path(out_dir) / model_name
    rng = np.random.default_rng(seed + 1)

    # ── 1. Load / build corpus ─────────────────
    corpus_raw, corpus_labels = build_corpus(
        model_name=model_name,
        num_packets=num_packets,
        device=device,
        corpus_size=corpus_size,
        out_dir=out_dir,
        batch_size=batch_size,
        hf_token=hf_token,
    )

    # ── 2. Apply transform ─────────────────────
    transform, corpus_transformed = apply_transform(
        model_name=model_name,
        num_packets=num_packets,
        out_dir=out_dir,
        use_bitmask=use_bitmask,
        use_randperm=use_randperm,
        seed=seed,
    )

    # ── 3. Build FAISS index ───────────────────
    index = build_index(corpus_transformed, device=device)

    # ── 4. Load model for query encoding ──────
    model = load_model(model_name, num_packets, device)
    model.dims.print()

    # ── 5. Get query loader ────────────────────
    loader = _get_query_loader(
        model_name, query_size, batch_size, hf_token
    )

    all_results = []

    for lr in loss_rates:
        print(f"\n  [alignment_test] loss_rate={lr:.2f} ...")
        per_query = []
        t_start = time.perf_counter()

        for batch_x, _ in loader:
            for i in range(len(batch_x)):
                if len(per_query) >= query_size:
                    break

                # Encode
                act = model.encoder(batch_x[i:i+1])          # (d,)

                # Encode to packets (permute + chunk + normalise + sign-mask)
                packets = activate_to_packets(act, transform) # (P, d_c)

                # Simulate loss + reordering
                received, true_pos = simulate_loss(packets, lr, rng)
                if len(received) == 0:
                    continue

                # Stage 1: corpus retrieval
                t0 = time.perf_counter()
                m_star, benefit, ms = index.retrieve(received)
                # Stage 2: position assignment
                pred_pos = assign_positions(benefit)
                elapsed = (time.perf_counter() - t0) * 1e3

                all_c, frac_c = position_accuracy(pred_pos, true_pos)
                per_query.append({
                    "all_correct": all_c,
                    "frac_correct": frac_c,
                    "align_ms": elapsed,
                })

            if len(per_query) >= query_size:
                break

        result = {
            "model_name": model_name,
            "loss_rate": lr,
            "use_bitmask": use_bitmask,
            "use_randperm": use_randperm,
            "all_positions_correct_pct": float(
                np.mean([r["all_correct"] for r in per_query]) * 100),
            "position_fraction_pct": float(
                np.mean([r["frac_correct"] for r in per_query]) * 100),
            "mean_alignment_ms": float(
                np.mean([r["align_ms"] for r in per_query])),
            "n_queries": len(per_query),
        }
        all_results.append(result)
        elapsed_total = time.perf_counter() - t_start
        print(f"    all_correct={result['all_positions_correct_pct']:.1f}%  "
              f"pos_frac={result['position_fraction_pct']:.1f}%  "
              f"align_ms={result['mean_alignment_ms']:.2f}  "
              f"({elapsed_total:.1f}s total)")

    model.offload()

    # ── 6. Plots ───────────────────────────────
    plot_alignment_bars(all_results, out_path, model_name)

    # ── 7. Terminal table ──────────────────────
    _print_alignment_table(all_results)

    return all_results


def _print_alignment_table(results: List[Dict[str, Any]]) -> None:
    hdr = ["loss", "all_pos%", "pos_frac%", "align_ms", "n_queries"]
    w   = [6, 10, 10, 10, 10]
    sep = "+" + "+".join("-" * (x + 2) for x in w) + "+"
    def row(vals):
        parts = []
        for v, wd in zip(vals, w):
            s = f"{v:.2f}" if isinstance(v, float) else str(v)
            parts.append(f" {s:<{wd}} ")
        return "|" + "|".join(parts) + "|"
    print(f"\n{'─'*60}")
    print("  Alignment Test Results")
    print(sep)
    print(row(hdr))
    print(sep)
    for r in sorted(results, key=lambda x: x["loss_rate"]):
        print(row([r["loss_rate"], r["all_positions_correct_pct"],
                   r["position_fraction_pct"], r["mean_alignment_ms"],
                   r["n_queries"]]))
    print(sep)


def _get_query_loader(model_name, query_size, batch_size, hf_token):
    if model_name == "yolo":
        from models.yolo import get_coco128_loader
        return get_coco128_loader(batch_size=batch_size, image_size=640)

    from corpus.build import _get_imagenet_loader
    return _get_imagenet_loader(
        split="validation",
        n=query_size,
        batch_size=batch_size,
        image_size=224,
        hf_token=hf_token,
    )
