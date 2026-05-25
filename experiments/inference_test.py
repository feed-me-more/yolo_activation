"""
experiments/inference_test.py  –  Test 2: inference accuracy.

5 conditions per loss rate:
  normal        – clean, no loss, no permutation
  oracle        – true positions known, lost chunks zeroed
  realigned     – our algorithm, lost chunks zeroed
  reconstructed – our algorithm, lost chunks filled from corpus
  random        – average of N random position assignments, lost chunks zeroed
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
from alignment.hungarian import assign_positions
from utils.packet import (
    PacketTransform, activation_to_packets,
    simulate_loss, normalise_chunks, reconstruct_activation,
)
from utils.plotting import (
    plot_inference_bars, plot_phase_transition,
    plot_alignment_time, print_summary_table,
)
import json


def run_inference_test(
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
    num_random_trials: int = 10,
) -> List[Dict[str, Any]]:

    out_path = Path(out_dir) / model_name
    out_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed + 99)

    # ── 1. Corpus ──────────────────────────────
    corpus_raw, corpus_labels = build_corpus(
        model_name=model_name,
        num_packets=num_packets,
        device=device,
        corpus_size=corpus_size,
        out_dir=out_dir,
        batch_size=batch_size,
        hf_token=hf_token,
    )

    # ── 2. Transform ───────────────────────────
    transform, corpus_transformed = apply_transform(
        model_name=model_name,
        num_packets=num_packets,
        out_dir=out_dir,
        use_bitmask=use_bitmask,
        use_randperm=use_randperm,
        seed=seed,
    )

    # ── 3. FAISS index ─────────────────────────
    index = build_index(corpus_transformed, device=device)

    # ── 4. Model ───────────────────────────────
    model = load_model(model_name, num_packets, device)
    model.dims.print()

    # ── 5. Raw corpus chunks for reconstruction
    # shape (M, P, d_c) unmasked
    # We store the raw corpus chunks (without sign mask) for imputation
    P, d_c = num_packets, transform.d_c
    corpus_chunks_raw = corpus_raw.reshape(corpus_raw.shape[0], P, d_c)  # (M, P, d_c)

    # ── 6. Query loader ────────────────────────
    loader = _get_query_loader(model_name, query_size, batch_size, hf_token)

    all_results = []

    for lr in loss_rates:
        print(f"\n{'─'*60}")
        print(f"  [inference_test]  model={model_name}  loss_rate={lr:.2f}")
        print(f"{'─'*60}")

        records: List[Dict[str, Any]] = []
        t_start = time.perf_counter()
        n_done  = 0

        for batch_x, batch_y in loader:
            for i in range(len(batch_x)):
                if n_done >= query_size:
                    break

                x   = batch_x[i:i+1]
                lbl = int(batch_y[i])

                act = model.encoder(x)                # (d,)

                # ── Normal (clean) ─────────────
                logits_normal = model.decoder(act)
                pred_normal   = int(logits_normal.argmax())

                # ── Packets ────────────────────
                packets = activation_to_packets(act, transform)  # (P, d_c)
                received, true_pos = simulate_loss(packets, lr, rng)
                n_recv = len(received)

                # ── Oracle (true positions, zero fill) ─
                act_oracle = _fill_activation(
                    received, true_pos, None, transform, "zeros", P, d_c
                )
                logits_oracle = model.decoder(act_oracle)
                pred_oracle   = int(logits_oracle.argmax())

                # ── Alignment (Stage 1 + 2) ────
                t0 = time.perf_counter()
                if n_recv > 0:
                    m_star, benefit, _ = index.retrieve(received)
                    pred_pos = assign_positions(benefit)
                else:
                    m_star, pred_pos = 0, np.array([], dtype=np.int32)
                align_ms = (time.perf_counter() - t0) * 1e3

                # ── Realigned (predicted positions, zero fill) ─
                act_realigned = _fill_activation(
                    received, pred_pos, None, transform, "zeros", P, d_c
                )
                logits_realigned = model.decoder(act_realigned)
                pred_realigned   = int(logits_realigned.argmax())

                # ── Reconstructed (predicted positions, corpus fill) ─
                corpus_entry = corpus_chunks_raw[m_star]   # (P, d_c)
                act_reconstructed = _fill_activation(
                    received, pred_pos, corpus_entry, transform, "corpus", P, d_c
                )
                logits_reconstructed = model.decoder(act_reconstructed)
                pred_reconstructed   = int(logits_reconstructed.argmax())

                # ── Random (avg of N trials) ───
                rand_correct = []
                for _ in range(num_random_trials):
                    rand_pos = _random_positions(n_recv, P, rng)
                    act_rand = _fill_activation(
                        received, rand_pos, None, transform, "zeros", P, d_c
                    )
                    logits_rand = model.decoder(act_rand)
                    rand_correct.append(int(logits_rand.argmax()) == lbl)
                rand_top1 = float(np.mean(rand_correct))

                records.append({
                    "label":           lbl,
                    "normal_top1":     float(pred_normal   == lbl),
                    "oracle_top1":     float(pred_oracle   == lbl),
                    "realigned_top1":  float(pred_realigned == lbl),
                    "reconstructed_top1": float(pred_reconstructed == lbl),
                    "random_top1":     rand_top1,
                    "align_ms":        align_ms,
                    "n_recv":          n_recv,
                })
                n_done += 1

                if n_done % 50 == 0:
                    _print_progress(n_done, query_size, records, lr)

            if n_done >= query_size:
                break

        elapsed = time.perf_counter() - t_start
        result  = _aggregate(model_name, lr, records, elapsed)
        all_results.append(result)

        # Save per-loss JSON
        (out_path / f"inference_{lr:.2f}.json").write_text(
            json.dumps(result, indent=2)
        )

    model.offload()

    # ── 7. Plots ───────────────────────────────
    plot_inference_bars(all_results, out_path, model_name)
    plot_phase_transition(all_results, out_path, model_name)
    plot_alignment_time(all_results, out_path, model_name)

    # ── 8. Terminal table ──────────────────────
    print_summary_table(all_results, model_name)

    # Save combined JSON
    (out_path / "inference_all.json").write_text(
        json.dumps(all_results, indent=2)
    )

    return all_results


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _fill_activation(
    received: np.ndarray,      # (n_recv, d_c)
    positions: np.ndarray,     # (n_recv,) predicted or true positions
    corpus_entry,              # (P, d_c) or None
    transform: PacketTransform,
    fill: str,                 # "zeros" | "corpus"
    P: int,
    d_c: int,
) -> np.ndarray:
    """Build full (d,) activation for decoder input."""
    full_chunks = np.zeros((P, d_c), dtype=np.float32)

    if fill == "corpus" and corpus_entry is not None:
        full_chunks[:] = corpus_entry

    for i, j in enumerate(positions):
        if 0 <= j < P:
            full_chunks[j] = transform.decode_chunk(received[i], j)

    return full_chunks.flatten()


def _random_positions(n_recv: int, P: int, rng: np.random.Generator) -> np.ndarray:
    """Sample a random injective mapping from n_recv packets to P positions."""
    return rng.choice(P, size=n_recv, replace=False).astype(np.int32)


def _aggregate(
    model_name: str,
    lr: float,
    records: List[Dict],
    elapsed: float,
) -> Dict[str, Any]:
    def mean(key): return float(np.mean([r[key] for r in records]) * 100)
    return {
        "model_name":           model_name,
        "loss_rate":            lr,
        "normal_top1":          mean("normal_top1"),
        "oracle_top1":          mean("oracle_top1"),
        "realigned_top1":       mean("realigned_top1"),
        "reconstructed_top1":   mean("reconstructed_top1"),
        "random_top1":          mean("random_top1"),
        "mean_alignment_ms":    float(np.mean([r["align_ms"] for r in records])),
        "all_positions_correct_pct": 0.0,   # filled by alignment_test
        "position_fraction_pct":     0.0,
        "n_queries":            len(records),
        "elapsed_s":            elapsed,
    }


def _print_progress(n_done, total, records, lr):
    def m(k): return np.mean([r[k] for r in records]) * 100
    print(
        f"  [{n_done}/{total}  loss={lr:.2f}]  "
        f"normal={m('normal_top1'):.1f}  "
        f"oracle={m('oracle_top1'):.1f}  "
        f"realign={m('realigned_top1'):.1f}  "
        f"reconstr={m('reconstructed_top1'):.1f}  "
        f"random={m('random_top1'):.1f}"
    )


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
