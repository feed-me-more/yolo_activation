"""
utils/plotting.py  –  all visualisation in one place.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Any

import numpy as np


# ─────────────────────────────────────────────
# Inference accuracy bar chart
# ─────────────────────────────────────────────

def plot_inference_bars(results: List[Dict[str, Any]], out_dir: Path, model_name: str) -> None:
    """
    One subplot per loss rate.  5 bars: Normal / Oracle / Realigned /
    Reconstructed / Random.
    """
    import matplotlib.pyplot as plt

    loss_rates = sorted({r["loss_rate"] for r in results})
    n = len(loss_rates)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5), sharey=True)
    if n == 1:
        axes = [axes]

    conditions  = ["normal", "oracle", "realigned", "reconstructed", "random"]
    labels      = ["Normal", "Oracle", "Realigned", "Reconstructed", "Random"]
    colors      = ["#4C78A8", "#59A14F", "#F28E2B", "#76B7B2", "#E15759"]

    row_by_loss = {r["loss_rate"]: r for r in results}

    for ax, lr in zip(axes, loss_rates):
        row = row_by_loss[lr]
        vals = [row[f"{c}_top1"] for c in conditions]
        bars = ax.bar(labels, vals, color=colors, width=0.6)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.8,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=8)
        ax.set_ylim(0, 105)
        ax.set_title(f"loss = {lr:.0%}", fontsize=11)
        ax.set_ylabel("Top-1 accuracy (%)")
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="x", rotation=25)

    fig.suptitle(f"{model_name}  –  inference accuracy under packet loss", fontsize=13)
    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{model_name}_inference.png", dpi=150)
    fig.savefig(out_dir / f"{model_name}_inference.pdf")
    plt.close(fig)
    print(f"  [plot] saved {out_dir}/{model_name}_inference.png")


# ─────────────────────────────────────────────
# Alignment quality bar chart
# ─────────────────────────────────────────────

def plot_alignment_bars(results: List[Dict[str, Any]], out_dir: Path, model_name: str) -> None:
    """
    Two metrics per loss rate:
      - all-positions-correct rate  (%)
      - per-chunk position accuracy (%)
    """
    import matplotlib.pyplot as plt

    loss_rates  = sorted({r["loss_rate"] for r in results})
    row_by_loss = {r["loss_rate"]: r for r in results}

    all_correct = [row_by_loss[lr]["all_positions_correct_pct"] for lr in loss_rates]
    pos_frac    = [row_by_loss[lr]["position_fraction_pct"]     for lr in loss_rates]

    x     = np.arange(len(loss_rates))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(6, 2 * len(loss_rates)), 4))

    b1 = ax.bar(x - width / 2, all_correct, width, label="All positions correct", color="#4C78A8")
    b2 = ax.bar(x + width / 2, pos_frac,    width, label="Per-chunk accuracy",    color="#F28E2B")

    for bar in list(b1) + list(b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                f"{h:.1f}", ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{lr:.0%}" for lr in loss_rates])
    ax.set_xlabel("Packet loss rate")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 110)
    ax.set_title(f"{model_name}  –  alignment quality")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{model_name}_alignment.png", dpi=150)
    fig.savefig(out_dir / f"{model_name}_alignment.pdf")
    plt.close(fig)
    print(f"  [plot] saved {out_dir}/{model_name}_alignment.png")


# ─────────────────────────────────────────────
# Phase-transition line plot
# ─────────────────────────────────────────────

def plot_phase_transition(results: List[Dict[str, Any]], out_dir: Path, model_name: str) -> None:
    """
    Accuracy vs loss-rate for all 5 conditions on one plot.
    """
    import matplotlib.pyplot as plt

    loss_rates  = sorted({r["loss_rate"] for r in results})
    row_by_loss = {r["loss_rate"]: r for r in results}

    conditions = ["normal", "oracle", "realigned", "reconstructed", "random"]
    labels     = ["Normal", "Oracle", "Realigned", "Reconstructed", "Random"]
    colors     = ["#4C78A8", "#59A14F", "#F28E2B", "#76B7B2", "#E15759"]
    markers    = ["o", "s", "^", "D", "x"]

    fig, ax = plt.subplots(figsize=(8, 5))
    for cond, lbl, col, mk in zip(conditions, labels, colors, markers):
        vals = [row_by_loss[lr][f"{cond}_top1"] for lr in loss_rates]
        ax.plot(loss_rates, vals, marker=mk, label=lbl, color=col, linewidth=2)

    ax.set_xlabel("Packet loss rate", fontsize=12)
    ax.set_ylabel("Top-1 accuracy (%)", fontsize=12)
    ax.set_title(f"{model_name}  –  phase transition", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, 105)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{model_name}_phase_transition.png", dpi=150)
    fig.savefig(out_dir / f"{model_name}_phase_transition.pdf")
    plt.close(fig)
    print(f"  [plot] saved {out_dir}/{model_name}_phase_transition.png")


# ─────────────────────────────────────────────
# Alignment time line plot
# ─────────────────────────────────────────────

def plot_alignment_time(results: List[Dict[str, Any]], out_dir: Path, model_name: str) -> None:
    import matplotlib.pyplot as plt

    loss_rates  = sorted({r["loss_rate"] for r in results})
    row_by_loss = {r["loss_rate"]: r for r in results}

    align_ms   = [row_by_loss[lr]["mean_alignment_ms"] for lr in loss_rates]
    realign_t1 = [row_by_loss[lr]["realigned_top1"]    for lr in loss_rates]

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax2 = ax1.twinx()

    ax1.plot(loss_rates, realign_t1, marker="o", color="#F28E2B",
             linewidth=2, label="Realigned Top-1 (%)")
    ax2.plot(loss_rates, align_ms,   marker="s", color="#4C78A8",
             linewidth=2, label="Alignment time (ms)")

    ax1.set_xlabel("Packet loss rate", fontsize=11)
    ax1.set_ylabel("Realigned Top-1 (%)", color="#F28E2B", fontsize=11)
    ax2.set_ylabel("Alignment time (ms)", color="#4C78A8", fontsize=11)
    ax1.set_title(f"{model_name}  –  accuracy vs alignment latency", fontsize=12)
    ax1.grid(alpha=0.25)

    lines  = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels)

    fig.tight_layout()
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{model_name}_time_tradeoff.png", dpi=150)
    plt.close(fig)
    print(f"  [plot] saved {out_dir}/{model_name}_time_tradeoff.png")


# ─────────────────────────────────────────────
# Terminal summary table
# ─────────────────────────────────────────────

def print_summary_table(results: List[Dict[str, Any]], model_name: str) -> None:
    cols  = ["loss_rate", "normal_top1", "oracle_top1",
             "realigned_top1", "reconstructed_top1", "random_top1",
             "all_positions_correct_pct", "position_fraction_pct",
             "mean_alignment_ms"]
    hdr   = ["loss", "normal", "oracle", "realign", "reconstr",
             "random", "all_pos%", "pos_frac%", "align_ms"]
    width = [6, 8, 8, 9, 9, 8, 9, 10, 10]

    sep = "+" + "+".join("-" * (w + 2) for w in width) + "+"
    def row_str(vals):
        parts = []
        for v, w in zip(vals, width):
            s = f"{v:.2f}" if isinstance(v, float) else str(v)
            parts.append(f" {s:<{w}} ")
        return "|" + "|".join(parts) + "|"

    print(f"\n{'─'*80}")
    print(f"  Results: {model_name}")
    print(sep)
    print(row_str(hdr))
    print(sep)
    for r in sorted(results, key=lambda x: x["loss_rate"]):
        vals = [r[c] for c in cols]
        print(row_str(vals))
    print(sep)
