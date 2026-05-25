"""
alignment/hungarian.py  –  Stage 2: position assignment via bipartite matching.

Given the benefit matrix B of shape (P-k, P) from Stage 1, find the
injective assignment sigma : [P-k] -> [P] that maximises sum B[i, sigma(i)].

Uses scipy.optimize.linear_sum_assignment (Hungarian algorithm).
Falls back to greedy if scipy is unavailable.
"""
from __future__ import annotations

import numpy as np
from typing import Tuple


# ─────────────────────────────────────────────
# Hungarian (exact)
# ─────────────────────────────────────────────

def hungarian_assignment(benefit: np.ndarray) -> np.ndarray:
    """
    benefit : (n_recv, P)  – higher is better
    Returns assigned_positions : (n_recv,)  integer positions in [0, P)

    scipy.linear_sum_assignment minimises cost, so negate benefit.
    The matrix may be wide (n_recv < P); scipy handles this correctly.
    """
    try:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(-benefit)
        # row_ind will be [0, 1, ..., n_recv-1] since n_recv <= P
        return col_ind.astype(np.int32)
    except ImportError:
        return greedy_assignment(benefit)


# ─────────────────────────────────────────────
# Greedy (fast fallback)
# ─────────────────────────────────────────────

def greedy_assignment(benefit: np.ndarray) -> np.ndarray:
    """
    benefit : (n_recv, P)
    Greedy: assign each received packet to its argmax position,
    with conflict resolution (each position used at most once).
    """
    n_recv, P = benefit.shape
    assigned   = np.full(n_recv, -1, dtype=np.int32)
    used       = set()

    # Sort packets by their max-benefit score (most confident first)
    confidence = benefit.max(axis=1)
    order      = np.argsort(-confidence)

    for i in order:
        # Best available position for packet i
        sorted_pos = np.argsort(-benefit[i])
        for j in sorted_pos:
            if j not in used:
                assigned[i] = j
                used.add(j)
                break

    return assigned


# ─────────────────────────────────────────────
# Unified entry point
# ─────────────────────────────────────────────

def assign_positions(
    benefit: np.ndarray,    # (n_recv, P)
    method: str = "hungarian",
) -> np.ndarray:
    """
    Returns assigned_positions : (n_recv,)  in [0, P).
    method : "hungarian" | "greedy"
    """
    if method == "greedy":
        return greedy_assignment(benefit)
    return hungarian_assignment(benefit)


# ─────────────────────────────────────────────
# Accuracy helpers
# ─────────────────────────────────────────────

def position_accuracy(
    predicted: np.ndarray,   # (n_recv,)
    true_pos: np.ndarray,    # (n_recv,)
) -> Tuple[float, float]:
    """
    Returns
    -------
    all_correct   : 1.0 if all positions match, else 0.0
    frac_correct  : fraction of packets at correct position
    """
    correct      = (predicted == true_pos)
    frac_correct = float(correct.mean())
    all_correct  = float(correct.all())
    return all_correct, frac_correct
