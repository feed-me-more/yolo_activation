"""
utils/detection_compare.py  –  per-frame detection accuracy against a
self-generated reference (the sender's clean, undamaged-activation output).

No COCO ground truth needed. The "reference" is simply what the model would
have detected if every packet had arrived — computed once on the sender from
the full activation before any chunking/loss/reconstruction happens.

This answers: "how much did packet loss + corpus-fill degrade detection
quality, frame by frame, relative to perfect transmission?"

Metric returned per frame:
  precision   : of the boxes the receiver found, what fraction match a
                reference box (same class, IoU >= threshold)?
  recall      : of the reference boxes, what fraction did the receiver
                still find?
  f1          : harmonic mean of precision/recall
  mean_iou    : average IoU of matched pairs (how well boxes line up,
                independent of whether the box count was right)
  conf_delta  : mean confidence of receiver detections minus mean
                confidence of reference detections for matched pairs
                (are we less *sure* even when we get the box right?)
"""
from __future__ import annotations

import numpy as np
from typing import Dict, Any
from collections import defaultdict


def average(result_metric):

    n = len(result_metric)
    mean = ["precision", "recall", "f1", "mean_iou"]
    totals = defaultdict(float)
    sum_sq_conf = 0.0

    for metrics in result_metric:
        for key in mean:
            totals[f"avg_{key}"] += metrics[key]
        sum_sq_conf += metrics["conf_delta"] ** 2
        totals["total_diff"] += abs(metrics["diff"])
        totals["total_diff_matched"] += abs(metrics["diff_matched"])

    avg_metrics = {
        f"avg_{key}": totals[f"avg_{key}"] / n
        for key in mean
    }

    rmse_conf = (sum_sq_conf / n) ** 0.5
    avg_metrics["rmse_conf_delta"] = rmse_conf
    avg_metrics["total_diff"] = totals["total_diff"]
    avg_metrics["total_diff_matched"] = totals["total_diff_matched"]

    return avg_metrics


def _iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """boxes_a: (n,4), boxes_b: (m,4)  x1y1x2y2  →  (n, m) IoU matrix."""
    if boxes_a.shape[0] == 0 or boxes_b.shape[0] == 0:
        return np.zeros((boxes_a.shape[0], boxes_b.shape[0]), dtype=np.float32)

    a = boxes_a[:, None, :]   # (n,1,4)
    b = boxes_b[None, :, :]   # (1,m,4)

    inter_x1 = np.maximum(a[..., 0], b[..., 0])
    inter_y1 = np.maximum(a[..., 1], b[..., 1])
    inter_x2 = np.minimum(a[..., 2], b[..., 2])
    inter_y2 = np.minimum(a[..., 3], b[..., 3])

    inter_w = np.clip(inter_x2 - inter_x1, 0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0, None)
    inter = inter_w * inter_h

    area_a = np.clip(boxes_a[:, 2] - boxes_a[:, 0], 0, None) * np.clip(boxes_a[:, 3] - boxes_a[:, 1], 0, None)
    area_b = np.clip(boxes_b[:, 2] - boxes_b[:, 0], 0, None) * np.clip(boxes_b[:, 3] - boxes_b[:, 1], 0, None)
    union = area_a[:, None] + area_b[None, :] - inter

    return np.where(union > 0, inter / union, 0.0).astype(np.float32)


def compare_detections(
    received_det: np.ndarray,    # (n_recv, 6)  x1,y1,x2,y2,conf,cls  — what receiver decoded
    reference_det: np.ndarray,   # (n_ref, 6)  same format            — clean / undamaged decode
    iou_thresh: float = 0.5,
) -> Dict[str, Any]:
    """
    Greedy class-aware matching, highest-confidence reference box matched first
    (mirrors standard detection-eval convention). Pure numpy, no torch needed,
    cheap enough to run every frame.
    """
    n_recv = received_det.shape[0]
    n_ref = reference_det.shape[0]

    if n_ref == 0 and n_recv == 0:
        print("Nothing to compare")
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "mean_iou": 1.0,
                "conf_delta": 0.0, "n_received": 0, "n_reference": 0, "n_matched": 0, "diff": n_recv - n_ref, "diff_matched": n_ref}

    if n_ref == 0:
        # reference saw nothing; any receiver box is a false positive
        print("No reference data")
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0, "mean_iou": 0.0,
                "conf_delta": 0.0, "n_received": n_recv, "n_reference": 0, "n_matched": 0, "diff": n_recv - n_ref, "diff_matched": n_ref}

    if n_recv == 0:
        # receiver found nothing; every reference box is a miss
        print("No received data")
        return {"precision": 1.0, "recall": 0.0, "f1": 0.0, "mean_iou": 0.0,
                "conf_delta": 0.0, "n_received": 0, "n_reference": n_ref, "n_matched": 0, "diff": n_recv - n_ref, "diff_matched": n_ref}

    iou = _iou_matrix(reference_det[:, :4], received_det[:, :4])     # (n_ref, n_recv)
    cls_match = (reference_det[:, 5][:, None] == received_det[:, 5][None, :])
    iou = iou * cls_match

    order = np.argsort(-reference_det[:, 4])    # most confident reference boxes matched first
    matched_recv = set()
    matches = []   # list of (g_idx, r_idx, iou_value)

    for g_idx in order:
        best_r, best_iou = -1, iou_thresh
        for r_idx in range(n_recv):
            if r_idx in matched_recv:
                continue
            if iou[g_idx, r_idx] >= best_iou:
                best_iou = iou[g_idx, r_idx]
                best_r = r_idx
        if best_r >= 0:
            matched_recv.add(best_r)
            matches.append((g_idx, best_r, best_iou))

    n_matched = len(matches)
    precision = n_matched / n_recv
    recall = n_matched / n_ref
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou = float(np.mean([m[2] for m in matches])) if matches else 0.0

    conf_delta = 0.0
    if matches:
        recv_confs = np.array([received_det[r, 4] for _, r, _ in matches])
        ref_confs = np.array([reference_det[g, 4] for g, _, _ in matches])
        conf_delta = float(np.mean(recv_confs - ref_confs))

    return {
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "mean_iou": mean_iou, "conf_delta": conf_delta,
        "n_received": n_recv, "n_reference": n_ref, "n_matched": n_matched,"diff": n_recv - n_ref, "diff_matched": n_ref - n_matched
    }