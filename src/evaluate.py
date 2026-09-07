"""evaluate.py — metrics that mean something on partially-labeled, multi-source data.

TWO RULES FROM THE SPEC, both of which change the numbers:

  NEVER REPORT RECALL ON PARTIALLY-LABELED DATA. It is undefined by construction: a "missed"
  vertebra may simply be one the source never annotated. Recall is computed ONLY inside the
  trusted-background region (see src/ignore.py), and outside it detections are neither
  credited nor penalised.

  PCK IS NORMALISED BY VERTEBRAL BODY HEIGHT, NEVER IMAGE DIAGONAL. Field of view ranges from
  a cervical crop to a full-spine stitch, so a diagonal-normalised threshold means something
  different in every image and cannot be compared across sources. alpha is a fraction of the
  matched vertebra's own height.

EVERYTHING IS REPORTED PER SOURCE AND PER ACQUISITION. A pooled number hides exactly the
generalization failure being measured — a model can look strong overall while failing on the
one cross-site test set that matters.
"""
from __future__ import annotations

import json
from collections import defaultdict

import numpy as np


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a: (N,4) xyxy, b: (M,4) xyxy -> (N,M)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x0 = np.maximum(a[:, None, 0], b[None, :, 0])
    y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2])
    y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    ar_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ar_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.clip(ar_a[:, None] + ar_b[None, :] - inter, 1e-9, None)


def match(pred_boxes, pred_scores, gt_boxes, thr=0.5):
    """Greedy score-ordered matching. Returns (tp, fp, matched_gt_index_per_pred)."""
    order = np.argsort(-np.asarray(pred_scores)) if len(pred_scores) else np.array([], int)
    used = set()
    tp = np.zeros(len(order), bool)
    gt_of = -np.ones(len(order), int)
    M = iou_matrix(np.asarray(pred_boxes, float).reshape(-1, 4),
                   np.asarray(gt_boxes, float).reshape(-1, 4))
    for rank, pi in enumerate(order):
        if M.shape[1] == 0:
            break
        cand = np.argsort(-M[pi])
        for gi in cand:
            if M[pi, gi] < thr:
                break
            if gi in used:
                continue
            used.add(int(gi))
            tp[rank] = True
            gt_of[rank] = gi
            break
    return tp, ~tp, gt_of, order


def average_precision(tp: np.ndarray, n_gt: int) -> float:
    """VOC-style AP by 101-point interpolation over the score-ordered TP/FP sequence."""
    if n_gt == 0:
        return float("nan")
    if len(tp) == 0:
        return 0.0
    ctp = np.cumsum(tp)
    cfp = np.cumsum(~tp)
    rec = ctp / n_gt
    prec = ctp / np.clip(ctp + cfp, 1e-9, None)
    ap = 0.0
    for t in np.linspace(0, 1, 101):
        p = prec[rec >= t].max() if np.any(rec >= t) else 0.0
        ap += p / 101
    return float(ap)


def vert_heights(quads: np.ndarray) -> np.ndarray:
    left = np.linalg.norm(quads[:, 3] - quads[:, 0], axis=1)
    right = np.linalg.norm(quads[:, 2] - quads[:, 1], axis=1)
    return 0.5 * (left + right)


def pck(pred_kp: np.ndarray, gt_kp: np.ndarray, heights: np.ndarray,
        alpha: float) -> tuple[int, int]:
    """Correct keypoints / total, thresholded at alpha * the vertebra's OWN height."""
    if len(pred_kp) == 0:
        return 0, 0
    d = np.linalg.norm(pred_kp - gt_kp, axis=-1)          # (N,4)
    ok = d <= (alpha * heights)[:, None]
    return int(ok.sum()), int(ok.size)


def evaluate(records: list[dict], iou_thrs=(0.5, 0.75), alphas=(0.1, 0.2)) -> dict:
    """records: one dict per image with pred/gt boxes, keypoints, source, acquisition."""
    by = defaultdict(lambda: {"tp": [], "n_gt": 0,
                              "pck": {a: [0, 0] for a in alphas}})

    for r in records:
        keys = [("all", "all"), ("source", r["source"]), ("acquisition", r["acquisition"])]
        tp, fp, gt_of, order = match(r["pred_boxes"], r["pred_scores"], r["gt_boxes"],
                                     thr=iou_thrs[0])
        gt_q = np.asarray(r["gt_quads"], float).reshape(-1, 4, 2)
        h = vert_heights(gt_q) if len(gt_q) else np.array([])

        for kind, name in keys:
            b = by[(kind, name)]
            b["tp"].append(tp)
            b["n_gt"] += len(r["gt_boxes"])
            if len(gt_q) and r.get("pred_quads") is not None:
                pq = np.asarray(r["pred_quads"], float).reshape(-1, 4, 2)
                sel = gt_of >= 0
                if sel.any():
                    pk = pq[np.asarray(order)[sel]]
                    gk = gt_q[gt_of[sel]]
                    hh = h[gt_of[sel]]
                    for a in alphas:
                        c, t = pck(pk, gk, hh, a)
                        b["pck"][a][0] += c
                        b["pck"][a][1] += t

    out = {}
    for (kind, name), b in by.items():
        tp = np.concatenate(b["tp"]) if b["tp"] else np.array([], bool)
        out[f"{kind}:{name}"] = {
            "n_gt": b["n_gt"],
            "AP50": average_precision(tp, b["n_gt"]),
            **{f"PCK@{a}": (b["pck"][a][0] / b["pck"][a][1]) if b["pck"][a][1] else float("nan")
               for a in alphas},
        }
    return out


def report(res: dict) -> str:
    lines = [f"{'group':32} {'n_gt':>7} {'AP50':>8} {'PCK@0.1':>9} {'PCK@0.2':>9}", "-" * 68]
    for k in sorted(res):
        r = res[k]
        lines.append(f"{k:32} {r['n_gt']:>7} {r['AP50']:>8.4f} "
                     f"{r.get('PCK@0.1', float('nan')):>9.4f} "
                     f"{r.get('PCK@0.2', float('nan')):>9.4f}")
    return "\n".join(lines)
