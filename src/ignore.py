"""ignore.py — where "no ground truth" may be read as "background", and where it may not.

THE PROBLEM THIS SOLVES. Training data is a union of partially-labeled datasets. CSXA labels
C3-C7 and nothing else; the thoracic vertebrae at the bottom of a CSXA film are real
vertebrae with no annotation. Train them as background and the model learns to suppress
true positives — the loss actively teaches it that a vertebra is not a vertebra.

THE GEOMETRY.

    hull          = convex_hull(all annotated corners)
    margin        = 0.25 * median annotated vertebral height
    trusted_bg    = erode(hull, margin)
    ignore_region = image \\ trusted_bg

ERODE, NOT DILATE, and the direction is the whole point. The vertebra immediately adjacent to
the annotated run abuts the hull and bleeds inside it. Dilating would swallow that neighbour
into trusted background — precisely the voxels that must not be called background. Eroding
pushes the trusted boundary inward so the neighbour lands in the ignore zone.

A CONVEX HULL, not an axis-aligned band above and below the annotated span: a scoliotic or
rotated spine runs diagonally, and a band either clips real annotation or trusts far too much.

Tune `margin_frac` by rendering overlays (`render_overlay`) before training anything.
"""
from __future__ import annotations

import cv2
import numpy as np

MARGIN_FRAC = 0.25


def vertebral_heights(quads: np.ndarray) -> np.ndarray:
    """Mean of the two lateral edge lengths per instance. quads: (N,4,2) in slot order."""
    if len(quads) == 0:
        return np.array([])
    left = np.linalg.norm(quads[:, 3] - quads[:, 0], axis=1)
    right = np.linalg.norm(quads[:, 2] - quads[:, 1], axis=1)
    return 0.5 * (left + right)


def annotated_hull(quads: np.ndarray) -> np.ndarray | None:
    """Convex hull of every annotated corner, as an (M,2) polygon."""
    if len(quads) == 0:
        return None
    pts = quads.reshape(-1, 2).astype(np.float32)
    if len(pts) < 3:
        return None
    return cv2.convexHull(pts).reshape(-1, 2)


def _erode_polygon(poly: np.ndarray, margin: float, shape: tuple[int, int]) -> np.ndarray:
    """Erode a polygon by `margin` px, via a mask (robust to non-convex results)."""
    h, w = shape
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [poly.astype(np.int32)], 1)
    k = max(1, int(round(margin)))
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
    return cv2.erode(m, ker)


def trusted_background(quads: np.ndarray, shape: tuple[int, int],
                       margin_frac: float = MARGIN_FRAC) -> np.ndarray:
    """Binary mask, 1 where an unmatched anchor may legitimately be trained as background."""
    h, w = shape
    hull = annotated_hull(quads)
    if hull is None:
        # nothing annotated: nothing can be trusted as background
        return np.zeros((h, w), np.uint8)
    heights = vertebral_heights(quads)
    margin = margin_frac * float(np.median(heights)) if len(heights) else 0.0
    return _erode_polygon(hull, margin, (h, w))


def ignore_mask(quads: np.ndarray, shape: tuple[int, int],
                margin_frac: float = MARGIN_FRAC) -> np.ndarray:
    """1 where the image has no trustworthy supervision — the complement of trusted bg."""
    return 1 - trusted_background(quads, shape, margin_frac)


def ignore_polygons(quads: np.ndarray, shape: tuple[int, int],
                    margin_frac: float = MARGIN_FRAC) -> list[np.ndarray]:
    """The ignore region as polygons, which is what the loss patch carries per sample.

    Polygons rather than a mask because every geometric augmentation has to transform them
    alongside the boxes, and warping a mask per augmentation is far more expensive than
    warping a handful of points.
    """
    m = ignore_mask(quads, shape, margin_frac)
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in cnts:
        eps = 0.002 * cv2.arcLength(c, True)
        out.append(cv2.approxPolyDP(c, eps, True).reshape(-1, 2).astype(np.float32))
    return out


def crop_box_v0(quads: np.ndarray, shape: tuple[int, int],
                dilate_frac: float = 0.5) -> tuple[int, int, int, int]:
    """v0 of the staged plan: crop so unannotated vertebrae fall outside the frame.

    Zero code in the training loop, and it is a SMOKE TEST rather than a result — it biases
    the model toward vertebrae near image borders, because after cropping every annotated run
    ends at one. Use it to prove the pipeline runs end to end, then move to the loss mask.
    """
    h, w = shape
    hull = annotated_hull(quads)
    if hull is None:
        return 0, 0, w, h
    heights = vertebral_heights(quads)
    pad = dilate_frac * float(np.median(heights)) if len(heights) else 0.0
    x0 = max(0, int(hull[:, 0].min() - pad))
    y0 = max(0, int(hull[:, 1].min() - pad))
    x1 = min(w, int(hull[:, 0].max() + pad))
    y1 = min(h, int(hull[:, 1].max() + pad))
    return x0, y0, x1, y1


def render_overlay(img: np.ndarray, quads: np.ndarray,
                   margin_frac: float = MARGIN_FRAC) -> np.ndarray:
    """Boxes and the ignore region on one picture. Look at 50 of these per source first.

    The margin is a judgement call per source, and the only way to set it is to see whether
    the neighbouring unannotated vertebra actually lands inside the ignore zone.
    """
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    ig = ignore_mask(quads, img.shape[:2], margin_frac)
    tint = vis.copy()
    tint[ig.astype(bool)] = (0, 0, 200)
    vis = cv2.addWeighted(vis, 0.72, tint, 0.28, 0)
    for q in quads.astype(np.int32):
        cv2.polylines(vis, [q], True, (0, 230, 0), 2)
        for k, (x, y) in enumerate(q):
            cv2.circle(vis, (int(x), int(y)), 4, (0, 200, 255), -1)
            cv2.putText(vis, str(k), (int(x) + 5, int(y) - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 255), 1)
    hull = annotated_hull(quads)
    if hull is not None:
        cv2.polylines(vis, [hull.astype(np.int32)], True, (255, 200, 0), 2)
    return vis
