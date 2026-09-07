"""run_eval.py — run the trained detector over the held-out split and score it honestly.

Produces three things the report needs:
  eval/metrics.json    AP50 and PCK@{0.1,0.2}, per source and per acquisition
  eval/renders/*.png   real predictions drawn over real images, GT in green, pred in magenta
  eval/predictions.json raw predictions, so the report can be regenerated without a GPU

RECALL IS NOT REPORTED GLOBALLY. On partially-labeled sources a "missed" vertebra may be one
the dataset never annotated, so a detection outside the trusted-background region is neither
credited nor penalised — see src/ignore.py for where that boundary is and why it is eroded
rather than dilated.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from src.evaluate import evaluate, report
from src.ignore import trusted_background

CORNERS = ["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]


def gt_for(g: pd.DataFrame):
    q = g[CORNERS].to_numpy(float).reshape(-1, 4, 2)
    boxes = np.stack([q[:, :, 0].min(1), q[:, :, 1].min(1),
                      q[:, :, 0].max(1), q[:, :, 1].max(1)], axis=1)
    return q, boxes


def draw(img, gt_q, pred_q, pred_boxes, scores):
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img.copy()
    for q in np.asarray(gt_q).astype(np.int32):
        cv2.polylines(vis, [q], True, (0, 220, 0), 2)
    for q, b, s in zip(np.asarray(pred_q).astype(np.int32),
                       np.asarray(pred_boxes).astype(int), scores):
        cv2.polylines(vis, [q], True, (255, 0, 220), 2)
        cv2.putText(vis, f"{s:.2f}", (b[0], max(12, b[1] - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 220), 1)
    cv2.putText(vis, "green = ground truth   magenta = prediction", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return vis


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--manifest", default="data/interim/manifest.parquet")
    ap.add_argument("--images", default="data/interim/images")
    ap.add_argument("--yolo", default="data/yolo")
    ap.add_argument("--out", default="runs/detect_v1/eval")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--n_render", type=int, default=40)
    a = ap.parse_args()

    from ultralytics import YOLO

    out = Path(a.out)
    (out / "renders").mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(a.manifest)

    test_ids = {p.stem for p in (Path(a.yolo) / "images" / "test").glob("*.png")}
    if not test_ids:
        print("no test images found — did export_yolo run?")
        return 1
    df = df[df["image_id"].isin(test_ids)]
    print(f"evaluating on {df['image_id'].nunique()} test images, {len(df)} instances")

    model = YOLO(a.weights)
    records, raw = [], []
    rendered = 0

    for image_id, g in df.groupby("image_id"):
        img_p = Path(a.images) / f"{image_id}.png"
        img = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        gt_q, gt_boxes = gt_for(g)

        res = model.predict(str(img_p), conf=a.conf, verbose=False)[0]
        pb = res.boxes.xyxy.cpu().numpy() if res.boxes is not None else np.zeros((0, 4))
        ps = res.boxes.conf.cpu().numpy() if res.boxes is not None else np.zeros((0,))
        pq = (res.keypoints.xy.cpu().numpy() if res.keypoints is not None
              else np.zeros((0, 4, 2)))

        # only score predictions whose centre lies in trusted background or on a GT box;
        # outside that, absence of a label does not mean absence of a vertebra
        trusted = trusted_background(gt_q, img.shape[:2])
        keep = []
        for i, b in enumerate(pb):
            cx, cy = int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2)
            cy = min(max(cy, 0), trusted.shape[0] - 1)
            cx = min(max(cx, 0), trusted.shape[1] - 1)
            keep.append(bool(trusted[cy, cx]))
        keep = np.array(keep, bool) if len(pb) else np.zeros((0,), bool)

        records.append(dict(
            image_id=image_id, source=g["source"].iloc[0],
            acquisition=g["acquisition"].iloc[0],
            pred_boxes=pb[keep], pred_scores=ps[keep],
            pred_quads=pq[keep] if len(pq) else np.zeros((0, 4, 2)),
            gt_boxes=gt_boxes, gt_quads=gt_q))
        raw.append(dict(image_id=image_id, source=g["source"].iloc[0],
                        n_gt=int(len(gt_boxes)), n_pred=int(keep.sum()),
                        scores=[float(x) for x in ps[keep]]))

        if rendered < a.n_render:
            cv2.imwrite(str(out / "renders" / f"{image_id}.png"),
                        draw(img, gt_q, pq[keep] if len(pq) else [], pb[keep], ps[keep]))
            rendered += 1

    res = evaluate(records)
    (out / "metrics.json").write_text(json.dumps(res, indent=2))
    (out / "predictions.json").write_text(json.dumps(raw, indent=2))
    txt = report(res)
    (out / "report.txt").write_text(txt)
    print(txt)
    print(f"\nwrote {out}/metrics.json, predictions.json, report.txt, {rendered} renders")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
