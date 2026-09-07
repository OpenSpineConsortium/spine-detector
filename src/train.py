"""train.py — the detection head. Class-agnostic vertebra + four corners.

THE AUGMENTATION SETTINGS ARE THE POINT OF THIS FILE. Ultralytics' defaults are tuned for
COCO and two of them are actively wrong for spine radiographs:

  fliplr=0.0. Ultralytics defaults to 0.5. On a lateral radiograph a horizontal flip is
  anatomically impossible and it swaps the anterior/posterior meaning of corner slots 0/3
  against 1/2; on an AP it mirrors scoliosis convexity and laterality. There is no error
  message — the keypoint head simply fails to converge, and the run looks like a plateau
  rather than a bug. This single line is the most expensive thing to get wrong here.

  mosaic=0.0, at least for the first runs. Mosaic composites four images and would composite
  their ignore regions with them, confounding the first look at whether the data is right.

ROTATION IS PER-ACQUISITION and cannot be one global number: a C-arm is not aligned to the
patient axis and genuinely produces any orientation, so intraoperative fluoroscopy wants
±180°, while applying ±180° to standing radiographs would destroy them. Ultralytics takes one
`degrees` for the whole run, so the standing value is set here and the intraop augmentation is
applied per-sample upstream (see src/augment.py).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/yolo/data.yaml")
    ap.add_argument("--model", default="yolo11m-pose.pt")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="0")
    ap.add_argument("--project", default="runs")
    ap.add_argument("--name", default="detect_v1")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(a.model)
    results = model.train(
        data=a.data,
        epochs=a.epochs,
        imgsz=a.imgsz,
        batch=a.batch,
        device=a.device,
        project=a.project,
        name=a.name,
        workers=a.workers,
        resume=a.resume,
        pretrained=True,

        # ---- geometry -------------------------------------------------------
        # A horizontal flip is anatomically impossible on a spine radiograph and
        # silently destroys corner-slot semantics. Never enable this.
        fliplr=0.0,
        flipud=0.0,
        # standing/supine/film; intraop fluoro is handled per-sample upstream
        degrees=15.0,
        translate=0.10,
        scale=0.50,          # 0.5-1.5x, part of the domain-gap policy
        shear=2.0,
        perspective=0.0,

        # ---- compositing ----------------------------------------------------
        # off for the first runs: it would composite ignore regions too
        mosaic=0.0,
        mixup=0.0,
        copy_paste=0.0,
        erasing=0.0,         # cutout stays off until ignore regions are verified

        # ---- photometric: these target the in-domain-good/out-of-domain-bad gap
        hsv_h=0.0,           # grayscale; hue jitter is meaningless
        hsv_s=0.0,
        hsv_v=0.4,           # brightness/contrast variation stands in for windowing
        bgr=0.0,

        # ---- schedule -------------------------------------------------------
        optimizer="AdamW",
        lr0=1e-3,
        lrf=0.01,
        warmup_epochs=3.0,
        patience=30,
        cos_lr=True,
        seed=0,
        deterministic=False,
        val=True,
        plots=True,
    )

    out = Path(a.project) / a.name
    summary = {
        "model": a.model, "epochs": a.epochs, "imgsz": a.imgsz, "batch": a.batch,
        "save_dir": str(results.save_dir) if hasattr(results, "save_dir") else str(out),
    }
    (out / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
