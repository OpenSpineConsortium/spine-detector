"""mendeley.py — Mendeley spondylolisthesis landmarks (716 images) -> manifest rows.

The closest thing to a drop-in source: Keypoint R-CNN JSON with BOTH bounding boxes and four
corner keypoints per vertebra, L3-S1.

THE LEAKAGE HAZARD IS THE WHOLE REASON THIS CONVERTER IS CAREFUL. 508 of the 716 images were
filtered out of BUU; only the 208 Honduran images are independent of it. Train on BUU and
evaluate on the BUU-derived half and the number is meaningless — the same patients are on
both sides. So `provenance` is tagged per image, and the 208 Honduran images are the only
genuine cross-site lateral lumbar test set that exists anywhere in this project.

Provenance is decided by filename convention where the release marks it, and otherwise by
matching against the BUU patient identifiers actually present on disk. If neither is
available the image is tagged `buu_derived` — the CONSERVATIVE choice, because wrongly
calling a BUU image Honduran would contaminate the test set, while wrongly calling a
Honduran image BUU only costs a little test data.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..manifest import canonical_slots, validate_instance

LEVELS = ["L3", "L4", "L5", "S1"]
BUU_STEM = re.compile(r"^\d{4}-[MF]-\d+Y[01]", re.I)


def _corners_from_obj(o: dict) -> np.ndarray | None:
    """Pull four (x, y) corners out of one annotation object, whatever it calls them."""
    for key in ("keypoints", "corners", "points", "landmarks"):
        if key in o and o[key]:
            pts = np.asarray(o[key], dtype=float)
            if pts.ndim == 1:
                if pts.size % 3 == 0 and pts.size >= 12:
                    pts = pts.reshape(-1, 3)[:, :2]
                elif pts.size % 2 == 0 and pts.size >= 8:
                    pts = pts.reshape(-1, 2)
                else:
                    continue
            elif pts.ndim == 2 and pts.shape[1] == 3:
                pts = pts[:, :2]
            if len(pts) >= 4:
                return pts[:4]
    return None


def _provenance(stem: str, buu_patients: set[str] | None) -> str:
    """Honduran, or re-filtered BUU? Wrong in the Honduran direction contaminates the test set."""
    s = stem.lower()
    if "hond" in s:
        return "honduran"
    if BUU_STEM.match(stem):
        return "buu_derived"
    if buu_patients:
        pid = stem.split("-")[0]
        if pid in buu_patients:
            return "buu_derived"
        return "honduran"
    # unknown provenance -> assume BUU-derived, never risk contaminating the test set
    return "buu_derived"


def convert(raw_dir: str | Path, buu_patients: set[str] | None = None,
            image_size_lookup=None) -> pd.DataFrame:
    raw = Path(raw_dir)
    rows, rejected = [], []

    for jf in sorted(raw.rglob("*.json")):
        try:
            blob = json.loads(jf.read_text())
        except Exception:
            continue

        # the release has been seen both as one big file and as per-image files
        if isinstance(blob, dict) and "images" in blob and "annotations" in blob:
            imgs = {i["id"]: i for i in blob["images"]}
            grouped: dict[int, list] = {}
            for an in blob["annotations"]:
                grouped.setdefault(an["image_id"], []).append(an)
            items = [(imgs[k]["file_name"], v) for k, v in grouped.items() if k in imgs]
        elif isinstance(blob, dict):
            items = [(blob.get("file_name", jf.stem), blob.get("annotations", [blob]))]
        elif isinstance(blob, list):
            items = [(jf.stem, blob)]
        else:
            continue

        for fname, objs in items:
            stem = Path(fname).stem
            prov = _provenance(stem, buu_patients)
            image_id = f"mendeley_{stem}"
            w, h = image_size_lookup(stem) if image_size_lookup else (0, 0)

            for i, o in enumerate(objs):
                pts = _corners_from_obj(o if isinstance(o, dict) else {})
                if pts is None:
                    rejected.append((f"{stem}[{i}]", "no_corners"))
                    continue
                q = canonical_slots(pts)
                if w and h:
                    why = validate_instance(q, w, h)
                    if why:
                        rejected.append((f"{stem}[{i}]", why))
                        continue
                rows.append({
                    "image_id": image_id,
                    "source": "mendeley",
                    "provenance": prov,
                    "orig_path": str(jf.relative_to(raw)),
                    "view": "LAT",
                    "acquisition": "standing_dr",
                    "region": "lumbar",
                    "facing": "left",
                    "width": w, "height": h,
                    "mm_per_px": None,
                    "patient_id": f"mendeley_{prov}_{stem.split('-')[0]}",
                    "instance_id": i,
                    "level": LEVELS[i] if i < len(LEVELS) else "unknown",
                    "level_source": "inferred_from_position",
                    "x1": q[0, 0], "y1": q[0, 1], "x2": q[1, 0], "y2": q[1, 1],
                    "x3": q[2, 0], "y3": q[2, 1], "x4": q[3, 0], "y4": q[3, 1],
                    "kp_visible": [1, 1, 1, 1],
                })

    df = pd.DataFrame(rows)
    if len(df):
        n_h = (df["provenance"] == "honduran").sum()
        print(f"  mendeley: {len(df)} instances  honduran={n_h}  buu_derived={len(df)-n_h}")
    if rejected:
        from collections import Counter
        print(f"  mendeley: rejected {len(rejected)}")
        for why, c in Counter(w for _, w in rejected).most_common():
            print(f"      {why}: {c}")
    return df
