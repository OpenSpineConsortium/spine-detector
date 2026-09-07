"""nhanes.py — NHANES II annotated subset (Zenodo 10223910) -> manifest rows.

WHAT THE ZIP ACTUALLY CONTAINS, having read it rather than the papers. The spec warned that
published descriptions disagree, and they do: this release is neither the 9-point morphometric
model nor a landmark CSV. It is 543 pickles — 275 cervical, 268 lumbar — each holding

    {'sample_id': '00223',
     'image':     ndarray (1755, 1462) float,      # the fullres NLM frame
     'masks':     {'C2': ndarray, 'C3': ..., 'vertebrae': ..., 'C2_lower': ...}}

Two consequences, both good:

  THE IMAGES SHIP WITH THE COORDINATES. The spec's coordinate-frame hazard — annotations
  valid for the NLM fullres frame being paired against the CDC 16-bit TIFF release — cannot
  arise, because the pixels and the masks are in the same file and the same frame.

  LEVELS ARE NAMED, NOT POSITIONAL. The mask dict is keyed by vertebra ('C2'…'C7', 'T1'),
  so `level_source` is 'annotated'. That is what makes NHANES the bridge between the
  cervical-only and lumbar-only sources: it is the one source labelling both under a single
  protocol.

CORNERS ARE DERIVED, AND THAT IS RECORDED. There are no annotated corners here; the four
corners come from the minimum-area rectangle of each per-level mask. That is a faithful
summary of a radiologist's mask but it is not a radiologist's corner, so these instances
carry kp_visible=1 (the geometry is real) while the fact of derivation is documented here and
in the acquisition field ('film' — 1976-80 film, digitised early 1990s).

NOT EVERY KEY IS A VERTEBRA. 'vertebrae' is the union of all of them and 'C2_lower' is a
sub-part of C2; both are skipped, or the union would be trained as one enormous instance.

ACQUISITION POSTURE, worth knowing before anything curvature-derived is computed from this:
cervical films were shot in MAXIMUM FLEXION with a weight in each hand to depress the
shoulders. This is a flexion distribution, not neutral standing.
"""
from __future__ import annotations

import io
import pickle
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ..manifest import LEVELS, canonical_slots, validate_instance

# keys that are not a single vertebral body
NOT_A_VERTEBRA = {"vertebrae", "spine", "background"}
MIN_MASK_PX = 200


def _level_of(key: str) -> str | None:
    """Map a mask key to a canonical level, or None if it is not one vertebra."""
    k = key.strip()
    if k.lower() in NOT_A_VERTEBRA:
        return None
    if k.endswith("_lower") or k.endswith("_upper"):
        return None                      # sub-parts of a body, not the body
    return k if k in LEVELS else None


def _corners_from_mask(m: np.ndarray) -> np.ndarray | None:
    """Minimum-area rectangle of a binary mask -> four corners in canonical slots."""
    b = (np.asarray(m) > 0).astype(np.uint8)
    if b.sum() < MIN_MASK_PX:
        return None
    cnts, _ = cv2.findContours(b, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    box = cv2.boxPoints(cv2.minAreaRect(c))
    return canonical_slots(box)


def convert(zip_path: str | Path, images_out: str | Path | None = None,
            limit: int | None = None) -> pd.DataFrame:
    """Read the zip; optionally write each image as PNG into `images_out`."""
    zp = Path(zip_path)
    out_dir = Path(images_out) if images_out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    rows, rejected = [], []
    with zipfile.ZipFile(zp) as z:
        names = [n for n in z.namelist() if n.endswith(".pkl")]
        if limit:
            names = names[:limit]
        for n in names:
            try:
                o = pickle.load(io.BytesIO(z.read(n)))
            except Exception as exc:                                  # noqa: BLE001
                rejected.append((n, f"unpickle:{type(exc).__name__}"))
                continue

            sid = str(o.get("sample_id", Path(n).stem))
            region = "cervical" if "cervical" in n else "lumbar"
            img = np.asarray(o["image"])
            h, w = img.shape[:2]
            image_id = f"nhanes_{region}_{sid}"

            if out_dir:
                x = img.astype(np.float32)
                lo, hi = np.percentile(x, [0.5, 99.5])
                x = np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)
                png = (x * 255).astype(np.uint8)
                png = cv2.createCLAHE(2.0, (8, 8)).apply(png)
                cv2.imwrite(str(out_dir / f"{image_id}.png"), png)

            masks = o.get("masks", {}) or {}
            inst = 0
            for key, m in masks.items():
                lvl = _level_of(key)
                if lvl is None:
                    continue
                q = _corners_from_mask(m)
                if q is None:
                    rejected.append((f"{image_id}:{key}", "mask_too_small"))
                    continue
                why = validate_instance(q, w, h)
                if why:
                    rejected.append((f"{image_id}:{key}", why))
                    continue
                rows.append({
                    "image_id": image_id,
                    "source": "nhanes",
                    "provenance": "nhanes",
                    "orig_path": n,
                    "view": "LAT",
                    "acquisition": "film",
                    "region": region,
                    "facing": "left",
                    "width": w, "height": h,
                    "mm_per_px": None,
                    # one participant can have BOTH a cervical and a lumbar film;
                    # the patient, not the image, is the split unit
                    "patient_id": f"nhanes_{sid}",
                    "instance_id": inst,
                    "level": lvl,
                    "level_source": "annotated",     # the mask dict is keyed by level
                    "x1": q[0, 0], "y1": q[0, 1], "x2": q[1, 0], "y2": q[1, 1],
                    "x3": q[2, 0], "y3": q[2, 1], "x4": q[3, 0], "y4": q[3, 1],
                    "kp_visible": [1, 1, 1, 1],
                })
                inst += 1

    df = pd.DataFrame(rows)
    if len(df):
        print(f"  nhanes: {df['image_id'].nunique()} images, {len(df)} instances")
        print(f"          regions {df['region'].value_counts().to_dict()}")
        print(f"          levels  {df['level'].value_counts().to_dict()}")
    if rejected:
        from collections import Counter
        print(f"  nhanes: rejected {len(rejected)}")
        for why, c in Counter(w for _, w in rejected).most_common(6):
            print(f"      {why}: {c}")
    return df
