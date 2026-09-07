"""buu.py — BUU-LSPINEv1 (Burapha) -> manifest rows.

3,600 patients, 7,200 images: an AP and a lateral for each. Annotation is VERTEBRAL EDGE
LINES, not corners — 11 lines on a lateral (L1-L5 + S1) and 10 on an AP (L1-L5), each line
given as a left and a right endpoint. Two consecutive lines bound one vertebra, so the upper
edge and the lower edge together are the four corners.

FILENAME CARRIES EVERYTHING: `0005-F-025Y1` is patient 0005, female, 25 years, and the final
character is the view — `1` lateral, `0` AP. That final character is the only view signal, so
it is parsed rather than guessed.

TWO THINGS THAT MATTER DOWNSTREAM:

  PATIENT PAIRING. One patient contributes an AP and a lateral. `patient_id` is the patient,
  not the image, or a split puts the same spine on both sides of the evaluation.

  LEVEL IS POSITIONAL. BUU gives a contiguous run of edge lines in cranio-caudal order; it
  does not label each vertebra. The levels are assigned by counting down the run, so
  `level_source` is `inferred_from_position` and never `annotated`. On a transitional spine
  that count is exactly what is in doubt, which is the whole reason the field exists.

  Laterals ship PRE-FLIPPED to left-facing, so `facing` is recorded as 'left' and the
  normaliser leaves them alone.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from ..manifest import canonical_slots, validate_instance

# lateral runs L1..L5 + S1; AP runs L1..L5
LEVELS_LAT = ["L1", "L2", "L3", "L4", "L5", "S1"]
LEVELS_AP = ["L1", "L2", "L3", "L4", "L5"]

STEM_RE = re.compile(r"^(?P<pid>\d+)-(?P<sex>[MF])-(?P<age>\d+)Y(?P<view>[01])", re.I)


def parse_stem(stem: str) -> dict | None:
    m = STEM_RE.match(stem)
    if not m:
        return None
    return {
        "patient": m.group("pid"),
        "sex": m.group("sex").upper(),
        "age": int(m.group("age")),
        "view": "LAT" if m.group("view") == "1" else "AP",
    }


def read_edge_lines(path: Path) -> np.ndarray | None:
    """Read one annotation file -> (n_lines, 4) as [xL, yL, xR, yR], cranio-caudal.

    BUU's CSVs have been seen with and without a header and with varying column counts, so
    numeric columns are taken positionally after dropping anything non-numeric. Four numbers
    per row is the contract; anything else is a parsing failure and is reported, not guessed.
    """
    try:
        raw = pd.read_csv(path, header=None, comment="#")
    except Exception:
        return None
    num = raw.apply(pd.to_numeric, errors="coerce")
    num = num.dropna(axis=1, how="all").dropna(axis=0, how="any")
    if num.shape[1] < 4:
        return None
    arr = num.to_numpy(dtype=float)[:, :4]
    if len(arr) < 2:
        return None
    # cranio-caudal: smaller y is more superior in image coordinates
    return arr[np.argsort(arr[:, 1])]


def lines_to_quads(lines: np.ndarray) -> np.ndarray:
    """Edge lines -> one quadrilateral per vertebra, pairing lines (0,1), (2,3), (4,5)…

    PAIRING IS STEP-2, NOT CONSECUTIVE, and the difference is not cosmetic. Each vertebra
    contributes TWO lines, its superior and inferior endplate; the gap between one vertebra's
    inferior line and the next one's superior line is the DISC SPACE. Walking consecutive
    pairs would emit a quadrilateral for every disc as well as every body — roughly doubling
    the instance count with objects that are not vertebrae, and teaching the detector that
    an intervertebral space is a detection target.

    Verified against the shipped files: an AP has 10 lines for L1-L5 and the y-values step
    167 -> 313 (body), 313 -> 344 (disc), 344 -> 498 (body). A lateral has 11 — five bodies
    plus a single S1 endplate — so a trailing unpaired line is dropped rather than fabricated
    into a partial vertebra.
    """
    quads = []
    for i in range(0, len(lines) - 1, 2):
        up, lo = lines[i], lines[i + 1]
        pts = [(up[0], up[1]), (up[2], up[3]), (lo[2], lo[3]), (lo[0], lo[1])]
        quads.append(canonical_slots(pts))
    return np.array(quads) if quads else np.zeros((0, 4, 2))


def convert(raw_dir: str | Path, image_size_lookup=None) -> pd.DataFrame:
    """Walk data/raw/buu_v1 and emit manifest rows.

    `image_size_lookup(stem) -> (w, h)` is injected so this stays testable without images.
    """
    raw = Path(raw_dir)
    rows, rejected = [], []

    ann_files = [p for p in raw.rglob("*") if p.suffix.lower() in (".csv", ".txt")]
    for ann in sorted(ann_files):
        info = parse_stem(ann.stem)
        if info is None:
            continue
        lines = read_edge_lines(ann)
        if lines is None:
            rejected.append((ann.name, "unparsable_edge_lines"))
            continue

        quads = lines_to_quads(lines)
        levels = LEVELS_LAT if info["view"] == "LAT" else LEVELS_AP
        image_id = f"buu_{ann.stem}"
        w, h = image_size_lookup(ann.stem) if image_size_lookup else (0, 0)

        for i, q in enumerate(quads):
            if w and h:
                why = validate_instance(q, w, h)
                if why:
                    rejected.append((f"{ann.stem}[{i}]", why))
                    continue
            rows.append({
                "image_id": image_id,
                "source": "buu",
                "provenance": "buu",
                "orig_path": str(ann.relative_to(raw)),
                "view": info["view"],
                "acquisition": "standing_dr",
                "region": "lumbar",
                # BUU ships every lateral already flipped to left-facing
                "facing": "left",
                "width": w, "height": h,
                "mm_per_px": None,
                "patient_id": f"buu_{info['patient']}",
                "instance_id": i,
                "level": levels[i] if i < len(levels) else "unknown",
                # a contiguous run counted from the top, NOT a per-vertebra label
                "level_source": "inferred_from_position",
                "x1": q[0, 0], "y1": q[0, 1], "x2": q[1, 0], "y2": q[1, 1],
                "x3": q[2, 0], "y3": q[2, 1], "x4": q[3, 0], "y4": q[3, 1],
                "kp_visible": [1, 1, 1, 1],
            })

    df = pd.DataFrame(rows)
    if rejected:
        n = len(rejected)
        tot = n + len(rows)
        print(f"  buu: rejected {n}/{tot} ({100*n/max(tot,1):.1f}%)")
        from collections import Counter
        for why, c in Counter(w for _, w in rejected).most_common():
            print(f"      {why}: {c}")
        # a spike here is a parsing bug, not bad data
    return df
