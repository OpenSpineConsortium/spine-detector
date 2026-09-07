"""aasce.py — AASCE MICCAI 2019 (SpineWeb Dataset 16) -> manifest rows.

609 standing AP full-spine radiographs (481 labelled training images survive on the Kaggle
mirror; the SpineWeb host is down). Each image carries 68 landmarks: four corners on each
of 17 vertebrae, T1 to L5, top to bottom, plus three Cobb angles read by two radiologists.

FORMAT, AS SHIPPED BY THE MIRROR. `train_txt/filenames.csv` lists the image files in row
order; `train_txt/landmarks.csv` has one row per image with 136 numbers: the 68 x values
first, then the 68 y values, each normalised to the image width or height (0..1);
`train_txt/angles.csv` has the three Cobb angles (PT, MT, TL) per row. Within a vertebra the
four corners run top-left, top-right, bottom-left, bottom-right.

TWO THINGS THAT MATTER DOWNSTREAM:
  LEVEL IS ASSUMED, NOT ANNOTATED. The top annotated vertebra is taken to be T1 and every
  level below it is a count. A film that starts elsewhere, or a spine with a thirteenth
  rib-bearing segment or a sixth lumbar body, shifts every name below the fault, and the
  published annotations carry documented ordering errors. So `level_source` is
  `inferred_from_position`, and the level head must never treat these names as truth.
  PATIENT PAIRING. Files come as `<session> A AP.jpg` / `<session> B AP.jpg`. Whether A and
  B are two exposures of one patient or two patients is not documented; they are grouped
  under one `patient_id` so a split can never put the pair on both sides.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

LEVELS = tuple(f"T{i}" for i in range(1, 13)) + tuple(f"L{i}" for i in range(1, 6))   # 17
STEM_RE = re.compile(r"^(?P<session>.+?)\s+(?P<shot>[A-Z])\s+AP$", re.I)


def read_index(txt_dir: Path):
    names = [l.strip() for l in (txt_dir / "filenames.csv").read_text().splitlines() if l.strip()]
    lm = np.loadtxt(txt_dir / "landmarks.csv", delimiter=",")
    if lm.ndim == 1:
        lm = lm[None, :]
    angles = None
    ap = txt_dir / "angles.csv"
    if ap.exists():
        angles = np.loadtxt(ap, delimiter=",")
    if len(names) != lm.shape[0] or lm.shape[1] != 136:
        raise ValueError(f"filenames {len(names)} vs landmarks {lm.shape}")
    return names, lm, angles


def patient_of(stem: str) -> str:
    m = STEM_RE.match(stem)
    return m.group("session") if m else stem


def quads_for(row: np.ndarray, width: int, height: int):
    """68 normalised landmarks -> 17 quads of four (x, y) pixel corners, top to bottom."""
    xs, ys = row[:68] * width, row[68:] * height
    pts = np.stack([xs, ys], 1).reshape(17, 4, 2)          # TL, TR, BL, BR per vertebra
    return [pts[i] for i in range(17)]
