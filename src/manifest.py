"""manifest.py — the schema every converter writes to, and the validators that police it.

One row per VERTEBRA INSTANCE, not per image. Six sources with six different annotation
formats all land here, and nothing downstream reads a source's native format again.

Two fields carry more weight than their size suggests:

  `level_source` distinguishes a level that was ANNOTATED from one INFERRED FROM POSITION.
  BUU gives contiguous L1-L5 runs by row order rather than per-vertebra labels; AASCE assumes
  its top annotated vertebra is T1, which silently shifts every level below it on any film
  that starts elsewhere or has a 13th rib-bearing segment. Recording how we know is what
  keeps an inference from being laundered into a label.

  `patient_id` is what splits are made on. BUU pairs an AP and a lateral per patient and
  NHANES has participants with both a cervical and a lumbar film, so splitting by IMAGE puts
  the same spine in train and test.

  `provenance` exists for one source: 508 of Mendeley's 716 images are re-filtered BUU. Only
  the 208 Honduran images are independent, and they are the only genuine cross-site lateral
  lumbar test set that exists. Losing that distinction destroys the evaluation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

# ----------------------------------------------------------------- vocabularies
# "buu" is the 77-file partial Drive download; "buu_2000" the official 2,000-pair subset;
# "buu_3600" the full release when its request form is approved. Same format, distinct
# deliveries, one spine never ingested twice (build_dataset.do_buu).
SOURCES = ("csxa", "nhanes", "buu", "buu_2000", "buu_3600", "mendeley", "cm_spine", "aasce", "vindr")
VIEWS = ("AP", "LAT")
ACQUISITIONS = ("standing_dr", "supine_dr", "intraop_fluoro", "film", "drr")
REGIONS = ("cervical", "thoracic", "lumbar", "full_spine")
FACINGS = ("left", "right")
LEVEL_SOURCES = ("annotated", "inferred_from_position", "unknown")

LEVELS = (
    tuple(f"C{i}" for i in range(1, 8))
    + tuple(f"T{i}" for i in range(1, 13))
    + tuple(f"L{i}" for i in range(1, 7))          # L6 exists; LSTV is a focus here
    + ("S1", "unknown")
)

CORNER_COLS = [f"{a}{i}" for i in (1, 2, 3, 4) for a in ("x", "y")]
CORNER_COLS = ["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]

COLUMNS = [
    "image_id", "source", "provenance", "orig_path", "view", "acquisition", "region",
    "facing", "width", "height", "mm_per_px", "patient_id", "instance_id",
    "level", "level_source", *CORNER_COLS, "kp_visible",
]

# ------------------------------------------------------------------- geometry
# A vertebral body is roughly as wide as it is tall on a lateral and wider on an AP.
# The band is deliberately loose: it is a parsing-bug detector, not an anatomy filter.
ASPECT_MIN, ASPECT_MAX = 0.3, 4.0
MIN_AREA_FRAC = 0.0002          # 0.02% of image area


@dataclass
class Rejection:
    image_id: str
    instance_id: int
    reason: str


def canonical_slots(pts: Sequence[Sequence[float]]) -> np.ndarray:
    """Sort four corners into canonical slot order by IMAGE GEOMETRY.

        slot 0 = superior, smaller x      slot 1 = superior, larger x
        slot 3 = inferior, smaller x      slot 2 = inferior, larger x

    Never trust a source's own ordering — CSXA names its points, BUU gives edge lines,
    AASCE uses a fixed index order with documented errors, and Mendeley uses its own. Sorting
    geometrically is what makes one keypoint slot mean one thing across all of them.

    Slots are geometric, not anatomical. On a lateral, 0/3 are anterior or posterior
    depending on facing; on an AP they are the left and right ends of an endplate. The
    anatomical reading is decoded downstream from `view` and `facing`.

    Image convention: y increases DOWNWARD, so superior = smaller y.
    """
    p = np.asarray(pts, dtype=float).reshape(4, 2)
    top = p[np.argsort(p[:, 1])[:2]]
    bot = p[np.argsort(p[:, 1])[2:]]
    s0, s1 = top[np.argsort(top[:, 0])]
    s3, s2 = bot[np.argsort(bot[:, 0])]
    return np.array([s0, s1, s2, s3], dtype=float)


def _seg_cross(a, b, c, d) -> bool:
    """Do open segments ab and cd properly intersect?"""
    def orient(p, q, r):
        # float(), because p/q/r arrive as numpy arrays and `bool - bool` is not
        # subtraction in numpy — it raises rather than giving the sign.
        v = float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))
        return (v > 1e-9) - (v < -1e-9)
    d1, d2 = orient(c, d, a), orient(c, d, b)
    d3, d4 = orient(a, b, c), orient(a, b, d)
    return d1 * d2 < 0 and d3 * d4 < 0


def is_simple(quad: np.ndarray) -> bool:
    """A quadrilateral in slot order 0-1-2-3 must not self-intersect."""
    return not (_seg_cross(quad[0], quad[1], quad[2], quad[3])
                or _seg_cross(quad[1], quad[2], quad[3], quad[0]))


def quad_area(quad: np.ndarray) -> float:
    """Shoelace area of the quadrilateral."""
    x, y = quad[:, 0], quad[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def quad_aspect(quad: np.ndarray) -> float:
    """Width / height, from mean opposing edge lengths.

    Defined explicitly because 'aspect ratio' is ambiguous and the accept band is
    asymmetric: getting the convention backwards silently rejects every wide vertebra.
    """
    w = 0.5 * (np.linalg.norm(quad[1] - quad[0]) + np.linalg.norm(quad[2] - quad[3]))
    h = 0.5 * (np.linalg.norm(quad[3] - quad[0]) + np.linalg.norm(quad[2] - quad[1]))
    return float(w / h) if h > 1e-9 else float("inf")


def validate_instance(quad: np.ndarray, width: int, height: int) -> str | None:
    """Return a rejection reason, or None if the instance is usable."""
    if not np.isfinite(quad).all():
        return "non_finite_coords"
    if not is_simple(quad):
        return "self_intersecting"
    area = quad_area(quad)
    if area < MIN_AREA_FRAC * width * height:
        return "area_too_small"
    asp = quad_aspect(quad)
    if not (ASPECT_MIN <= asp <= ASPECT_MAX):
        return "aspect_out_of_range"
    return None


# -------------------------------------------------------------------- schema
def validate_manifest(df: pd.DataFrame) -> list[str]:
    """Structural checks. Returns a list of problems; empty means clean."""
    problems: list[str] = []

    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        problems.append(f"missing columns: {missing}")
        return problems

    def bad(col, allowed):
        got = set(df[col].dropna().unique()) - set(allowed)
        if got:
            problems.append(f"{col}: unexpected values {sorted(got)}")

    bad("source", SOURCES)
    bad("view", VIEWS)
    bad("acquisition", ACQUISITIONS)
    bad("region", REGIONS)
    bad("facing", FACINGS)
    bad("level", LEVELS)
    bad("level_source", LEVEL_SOURCES)

    if df["level_source"].isna().any():
        problems.append("level_source is null somewhere — record how the level is known")
    lvl_known = df["level"].notna() & (df["level"] != "unknown")
    if (lvl_known & (df["level_source"] == "unknown")).any():
        problems.append("a level is given while level_source says 'unknown'")

    if df["patient_id"].isna().any():
        problems.append("patient_id is null somewhere — splits would leak")

    dup = df.duplicated(subset=["image_id", "instance_id"]).sum()
    if dup:
        problems.append(f"{dup} duplicate (image_id, instance_id) pairs")

    for col in ("width", "height"):
        if (df[col] <= 0).any():
            problems.append(f"{col} has non-positive values")

    # provenance must be set for mendeley and equal to source elsewhere
    m = df["source"] == "mendeley"
    if m.any():
        got = set(df.loc[m, "provenance"].dropna().unique())
        if not got <= {"honduran", "buu_derived"}:
            problems.append(f"mendeley provenance must be honduran|buu_derived, got {sorted(got)}")
    off = df.loc[~m]
    if (off["provenance"] != off["source"]).any():
        problems.append("provenance must equal source for non-mendeley rows")

    kp = df["kp_visible"]
    if kp.map(lambda v: v is None or len(v) != 4).any():
        problems.append("kp_visible must be length-4 per row")

    return problems


def empty_manifest() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype="object") for c in COLUMNS})
