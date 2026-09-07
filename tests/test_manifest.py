"""Tests for the manifest schema and the geometry validators.

These run before any converter exists, because every converter's output is only as
trustworthy as the checks it is measured against. The cases below are the specific failure
modes the spec calls out, not generic coverage:

  * slot order must come from image geometry, never from a source's own ordering;
  * a source that ships corners in a rotated or reversed order must still land in the same
    slots, because the keypoint head learns slots;
  * self-intersecting quads (the classic symptom of a mis-parsed edge-line file) must be
    rejected rather than trained on;
  * `level_source` must not be able to launder an inference into a label.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.manifest import (
    ASPECT_MAX, ASPECT_MIN, COLUMNS, canonical_slots, empty_manifest, is_simple,
    quad_area, quad_aspect, validate_instance, validate_manifest,
)

# a plausible lateral vertebral body: 40 wide, 30 tall, y increases downward
SQUARE = [(100, 100), (140, 100), (140, 130), (100, 130)]


# ------------------------------------------------------------------ slot order
def test_canonical_slots_basic():
    q = canonical_slots(SQUARE)
    assert tuple(q[0]) == (100, 100)      # superior, smaller x
    assert tuple(q[1]) == (140, 100)      # superior, larger x
    assert tuple(q[2]) == (140, 130)      # inferior, larger x
    assert tuple(q[3]) == (100, 130)      # inferior, smaller x


@pytest.mark.parametrize("shift", range(4))
def test_canonical_slots_is_rotation_invariant(shift):
    """A source listing the same quad from a different starting corner must not move slots."""
    rotated = SQUARE[shift:] + SQUARE[:shift]
    assert np.allclose(canonical_slots(rotated), canonical_slots(SQUARE))


def test_canonical_slots_is_winding_invariant():
    """Clockwise vs counter-clockwise ordering must also land identically."""
    assert np.allclose(canonical_slots(SQUARE[::-1]), canonical_slots(SQUARE))


def test_canonical_slots_handles_tilt():
    """A tilted vertebra still sorts superior/inferior correctly."""
    tilted = [(100, 100), (140, 108), (138, 138), (98, 130)]
    q = canonical_slots(tilted)
    assert q[0][1] < q[3][1] and q[1][1] < q[2][1]      # both superior above both inferior
    assert q[0][0] < q[1][0] and q[3][0] < q[2][0]      # left of right, on each edge


# -------------------------------------------------------------------- geometry
def test_simple_quad_accepted():
    assert is_simple(canonical_slots(SQUARE))


def test_self_intersecting_rejected():
    """Slots 2 and 3 swapped makes a bow-tie — the signature of a mis-paired edge line."""
    bowtie = np.array([(100, 100), (140, 100), (100, 130), (140, 130)], dtype=float)
    assert not is_simple(bowtie)
    assert validate_instance(bowtie, 1000, 1000) == "self_intersecting"


def test_area_and_aspect():
    q = canonical_slots(SQUARE)
    assert quad_area(q) == pytest.approx(40 * 30)
    assert quad_aspect(q) == pytest.approx(40 / 30)


def test_tiny_instance_rejected():
    tiny = canonical_slots([(0, 0), (3, 0), (3, 3), (0, 3)])
    assert validate_instance(tiny, 2048, 2048) == "area_too_small"


def test_aspect_band_edges():
    """A sliver in either direction is a parsing bug, not a vertebra."""
    wide = canonical_slots([(0, 0), (500, 0), (500, 40), (0, 40)])       # aspect 12.5
    tall = canonical_slots([(0, 0), (40, 0), (40, 500), (0, 500)])       # aspect 0.08
    assert quad_aspect(wide) > ASPECT_MAX
    assert quad_aspect(tall) < ASPECT_MIN
    assert validate_instance(wide, 2048, 2048) == "aspect_out_of_range"
    assert validate_instance(tall, 2048, 2048) == "aspect_out_of_range"


def test_realistic_instance_accepted():
    q = canonical_slots([(820, 1180), (1015, 1168), (1022, 1330), (826, 1341)])
    assert validate_instance(q, 2048, 2048) is None


def test_non_finite_rejected():
    q = np.array([(0, 0), (40, 0), (np.nan, 30), (0, 30)], dtype=float)
    assert validate_instance(q, 2048, 2048) == "non_finite_coords"


# ---------------------------------------------------------------------- schema
def _row(**over):
    r = {
        "image_id": "csxa_0001", "source": "csxa", "provenance": "csxa",
        "orig_path": "csxa/0001.jpg", "view": "LAT", "acquisition": "standing_dr",
        "region": "cervical", "facing": "left", "width": 2048, "height": 2048,
        "mm_per_px": 0.14, "patient_id": "csxa_p0001", "instance_id": 0,
        "level": "C3", "level_source": "annotated",
        "x1": 100, "y1": 100, "x2": 140, "y2": 100,
        "x3": 140, "y3": 130, "x4": 100, "y4": 130,
        "kp_visible": [1, 1, 1, 1],
    }
    r.update(over)
    return r


def test_clean_manifest_validates():
    assert validate_manifest(pd.DataFrame([_row()])) == []


def test_empty_manifest_has_every_column():
    assert list(empty_manifest().columns) == COLUMNS


def test_unknown_level_vocabulary_rejected():
    p = validate_manifest(pd.DataFrame([_row(level="C9")]))
    assert any("level" in x for x in p)


def test_l6_is_legal():
    """LSTV is a focus of this project; L6 must not be a schema error."""
    assert validate_manifest(pd.DataFrame([_row(level="L6")])) == []


def test_level_without_source_rejected():
    """An inferred level must never be recorded as if it were annotated."""
    p = validate_manifest(pd.DataFrame([_row(level="T7", level_source="unknown")]))
    assert any("level_source" in x for x in p)


def test_duplicate_instance_rejected():
    df = pd.DataFrame([_row(), _row()])
    assert any("duplicate" in x for x in validate_manifest(df))


def test_null_patient_id_rejected():
    p = validate_manifest(pd.DataFrame([_row(patient_id=None)]))
    assert any("patient_id" in x for x in p)


def test_mendeley_provenance_enforced():
    """The 208 Honduran images are the only cross-site test set; losing the tag is fatal."""
    ok = _row(source="mendeley", provenance="honduran", region="lumbar", level="L3")
    assert validate_manifest(pd.DataFrame([ok])) == []
    bad = _row(source="mendeley", provenance="mendeley", region="lumbar", level="L3")
    assert any("provenance" in x for x in validate_manifest(pd.DataFrame([bad])))


def test_non_mendeley_provenance_must_equal_source():
    bad = _row(provenance="honduran")
    assert any("provenance" in x for x in validate_manifest(pd.DataFrame([bad])))


def test_kp_visible_length_enforced():
    p = validate_manifest(pd.DataFrame([_row(kp_visible=[1, 1, 1])]))
    assert any("kp_visible" in x for x in p)


def test_boxes_only_source_is_legal():
    """A source with boxes but no corners contributes v=0 and still trains detection."""
    assert validate_manifest(pd.DataFrame([_row(kp_visible=[0, 0, 0, 0])])) == []
