"""normalize.py — every source becomes one thing: 8-bit PNG, longest edge 2048.

Three transforms, each recorded so native-resolution coordinates can be recovered:

  FACING. BUU ships every lateral pre-flipped to left-facing; CSXA and NHANES are
  unverified. If facing is mixed, corner slot semantics flip between images and the
  keypoint head learns nothing — slot 0 is anterior on one film and posterior on the next.
  Everything is normalised to left-facing and the original is kept in the manifest.

  WINDOWING. NHANES is 16-bit film scan with wide, uneven dynamic range; CSXA and BUU are
  already 8-bit and pre-windowed. Percentile windowing then CLAHE, applied only where the
  source needs it — re-windowing an already-windowed JPEG just crushes it.

  SCALE. Longest edge to 2048, aspect preserved. Coordinates scale by the same factor.

WHY NOT torchvision/PIL RESIZE ALONE: the geometry has to move with the pixels, so every
function here returns the transform it applied and `apply_to_points` replays it. A resize
that forgets its keypoints is the single most common silent corruption in this kind of
pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import cv2
import numpy as np

TARGET_LONG_EDGE = 2048


@dataclass
class Transform:
    """Everything needed to map native coordinates into the normalised image."""
    flipped: bool          # horizontal flip applied (facing normalisation)
    orig_w: int
    orig_h: int
    scale: float           # applied AFTER any flip
    out_w: int
    out_h: int

    def apply_to_points(self, pts: np.ndarray) -> np.ndarray:
        """(N,2) native pixel coords -> normalised image coords."""
        p = np.asarray(pts, dtype=float).copy()
        if self.flipped:
            p[:, 0] = (self.orig_w - 1) - p[:, 0]
        return p * self.scale

    def invert_points(self, pts: np.ndarray) -> np.ndarray:
        """Normalised image coords -> native pixel coords."""
        p = np.asarray(pts, dtype=float) / self.scale
        if self.flipped:
            p[:, 0] = (self.orig_w - 1) - p[:, 0]
        return p

    def as_dict(self):
        return asdict(self)


def to_uint8(img: np.ndarray, *, already_windowed: bool,
             lo_pct: float = 0.5, hi_pct: float = 99.5,
             clahe_clip: float = 2.0) -> np.ndarray:
    """16-bit film scan or pre-windowed 8-bit -> uint8, contrast comparable across sources."""
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if already_windowed and img.dtype == np.uint8:
        return img
    x = img.astype(np.float32)
    lo, hi = np.percentile(x, [lo_pct, hi_pct])
    if hi <= lo:
        lo, hi = float(x.min()), float(max(x.max(), x.min() + 1))
    x = np.clip((x - lo) / (hi - lo), 0, 1)
    x = (x * 255).astype(np.uint8)
    return cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8)).apply(x)


def detect_facing(img: np.ndarray) -> str:
    """Guess which way a lateral faces, by where the dense vertebral column sits.

    On a lateral the spine is the brightest sustained vertical structure and sits POSTERIOR;
    soft tissue and air fill the anterior half. So the heavier column-side is the posterior
    side, and facing is the opposite.

    This is a heuristic and it is used only to make a source self-consistent, never to
    overrule a source that documents its own facing (BUU does). Verify on a visual sample
    before trusting it on a new source.
    """
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = g.shape
    band = g[int(0.25 * h):int(0.75 * h), :]
    col = band.mean(axis=0).astype(np.float64)
    half = w // 2
    return "left" if col[:half].sum() >= col[half:].sum() else "right"


def normalize_image(img: np.ndarray, *, facing: str | None,
                    target_facing: str = "left",
                    already_windowed: bool,
                    long_edge: int = TARGET_LONG_EDGE) -> tuple[np.ndarray, Transform]:
    """Window, flip to a single facing, resize. Returns the image and its Transform."""
    g = to_uint8(img, already_windowed=already_windowed)
    oh, ow = g.shape[:2]

    flip = bool(facing is not None and facing != target_facing)
    if flip:
        g = cv2.flip(g, 1)

    scale = long_edge / max(oh, ow)
    if abs(scale - 1.0) > 1e-6:
        g = cv2.resize(g, (int(round(ow * scale)), int(round(oh * scale))),
                       interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)

    return g, Transform(flipped=flip, orig_w=ow, orig_h=oh, scale=scale,
                        out_w=g.shape[1], out_h=g.shape[0])


# ------------------------------------------------------------------------ DICOM
def read_dicom(path) -> tuple[np.ndarray, dict]:
    """Read a clinical DICOM the way inference must, or the model silently sees noise.

    Three things, all of which produce a plausible-looking image when got wrong:
      1. apply the VOI LUT / stored window — min-max on raw 12-16 bit values gives a
         completely different contrast distribution from a pre-windowed training JPEG;
      2. MONOCHROME1 is INVERTED (bone dark, background bright) and must be flipped;
      3. facing still has to be detected and normalised.
    """
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut

    ds = pydicom.dcmread(str(path))
    arr = apply_voi_lut(ds.pixel_array, ds)
    photometric = str(getattr(ds, "PhotometricInterpretation", "MONOCHROME2"))
    if photometric == "MONOCHROME1":
        arr = arr.max() - arr
    meta = {
        "photometric": photometric,
        "inverted": photometric == "MONOCHROME1",
        "rows": int(getattr(ds, "Rows", arr.shape[0])),
        "cols": int(getattr(ds, "Columns", arr.shape[1])),
        "manufacturer": str(getattr(ds, "Manufacturer", "")),
    }
    return arr, meta
