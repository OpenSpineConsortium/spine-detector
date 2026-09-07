"""export_yolo.py — manifest + normalised images -> ultralytics pose format.

LABEL LINE, one per vertebra instance:

    0  cx cy w h  x1 y1 v1  x2 y2 v2  x3 y3 v3  x4 y4 v4

all normalised to [0,1] by image size. Class is always 0 — the detector is class-agnostic
by design (see the spec's §1: per-level classes confound level identity with source dataset,
because class C3 appears only in CSXA and T7 only in AASCE).

`v` is the ultralytics visibility flag, and it does real work here: a source that gives boxes
but no corners writes v=0 on all four and still trains the detection head, because
ultralytics masks the keypoint loss wherever v == 0. No loss modification is needed for that
case — only for ignore regions, which are a separate problem (see src/ignore.py).

BOX is the axis-aligned bounding box of the four corners, dilated 5%.

SPLITS ARE PATIENT-DISJOINT. BUU pairs an AP and a lateral per patient and NHANES has
participants with both a cervical and a lumbar film, so splitting by image puts the same
spine in train and test. The Mendeley Honduran subset is forced entirely into `test`: it is
the only genuine cross-site lateral lumbar evaluation that exists, and any of it leaking into
training destroys the one number worth reporting.
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

BOX_DILATE = 0.05
CORNER_COLS = ["x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"]


def quads_of(df: pd.DataFrame) -> np.ndarray:
    return df[CORNER_COLS].to_numpy(dtype=float).reshape(-1, 4, 2)


def label_lines(df_img: pd.DataFrame, w: int, h: int) -> list[str]:
    lines = []
    for _, r in df_img.iterrows():
        q = np.array([[r.x1, r.y1], [r.x2, r.y2], [r.x3, r.y3], [r.x4, r.y4]], float)
        x0, y0 = q[:, 0].min(), q[:, 1].min()
        x1, y1 = q[:, 0].max(), q[:, 1].max()
        bw, bh = (x1 - x0), (y1 - y0)
        x0 -= BOX_DILATE * bw / 2; x1 += BOX_DILATE * bw / 2
        y0 -= BOX_DILATE * bh / 2; y1 += BOX_DILATE * bh / 2
        x0, y0 = max(x0, 0.0), max(y0, 0.0)
        x1, y1 = min(x1, w - 1.0), min(y1, h - 1.0)
        cx, cy = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h
        bw, bh = (x1 - x0) / w, (y1 - y0) / h
        if bw <= 0 or bh <= 0:
            continue
        vis = r.kp_visible if isinstance(r.kp_visible, (list, tuple, np.ndarray)) else [1, 1, 1, 1]
        kp = []
        for (kx, ky), v in zip(q, vis):
            kp += [f"{np.clip(kx / w, 0, 1):.6f}", f"{np.clip(ky / h, 0, 1):.6f}", str(int(v))]
        lines.append(" ".join(["0", f"{cx:.6f}", f"{cy:.6f}", f"{bw:.6f}", f"{bh:.6f}", *kp]))
    return lines


def assign_splits(df: pd.DataFrame, val_frac=0.1, test_frac=0.1, seed=0) -> dict[str, str]:
    """Patient-level split. The Honduran images are forced to test, whole."""
    rng = np.random.default_rng(seed)
    split_of: dict[str, str] = {}

    honduran = set(df.loc[df["provenance"] == "honduran", "patient_id"].unique())
    for p in honduran:
        split_of[p] = "test"

    rest = [p for p in df["patient_id"].unique() if p not in honduran]
    # stratify by source so every split keeps every source's acquisition style
    by_src = defaultdict(list)
    src_of = df.groupby("patient_id")["source"].first().to_dict()
    for p in rest:
        by_src[src_of[p]].append(p)
    for src, pats in by_src.items():
        pats = sorted(pats)
        rng.shuffle(pats)
        n = len(pats)
        n_test = int(round(test_frac * n))
        n_val = int(round(val_frac * n))
        for p in pats[:n_test]:
            split_of[p] = "test"
        for p in pats[n_test:n_test + n_val]:
            split_of[p] = "val"
        for p in pats[n_test + n_val:]:
            split_of[p] = "train"
    return split_of


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/interim/manifest.parquet")
    ap.add_argument("--images", default="data/interim/images")
    ap.add_argument("--out", default="data/yolo")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--symlink", action="store_true",
                    help="link images instead of copying (saves a full second copy)")
    a = ap.parse_args()

    df = pd.read_parquet(a.manifest)
    out = Path(a.out)
    for sub in ("images", "labels"):
        for sp in ("train", "val", "test"):
            (out / sub / sp).mkdir(parents=True, exist_ok=True)

    split_of = assign_splits(df, seed=a.seed)
    df["split"] = df["patient_id"].map(split_of)

    counts = defaultdict(lambda: defaultdict(int))
    n_img = defaultdict(int)
    for image_id, g in df.groupby("image_id"):
        sp = g["split"].iloc[0]
        w, h = int(g["width"].iloc[0]), int(g["height"].iloc[0])
        src_img = Path(a.images) / f"{image_id}.png"
        if not src_img.exists():
            continue
        dst_img = out / "images" / sp / f"{image_id}.png"
        if not dst_img.exists():
            if a.symlink:
                dst_img.symlink_to(src_img.resolve())
            else:
                shutil.copy2(src_img, dst_img)
        lines = label_lines(g, w, h)
        (out / "labels" / sp / f"{image_id}.txt").write_text("\n".join(lines) + "\n")
        counts[sp][g["source"].iloc[0]] += len(lines)
        n_img[sp] += 1

    yaml = out / "data.yaml"
    yaml.write_text(
        f"path: {out.resolve()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n\n"
        "kpt_shape: [4, 3]\n"
        "flip_idx: [1, 0, 3, 2]\n"      # only meaningful if fliplr is ever enabled; it is not
        "names:\n  0: vertebra\n"
    )

    print(f"wrote {yaml}")
    for sp in ("train", "val", "test"):
        tot = sum(counts[sp].values())
        print(f"  {sp:5} {n_img[sp]:>6} images  {tot:>7} instances  "
              f"{dict(sorted(counts[sp].items()))}")
    hon = df[df['provenance'] == 'honduran']
    if len(hon):
        assert set(hon['split']) == {'test'}, "Honduran images leaked out of test"
        print(f"  Honduran cross-site test set: {hon['image_id'].nunique()} images, "
              f"{len(hon)} instances — held out, verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
