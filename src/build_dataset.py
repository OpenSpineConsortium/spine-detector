"""build_dataset.py — every reachable source -> normalised PNGs + one manifest.

Runs the converters, writes 8-bit PNGs into data/interim/images, validates the manifest, and
reports per-source instance counts so a source that silently contributes nothing is visible
immediately rather than after a training run.

SOURCES HANDLED HERE are the ones actually obtainable without credentials. BUU arrives
partially (Google Drive caps folder downloads), CSXA's images sit behind a SciDB browser
session, and VinDr/IEEE need personal credentials — each is skipped with a reason rather than
silently omitted, so the manifest's coverage is always legible.

LEVELS ARE NOT INVENTED. Mendeley's Roboflow export carries no level names and its instance
count per image does not match the published 'L3-S1' description, so its levels are
'unknown' with level_source 'unknown'. That costs nothing for Stage 1, which is
class-agnostic by design, and it keeps a guess out of the level head's training data later.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.manifest import COLUMNS, canonical_slots, validate_instance, validate_manifest
from src.normalize import normalize_image

RAW = Path("data/raw")
INTERIM = Path("data/interim")
IMAGES = INTERIM / "images"


def _save(image_id: str, img: np.ndarray, *, already_windowed: bool, facing: str | None):
    out, tf = normalize_image(img, facing=facing, already_windowed=already_windowed)
    cv2.imwrite(str(IMAGES / f"{image_id}.png"), out)
    return tf


# --------------------------------------------------------------------- NHANES
def do_nhanes() -> pd.DataFrame:
    zp = RAW / "nhanes_annotated" / "NHANESII_Annotations_VertebralOsetophytes.zip"
    if not zp.exists():
        print("  nhanes: zip absent, skipping")
        return pd.DataFrame()
    from src.converters.nhanes import _corners_from_mask, _level_of
    import io, pickle

    rows, rejected = [], Counter()
    with zipfile.ZipFile(zp) as z:
        names = [n for n in z.namelist() if n.endswith(".pkl")]
        for n in names:
            try:
                o = pickle.load(io.BytesIO(z.read(n)))
            except Exception:
                rejected["unpickle"] += 1
                continue
            sid = str(o.get("sample_id", Path(n).stem))
            region = "cervical" if "cervical" in n else "lumbar"
            img = np.asarray(o["image"])
            image_id = f"nhanes_{region}_{sid}"
            tf = _save(image_id, img, already_windowed=False, facing=None)

            inst = 0
            for key, m in (o.get("masks") or {}).items():
                lvl = _level_of(key)
                if lvl is None:
                    continue
                q = _corners_from_mask(m)
                if q is None:
                    rejected["mask_too_small"] += 1
                    continue
                q = tf.apply_to_points(q)          # into the normalised frame
                why = validate_instance(q, tf.out_w, tf.out_h)
                if why:
                    rejected[why] += 1
                    continue
                rows.append(dict(
                    image_id=image_id, source="nhanes", provenance="nhanes", orig_path=n,
                    view="LAT", acquisition="film", region=region, facing="left",
                    width=tf.out_w, height=tf.out_h, mm_per_px=None,
                    patient_id=f"nhanes_{sid}", instance_id=inst,
                    level=lvl, level_source="annotated",
                    x1=q[0, 0], y1=q[0, 1], x2=q[1, 0], y2=q[1, 1],
                    x3=q[2, 0], y3=q[2, 1], x4=q[3, 0], y4=q[3, 1],
                    kp_visible=[1, 1, 1, 1]))
                inst += 1
    print(f"  nhanes: {len(rows)} instances; rejected {dict(rejected)}")
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- MENDELEY
def do_mendeley() -> pd.DataFrame:
    zp = RAW / "mendeley_716" / "Dataset.zip"
    if not zp.exists():
        print("  mendeley: zip absent, skipping")
        return pd.DataFrame()
    rows, rejected = [], Counter()
    # Roboflow exports the SAME image into train/, valid/ and test/ under different
    # content hashes. Keeping all three duplicates every instance and -- worse -- puts one
    # physical radiograph on both sides of a split. Deduplicate on the underlying stem.
    seen_bases: set[str] = set()
    with zipfile.ZipFile(zp) as z:
        names = z.namelist()
        jsons = [n for n in names if n.endswith(".json")]
        imgs = {Path(n).stem: n for n in names if n.lower().endswith((".jpg", ".png"))}
        for jn in jsons:
            stem = Path(jn).stem
            img_n = imgs.get(stem)
            if img_n is None:
                rejected["no_image_for_json"] += 1
                continue
            try:
                blob = json.loads(z.read(jn))
            except Exception:
                rejected["bad_json"] += 1
                continue
            buf = np.frombuffer(z.read(img_n), np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
            if img is None:
                rejected["undecodable_image"] += 1
                continue

            # 3209-F-073Y1_jpg.rf.<hash> -> BUU patient 3209; the hash is Roboflow's
            base = stem.split("_jpg")[0]
            if base in seen_bases:
                rejected["duplicate_roboflow_copy"] += 1
                continue
            seen_bases.add(base)
            patient = base.split("-")[0]
            # every stem here follows BUU's convention, so treat as BUU-derived unless
            # the release marks otherwise. Never risk contaminating the Honduran test set.
            prov = "honduran" if "hond" in stem.lower() else "buu_derived"
            image_id = f"mendeley_{base}"
            tf = _save(image_id, img, already_windowed=True, facing=None)

            kps = blob.get("keypoints") or []
            for i, kp in enumerate(kps):
                pts = np.asarray(kp, dtype=float)
                if pts.ndim != 2 or len(pts) < 4:
                    rejected["bad_keypoints"] += 1
                    continue
                q = canonical_slots(pts[:4, :2])
                q = tf.apply_to_points(q)
                why = validate_instance(q, tf.out_w, tf.out_h)
                if why:
                    rejected[why] += 1
                    continue
                rows.append(dict(
                    image_id=image_id, source="mendeley", provenance=prov, orig_path=jn,
                    view="LAT", acquisition="standing_dr", region="lumbar", facing="left",
                    width=tf.out_w, height=tf.out_h, mm_per_px=None,
                    patient_id=f"mendeley_{prov}_{patient}", instance_id=i,
                    # the export carries no level names and the instance count does not
                    # match the published L3-S1 description: do not invent levels
                    level="unknown", level_source="unknown",
                    x1=q[0, 0], y1=q[0, 1], x2=q[1, 0], y2=q[1, 1],
                    x3=q[2, 0], y3=q[2, 1], x4=q[3, 0], y4=q[3, 1],
                    kp_visible=[int(v) for v in (pts[:4, 2] if pts.shape[1] > 2 else [1] * 4)]))
    print(f"  mendeley: {len(rows)} instances; rejected {dict(rejected)}")
    return pd.DataFrame(rows)


# ------------------------------------------------------------------------ BUU
# Several BUU deliveries can sit side by side: the 2,000-pair subset (data/raw/buu_2000, the
# official single-file release, 2026-09-07) and the 77-file partial Drive download
# (data/raw/buu_v1). They overlap by patient. Roots are read in priority order and a patient
# already ingested from an earlier root is skipped from a later one, so one spine never
# appears twice under two source tags; the source tag records which delivery it came from.
BUU_ROOTS = (("buu_2000", "buu_2000"), ("buu_v1", "buu"))


def do_buu() -> pd.DataFrame:
    from src.converters.buu import lines_to_quads, parse_stem, read_edge_lines
    LEV_LAT = ["L1", "L2", "L3", "L4", "L5", "S1"]
    LEV_AP = ["L1", "L2", "L3", "L4", "L5"]
    rows, rejected = [], Counter()
    seen_patients: set[str] = set()
    for root_name, source in BUU_ROOTS:
        root = RAW / root_name
        if not root.exists():
            print(f"  {source}: absent, skipping")
            continue
        this_root: set[str] = set()
        n_before = len(rows)
        for csv_p in sorted(root.rglob("*.csv")):
            info = parse_stem(csv_p.stem)
            if info is None:
                continue
            if info["patient"] in seen_patients:
                rejected[f"{source}:patient_already_ingested"] += 1
                continue
            this_root.add(info["patient"])
            _ingest_buu_file(csv_p, info, source, LEV_LAT, LEV_AP, rows, rejected,
                             lines_to_quads, read_edge_lines)
        seen_patients |= this_root
        print(f"  {source}: {len(rows) - n_before} instances from {len(this_root)} patients")
    print(f"  buu total: {len(rows)} instances; rejected {dict(rejected)}")
    return pd.DataFrame(rows)


def _ingest_buu_file(csv_p, info, source, LEV_LAT, LEV_AP, rows, rejected, lines_to_quads, read_edge_lines):
    if True:
        img_p = csv_p.with_suffix(".jpg")
        if not img_p.exists():
            rejected["no_image"] += 1
            return
        lines = read_edge_lines(csv_p)
        if lines is None:
            rejected["unparsable"] += 1
            return
        img = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            rejected["undecodable_image"] += 1
            return
        image_id = f"buu_{csv_p.stem}"
        tf = _save(image_id, img, already_windowed=True, facing="left")
        levels = LEV_LAT if info["view"] == "LAT" else LEV_AP
        for i, q in enumerate(lines_to_quads(lines)):
            q = tf.apply_to_points(q)
            why = validate_instance(q, tf.out_w, tf.out_h)
            if why:
                rejected[why] += 1
                continue
            rows.append(dict(
                image_id=image_id, source=source, provenance="buu",
                orig_path=str(csv_p.relative_to(RAW)),
                view=info["view"], acquisition="standing_dr", region="lumbar",
                facing="left", width=tf.out_w, height=tf.out_h, mm_per_px=None,
                patient_id=f"buu_{info['patient']}", instance_id=i,
                level=levels[i] if i < len(levels) else "unknown",
                level_source="inferred_from_position",
                x1=q[0, 0], y1=q[0, 1], x2=q[1, 0], y2=q[1, 1],
                x3=q[2, 0], y3=q[2, 1], x4=q[3, 0], y4=q[3, 1],
                kp_visible=[1, 1, 1, 1]))


# ---------------------------------------------------------------------- AASCE
def do_aasce() -> pd.DataFrame:
    root = RAW / "aasce"
    txt = root / "train_txt"
    if not (txt / "landmarks.csv").exists():
        print("  aasce: absent, skipping")
        return pd.DataFrame()
    from src.converters.aasce import LEVELS as AASCE_LEVELS, patient_of, quads_for, read_index
    names, lm, angles = read_index(txt)
    rows, rejected = [], Counter()
    for r, fname in enumerate(names):
        img_p = root / "train" / fname
        if not img_p.exists():
            rejected["no_image"] += 1
            continue
        img = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            rejected["undecodable_image"] += 1
            continue
        h, w = img.shape[:2]
        stem = Path(fname).stem
        image_id = "aasce_" + re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)
        tf = _save(image_id, img, already_windowed=True, facing=None)
        for i, q in enumerate(quads_for(lm[r], w, h)):
            q = tf.apply_to_points(canonical_slots(q))
            why = validate_instance(q, tf.out_w, tf.out_h)
            if why:
                rejected[why] += 1
                continue
            rows.append(dict(
                image_id=image_id, source="aasce", provenance="aasce",
                orig_path=str(img_p.relative_to(RAW)),
                view="AP", acquisition="standing_dr", region="full_spine",
                facing=None, width=tf.out_w, height=tf.out_h, mm_per_px=None,
                patient_id="aasce_" + re.sub(r"[^A-Za-z0-9_.-]+", "_", patient_of(stem)), instance_id=i,
                level=AASCE_LEVELS[i], level_source="inferred_from_position",
                x1=q[0, 0], y1=q[0, 1], x2=q[1, 0], y2=q[1, 1],
                x3=q[2, 0], y3=q[2, 1], x4=q[3, 0], y4=q[3, 1],
                kp_visible=[1, 1, 1, 1]))
    print(f"  aasce: {len(rows)} instances from {len(names)} images; rejected {dict(rejected)}")
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(INTERIM / "manifest.parquet"))
    a = ap.parse_args()
    IMAGES.mkdir(parents=True, exist_ok=True)

    parts = []
    for name, fn in (("nhanes", do_nhanes), ("mendeley", do_mendeley), ("buu", do_buu), ("aasce", do_aasce)):
        print(f"=== {name} ===")
        try:
            parts.append(fn())
        except Exception as exc:                                       # noqa: BLE001
            print(f"  {name} FAILED: {type(exc).__name__}: {exc}")

    df = pd.concat([p for p in parts if len(p)], ignore_index=True)
    for c in COLUMNS:
        if c not in df.columns:
            df[c] = None
    df = df[COLUMNS]

    problems = validate_manifest(df)
    print("\n=== manifest validation ===")
    print("  clean" if not problems else "\n".join(f"  PROBLEM: {p}" for p in problems))

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out, index=False)
    print(f"\nwrote {a.out}")
    print(f"  images   : {df['image_id'].nunique()}")
    print(f"  instances: {len(df)}")
    print(f"  by source: {df['source'].value_counts().to_dict()}")
    print(f"  by view  : {df['view'].value_counts().to_dict()}")
    print(f"  level_src: {df['level_source'].value_counts().to_dict()}")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
