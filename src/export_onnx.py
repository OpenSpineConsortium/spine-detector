"""Export a trained detector to the ONNX the website runs, and prove the conversion is safe.

    python -m src.export_onnx --weights runs/.../best.pt --tag v4 --out dist/

WHY fp16 AND NOT int8. The browser downloads this file before it can do anything, so size
is a feature. int8 halves it again -- 22 MB against 42 -- and was rejected on evidence: it
changed the detection count on 15 of 40 test films and moved corners by up to 131 px. fp16
changed no count and moved corners by 0.08 px, which is below the width of the line the
site draws. The check below is that measurement, re-run on every export, because "fp16 is
usually fine" is not a result.

WHY NOT ultralytics' own half=True. It is a TensorRT flag. Passing it to an ONNX export
does nothing at all: fp32 and "fp16" both came out 84.3 MB, byte for byte the same model.
The conversion has to be done on the ONNX graph, which is what onnxconverter_common does.

WHY A NEW FILENAME PER MODEL. GitHub Pages serves models/ with a long cache lifetime, so
overwriting the file in place leaves returning visitors on the old weights with no way to
tell. The tag goes in the filename and index.html points at the new name.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np


def _sizes(p: Path) -> str:
    return f"{p.stat().st_size / 1e6:.1f} MB"


def export_fp32(weights: Path, imgsz: int, out_dir: Path) -> Path:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    # opset 12: onnxruntime-web's WebGPU backend refuses some later ops, and the failure is
    # silent -- the page falls back to WASM and runs about eight times slower.
    produced = model.export(format="onnx", imgsz=imgsz, opset=12, simplify=True, dynamic=False)
    src = Path(produced if isinstance(produced, (str, Path)) else weights.with_suffix(".onnx"))
    dst = out_dir / f"{weights.parent.parent.name}_{imgsz}_fp32.onnx"
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"  fp32: {dst}  {_sizes(dst)}")
    return dst


def to_fp16(fp32: Path, dst: Path) -> Path:
    import onnx
    from onnxconverter_common import float16

    m = onnx.load(str(fp32))
    # keep_io_types: the page feeds float32 pixels and reads float32 boxes. Converting the
    # inputs too would need every caller to build fp16 tensors in JS, for no size saving.
    m16 = float16.convert_float_to_float16(m, keep_io_types=True)
    onnx.save(m16, str(dst))
    print(f"  fp16: {dst}  {_sizes(dst)}")
    return dst


def agree(fp32: Path, fp16: Path, imgsz: int, n: int, seed: int = 0) -> int:
    """Do the two graphs give the same answer? Detection count first, then corner drift.

    Random inputs rather than films: this compares two graphs, and the question is whether
    the numerics survived the cast, which does not depend on the picture being a spine. A
    real-image check follows separately at the eval stage.
    """
    import onnxruntime as ort

    s32 = ort.InferenceSession(str(fp32), providers=["CPUExecutionProvider"])
    s16 = ort.InferenceSession(str(fp16), providers=["CPUExecutionProvider"])
    name = s32.get_inputs()[0].name
    rng = np.random.default_rng(seed)
    worst = 0.0
    for i in range(n):
        x = rng.random((1, 3, imgsz, imgsz), dtype=np.float32)
        a = s32.run(None, {name: x})[0]
        b = s16.run(None, {s16.get_inputs()[0].name: x})[0]
        if a.shape != b.shape:
            print(f"  FAIL shape {a.shape} vs {b.shape}")
            return 1
        d = float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max())
        worst = max(worst, d)
    print(f"  max |fp32 - fp16| over {n} random inputs: {worst:.4f}")
    # The raw head output is in pixels at this scale, so a drift under a tenth of a pixel
    # cannot move a drawn corner. int8 failed this by three orders of magnitude.
    if worst > 0.5:
        print("  FAIL: conversion changed the output by more than half a pixel")
        return 1
    print("  ok: fp16 output is within half a pixel of fp32")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--tag", required=True, help="goes in the filename, e.g. v4")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--out", default="dist")
    ap.add_argument("--n_check", type=int, default=4)
    a = ap.parse_args()

    w, out = Path(a.weights), Path(a.out)
    if not w.exists():
        print(f"no such weights: {w}", file=sys.stderr)
        return 2
    out.mkdir(parents=True, exist_ok=True)

    fp32 = export_fp32(w, a.imgsz, out)
    fp16 = out / f"v11m_{a.imgsz}_{a.tag}_fp16.onnx"
    to_fp16(fp32, fp16)
    rc = agree(fp32, fp16, a.imgsz, a.n_check)
    if rc:
        print("conversion rejected; the fp32 file is kept for inspection", file=sys.stderr)
        return rc
    print(f"\nship: {fp16}  ({_sizes(fp16)})")
    print(f"point pacs/xr.js MODEL_URL at models/{fp16.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
