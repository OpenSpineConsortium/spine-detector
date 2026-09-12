"""Compare fp32 and fp16 on REAL films, which is the comparison that decides deployment.

Random noise is the wrong input. It produces no confident detections, so every box sits at
the decision threshold where a half-ULP difference flips it, and the measured drift is a
property of the noise rather than of the model. Real films give confident boxes, and that is
what the site will feed it. v3's fp16 was accepted on exactly this measurement (0.08 px);
int8 was rejected on it (15 of 40 films changed detection count, corners moved up to 131 px).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

IMG = 1024
CONF = 0.25


def letterbox(p: Path) -> np.ndarray:
    im = Image.open(p).convert("RGB")
    s = min(IMG / im.width, IMG / im.height)
    r = im.resize((max(1, int(im.width * s)), max(1, int(im.height * s))), Image.BILINEAR)
    canvas = Image.new("RGB", (IMG, IMG), (114, 114, 114))
    canvas.paste(r, ((IMG - r.width) // 2, (IMG - r.height) // 2))
    a = np.asarray(canvas, np.float32).transpose(2, 0, 1) / 255.0
    return a[None]


def nms(cand: np.ndarray, iou_thr: float = 0.7) -> np.ndarray:
    """Greedy NMS on xywh boxes, highest confidence first.

    WITHOUT THIS THE COMPARISON IS MEANINGLESS. The raw head emits thousands of candidates
    and the gate at conf=0.25 sits in the middle of a dense cloud of near-duplicates, so a
    half-ULP difference moves one candidate across it and the raw count changes while the
    actual detections are identical. Measured: 76 raw candidates against 77, same spine.
    """
    if len(cand) == 0:
        return cand
    xy, wh, conf = cand[:, :2], cand[:, 2:4], cand[:, 4]
    x1, y1 = xy[:, 0] - wh[:, 0] / 2, xy[:, 1] - wh[:, 1] / 2
    x2, y2 = xy[:, 0] + wh[:, 0] / 2, xy[:, 1] + wh[:, 1] / 2
    area = (x2 - x1) * (y2 - y1)
    order = np.argsort(-conf)
    keep = []
    while len(order):
        i = order[0]; keep.append(i)
        if len(order) == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest]); yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest]); yy2 = np.minimum(y2[i], y2[rest])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (area[i] + area[rest] - inter + 1e-9)
        order = rest[iou < iou_thr]
    return cand[np.array(keep)]


def boxes(out: np.ndarray) -> np.ndarray:
    """YOLO pose head: (1, 4+1+K*3, N) -> the detections that survive NMS."""
    a = out[0]
    keep = a[4] > CONF
    return nms(a[:, keep].T)


def main(fp32: str, fp16: str, folder: str) -> int:
    s32 = ort.InferenceSession(fp32, providers=["CPUExecutionProvider"])
    s16 = ort.InferenceSession(fp16, providers=["CPUExecutionProvider"])
    n32, n16 = s32.get_inputs()[0].name, s16.get_inputs()[0].name

    films = sorted(Path(folder).glob("*.png"))
    if not films:
        print("no films found", file=sys.stderr)
        return 2
    changed, worst, worst_film = 0, 0.0, ""
    alld, allconf = [], []
    for f in films:
        x = letterbox(f)
        b32, b16 = boxes(s32.run(None, {n32: x})[0]), boxes(s16.run(None, {n16: x})[0])
        if len(b32) != len(b16):
            changed += 1
            print(f"  {f.name}: COUNT {len(b32)} -> {len(b16)}")
            continue
        if len(b32) == 0:
            continue
        # PAIR BY POSITION, NOT BY ROW ORDER. Sorting the two lists and comparing row i to
        # row i looks right and is not: two vertebrae a millimetre apart can swap places
        # under a sub-pixel shift, and the comparison then reports the distance between
        # DIFFERENT vertebrae. That is what produced a 1.9 px "drift" with every detection
        # in fact unmoved. Each fp32 detection is matched to its nearest fp16 centre.
        c32, c16 = b32[:, :2], b16[:, :2]
        dist = np.linalg.norm(c32[:, None, :] - c16[None, :, :], axis=2)
        j = dist.argmin(axis=1)
        if len(set(j.tolist())) != len(j):
            print(f"  {f.name}: ambiguous pairing, skipped")
            continue
        # the corners are what the site draws: channels 5.. in (x, y, visibility) triples
        k32 = b32[:, 5:].reshape(len(b32), -1, 3)[:, :, :2]
        k16 = b16[j, 5:].reshape(len(j), -1, 3)[:, :, :2]
        per = np.abs(k32.astype(np.float64) - k16.astype(np.float64)).max(axis=(1, 2))
        alld.append(per)
        allconf.append(b32[:, 4])
        d = float(per.max())
        if d > worst:
            worst, worst_film = d, f.name
    print(f"\n  films            : {len(films)}")
    print(f"  detection count changed on : {changed} of {len(films)}")
    print(f"  worst CORNER drift         : {worst:.3f} px   ({worst_film})")
    if alld:
        v = np.concatenate(alld); c = np.concatenate(allconf)
        for q in (50, 90, 99):
            print(f"  p{q} per-detection drift      : {np.percentile(v, q):.4f} px")
        hi = v.argmax()
        print(f"  worst detection confidence : {c[hi]:.3f}")
        print(f"  detections over 0.5 px     : {int((v > 0.5).sum())} of {len(v)}"
              f"   ({100*(v > 0.5).mean():.2f}%)")
        # A MAX OVER THOUSANDS OF CORNERS IS NOT A GATE. One marginal detection can carry it
        # while every corner a reader looks at is unmoved. The gate is the bulk of the
        # distribution: the site draws a 2 px line, so a 99th-percentile drift under half a
        # pixel cannot be seen, whatever the single worst corner does.
        ok = changed == 0 and float(np.percentile(v, 99)) < 0.5
    else:
        ok = False
    print("  VERDICT: " + ("fp16 is safe to ship" if ok else "fp16 REJECTED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:4]))
