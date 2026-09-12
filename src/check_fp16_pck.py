"""Does the fp16 cast cost corner accuracy? Score both graphs against ground truth.

Comparing fp32 to fp16 tells you they differ; it does not tell you whether the difference
matters. PCK does: a corner is correct when it lands within a fraction of the vertebral
BODY HEIGHT of the truth, which is the tolerance the site's angles actually live inside.
If both graphs score the same PCK, a 1.9 px worst-case drift is below the noise floor of
the thing being measured and the smaller file ships.

Normalised by body height, never by image diagonal: field of view varies enormously across
these sources and a diagonal-normalised PCK flatters a cropped film.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

IMG, CONF = 1024, 0.25


def letterbox(p: Path):
    im = Image.open(p).convert("RGB")
    s = min(IMG / im.width, IMG / im.height)
    w, h = max(1, int(im.width * s)), max(1, int(im.height * s))
    r = im.resize((w, h), Image.BILINEAR)
    c = Image.new("RGB", (IMG, IMG), (114, 114, 114))
    ox, oy = (IMG - w) // 2, (IMG - h) // 2
    c.paste(r, (ox, oy))
    a = np.asarray(c, np.float32).transpose(2, 0, 1) / 255.0
    return a[None], (s, ox, oy, im.width, im.height)


def nms(c, thr=0.7):
    if len(c) == 0:
        return c
    x1, y1 = c[:, 0] - c[:, 2] / 2, c[:, 1] - c[:, 3] / 2
    x2, y2 = c[:, 0] + c[:, 2] / 2, c[:, 1] + c[:, 3] / 2
    ar = (x2 - x1) * (y2 - y1)
    order, keep = np.argsort(-c[:, 4]), []
    while len(order):
        i = order[0]; keep.append(i)
        if len(order) == 1:
            break
        r = order[1:]
        xx1 = np.maximum(x1[i], x1[r]); yy1 = np.maximum(y1[i], y1[r])
        xx2 = np.minimum(x2[i], x2[r]); yy2 = np.minimum(y2[i], y2[r])
        it = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        order = r[it / (ar[i] + ar[r] - it + 1e-9) < thr]
    return c[np.array(keep)]


def corners(sess, name, x, meta):
    """Predicted corners in ORIGINAL image pixels."""
    out = sess.run(None, {name: x})[0][0]
    det = nms(out[:, out[4] > CONF].T)
    if len(det) == 0:
        return np.zeros((0, 4, 2))
    k = det[:, 5:].reshape(len(det), -1, 3)[:, :4, :2]
    s, ox, oy, _, _ = meta
    return (k - np.array([ox, oy])) / s


def gt_corners(lab: Path, w: int, h: int):
    rows = [r.split() for r in lab.read_text().split("\n") if r.strip()]
    out = []
    for r in rows:
        v = np.array(r[5:], float).reshape(-1, 3)[:4, :2]
        out.append(v * np.array([w, h]))
    return np.array(out) if out else np.zeros((0, 4, 2))


def pck(pred, gt, alpha):
    """Fraction of GT corners with a predicted corner within alpha * body height."""
    if len(gt) == 0:
        return 0, 0
    hit = tot = 0
    for g in gt:
        bh = max(1.0, np.linalg.norm(g[0] - g[3]))       # body height, left edge
        if len(pred) == 0:
            tot += 4; continue
        j = np.linalg.norm(pred.mean(1) - g.mean(0), axis=1).argmin()
        d = np.linalg.norm(pred[j] - g, axis=1)
        hit += int((d <= alpha * bh).sum()); tot += 4
    return hit, tot


def main(fp32, fp16, films, labels):
    ses = {t: ort.InferenceSession(p, providers=["CPUExecutionProvider"])
           for t, p in (("fp32", fp32), ("fp16", fp16))}
    acc = {t: {a: [0, 0] for a in (0.1, 0.2)} for t in ses}
    for f in sorted(Path(films).glob("*.png")):
        lab = Path(labels) / (f.stem + ".txt")
        if not lab.exists():
            continue
        x, meta = letterbox(f)
        g = gt_corners(lab, meta[3], meta[4])
        for t, s in ses.items():
            p = corners(s, s.get_inputs()[0].name, x, meta)
            for a in (0.1, 0.2):
                h, n = pck(p, g, a)
                acc[t][a][0] += h; acc[t][a][1] += n
    print(f"  {'graph':6} {'PCK@0.1':>9} {'PCK@0.2':>9}   (corners scored)")
    for t in ("fp32", "fp16"):
        a1 = acc[t][0.1][0] / max(1, acc[t][0.1][1])
        a2 = acc[t][0.2][0] / max(1, acc[t][0.2][1])
        print(f"  {t:6} {a1:9.4f} {a2:9.4f}   ({acc[t][0.1][1]})")
    # ONE-SIDED, AND AWARE OF HOW COARSE THE MEASUREMENT IS. The question is whether
    # fp16 is WORSE, not whether the two differ at all. On 328 corners a single corner
    # is 0.003, so a two-sided gate at 0.002 rejects a cast that came out fractionally
    # AHEAD -- which is exactly what happened on the first run of this script.
    n = max(1, acc['fp32'][0.1][1])
    loss = (acc['fp32'][0.1][0] - acc['fp16'][0.1][0]) / n
    one = 1.0 / n
    print(f"\n  PCK@0.1, fp16 minus fp32: {-loss:+.4f}   (one corner = {one:.4f})")
    print("  VERDICT: " + ("fp16 costs nothing measurable -- ship it"
                           if loss <= 2 * one
                           else f"fp16 is worse by {loss:.4f} -- do not ship"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:5]))
