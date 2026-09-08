#!/usr/bin/env python3
"""Throttled Google Drive folder download for BUU-LSPINE (7,200 images + CSVs). gdown's folder
mode fires requests as fast as it can and Drive answers "many accesses" after ~70 files.
This lists the folder once, then fetches one file at a time with a pause, skips files
already on disk, and retries a failed file after a longer pause. Resumable: rerun anytime.

    python download_buu_drive.py --out data/raw/buu_drive --sleep 2.5
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import gdown

FOLDER = "https://drive.google.com/drive/folders/1pSqtxYT_B2GHweoDAVB_32JNsHykuZK4"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--sleep", type=float, default=2.5)
    ap.add_argument("--max_fail", type=int, default=25, help="stop after this many consecutive failures (quota)")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    listing = gdown.download_folder(FOLDER, output=str(a.out), skip_download=True, quiet=True, remaining_ok=True) \
        if "remaining_ok" in gdown.download_folder.__code__.co_varnames else \
        gdown.download_folder(FOLDER, output=str(a.out), skip_download=True, quiet=True)
    files = [(f.id, Path(f.path)) for f in listing] if listing else []
    print(f"{len(files)} files in the folder listing", flush=True)
    done = fail = streak = 0
    for fid, path in files:
        if path.exists() and path.stat().st_size > 0:
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        ok = False
        for attempt in range(3):
            try:
                r = gdown.download(id=fid, output=str(path), quiet=True)
                ok = r is not None and path.exists() and path.stat().st_size > 0
            except Exception as e:  # noqa: BLE001
                print("error", path.name, repr(e)[:120], flush=True)
            if ok:
                break
            time.sleep(30 * (attempt + 1))
        if ok:
            done += 1; streak = 0
        else:
            fail += 1; streak += 1
            if path.exists():
                path.unlink()
            if streak >= a.max_fail:
                print(f"{streak} consecutive failures: Drive quota; stop and rerun later", flush=True)
                break
        if done % 100 == 0 and ok:
            print(f"downloaded {done}, failed {fail}, {time.strftime('%H:%M')}", flush=True)
        time.sleep(a.sleep)
    n = sum(1 for _ in a.out.rglob("*") if _.is_file())
    print(f"done {done} failed {fail}; {n} files on disk", flush=True)
    return 0 if fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
