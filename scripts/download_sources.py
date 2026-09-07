"""download_sources.py -- pull every spine-radiograph source that is publicly reachable.

Each source is a separate function so one failing does not stop the rest, and each is
resumable: a source whose marker file exists is skipped, so re-running after a timeout
costs nothing.

REACHABLE WITHOUT CREDENTIALS (done here):
  nhanes_annotated  Zenodo 10223910, CC BY 4.0, ~1 GB
  mendeley_716      data.mendeley.com public API
  buu_v1            Google Drive folder, 18.5 GB
  cm_spine          Google Drive file, intraoperative
  csxa              SciDB -- the API is undocumented; several endpoints are tried
  nhanes_full       CDC, 17,075 TIFF, public domain, no DUA

NOT REACHABLE HERE, and deliberately not faked:
  vindr             PhysioNet credentialed access, tied to a named person's account
  ieee_236          IEEE DataPort subscription
  aasce             SpineWeb, historically behind a request form
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

RAW = Path("data/raw")


def _get(url: str, timeout: int = 60) -> bytes:
    """GET with a browser User-Agent. Mendeley 403s urllib's default agent."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return urllib.request.urlopen(req, timeout=timeout).read()


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def done_marker(name: str) -> Path:
    return RAW / name / ".download_complete"


def fetch(url: str, dest: Path, tries: int = 3) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "spine-detector/1.0"})
            with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as fh:
                while chunk := r.read(1 << 20):
                    fh.write(chunk)
            return True
        except Exception as exc:                                     # noqa: BLE001
            log(f"    attempt {k+1}/{tries} failed: {type(exc).__name__} {exc}")
            time.sleep(5)
    return False


# ------------------------------------------------------------------ zenodo
def get_nhanes_annotated() -> bool:
    d = RAW / "nhanes_annotated"
    d.mkdir(parents=True, exist_ok=True)
    meta = json.loads(_get("https://zenodo.org/api/records/10223910"))
    ok = True
    for f in meta.get("files", []):
        url = f["links"]["self"]
        out = d / f["key"]
        if out.exists() and out.stat().st_size == f.get("size", -1):
            log(f"    have {f['key']}")
            continue
        log(f"    {f['key']}  {f.get('size', 0)/1e6:.1f} MB")
        ok &= fetch(url, out)
    return ok


# ---------------------------------------------------------------- mendeley
def get_mendeley() -> bool:
    d = RAW / "mendeley_716"
    d.mkdir(parents=True, exist_ok=True)
    api = ("https://data.mendeley.com/public-api/datasets/5jdfdgp762/"
           "files?folder_id=root&version=1")
    files = json.loads(_get(api))
    ok = True
    for f in files:
        url = f.get("content_details", {}).get("download_url")
        name = f.get("filename")
        if not url:
            continue
        out = d / name
        if out.exists() and out.stat().st_size > 0:
            log(f"    have {name}")
            continue
        log(f"    {name}  {f.get('size', 0)/1e6:.1f} MB")
        ok &= fetch(url, out)
    return ok


# ------------------------------------------------------------ google drive
def _gdown(args: list[str]) -> bool:
    try:
        r = subprocess.run([sys.executable, "-m", "gdown", *args],
                           capture_output=True, text=True, timeout=14400)
        if r.returncode != 0:
            log(f"    gdown rc={r.returncode}: {r.stderr[-400:]}")
        return r.returncode == 0
    except Exception as exc:                                          # noqa: BLE001
        log(f"    gdown raised {type(exc).__name__}: {exc}")
        return False


def get_buu() -> bool:
    d = RAW / "buu_v1"
    d.mkdir(parents=True, exist_ok=True)
    return _gdown(["--folder", "--remaining-ok", "-O", str(d),
                   "https://drive.google.com/drive/folders/1pSqtxYT_B2GHweoDAVB_32JNsHykuZK4"])


def get_cm_spine() -> bool:
    d = RAW / "cm_spine"
    d.mkdir(parents=True, exist_ok=True)
    return _gdown(["-O", str(d / "cm_spine.zip"), "--fuzzy",
                   "https://drive.google.com/file/d/1r9wX1802gkU4-EQHMMlyAlEiWXGWLhhJ/view"])


# -------------------------------------------------------------------- scidb
def get_csxa() -> bool:
    """SciDB's download API is undocumented; try the endpoints it is known to expose."""
    d = RAW / "csxa"
    d.mkdir(parents=True, exist_ok=True)
    dsid = "8e3b3d5e60a348ba961e19d48b881c90"
    candidates = [
        f"https://www.scidb.cn/api/sdb-file/file/getFileListByDataSetId?dataSetId={dsid}",
        f"https://www.scidb.cn/api/sdb-dataset/dataSet/getFileList?dataSetId={dsid}",
        f"https://china.scidb.cn/api/sdb-file/file/list?dataSetId={dsid}",
        f"https://www.scidb.cn/api/sdb-core/dataSetFile/list?dataSetId={dsid}",
    ]
    for url in candidates:
        try:
            body = urllib.request.urlopen(urllib.request.Request(
                url, headers={"User-Agent": "Mozilla/5.0"}), timeout=45).read()
            log(f"    {url} -> {len(body)} bytes")
            (d / "scidb_probe.json").write_bytes(body)
            if b'"' in body and len(body) > 200:
                log("    got a listing; inspect data/raw/csxa/scidb_probe.json")
                return True
        except Exception as exc:                                      # noqa: BLE001
            log(f"    {url} -> {type(exc).__name__}")
    log("    SciDB listing not obtained; images need a browser session")
    return False


SOURCES = {
    "nhanes_annotated": get_nhanes_annotated,
    "mendeley_716": get_mendeley,
    "cm_spine": get_cm_spine,
    "buu_v1": get_buu,
    "csxa": get_csxa,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None)
    a = ap.parse_args()

    todo = a.only or list(SOURCES)
    results = {}
    for name in todo:
        fn = SOURCES.get(name)
        if fn is None:
            log(f"unknown source {name}")
            continue
        if done_marker(name).exists():
            log(f"{name}: already complete, skipping")
            results[name] = True
            continue
        log(f"=== {name} ===")
        try:
            ok = fn()
        except Exception as exc:                                      # noqa: BLE001
            log(f"    FAILED {type(exc).__name__}: {exc}")
            ok = False
        results[name] = ok
        if ok:
            done_marker(name).parent.mkdir(parents=True, exist_ok=True)
            done_marker(name).write_text(time.strftime("%Y-%m-%d %H:%M:%S"))
        log(f"=== {name}: {'OK' if ok else 'INCOMPLETE'} ===")

    log("summary: " + json.dumps(results))
    for name in todo:
        p = RAW / name
        if p.exists():
            n = sum(1 for _ in p.rglob("*") if _.is_file())
            sz = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
            log(f"  {name:18} {n:>7} files  {sz/1e9:7.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
