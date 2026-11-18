"""Download the raw data of Joshi et al., "Observing the quantum Mpemba effect in quantum simulations",
PRL 133, 010402 (2024), from Zenodo and verify the md5 checksum.

Record 10.5281/zenodo.10390690, licence CC BY 4.0. The archive lands in data/raw/mpemba2024/ (git-ignored); only the
readme, the applied random unitaries and the dephasing-only runs (interactions off, three tilt angles) are extracted.
Run:  python data/fetch_mpemba2024.py
"""

import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://zenodo.org/api/records/10390690/files/MpembaDataUpload.zip/content"
MD5 = "5b605989f93f573ef17dc3dc97b10ae0"
PREFIXES = ("MpembaDataUpload/Readme.txt", "MpembaDataUpload/AppliedRandomUnitaries/",
            "MpembaDataUpload/RawData/DephasingData/")
RAW = Path(__file__).resolve().parent / "raw" / "mpemba2024"


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    target = RAW / "MpembaDataUpload.zip"
    if not target.exists() or md5(target) != MD5:
        req = urllib.request.Request(URL, headers={"User-Agent": "research-data-fetch/1.0"})
        with urllib.request.urlopen(req, timeout=120) as r, open(target, "wb") as out:
            for chunk in iter(lambda: r.read(1 << 22), b""):
                out.write(chunk)
    if md5(target) != MD5:
        print("checksum mismatch")
        return 1
    with zipfile.ZipFile(target) as z:
        members = [n for n in z.namelist() if n.startswith(PREFIXES) and not n.endswith("/")]
        for name in members:
            z.extract(name, RAW)
    print(f"ok  MpembaDataUpload.zip ({len(members)} files extracted)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
