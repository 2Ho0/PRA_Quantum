"""Download the hardware shots of Phasecraft and Quantinuum, arXiv:2510.26300 ("Fermionic dynamics on a trapped-ion
quantum computer beyond exact classical simulation"), from Zenodo and verify the md5 checksum.

Record 10.5281/zenodo.17881394, licence CC BY 4.0. The archive lands in data/raw/phasecraft2025/ (git-ignored); only
README.md and the two shots.h5 files (U = 0 and U = 4) are extracted, the simulated expectation values are not.
Run:  python data/fetch_phasecraft2025.py
"""

import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

URL = "https://zenodo.org/api/records/17881394/files/fermionic_dynamics.zip/content"
MD5 = "fdf4b15ddf734bedbe44279a3cf5c5c3"
MEMBERS = ["fermionic_dynamics/README.md", "fermionic_dynamics/U_0/shots.h5", "fermionic_dynamics/U_4/shots.h5"]
RAW = Path(__file__).resolve().parent / "raw" / "phasecraft2025"


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    target = RAW / "fermionic_dynamics.zip"
    if not target.exists() or md5(target) != MD5:
        req = urllib.request.Request(URL, headers={"User-Agent": "research-data-fetch/1.0"})
        with urllib.request.urlopen(req, timeout=120) as r, open(target, "wb") as out:
            for chunk in iter(lambda: r.read(1 << 22), b""):
                out.write(chunk)
    if md5(target) != MD5:
        print("checksum mismatch")
        return 1
    with zipfile.ZipFile(target) as z:
        for name in MEMBERS:
            z.extract(name, RAW)
    print("ok  fermionic_dynamics.zip (README.md and shots.h5 extracted)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
