"""Download data files of Andersen et al., Nature 638, 79 (2025) from Zenodo and verify their md5 checksums.

Record 10.5281/zenodo.14060446 (Google Quantum AI, analog XY simulator on superconducting qubits), licence CC BY 4.0.
Files land in data/raw/andersen2025/ (git-ignored). The files are Python pickles: never open them with pickle.load;
use qerr.googledata.safe_load, which refuses every global outside a whitelist of numpy array constructors.

Run:  python data/fetch_andersen2025.py [NAME ...]      (default: Fig2)
"""

import hashlib
import sys
import urllib.request
from pathlib import Path

RECORD = "https://zenodo.org/api/records/14060446/files/{}/content"
FILES = {
    "Fig2": "5202caa0f517d42c92691d24dec4b7b6",
    "Fig3_1": "2c23ccb8a2837f3481134c299da6f55a",
    "Fig3_2": "b5f50de6a8a0f0c3bcc19a3c2dbc4548",
    "Fig4_1": "95476766b68d430d4b9fed24af111c0a",
    "Fig4_2": "f44183768f825ba0654334db8a6bcb46",
    "Fig5_1": "84b0341983a9ff8e48e3f278a19d4d06",
    "Fig5_2": "77d9d57094d3c99fe5a9d02a1e48f2a5",
    "Fig5_3": "5e12bc91462ef2751baf891371b7b1dd",
}
RAW = Path(__file__).resolve().parent / "raw" / "andersen2025"


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def download(name: str, target: Path) -> None:
    req = urllib.request.Request(RECORD.format(name), headers={"User-Agent": "research-data-fetch/1.0"})
    tmp = target.with_suffix(".part")
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as out:
        for chunk in iter(lambda: r.read(1 << 22), b""):
            out.write(chunk)
    tmp.replace(target)


def main() -> int:
    names = sys.argv[1:] or ["Fig2"]
    RAW.mkdir(parents=True, exist_ok=True)
    for name in names:
        digest = FILES[name]
        target = RAW / name
        if not target.exists() or md5(target) != digest:
            download(name, target)
        if md5(target) != digest:
            print(f"checksum mismatch for {name}")
            return 1
        print(f"ok  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
