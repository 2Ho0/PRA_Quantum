"""Download the raw data of Kraft et al., PRX 16, 031037 (2026) from Zenodo and verify the md5 checksums.

Record 10.5281/zenodo.17750419, licence CC BY 4.0. Files land in data/raw/ (git-ignored); each zip is
extracted to data/raw/<name>/. Run:  python data/fetch_beqs.py
"""

import hashlib
import sys
import urllib.request
import zipfile
from pathlib import Path

RECORD = "https://zenodo.org/api/records/17750419/files/{}/content"
FILES = {
    "ReadMe.txt": "88b46da9ca86628a5194d68d640b1bd9",
    "state1_2.zip": "c5e9a4fd4bb0919ae6b71278e8209bee",
    "NeelDataZ51.zip": "2dbb0f66caa9efe7039772a367b472ed",
    "DataHLBicL51Ions.zip": "c130a911c71a297dbfbeb7305cd6ec36",
}
RAW = Path(__file__).resolve().parent / "raw"


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def main() -> int:
    RAW.mkdir(parents=True, exist_ok=True)
    for name, digest in FILES.items():
        target = RAW / name
        if not target.exists() or md5(target) != digest:
            urllib.request.urlretrieve(RECORD.format(name), target)
        if md5(target) != digest:
            print(f"checksum mismatch for {name}")
            return 1
        if target.suffix == ".zip":
            with zipfile.ZipFile(target) as z:
                z.extractall(RAW / target.stem)
        print(f"ok  {name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
