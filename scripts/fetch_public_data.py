"""Download versioned public-data archives into the ignored local data folder."""

import argparse
import hashlib
import json
import os
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "keele": {
        "filename": "KEELE.zip",
        "url": "https://zenodo.org/records/3921794/files/KEELE.zip?download=1",
        "page": "https://zenodo.org/records/3921794",
        "license_note": "KEELE corpus: noncommercial use stated by the Zenodo record",
        "md5": "f5a87014bad14744660b90187de7d43f",
    },
    "cudb": {
        "filename": "cudb-1.0.0.zip",
        "url": "https://physionet.org/content/cudb/get-zip/1.0.0/",
        "page": "https://physionet.org/content/cudb/1.0.0/",
        "license_note": "Open Data Commons Attribution License v1.0",
    },
}


def verify(archive, dataset):
    info = SOURCES[dataset]
    if dataset == "keele":
        actual = hashlib.md5(archive.read_bytes()).hexdigest()
        if actual != info["md5"]:
            raise ValueError(f"KEELE.zip MD5 mismatch: {actual}")
    with zipfile.ZipFile(archive) as z:
        names = {Path(name).name: name for name in z.namelist()}
        if dataset == "keele":
            if "laryngograph.wav" not in names:
                raise ValueError("KEELE.zip lacks a laryngograph WAV")
        else:
            required = ("cu01.dat", "cu01.atr", "cu01.hea", "SHA256SUMS.txt")
            if any(name not in names for name in required):
                raise ValueError("CUDB ZIP lacks required cu01 files or SHA256SUMS.txt")
            sums = dict(
                line.split(maxsplit=1)[::-1]
                for line in z.read(names["SHA256SUMS.txt"]).decode().splitlines()
            )
            for name in required[:-1]:
                if hashlib.sha256(z.read(names[name])).hexdigest() != sums[name]:
                    raise ValueError(f"CUDB checksum mismatch: {name}")
    return hashlib.sha256(archive.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=SOURCES)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Verify an existing ZIP without writing",
    )
    args = parser.parse_args()
    info = SOURCES[args.dataset]
    path = args.data_dir / info["filename"]
    if not path.exists():
        if args.check_only:
            parser.error(f"missing {path}")
        args.data_dir.mkdir(parents=True, exist_ok=True)
        partial = path.with_suffix(path.suffix + ".part")
        request = urllib.request.Request(info["url"], headers={"User-Agent": "iceemdan-gpu/0.2.0"})
        with (
            urllib.request.urlopen(request, timeout=60) as response,
            partial.open("wb") as target,
        ):
            total = 0
            while block := response.read(1024 * 1024):
                total += len(block)
                if total > 100_000_000:
                    raise ValueError("Archive exceeds 100 MB download limit")
                target.write(block)
        checksum = verify(partial, args.dataset)
        os.replace(partial, path)
    else:
        checksum = verify(path, args.dataset)
    print(f"{args.dataset}: {path} sha256={checksum}")
    if not args.check_only:
        (args.data_dir / f"{args.dataset}.json").write_text(
            json.dumps(
                {
                    "dataset": args.dataset,
                    "versioned_url": info["url"],
                    "source_page": info["page"],
                    "license_note": info["license_note"],
                    "sha256": checksum,
                    "bytes": path.stat().st_size,
                    "checked_utc": datetime.now(timezone.utc).isoformat(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
