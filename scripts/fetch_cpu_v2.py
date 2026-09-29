"""Fetch and verify the pinned public CPU source used by the matched sweep."""

import argparse
import hashlib
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
URL = "https://raw.githubusercontent.com/KimiK1983/iceemdan-cpu/v2.0.0/ICEEMDAN.py"
SHA256 = "2de7564f9f01560ff1d3b1d87af12e88e34b3647152616dce5d2e90c926b695f"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "ICEEMDAN_cpu_v2.0.0.py")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        payload = args.output.read_bytes()
    elif args.check_only:
        parser.error(f"missing {args.output}")
    else:
        with urllib.request.urlopen(URL, timeout=60) as response:
            payload = response.read(1_000_001)
        if len(payload) > 1_000_000:
            raise ValueError("CPU source exceeds 1 MB limit")
    actual = hashlib.sha256(payload).hexdigest()
    if actual != SHA256:
        raise ValueError(f"CPU source SHA-256 mismatch: {actual}")
    if not args.output.exists():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        partial = args.output.with_suffix(args.output.suffix + ".part")
        partial.write_bytes(payload)
        os.replace(partial, args.output)
    print(f"{args.output} sha256={actual}")


if __name__ == "__main__":
    main()
