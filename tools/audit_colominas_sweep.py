"""Audit the complete, ordered 500-pair matched Colominas JSONL."""

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import run_colominas_cupy_sweep as sweep


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=sweep.ROOT / "results" / "colominas_500_pairs.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=sweep.ROOT / "results" / "colominas_500_audit.json",
    )
    parser.add_argument(
        "--cpu-source",
        type=Path,
        default=sweep.ROOT.parent / "iceemdan-cpu-public" / "ICEEMDAN.py",
    )
    args = parser.parse_args()
    rows = sweep.load_rows(args.input)
    pairs = [(size, seed) for size in sweep.SIZES for seed in range(100)]
    cpu_sha = sweep.sha256(args.cpu_source.read_bytes())
    protocol_sha = sweep.protocol_sha256(cpu_sha)
    selection_sha = sweep.sha256(json.dumps(pairs).encode())
    sweep.checked_prefix(rows, pairs, selection_sha, protocol_sha, cpu_sha, 1000)
    if len(rows) != 500 or len({(row["I"], row["seed"]) for row in rows}) != 500:
        raise ValueError("Expected 500 unique ordered pairs.")

    by_size = {}
    for size in sweep.SIZES:
        block = [row for row in rows if row["I"] == size]
        if len(block) != 100:
            raise ValueError(f"Expected 100 pairs for I={size}.")
        for row in block:
            if (
                not row["finite"]
                or row["shape_cpu"] != row["shape_gpu"]
                or row["diagnostics_cpu"] != row["diagnostics_gpu"]
                or any(
                    not math.isfinite(row[key]) for key in ("max_abs_difference", "difference_l2")
                )
                or any(
                    not math.isfinite(row[route][metric])
                    for route in ("cpu", "gpu")
                    for metric in sweep.METRICS
                )
                or any(
                    not math.isclose(
                        row["cpu"][metric],
                        row["gpu"][metric],
                        rel_tol=sweep.RTOL,
                        abs_tol=sweep.ATOL,
                    )
                    for metric in sweep.METRICS
                )
            ):
                raise ValueError(f"Invalid numerical comparison at I={size}, seed={row['seed']}.")
        by_size[str(size)] = {
            "pairs": len(block),
            "max_abs_difference": max(row["max_abs_difference"] for row in block),
            "max_difference_l2": max(row["difference_l2"] for row in block),
            "max_metric_absolute_difference": {
                metric: max(abs(row["cpu"][metric] - row["gpu"][metric]) for row in block)
                for metric in sweep.METRICS
            },
            "shapes": dict(Counter("x".join(map(str, row["shape_cpu"])) for row in block)),
            "stop_reasons": dict(Counter(row["diagnostics_cpu"]["stop_reason"] for row in block)),
            "stage_counts": dict(Counter(str(row["diagnostics_cpu"]["stages"]) for row in block)),
        }

    report = {
        "status": "PASS_500_OF_500",
        "pairs": len(rows),
        "failures": 0,
        "cpu_source_sha256": cpu_sha,
        "protocol_sha256": protocol_sha,
        "selection_sha256": selection_sha,
        "jsonl_sha256": sweep.sha256(args.input.read_bytes()),
        "rtol": sweep.RTOL,
        "atol": sweep.ATOL,
        "by_size": by_size,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
