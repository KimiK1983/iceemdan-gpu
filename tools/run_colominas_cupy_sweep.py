"""Resume Colominas comparisons only when CPU rows preserve the exact noise W."""

import argparse
import contextlib
import hashlib
import json
import math
import platform
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from iceemdan_cupy import ICEEMDAN, to_numpy
from iceemdan_cupy.runtime import cp

SIZES = (50, 100, 200, 400, 800)
METRICS = (
    "left_energy",
    "right_energy",
    "rrse_fast",
    "rrse_slow_residue",
    "rrse_reconstruction",
)
RTOL, ATOL = 1e-9, 1e-12


def protocol_sha256():
    digest = hashlib.sha256()
    digest.update(
        f"python={sys.version};platform={platform.platform()};numpy={np.__version__};cupy={cp.__version__}".encode()
    )
    files = [
        Path(__file__).resolve(),
        ROOT / "pyproject.toml",
        *(ROOT / "iceemdan_cupy").rglob("*.py"),
    ]
    for path in sorted(files):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def baseline_sha256(row):
    values = {key: row[key] for key in ("modes", "stop_reason", *METRICS)}
    return hashlib.sha256(json.dumps(values, sort_keys=True, allow_nan=False).encode()).hexdigest()


def reusable(row, cpu, noise_hash, baseline_hash, protocol_hash):
    if (
        row.get("pass") is not True
        or row.get("noise_sha256") != noise_hash
        or row.get("baseline_sha256") != baseline_hash
        or row.get("protocol_sha256") != protocol_hash
    ):
        return False
    try:
        return (
            row["modes_cpu"] == row["modes_gpu"] == cpu["modes"]
            and row["stop_cpu"] == row["stop_gpu"] == cpu["stop_reason"]
            and all(
                row["cpu"][key] == cpu[key]
                and type(row["gpu"][key]) in (int, float)
                and math.isfinite(row["gpu"][key])
                and np.isclose(row["gpu"][key], cpu[key], rtol=RTOL, atol=ATOL)
                for key in METRICS
            )
        )
    except (KeyError, TypeError, ValueError):
        return False


def signal_parts():
    n = np.arange(1, 1001)
    fast = np.zeros(1000)
    fast[500:750] = np.sin(2 * np.pi * 0.255 * (n[500:750] - 501))
    slow = np.sin(2 * np.pi * 0.065 * (n - 1))
    return fast, slow, fast + slow


def load_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@contextlib.contextmanager
def keep_awake_on_ac(enabled):
    if not enabled or sys.platform != "win32":
        yield
        return

    import ctypes

    class PowerStatus(ctypes.Structure):
        _fields_ = [
            ("ac", ctypes.c_ubyte),
            ("battery", ctypes.c_ubyte),
            ("percent", ctypes.c_ubyte),
            ("reserved", ctypes.c_ubyte),
            ("seconds_left", ctypes.c_uint),
            ("seconds_full", ctypes.c_uint),
        ]

    stop = threading.Event()

    def monitor():
        kernel = ctypes.windll.kernel32
        try:
            while not stop.is_set():
                status = PowerStatus()
                on_ac = kernel.GetSystemPowerStatus(ctypes.byref(status)) and status.ac == 1
                kernel.SetThreadExecutionState(0x80000001 if on_ac else 0x80000000)
                stop.wait(30)
        finally:
            kernel.SetThreadExecutionState(0x80000000)

    worker = threading.Thread(target=monitor, daemon=True)
    worker.start()
    try:
        yield
    finally:
        stop.set()
        worker.join()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=ROOT.parent / "iceemdan_results" / "colominas_100x5_iceemdan.jsonl",
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results" / "colominas_100x5_cupy.jsonl"
    )
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--keep-awake-on-ac", action="store_true")
    args = parser.parse_args()

    cpu_rows = load_rows(args.baseline)
    baseline = {(row["I"], row["seed"]): row for row in cpu_rows}
    pairs = [(size, seed) for seed in range(100) for size in SIZES]
    if len(cpu_rows) != 500 or set(baseline) != set(pairs):
        raise ValueError("The CPU baseline must contain each of the 500 (I, seed) pairs once.")
    if any("W" not in row for row in cpu_rows):
        print(
            json.dumps(
                {
                    "status": "NON_REPRODUCIBLE",
                    "reason": "Historical CPU rows do not preserve W; PCG64 and Philox differ at equal seeds.",
                }
            ),
            flush=True,
        )
        return 2
    if args.max_runs is not None and args.max_runs < 1:
        raise ValueError("--max-runs must be positive.")
    fast, slow, x = signal_parts()
    protocol_hash = protocol_sha256()
    noise_hashes, baseline_hashes = {}, {}
    for key, cpu in baseline.items():
        size, seed = key
        W = np.asarray(cpu["W"], dtype=np.float64)
        if W.shape != (size, len(x)) or not np.isfinite(W).all():
            raise ValueError(f"Invalid baseline W for I={size}, seed={seed}.")
        if (
            type(cpu.get("modes")) is not int
            or cpu["modes"] < 0
            or not isinstance(cpu.get("stop_reason"), str)
            or not cpu["stop_reason"]
            or any(
                type(cpu.get(name)) not in (int, float) or not math.isfinite(cpu[name])
                for name in METRICS
            )
        ):
            raise ValueError(f"Invalid baseline metrics or states for I={size}, seed={seed}.")
        noise_hashes[key] = hashlib.sha256(W.tobytes()).hexdigest()
        baseline_hashes[key] = baseline_sha256(cpu)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    existing = load_rows(args.output) if args.output.exists() else []
    done = {
        (row["I"], row["seed"])
        for row in existing
        if (row["I"], row["seed"]) in baseline
        and reusable(
            row,
            baseline[row["I"], row["seed"]],
            noise_hashes[row["I"], row["seed"]],
            baseline_hashes[row["I"], row["seed"]],
            protocol_hash,
        )
    }
    pending = [pair for pair in pairs if pair not in done]
    print(f"Completed {len(done)}/500; pending {len(pending)}", flush=True)

    with keep_awake_on_ac(args.keep_awake_on_ac), args.output.open("a", encoding="utf-8") as log:
        for size, seed in pending[: args.max_runs]:
            start = time.perf_counter()
            cpu = baseline[size, seed]
            W = np.asarray(cpu["W"], dtype=np.float64)
            model = ICEEMDAN(trials=size, epsilon=0.2, seed=seed)
            parts = to_numpy(model(x, noise=W))
            first = parts[0]
            residue = x - first
            gpu = {
                "left_energy": float(np.mean(first[10:490] ** 2)),
                "right_energy": float(np.mean(first[760:990] ** 2)),
                "rrse_fast": float(np.linalg.norm(first - fast) / np.linalg.norm(fast)),
                "rrse_slow_residue": float(np.linalg.norm(residue - slow) / np.linalg.norm(slow)),
                "rrse_reconstruction": float(
                    np.linalg.norm(parts.sum(axis=0) - x) / np.linalg.norm(x)
                ),
            }
            differences = {key: abs(gpu[key] - cpu[key]) for key in METRICS}
            passed = (
                np.isfinite(parts).all()
                and all(np.isfinite(cpu[key]) for key in METRICS)
                and len(parts) - 1 == cpu["modes"]
                and model.diagnostics_["stop_reason"] == cpu["stop_reason"]
                and all(np.isclose(gpu[key], cpu[key], rtol=RTOL, atol=ATOL) for key in METRICS)
            )
            row = {
                "I": size,
                "seed": seed,
                "noise_sha256": noise_hashes[size, seed],
                "baseline_sha256": baseline_hashes[size, seed],
                "protocol_sha256": protocol_hash,
                "pass": bool(passed),
                "seconds_gpu": time.perf_counter() - start,
                "modes_gpu": len(parts) - 1,
                "modes_cpu": cpu["modes"],
                "stop_gpu": model.diagnostics_["stop_reason"],
                "stop_cpu": cpu["stop_reason"],
                "gpu": gpu,
                "cpu": {key: cpu[key] for key in METRICS},
                "absolute_differences": differences,
            }
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(
                f"I={size}, seed={seed}: {'PASS' if passed else 'FAIL'}, "
                f"{row['seconds_gpu']:.1f}s, max metric difference={max(differences.values()):.3g}",
                flush=True,
            )
            if not passed:
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
