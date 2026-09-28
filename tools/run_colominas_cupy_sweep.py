"""Compare public CPU v2.0.0 with batched CUDA using identical explicit noise."""

import argparse
import hashlib
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from iceemdan_cupy import ICEEMDAN, to_numpy
from iceemdan_cupy.runtime import cp

SIZES = (50, 100, 200, 400, 800)
METRICS = ("left_energy", "right_energy", "rrse_fast", "rrse_slow_residue", "rrse_reconstruction")
RTOL, ATOL = 1e-9, 1e-10


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def record_sha256(row):
    payload = {key: value for key, value in row.items() if key != "record_sha256"}
    return sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode())


def json_safe(value):
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def protocol_sha256(cpu_sha):
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {
                "cpu_sha256": cpu_sha,
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "cupy": cp.__version__,
                "rtol": RTOL,
                "atol": ATOL,
            },
            sort_keys=True,
        ).encode()
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


def signal_parts():
    n = np.arange(1, 1001)
    fast = np.zeros(1000)
    fast[500:750] = np.sin(2 * np.pi * 0.255 * (n[500:750] - 501))
    slow = np.sin(2 * np.pi * 0.065 * (n - 1))
    return fast, slow, fast + slow


def metrics(parts, fast, slow, x):
    first = parts[0]
    residue = x - first
    return {
        "left_energy": float(np.mean(first[10:490] ** 2)),
        "right_energy": float(np.mean(first[760:990] ** 2)),
        "rrse_fast": float(np.linalg.norm(first - fast) / np.linalg.norm(fast)),
        "rrse_slow_residue": float(np.linalg.norm(residue - slow) / np.linalg.norm(slow)),
        "rrse_reconstruction": float(np.linalg.norm(parts.sum(axis=0) - x) / np.linalg.norm(x)),
    }


def discrete(info):
    return {
        "stages": len(info["stages"]),
        "stop_reason": info["stop_reason"],
        "natural_termination": info["natural_termination"],
        "noise_mode_counts": info["noise_mode_counts"],
        "missing_noise_modes": [s["missing_noise_modes"] for s in info["stages"]],
        "sift_iterations": [s["sift_iterations"] for s in info["stages"]],
    }


def cpu_class(path):
    spec = importlib.util.spec_from_file_location("public_iceemdan_cpu", path)
    if spec is None or spec.loader is None:
        raise ValueError("Cannot load the public CPU source.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if module.__version__ != "2.0.0":
        raise ValueError("The CPU source must be public v2.0.0.")
    return module.ICEEMDAN


def compare(cpu_parts, gpu_parts, cpu_info, gpu_info, fast, slow, x):
    cpu_metrics = metrics(cpu_parts, fast, slow, x)
    gpu_metrics = metrics(gpu_parts, fast, slow, x)
    cpu_states, gpu_states = discrete(cpu_info), discrete(gpu_info)
    same_shape = cpu_parts.shape == gpu_parts.shape
    finite = bool(np.isfinite(cpu_parts).all() and np.isfinite(gpu_parts).all())
    max_abs = float(np.max(np.abs(cpu_parts - gpu_parts))) if same_shape else None
    norm = float(np.linalg.norm(cpu_parts - gpu_parts)) if same_shape else None
    failures = []
    if not same_shape:
        failures.append("shape")
    if not finite or not all(np.isfinite(list(cpu_metrics.values()) + list(gpu_metrics.values()))):
        failures.append("nonfinite")
    if same_shape and finite and not np.allclose(cpu_parts, gpu_parts, rtol=RTOL, atol=ATOL):
        failures.append("components")
    if cpu_states != gpu_states:
        failures.append("discrete_diagnostics")
    if not all(np.isclose(cpu_metrics[k], gpu_metrics[k], rtol=RTOL, atol=ATOL) for k in METRICS):
        failures.append("metrics")
    return {
        "pass": not failures,
        "failure_reasons": failures,
        "shape_cpu": list(cpu_parts.shape),
        "shape_gpu": list(gpu_parts.shape),
        "finite": finite,
        "max_abs_difference": max_abs,
        "difference_l2": norm,
        "cpu": cpu_metrics,
        "gpu": gpu_metrics,
        "diagnostics_cpu": cpu_states,
        "diagnostics_gpu": gpu_states,
    }


def load_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def checked_prefix(rows, pairs, selection_sha, protocol_sha, cpu_sha, n):
    if len(rows) > len(pairs):
        raise ValueError("Output has more rows than selected pairs.")
    for row, (size, seed) in zip(rows, pairs, strict=False):
        W = np.random.default_rng(seed).normal(size=(size, n))
        noise_sha = sha256(W.tobytes())
        baseline_sha = sha256(f"{cpu_sha}:{noise_sha}:epsilon=0.2:max_imf=-1".encode())
        if (
            (row.get("I"), row.get("seed")) != (size, seed)
            or row.get("selection_sha256") != selection_sha
            or row.get("protocol_sha256") != protocol_sha
            or row.get("cpu_source_sha256") != cpu_sha
            or row.get("noise_sha256") != noise_sha
            or row.get("baseline_sha256") != baseline_sha
            or row.get("pass") is not True
            or row.get("failure_reasons") != []
            or row.get("diagnostics_cpu") != row.get("diagnostics_gpu")
            or row.get("record_sha256") != record_sha256(row)
        ):
            raise ValueError(f"Resume mismatch at I={size}, seed={seed}; use a new output file.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cpu-source", type=Path, default=ROOT.parent / "iceemdan-cpu-public" / "ICEEMDAN.py"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results" / "colominas_public_matched.jsonl"
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    parser.add_argument("--seeds", type=int, nargs="+", default=range(100))
    parser.add_argument("--max-runs", type=int)
    args = parser.parse_args()
    if any(size not in SIZES for size in args.sizes) or len(set(args.sizes)) != len(args.sizes):
        parser.error("sizes must be unique members of 50,100,200,400,800")
    if any(seed not in range(100) for seed in args.seeds) or len(set(args.seeds)) != len(
        args.seeds
    ):
        parser.error("seeds must be unique integers from 0 through 99")
    if args.max_runs is not None and args.max_runs < 1:
        parser.error("--max-runs must be positive")

    pairs = [
        (size, seed)
        for size in SIZES
        for seed in range(100)
        if size in args.sizes and seed in args.seeds
    ]
    selection_sha = sha256(json.dumps(pairs).encode())
    cpu_sha = sha256(args.cpu_source.read_bytes())
    CPU = cpu_class(args.cpu_source)
    protocol_sha = protocol_sha256(cpu_sha)
    fast, slow, x = signal_parts()
    existing = load_rows(args.output) if args.output.exists() else []
    checked_prefix(existing, pairs, selection_sha, protocol_sha, cpu_sha, len(x))
    print(
        f"Completed {len(existing)}/{len(pairs)}; pending {len(pairs) - len(existing)}", flush=True
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pending = pairs[len(existing) :]
    with args.output.open("a", encoding="utf-8") as log:
        for size, seed in pending[: args.max_runs]:
            W = np.random.default_rng(seed).normal(size=(size, len(x)))
            noise_sha = sha256(W.tobytes())
            baseline_sha = sha256(f"{cpu_sha}:{noise_sha}:epsilon=0.2:max_imf=-1".encode())
            row = {
                "I": size,
                "seed": seed,
                "selection_sha256": selection_sha,
                "protocol_sha256": protocol_sha,
                "cpu_source_sha256": cpu_sha,
                "cpu_version": "2.0.0",
                "noise_source": "numpy.default_rng(seed).normal((I,1000))",
                "noise_sha256": noise_sha,
                "baseline_sha256": baseline_sha,
                "epsilon": 0.2,
                "max_imf": -1,
                "rtol": RTOL,
                "atol": ATOL,
                "gpu_route": "batch_emd=True,graph_control=False",
            }
            try:
                cpu = CPU(trials=size, epsilon=0.2, parallel=False)
                rows = iter(W)
                cpu.generate_noise = lambda scale, shape, noise_rows=rows: (
                    next(noise_rows).copy() * scale
                )
                start = time.perf_counter()
                cpu_parts = cpu(x, max_imf=-1)
                cpu_seconds = time.perf_counter() - start
                gpu = ICEEMDAN(trials=size, epsilon=0.2, batch_emd=True, graph_control=False)
                start = time.perf_counter()
                gpu_parts = to_numpy(gpu(x, max_imf=-1, noise=W))
                cp.cuda.Device().synchronize()
                gpu_seconds = time.perf_counter() - start
                row.update(seconds_cpu=cpu_seconds, seconds_gpu=gpu_seconds)
                row.update(
                    compare(cpu_parts, gpu_parts, cpu.diagnostics_, gpu.diagnostics_, fast, slow, x)
                )
            except Exception as exc:
                row.update(
                    {"pass": False, "failure_reasons": [type(exc).__name__], "error": str(exc)}
                )
            row = json_safe(row)
            row["record_sha256"] = record_sha256(row)
            log.write(json.dumps(row, allow_nan=False) + "\n")
            log.flush()
            print(
                f"I={size}, seed={seed}: {'PASS' if row['pass'] else 'FAIL'}; "
                f"CPU {row.get('seconds_cpu', 0):.1f}s, GPU {row.get('seconds_gpu', 0):.1f}s",
                flush=True,
            )
            if not row["pass"]:
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
