"""Cold-call CPU-serial versus CUDA-batch timing with the same explicit W."""

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from iceemdan_cupy import ICEEMDAN, to_numpy
from iceemdan_cupy.runtime import cp
from tools.run_colominas_cupy_sweep import SIZES, cpu_class, sha256, signal_parts


def timed_cpu(CPU, x, W):
    start = time.perf_counter()
    model = CPU(trials=len(W), epsilon=0.2, parallel=False)
    rows = iter(W)
    model.generate_noise = lambda scale, shape, noise_rows=rows: next(noise_rows).copy() * scale
    parts = model(x, max_imf=-1)
    return time.perf_counter() - start, parts


def timed_gpu(x, W):
    cp.cuda.Device().synchronize()
    start = time.perf_counter()
    model = ICEEMDAN(trials=len(W), epsilon=0.2, batch_emd=True, graph_control=False)
    parts = to_numpy(model(x, max_imf=-1, noise=W))
    cp.cuda.Device().synchronize()
    return time.perf_counter() - start, parts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cpu-source", type=Path, default=ROOT.parent / "iceemdan-cpu-public" / "ICEEMDAN.py"
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=SIZES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results" / "benchmark_cpu_batch.json"
    )
    args = parser.parse_args()
    if any(size not in SIZES for size in args.sizes) or len(set(args.sizes)) != len(args.sizes):
        parser.error("sizes must be unique members of 50,100,200,400,800")
    if args.seed not in range(100):
        parser.error("seed must be from 0 through 99")
    CPU = cpu_class(args.cpu_source)
    x = signal_parts()[2]
    cases = []
    for size in args.sizes:
        W = np.random.default_rng(args.seed).normal(size=(size, len(x)))
        timed_cpu(CPU, x, W)  # one warm-up per route; no warm-up is timed
        timed_gpu(x, W)
        cpu_times, gpu_times = [], []
        for _ in range(3):
            cpu_seconds, cpu_parts = timed_cpu(CPU, x, W)
            gpu_seconds, gpu_parts = timed_gpu(x, W)
            if not np.allclose(cpu_parts, gpu_parts, rtol=1e-9, atol=1e-10):
                raise ValueError(f"CPU/GPU mismatch during benchmark I={size}")
            cpu_times.append(cpu_seconds)
            gpu_times.append(gpu_seconds)
        cpu_med, gpu_med = statistics.median(cpu_times), statistics.median(gpu_times)
        case = {
            "I": size,
            "seed": args.seed,
            "noise_sha256": sha256(W.tobytes()),
            "cpu_seconds": cpu_times,
            "gpu_seconds": gpu_times,
            "cpu_median_seconds": cpu_med,
            "gpu_median_seconds": gpu_med,
            "cpu_over_gpu": cpu_med / gpu_med,
        }
        cases.append(case)
        print(json.dumps(case), flush=True)
    report = {
        "cpu_version": "2.0.0",
        "cpu_source_sha256": sha256(args.cpu_source.read_bytes()),
        "cpu_route": "serial",
        "gpu_route": "batch_emd=True,graph_control=False",
        "noise_source": "numpy.default_rng(seed).normal((I,1000))",
        "warmups_per_route": 1,
        "timed_repeats_per_route": 3,
        "timing_scope": "model construction, CPU/GPU transfers, decomposition, explicit GPU synchronization",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "cupy": cp.__version__,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
