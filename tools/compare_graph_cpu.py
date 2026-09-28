"""Cold full-run parity and timing against the supplied corrected CPU reference."""

import argparse
import json
import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from iceemdan_cupy import ICEEMDAN
from reference.ICEEMDAN_cpu import ICEEMDAN as CPU
from tools.benchmark_graph_control import paper_signal


def parity(gpu_components, cpu_components, gpu_info, cpu_info):
    same_shape = gpu_components.shape == cpu_components.shape
    finite = bool(np.isfinite(gpu_components).all() and np.isfinite(cpu_components).all())
    same_stop = gpu_info["stop_reason"] == cpu_info["stop_reason"]
    same_natural = gpu_info["natural_termination"] == cpu_info["natural_termination"]
    same_noise_counts = gpu_info["noise_mode_counts"] == cpu_info["noise_mode_counts"]
    same_sift_counts = [s["sift_iterations"] for s in gpu_info["stages"]] == [
        s["sift_iterations"] for s in cpu_info["stages"]
    ]
    components_close = bool(
        same_shape and finite and np.allclose(gpu_components, cpu_components, rtol=1e-9, atol=1e-10)
    )
    return (
        components_close and same_stop and same_natural and same_noise_counts and same_sift_counts
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, nargs="+", default=[50, 800])
    parser.add_argument("--output", type=Path, default=Path("results/graph_vs_cpu_parity.json"))
    args = parser.parse_args()
    if any(value < 1 for value in args.trials):
        parser.error("trials must be positive")
    signal = paper_signal()
    cases = []
    for trials in args.trials:
        noise = np.random.default_rng(7).normal(size=(trials, len(signal)))
        gpu = ICEEMDAN(trials=trials, batch_emd=True, graph_control=True)
        start = time.perf_counter()
        gpu_components = cp.asnumpy(gpu(signal, noise=noise))
        cp.cuda.Device().synchronize()
        graph_seconds = time.perf_counter() - start
        cpu = CPU(trials=trials)
        rows = iter(noise)
        cpu.generate_noise = lambda scale, size, noise_rows=rows: next(noise_rows).copy() * scale
        start = time.perf_counter()
        cpu_components = cpu(signal)
        cpu_seconds = time.perf_counter() - start
        passed = parity(gpu_components, cpu_components, gpu.diagnostics_, cpu.diagnostics_)
        same_shape = gpu_components.shape == cpu_components.shape
        case = {
            "trials": trials,
            "samples": len(signal),
            "cpu_seconds": cpu_seconds,
            "graph_seconds": graph_seconds,
            "cpu_over_graph": cpu_seconds / graph_seconds,
            "max_abs_difference": float(np.max(np.abs(gpu_components - cpu_components)))
            if same_shape
            else None,
            "graph_stop_reason": gpu.diagnostics_["stop_reason"],
            "cpu_stop_reason": cpu.diagnostics_["stop_reason"],
            "same_natural_termination": gpu.diagnostics_["natural_termination"]
            == cpu.diagnostics_["natural_termination"],
            "components": len(gpu_components) - 1,
            "same_noise_counts": gpu.diagnostics_["noise_mode_counts"]
            == cpu.diagnostics_["noise_mode_counts"],
            "same_sift_counts": [s["sift_iterations"] for s in gpu.diagnostics_["stages"]]
            == [s["sift_iterations"] for s in cpu.diagnostics_["stages"]],
            "finite": bool(np.isfinite(gpu_components).all() and np.isfinite(cpu_components).all()),
            "passed": passed,
        }
        print(json.dumps(case), flush=True)
        cases.append(case)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "signal": "paper",
                "noise": "default_rng(7)",
                "rtol": 1e-9,
                "atol": 1e-10,
                "cases": cases,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0 if cases and all(case["passed"] for case in cases) else 1


if __name__ == "__main__":
    raise SystemExit(main())
