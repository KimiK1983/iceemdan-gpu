"""Compare the first stage of the seven saved CPU stress experiments on CUDA."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from iceemdan_cupy import ICEEMDAN, to_numpy
from reference.ICEEMDAN_cpu import ICEEMDAN as CPU


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--skip", action="append", default=[])
    args = parser.parse_args()
    output = (
        ROOT
        / "results"
        / ("prior_stress_full.json" if args.full else "prior_stress_first_stage.json")
    )
    results = []
    with np.load(args.baseline) as saved:
        t = saved["t"]
        cases = [key for key in saved.files if key != "t" and "__" not in key]
        cases = [name for name in cases if name not in args.skip]
        if not cases:
            raise ValueError("No stress cases selected.")
        missing = [name for name in cases if name + "__W" not in saved]
        if missing:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps({"status": "NON_REPRODUCIBLE", "missing_W": missing}, indent=2),
                encoding="utf-8",
            )
            return 2
        for name in cases:
            x, prior = saved[name], saved[name + "__components"]
            W = saved[name + "__W"]
            if W.shape != (20, len(x)):
                raise ValueError(f"Saved W shape invalid for {name}.")
            reference = CPU(trials=20, epsilon=0.2)
            rows = iter(W)
            reference.generate_noise = lambda scale, size, rows=rows: next(rows).copy() * scale
            cpu_parts = reference(x, T=t, max_imf=-1 if args.full else 1)
            start = time.perf_counter()
            model = ICEEMDAN(trials=20, epsilon=0.2)
            gpu_parts = to_numpy(model(x, T=t, max_imf=-1 if args.full else 1, noise=W))
            cpu_check = cpu_parts if args.full else cpu_parts[:1]
            gpu_check = gpu_parts if args.full else gpu_parts[:1]
            prior_check = prior if args.full else prior[:1]
            same_shape = cpu_check.shape == gpu_check.shape == prior_check.shape
            row = {
                "case": name,
                "seconds_gpu": time.perf_counter() - start,
                "shape_cpu": list(cpu_check.shape),
                "shape_gpu": list(gpu_check.shape),
                "cpu_vs_saved_max_abs": float(np.max(np.abs(cpu_check - prior_check)))
                if same_shape
                else None,
                "gpu_vs_cpu_max_abs": float(np.max(np.abs(gpu_check - cpu_check)))
                if same_shape
                else None,
                "gpu_vs_saved_max_abs": float(np.max(np.abs(gpu_check - prior_check)))
                if same_shape
                else None,
                "finite": bool(
                    np.isfinite(cpu_check).all()
                    and np.isfinite(gpu_check).all()
                    and np.isfinite(prior_check).all()
                ),
                "same_saved": bool(
                    same_shape and np.allclose(cpu_check, prior_check, rtol=1e-9, atol=1e-10)
                ),
                "gpu_allclose": bool(
                    same_shape and np.allclose(gpu_check, cpu_check, rtol=1e-9, atol=1e-10)
                ),
            }
            row.update(
                stop_cpu=reference.diagnostics_["stop_reason"],
                stop_gpu=model.diagnostics_["stop_reason"],
                same_noise_counts=reference.diagnostics_["noise_mode_counts"]
                == model.diagnostics_["noise_mode_counts"],
                same_natural_termination=reference.diagnostics_["natural_termination"]
                == model.diagnostics_["natural_termination"],
                same_sift_counts=[s["sift_iterations"] for s in reference.diagnostics_["stages"]]
                == [s["sift_iterations"] for s in model.diagnostics_["stages"]],
            )
            row["passed"] = bool(
                row["finite"]
                and row["same_saved"]
                and row["gpu_allclose"]
                and row["same_noise_counts"]
                and row["same_natural_termination"]
                and row["same_sift_counts"]
                and row["stop_cpu"] == row["stop_gpu"]
            )
            results.append(row)
            output.write_text(json.dumps(results, indent=2), encoding="utf-8")
            print(json.dumps(row), flush=True)
    return 0 if results and all(r["passed"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
