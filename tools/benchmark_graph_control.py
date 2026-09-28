"""Compare batched ICEEMDAN with device-controlled stages using identical noise."""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from iceemdan_cupy import ICEEMDAN


def paper_signal():
    n = np.arange(1, 1001, dtype="float64")
    signal = np.sin(2 * np.pi * 0.065 * (n - 1))
    signal[500:750] += np.sin(2 * np.pi * 0.255 * (n[500:750] - 501))
    return signal


def run_case(trials, repeats, signal_kind="paper", max_imf=-1):
    if signal_kind == "paper":
        signal = paper_signal()
    else:
        t = np.arange(1000, dtype="float64")
        signal = np.sin(0.71 * t) + 0.5 * np.sin(0.21 * t) + 0.3 * np.sin(0.035 * t)
    noise = np.random.default_rng(7).normal(size=(trials, len(signal)))
    models = {
        "batched": ICEEMDAN(trials=trials, batch_emd=True),
        "graph": ICEEMDAN(trials=trials, batch_emd=True, graph_control=True),
    }
    samples = {name: [] for name in models}
    outputs = {}
    for name, model in models.items():
        outputs[name] = cp.asnumpy(
            model(signal, noise=noise, max_imf=max_imf)
        )  # Compile and warm both routes.
    for repeat in range(repeats):
        order = ("batched", "graph") if repeat % 2 == 0 else ("graph", "batched")
        for name in order:
            start = time.perf_counter()
            output = models[name](signal, noise=noise, max_imf=max_imf)
            cp.cuda.Device().synchronize()
            samples[name].append(time.perf_counter() - start)
            np.testing.assert_allclose(cp.asnumpy(output), outputs[name], rtol=0, atol=0)
    np.testing.assert_allclose(outputs["graph"], outputs["batched"], rtol=1e-9, atol=1e-10)
    assert (
        models["graph"].diagnostics_["stop_reason"] == models["batched"].diagnostics_["stop_reason"]
    )
    medians = {name: float(np.median(values)) for name, values in samples.items()}
    return {
        "signal": signal_kind,
        "max_imf": max_imf,
        "trials": trials,
        "samples": len(signal),
        "components": len(outputs["graph"]) - 1,
        "stop_reason": models["graph"].diagnostics_["stop_reason"],
        "max_abs_difference": float(np.max(np.abs(outputs["graph"] - outputs["batched"]))),
        "seconds": samples,
        "medians_seconds": medians,
        "graph_over_batched": medians["graph"] / medians["batched"],
    }


def plot_svg(cases, path):
    max_time = max(c["medians_seconds"][name] for c in cases for name in ("batched", "graph"))
    width = 670
    height = 100 + 96 * len(cases)
    plot_width = 390
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<text x="24" y="30" font-family="sans-serif" font-size="18" font-weight="600">ICEEMDAN: tiempo de extremo a extremo</text>',
        f'<text x="24" y="51" font-family="sans-serif" font-size="12" fill="#555">{cases[0]["signal"]}, max_imf={cases[0]["max_imf"]}; mediana de corridas calientes</text>',
    ]
    for row, case in enumerate(cases):
        top = 76 + 96 * row
        lines.append(
            f'<text x="24" y="{top + 15}" font-family="sans-serif" font-size="14">I={case["trials"]}</text>'
        )
        for offset, name, color in ((0, "batched", "#375b8e"), (30, "graph", "#159a82")):
            value = case["medians_seconds"][name]
            bar_width = max(1, round(plot_width * value / max_time))
            y = top + offset
            lines.append(f'<rect x="155" y="{y}" width="{bar_width}" height="22" fill="{color}"/>')
            lines.append(
                f'<text x="{160 + bar_width}" y="{y + 16}" font-family="sans-serif" font-size="12">{value:.3f} s</text>'
            )
    lines.extend(
        [
            f'<rect x="24" y="{height - 22}" width="12" height="12" fill="#375b8e"/>',
            f'<text x="42" y="{height - 11}" font-family="sans-serif" font-size="12">batched</text>',
            f'<rect x="125" y="{height - 22}" width="12" height="12" fill="#159a82"/>',
            f'<text x="143" y="{height - 11}" font-family="sans-serif" font-size="12">graph_control</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, nargs="+", default=[50, 800])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--signal", choices=("paper", "three-tone"), default="paper")
    parser.add_argument("--max-imf", type=int, default=-1)
    parser.add_argument("--output", type=Path, default=Path("results/graph_control_benchmark.json"))
    args = parser.parse_args()
    if args.repeats < 1 or any(value < 1 for value in args.trials):
        parser.error("repeats and trials must be positive")
    if args.max_imf == 0 or args.max_imf < -1:
        parser.error("max-imf must be -1 or positive")
    cases = [run_case(trials, args.repeats, args.signal, args.max_imf) for trials in args.trials]
    report = {
        "python": platform.python_version(),
        "cupy": cp.__version__,
        "cuda_runtime": cp.cuda.runtime.runtimeGetVersion(),
        "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode("utf-8"),
        "repeats": args.repeats,
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    plot_svg(cases, args.output.with_suffix(".svg"))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
