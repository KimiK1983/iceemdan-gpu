"""Analyze two candidate Keele EGG windows; data and outputs stay local."""

import argparse
import hashlib
import io
import json
import sys
import wave
import zipfile
from pathlib import Path

import cupy as cp
import numpy as np

from iceemdan_cupy import ICEEMDAN as Model
from iceemdan_cupy import __version__ as gpu_version
from iceemdan_cupy import to_numpy
from scripts.fetch_public_data import SOURCES, verify

ROOT = Path(__file__).resolve().parents[1]
WINDOWS = (("fig8", 17.275, 0.180), ("fig10", 25.925, 0.100))
RECORD = "KEELE/f1nw0000/laryngograph.wav"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "data" / "KEELE.zip")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "keele")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--save-local-arrays", action="store_true")
    parser.add_argument("--plot-local", action="store_true")
    args = parser.parse_args()
    if args.trials < 1:
        parser.error("trials must be positive")
    if not args.archive.is_file():
        parser.error(f"missing {args.archive}; run scripts.fetch_public_data keele")
    archive_sha256 = verify(args.archive, "keele")
    gpu_source_sha256 = hashlib.sha256(
        b"".join(path.read_bytes() for path in sorted((ROOT / "iceemdan_cupy").glob("*.py")))
    ).hexdigest()
    analysis_script_sha256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.out.mkdir(parents=True, exist_ok=True)
    results = []
    with zipfile.ZipFile(args.archive) as archive:
        record_bytes = archive.read(RECORD)
    with wave.open(io.BytesIO(record_bytes)) as wav:
        fs = wav.getframerate()
        if wav.getnchannels() != 1 or wav.getsampwidth() != 2:
            raise ValueError("expected mono 16-bit Keele EGG")
        for label, start_s, duration_s in WINDOWS:
            wav.setpos(round(start_s * fs))
            x = -np.frombuffer(wav.readframes(round(duration_s * fs)), dtype="<i2").astype(float)
            if len(x) != round(duration_s * fs):
                raise ValueError(f"incomplete window: {label}")
            t = np.arange(len(x)) / fs
            model = Model(
                trials=args.trials, epsilon=0.2, seed=0, batch_emd=True, graph_control=False
            )
            parts = to_numpy(model(x, T=t))
            row = {
                "window": label,
                "source_record": RECORD,
                "source_identification": "candidate from earlier visual matching; unconfirmed",
                "archive_url": SOURCES["keele"]["url"],
                "source_page": SOURCES["keele"]["page"],
                "license_note": SOURCES["keele"]["license_note"],
                "archive_sha256": archive_sha256,
                "record_sha256": hashlib.sha256(record_bytes).hexdigest(),
                "gpu_source_sha256": gpu_source_sha256,
                "analysis_script_sha256": analysis_script_sha256,
                "python_version": sys.version.split()[0],
                "numpy_version": np.__version__,
                "cupy_version": cp.__version__,
                "gpu_package_version": gpu_version,
                "gpu_route": "batch_emd=True,graph_control=False",
                "start_s": start_s,
                "duration_s": duration_s,
                "sample_rate_hz": fs,
                "archive_polarity_multiplier": -1,
                "trials": args.trials,
                "epsilon": 0.2,
                "seed": 0,
                "rng_mode": "cupy_philox",
                "components": len(parts) - 1,
                "stop_reason": model.diagnostics_["stop_reason"],
                "reconstruction_linf": float(np.max(np.abs(parts.sum(axis=0) - x))),
            }
            if args.save_local_arrays:
                np.savez_compressed(args.out / f"{label}.npz", x=x, t=t, parts=parts)
            if args.plot_local:
                import matplotlib.pyplot as plt

                labels = [
                    "signal",
                    *[f"component {i + 1}" for i in range(len(parts) - 1)],
                    "residue",
                ]
                fig, axes = plt.subplots(len(parts) + 1, sharex=True)
                for axis, values, axis_label in zip(axes, [x, *parts], labels, strict=True):
                    axis.plot(t, values, linewidth=0.7)
                    axis.set_ylabel(axis_label)
                axes[-1].set_xlabel("seconds")
                fig.tight_layout()
                fig.savefig(args.out / f"{label}.svg")
                plt.close(fig)
            results.append(row)
    (args.out / "metrics.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(results))


if __name__ == "__main__":
    main()
