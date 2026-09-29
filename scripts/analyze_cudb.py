"""Analyze a candidate CUDB ECG window; the paper does not identify a record."""

import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np

from iceemdan_cupy import ICEEMDAN as Model
from iceemdan_cupy import to_numpy
from scripts.fetch_public_data import SOURCES, verify

ROOT = Path(__file__).resolve().parents[1]


def decode_212(data):
    """Decode a single-channel WFDB format-212 sample stream."""
    if len(data) % 3:
        raise ValueError("format-212 byte count must be divisible by three")
    packed = np.frombuffer(data, dtype=np.uint8).reshape(-1, 3).astype(np.int16)
    first = packed[:, 0] + ((packed[:, 1] & 15) << 8)
    second = packed[:, 2] + ((packed[:, 1] & 240) << 4)
    samples = np.empty(first.size + second.size, dtype=np.int16)
    samples[0::2] = np.where(first >= 2048, first - 4096, first)
    samples[1::2] = np.where(second >= 2048, second - 4096, second)
    return samples


def vf_onsets(data):
    """Read WFDB annotation sample offsets for VFON (code 32)."""
    times = []
    index = sample = 0
    while index + 1 < len(data):
        b0, b1 = data[index : index + 2]
        index += 2
        kind, value = b1 >> 2, b0 + ((b1 & 3) << 8)
        if kind == 0 and value == 0:
            break
        if kind == 59:
            if index + 4 > len(data):
                raise ValueError("truncated WFDB skip annotation")
            a, b, c, d = data[index : index + 4]
            sample += ((b << 8 | a) << 16) | (d << 8 | c)
            index += 4
        elif kind == 63:
            index += value + value % 2
        elif kind not in (60, 61, 62):
            sample += value
            if kind == 32:
                times.append(sample)
    return times


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "data" / "cudb-1.0.0.zip")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "cudb")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--start", type=float, default=206.0)
    parser.add_argument("--duration", type=float, default=16.0)
    parser.add_argument("--save-local-arrays", action="store_true")
    parser.add_argument("--plot-local", action="store_true")
    args = parser.parse_args()
    if args.trials < 1 or args.start < 0 or args.duration <= 0:
        parser.error("trials and duration must be positive; start nonnegative")
    if not args.archive.is_file():
        parser.error(f"missing {args.archive}; run scripts.fetch_public_data cudb")
    archive_sha256 = verify(args.archive, "cudb")
    gpu_source_sha256 = hashlib.sha256(
        b"".join(path.read_bytes() for path in sorted((ROOT / "iceemdan_cupy").glob("*.py")))
    ).hexdigest()
    with zipfile.ZipFile(args.archive) as archive:
        names = {Path(name).name: name for name in archive.namelist()}
        header = archive.read(names["cu01.hea"]).decode("ascii").splitlines()[0].split()
        fs = int(header[2])
        raw = decode_212(archive.read(names["cu01.dat"]))
        onsets = [sample / fs for sample in vf_onsets(archive.read(names["cu01.atr"]))]
    start = round(args.start * fs)
    length = round(args.duration * fs)
    x = raw[start : start + length].astype(float)
    if len(x) != length:
        raise ValueError("requested window extends past record")
    t = np.arange(len(x)) / fs
    model = Model(trials=args.trials, epsilon=0.2, seed=0, batch_emd=True, graph_control=False)
    parts = to_numpy(model(x, T=t))
    row = {
        "candidate_record": "cu01",
        "source_identification": "hypothesis; paper record unspecified",
        "archive_url": SOURCES["cudb"]["url"],
        "archive_sha256": archive_sha256,
        "gpu_source_sha256": gpu_source_sha256,
        "gpu_route": "batch_emd=True,graph_control=False",
        "start_s": args.start,
        "duration_s": args.duration,
        "sample_rate_hz": fs,
        "vf_onsets_s": onsets,
        "trials": args.trials,
        "epsilon": 0.2,
        "seed": 0,
        "components": len(parts) - 1,
        "stop_reason": model.diagnostics_["stop_reason"],
        "reconstruction_linf": float(np.max(np.abs(parts.sum(axis=0) - x))),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(row, indent=2) + "\n", encoding="utf-8")
    if args.save_local_arrays:
        np.savez_compressed(args.out / "cu01_window.npz", x=x, t=t, parts=parts)
    if args.plot_local:
        import matplotlib.pyplot as plt

        labels = ["signal", *[f"component {i + 1}" for i in range(len(parts) - 1)], "residue"]
        fig, axes = plt.subplots(len(parts) + 1, sharex=True)
        for axis, values, axis_label in zip(axes, [x, *parts], labels, strict=True):
            axis.plot(t, values, linewidth=0.7)
            axis.set_ylabel(axis_label)
        axes[-1].set_xlabel("seconds")
        fig.tight_layout()
        fig.savefig(args.out / "cu01_window.svg")
        plt.close(fig)
    print(json.dumps(row))


if __name__ == "__main__":
    main()
