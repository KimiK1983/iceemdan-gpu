import json

import numpy as np
import pytest


def test_graph_cpu_parity_requires_components_and_discrete_states():
    from tools.compare_graph_cpu import parity

    a = np.array([[1.0, 2.0], [3.0, 4.0]])
    info = {
        "stop_reason": "max_imf",
        "natural_termination": False,
        "noise_mode_counts": [2],
        "stages": [{"sift_iterations": [3]}],
    }
    assert parity(a, a.copy(), info, info)
    assert not parity(a + 1, a, info, info)
    assert not parity(np.full_like(a, np.nan), a, info, info)
    assert not parity(a, a, {**info, "stop_reason": "residue_is_imf"}, info)
    assert not parity(a, a, {**info, "natural_termination": True}, info)
    assert not parity(a, a, {**info, "noise_mode_counts": [1]}, info)
    assert not parity(a, a, {**info, "stages": [{"sift_iterations": [4]}]}, info)


def test_saved_comparator_requires_original_noise(monkeypatch, tmp_path):
    import sys

    from tools import compare_prior_saved_full

    baseline = tmp_path / "baseline.npz"
    output = tmp_path / "report.json"
    np.savez(baseline, x=np.arange(4), t=np.arange(4), components=np.zeros((2, 4)))
    monkeypatch.setattr(sys, "argv", ["compare_prior_saved_full.py", str(baseline), str(output)])
    assert compare_prior_saved_full.main() == 2
    assert json.loads(output.read_text(encoding="utf-8"))["status"] == "NON_REPRODUCIBLE"


def test_stress_comparator_requires_noise_and_cases(monkeypatch, tmp_path):
    import sys

    from tools import compare_prior_stress

    baseline = tmp_path / "stress.npz"
    np.savez(baseline, t=np.arange(4), case=np.arange(4), case__components=np.zeros((2, 4)))
    monkeypatch.setattr(compare_prior_stress, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["compare_prior_stress.py", str(baseline)])
    assert compare_prior_stress.main() == 2
    report = tmp_path / "results" / "prior_stress_first_stage.json"
    assert json.loads(report.read_text(encoding="utf-8"))["status"] == "NON_REPRODUCIBLE"
    monkeypatch.setattr(sys, "argv", ["compare_prior_stress.py", str(baseline), "--skip", "case"])
    with pytest.raises(ValueError, match="No stress cases"):
        compare_prior_stress.main()


def test_colominas_history_without_noise_is_not_parity(monkeypatch, tmp_path):
    import sys

    from tools import run_colominas_cupy_sweep as sweep

    rows = [{"I": size, "seed": seed} for seed in range(100) for size in sweep.SIZES]
    monkeypatch.setattr(sweep, "load_rows", lambda path: rows)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_colominas_cupy_sweep.py", "--max-runs", "1", "--output", str(tmp_path / "out.jsonl")],
    )
    assert sweep.main() == 2
    assert not (tmp_path / "out.jsonl").exists()


def test_colominas_resume_rejects_nonfinite_baseline(monkeypatch, tmp_path):
    import hashlib
    import sys

    from tools import run_colominas_cupy_sweep as sweep

    banks = {size: np.ones((size, 4)) for size in sweep.SIZES}
    baseline = [
        {
            "I": size,
            "seed": seed,
            "W": banks[size],
            "modes": 1,
            "stop_reason": "max_imf",
            **{key: float("nan") for key in sweep.METRICS},
        }
        for seed in range(100)
        for size in sweep.SIZES
    ]
    existing = [
        {
            "I": size,
            "seed": seed,
            "pass": True,
            "noise_sha256": hashlib.sha256(banks[size].tobytes()).hexdigest(),
        }
        for seed in range(100)
        for size in sweep.SIZES
    ]
    base_path, output = tmp_path / "baseline.jsonl", tmp_path / "out.jsonl"
    output.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        sweep, "load_rows", lambda path: baseline if path == base_path else existing
    )
    monkeypatch.setattr(sweep, "signal_parts", lambda: (np.ones(4), np.ones(4), np.ones(4)))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_colominas_cupy_sweep.py",
            "--baseline",
            str(base_path),
            "--output",
            str(output),
            "--max-runs",
            "1",
        ],
    )
    with pytest.raises(ValueError, match="baseline"):
        sweep.main()


def test_colominas_resume_requires_matching_record_and_baseline(monkeypatch, tmp_path):
    import hashlib
    import sys

    from tools import run_colominas_cupy_sweep as sweep

    banks = {size: np.ones((size, 4)) for size in sweep.SIZES}
    baseline = [
        {
            "I": size,
            "seed": seed,
            "W": banks[size],
            "modes": 1,
            "stop_reason": "max_imf",
            **{key: 0.0 for key in sweep.METRICS},
        }
        for seed in range(100)
        for size in sweep.SIZES
    ]
    protocol = sweep.protocol_sha256()
    existing = [
        {
            "I": row["I"],
            "seed": row["seed"],
            "pass": True,
            "noise_sha256": hashlib.sha256(banks[row["I"]].tobytes()).hexdigest(),
            "baseline_sha256": sweep.baseline_sha256(row),
            "protocol_sha256": protocol,
            "modes_cpu": 1,
            "modes_gpu": 1,
            "stop_cpu": "max_imf",
            "stop_gpu": "max_imf",
            "cpu": {key: 0.0 for key in sweep.METRICS},
            "gpu": {key: 0.0 for key in sweep.METRICS},
        }
        for row in baseline
    ]
    base_path, output = tmp_path / "baseline.jsonl", tmp_path / "out.jsonl"
    output.write_text("", encoding="utf-8")
    monkeypatch.setattr(
        sweep, "load_rows", lambda path: baseline if path == base_path else existing
    )
    monkeypatch.setattr(sweep, "signal_parts", lambda: (np.ones(4), np.ones(4), np.ones(4)))
    monkeypatch.setattr(
        sweep, "ICEEMDAN", lambda **kwargs: (_ for _ in ()).throw(RuntimeError("recompute"))
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_colominas_cupy_sweep.py",
            "--baseline",
            str(base_path),
            "--output",
            str(output),
            "--max-runs",
            "1",
        ],
    )
    assert sweep.main() == 0

    gpu = existing[0].pop("gpu")
    with pytest.raises(RuntimeError, match="recompute"):
        sweep.main()
    existing[0]["gpu"] = gpu

    existing[0]["protocol_sha256"] = "stale"
    with pytest.raises(RuntimeError, match="recompute"):
        sweep.main()
    existing[0]["protocol_sha256"] = protocol

    baseline[0]["left_energy"] = 1.0  # W is unchanged.
    with pytest.raises(RuntimeError, match="recompute"):
        sweep.main()


@pytest.mark.parametrize(
    "kind", ["cubic", "pchip", "akima", "cubic_hermite", "linear", "slinear", "quadratic"]
)
@pytest.mark.parametrize("seed", [1, 5])
def test_identical_W_two_stages(cpu, gpu, cp, kind, seed):
    n = np.arange(192.0)
    x = np.sin(0.73 * n) + 0.5 * np.cos(0.14 * n)
    W = np.random.default_rng(seed).normal(size=(4, len(x)))
    a = cpu.ICEEMDAN(trials=4, emd_params={"spline_kind": kind})
    rows = iter(W)
    a.generate_noise = lambda scale, size: next(rows).copy() * scale
    b = gpu.ICEEMDAN(trials=4, emd_params={"spline_kind": kind})
    expected = a(x, max_imf=2)
    actual = b(x, max_imf=2, noise=W)
    # A failure is a failure: never xfail or widen tolerances after a branch flips.
    np.testing.assert_allclose(cp.asnumpy(actual), expected, rtol=1e-9, atol=1e-10)
    assert a.diagnostics_["noise_mode_counts"] == b.diagnostics_["noise_mode_counts"]
    for ra, rb in zip(a.diagnostics_["stages"], b.diagnostics_["stages"], strict=True):
        assert ra["sift_iterations"] == rb["sift_iterations"]


def test_paper_signal_full(cpu, gpu, cp):
    n = np.arange(1, 1001)
    x = np.sin(2 * np.pi * 0.065 * (n - 1))
    x[500:750] += np.sin(2 * np.pi * 0.255 * (n[500:750] - 501))
    W = np.random.default_rng(0).normal(size=(5, len(x)))
    a = cpu.ICEEMDAN(trials=5)
    rows = iter(W)
    a.generate_noise = lambda scale, size: next(rows).copy() * scale
    b = gpu.ICEEMDAN(trials=5)
    expected = a(x)
    actual = b(x, noise=W)
    np.testing.assert_allclose(cp.asnumpy(actual), expected, rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(cp.asnumpy(actual).sum(0), x, rtol=1e-13, atol=1e-13)
    assert a.diagnostics_["stop_reason"] == b.diagnostics_["stop_reason"]


@pytest.mark.parametrize("requested", [0, 1, 2])
def test_max_imf_getters_dtype_and_copies(gpu, cp, requested):
    n = np.arange(128.0)
    x = np.rint(10 * (np.sin(0.7 * n) + 0.5 * np.cos(0.14 * n))).astype("int16")
    e = gpu.ICEEMDAN(trials=3, seed=8, dtype="float32")
    out = e(x, max_imf=requested)
    a, r = e.get_imfs_and_residue()
    assert out.dtype.name == "float32"
    np.testing.assert_array_equal(cp.asnumpy(out), np.vstack((cp.asnumpy(a), cp.asnumpy(r))))
    out[:] = 0
    a[:] = 0
    r[:] = 0
    assert np.max(abs(cp.asnumpy(e.get_imfs_and_residue()[1]))) > 0


def test_invalid_configuration(gpu):
    for opts in [
        {"trials": 0},
        {"trials": True},
        {"epsilon": np.nan},
        {"parallel": True},
        {"dtype": "int32"},
        {"processes": 2},
        {"device": -1},
    ]:
        with pytest.raises((ValueError, TypeError)):
            gpu.ICEEMDAN(**opts)
