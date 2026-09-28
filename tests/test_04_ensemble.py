import copy

import numpy as np
import pytest


def signal():
    n = np.arange(128.0)
    return np.sin(0.77 * n) + 0.45 * np.cos(0.14 * n)


class Quarter:
    def __init__(self, cp):
        self.cp = cp

    def emd(self, x, t, max_imf=1):
        self.imfs = x[None, :] / 4
        self.residue = x * 0.75

    def get_imfs_and_residue(self):
        return self.imfs, self.residue


def test_fixed_ensemble_missing_mode(cpu, gpu, cp):
    x = signal()
    peak = np.max(abs(x))
    std = np.std(x / peak, ddof=1)
    r0 = x / peak / std
    n = np.arange(len(x))
    v = np.sin(0.9 * n)
    z = np.cos(0.21 * n)
    banks = [cp.asarray(np.stack((v, z))), cp.asarray(v[None, :])]
    e = gpu.ICEEMDAN(trials=2, ext_EMD=Quarter(cp))
    e._pre_decompose_noise_for_ceemdan = lambda *args: setattr(
        e, "all_noise_IMFs_for_CEEMDAN", banks
    )
    out = e(x, max_imf=2)
    r1 = 0.75 * (r0 + 0.2 * np.std(r0, ddof=1) * v / np.std(v, ddof=1))
    r2 = 0.75 * r1 + 0.75 * 0.2 * np.std(r1, ddof=1) * z / 2
    expected = np.stack((r0 - r1, r1 - r2, r2)) * std * peak
    np.testing.assert_allclose(cp.asnumpy(out), expected, rtol=1e-13, atol=1e-13)
    assert e.diagnostics_["stages"][1]["missing_noise_modes"] == 1
    assert e.diagnostics_["stages"][1]["ensemble_denominator"] == 2


def test_first_noise_std_per_realization(gpu, cp):
    x = signal()
    v = np.sin(np.arange(len(x)) * 0.8)
    banks = [cp.asarray(v[None, :]), cp.asarray((3 * v)[None, :])]
    e = gpu.ICEEMDAN(trials=2, ext_EMD=Quarter(cp))
    e._pre_decompose_noise_for_ceemdan = lambda *args: setattr(
        e, "all_noise_IMFs_for_CEEMDAN", banks
    )
    e(x, max_imf=1)
    np.testing.assert_allclose(
        e.diagnostics_["stages"][0]["perturbation_stds"], [0.2, 0.2], atol=1e-14
    )


def test_cpu_rng_mode_rejected(gpu):
    with pytest.raises(ValueError, match="cupy_philox"):
        gpu.ICEEMDAN(seed=42, rng_mode="numpy_pcg64_compat")


def test_default_rng_generates_on_device(gpu, cp, monkeypatch):
    def reject_cpu_rng(*args, **kwargs):
        raise AssertionError("CPU random generation is not allowed by default")

    monkeypatch.setattr(np.random, "default_rng", reject_cpu_rng)
    e = gpu.ICEEMDAN(seed=7)
    assert e._noise.mode == "cupy_philox"
    assert isinstance(e.generate_noise(1.0, 16), cp.ndarray)


def test_rng_reset_failure_and_epsilon_zero(gpu, cp):
    e = gpu.ICEEMDAN(seed=7, trials=3)
    a = e.generate_noise(1, 128)
    e.noise_seed(7)
    b = e.generate_noise(1, 128)
    np.testing.assert_array_equal(cp.asnumpy(a), cp.asnumpy(b))
    state = copy.deepcopy(e._noise.snapshot())
    e.epsilon = 0
    e(signal(), max_imf=1)
    assert repr(e._noise.snapshot()) == repr(state)
    e.epsilon = 0.2
    e._pre_decompose_noise_for_ceemdan = lambda *args: (_ for _ in ()).throw(
        RuntimeError("injected")
    )
    with pytest.raises(RuntimeError, match="injected"):
        e(signal(), max_imf=1)
    assert repr(e._noise.snapshot()) == repr(state)
    with pytest.raises(ValueError):
        e.get_imfs_and_residue()
    assert e.all_noise_IMFs_for_CEEMDAN == []


def test_supplied_W_not_mutated_or_rng_consumed(gpu, cp):
    x = signal()
    W = np.random.default_rng(5).normal(size=(3, len(x)))
    saved = W.copy()
    e = gpu.ICEEMDAN(trials=3, seed=11)
    state = repr(e._noise.snapshot())
    e(x, max_imf=1, noise=W)
    assert repr(e._noise.snapshot()) == state
    np.testing.assert_array_equal(W, saved)


def test_external_cpu_backend_rejected(cpu, gpu):
    e = gpu.ICEEMDAN(trials=2, ext_EMD=cpu.EMD())
    with pytest.raises((TypeError, gpu.BackendContractError)):
        e(signal(), max_imf=1)


def test_kahan_across_subchunks(gpu, cp):
    from iceemdan_cupy.ensemble import kahan_add

    values = np.random.default_rng(9).normal(size=(5, 64))
    total = cp.zeros(64)
    corr = cp.zeros(64)
    for row in values:
        total, corr = kahan_add(total, corr, cp.asarray(row))
    b = cp.zeros(64)
    d = cp.zeros(64)
    for chunk in [values[:2], values[2:]]:
        for row in chunk:
            b, d = kahan_add(b, d, cp.asarray(row))
    np.testing.assert_array_equal(cp.asnumpy(total), cp.asnumpy(b))
    np.testing.assert_array_equal(cp.asnumpy(corr), cp.asnumpy(d))
