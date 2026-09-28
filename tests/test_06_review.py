"""Review cases beyond the happy-path stage checks."""

import numpy as np
import pytest


def test_first_noise_std_must_be_finite(gpu, cp):
    n = np.arange(128.0)
    x = np.sin(0.7 * n) + 0.3 * np.cos(0.13 * n)
    e = gpu.ICEEMDAN(trials=2)
    # Use finite samples whose sample standard deviation overflows float64.
    large = cp.asarray(np.tile([1.795e308, -1.795e308], 64)[None, :])
    e._pre_decompose_noise_for_ceemdan = lambda *args: setattr(
        e, "all_noise_IMFs_for_CEEMDAN", [large, large]
    )
    with pytest.raises(gpu.BackendContractError, match="first noise IMF"):
        e(x, max_imf=1)


@pytest.mark.parametrize("kind", ["normal", "uniform"])
def test_zero_size_and_zero_scale_rng(gpu, cp, kind):
    e = gpu.ICEEMDAN(seed=1, noise_kind=kind, rng_mode="cupy_philox")
    assert e.generate_noise(1, (0,)).shape == (0,)
    np.testing.assert_array_equal(cp.asnumpy(e.generate_noise(0, 10)), np.zeros(10))


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_public_emd_precision_and_copies(gpu, cp, dtype):
    x = np.sin(np.arange(256.0) * 0.9) + 0.5 * np.cos(np.arange(256.0) * 0.13)
    e = gpu.EMD(DTYPE=dtype)
    a = e(x, max_imf=2)
    d, r = e.get_imfs_and_residue()
    assert a.dtype.name == dtype and d.dtype.name == dtype and r.dtype.name == dtype
    np.testing.assert_array_equal(cp.asnumpy(a), np.vstack((cp.asnumpy(d), cp.asnumpy(r))))
    a[:] = 0
    d[:] = 0
    r[:] = 0
    assert np.max(np.abs(cp.asnumpy(e.get_imfs_and_residue()[0]))) > 0


def test_original_near_tie_fixture_is_not_removed(cpu):
    from pathlib import Path

    z = np.load(Path(__file__).parent / "fixtures/first_divergent_mirror.npz")
    assert z.files


def test_external_float_or_complex_arrays_not_silently_cast(gpu, cp):
    from iceemdan_cupy.emd import backend_parts

    class Bad:
        def emd(self, s, t, max_imf):
            self.s = s

        def get_imfs_and_residue(self):
            return self.s.astype("complex128")[None, :] + 1j, self.s * 0 - 1j

    with pytest.raises(gpu.BackendContractError, match="real"):
        backend_parts(Bad(), cp.asarray(np.arange(10.0)), None, 1)


def test_safety_limit_remains_independent(gpu):
    e = gpu.ICEEMDAN(max_imf_iterations=1)
    with pytest.raises(ValueError, match="safety|limit"):
        e(np.sin(np.arange(30.0)), max_imf=2)
