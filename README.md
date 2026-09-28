# ICEEMDAN GPU

CUDA implementation of univariate ICEEMDAN with a batched cubic EMD. Install as `iceemdan-gpu`; import `iceemdan_cupy`.

The supported accelerated configuration is `batch_emd=True, graph_control=False`. The optional `graph_control=True` path is experimental. The last output row is always the residue; an ICEEMDAN component is not automatically an exact IMF.

## Requirements

Python 3.13, NumPy 2, CuPy `cupy-cuda12x==14.2.0`, and a compatible NVIDIA GPU and driver. Local CUDA validation is limited to Windows 11, Python 3.13.14, CuPy 14.2.0, CUDA runtime 12.9, and an RTX 5080 Laptop. Other platforms have not been validated for this release candidate.

```powershell
python -m pip install .
```

## Use

```python
import cupy as cp
from iceemdan_cupy import ICEEMDAN, to_numpy

t = cp.arange(1000, dtype=cp.float64) / 100
x = cp.sin(2 * cp.pi * 2 * t) + 0.4 * cp.sin(2 * cp.pi * 7 * t)
model = ICEEMDAN(trials=100, epsilon=0.2, seed=42, batch_emd=True)
components = model(x, T=t)
imfs, residue = model.get_imfs_and_residue()
host_components = to_numpy(components)
```

For CPU/GPU parity, provide the same explicit `noise=W` matrix to each implementation. The internal random generators are different. The test-only CPU reference is in `reference/ICEEMDAN_cpu.py`; SciPy is needed for that reference, not for GPU runtime. The batched path checks a lower bound on device memory before allocating, which does not guarantee every input fits in VRAM. See [batch limits](docs/BATCHED_EMD_GPU.md) and [third-party notices](THIRD_PARTY_NOTICES.md).

## Validate

```powershell
python -m pytest -q tests/test_07_batched_extrema.py tests/test_09_numeric_contracts.py
python -m ruff check iceemdan_cupy tests tools examples
python -m ruff format --check iceemdan_cupy tests tools examples
python -m mypy iceemdan_cupy
```

CUDA tests require CuPy 14.2.0 and an available GPU. `ICEEMDAN_TEST_MODE=cpu-surrogate` exercises only Python logic and does not establish CUDA behavior.

The local release candidate's checks and limits are recorded in [Validation status](docs/VALIDATION_STATUS.md).

Copyright 2025 Javier F. Santamaria and contributors. Licensed under Apache-2.0; see [LICENSE](LICENSE) and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
