# Copyright, provenance, and modifications

## Original ICEEMDAN wrapper and CPU reference

Original wrapper: Copyright 2025 Javier F. Santamaria. The copyright holder authorized distribution of his contributions to this GPU adaptation under Apache License, Version 2.0. The complete terms are in `LICENSE`.

`reference/ICEEMDAN_cpu.py` is a byte-for-byte frozen testing reference (SHA-256 `d697bfb841ab2b099a1961a4575b671ba08867748e5cee63225f55a4be365b3a`). Its historical header retains its original licensing language. The frozen file is included for comparison, not imported by the runtime package.

## PyEMD-derived geometry

Copyright 2017 Dawid Laszuk. Apache License, Version 2.0. The supplied CPU reference incorporated PyEMD v1.10.0 mirror and interpolation routines. `iceemdan_cupy/geometry.py` adapts that geometry for device arrays and CuPy interpolation; `iceemdan_cupy/batched_emd.py` implements corresponding batched CUDA kernels. These are modified derivatives, not an unchanged copy of PyEMD. The PyEMD package is not a runtime dependency and is not redistributed.

Upstream: https://github.com/laszukdawid/PyEMD

## CuPy

CuPy 14.2.0 is a pinned runtime dependency. Its wheel and sources are not redistributed here. API contracts used by this adaptation are inventoried in `docs/CUPY_API_LEDGER.json`.
