# Estado de la candidata local GPU 0.2.0

Fecha: 2026-09-28. Entorno: Windows 11, Python 3.13.14, CuPy 14.2.0, CUDA runtime 12.9 y RTX 5080 Laptop. Configuración respaldada: `batch_emd=True, graph_control=False`.

## Cobertura batch

| Caso | Prueba existente o añadida |
|---|---|
| Longitudes parciales y llamadas repetidas | `test_batched_repeated_calls_with_partial_lengths_match_cpu` (127, 129, 127 muestras; mismo objeto) |
| Mesetas y tramos de ceros | `test_batched_extrema_plateaus_and_zero_runs`, `test_batched_extrema_random_with_plateaus` |
| Terminación y prioridad de estados | `test_batched_reflection_terminal_row_is_initialized`, `test_batched_early_stop_codes_preserve_reason_priority`, `test_batched_imf_stop_codes_match_existing_reason` |
| Varias etapas con W común | `test_batched_ensemble_two_stages_matches_cpu`, `test_three_stages_shared_noise_matches_cpu_scalar_batch_graph` |
| Modos de ruido ausentes | `test_batched_ensemble_missing_noise_modes_keep_denominator` |
| Límite de memoria | `test_batched_emd_rejects_workspace_larger_than_device` |

Las comparaciones de componentes de ICEEMDAN usan `rtol=1e-9, atol=1e-10`; los motivos de parada, terminación, recuentos de ruido, modos ausentes e iteraciones se comparan exactamente cuando corresponden. Las pruebas de kernels tienen sus propias tolerancias documentadas en cada caso.

## Comprobaciones de esta candidata

- `python -m pytest -q tests/test_00_delivery.py tests/test_01_contracts_extrema.py tests/test_02_geometry.py tests/test_03_emd.py tests/test_07_batched_extrema.py tests/test_09_numeric_contracts.py`: **142 passed**, 1 aviso de localización de CUDA.
- Con `ICEEMDAN_TEST_MODE=cpu-surrogate`: `python -m pytest -q tests/test_00_delivery.py tests/test_01_contracts_extrema.py tests/test_02_geometry.py tests/test_03_emd.py tests/test_04_ensemble.py tests/test_05_integration.py tests/test_06_review.py -k 'not test_default_rng_generates_on_device'`: **126 passed, 1 deselected**, 1 aviso de ruido uniforme experimental. Esta prueba no acredita CUDA.
- `python -m ruff check iceemdan_cupy tests tools examples`: aprobado.
- `python -m ruff format --check iceemdan_cupy tests tools examples`: 32 archivos conformes.
- `python -m mypy iceemdan_cupy`: aprobado, 10 archivos fuente.
- `python tools/check_api_coverage.py`: 77 entradas documentadas, ninguna API CuPy usada sin registrar.
- `python tools/run_batch_sanitizer.py --timeout 600 --output-dir results/batch_sanitizer/gate_5ff942d` con Compute Sanitizer 2026.3: **16 PASS**. Los cuatro casos (`N/I/max_imf`: 127/1/1, 129/5/2, 192/4/3, 1000/50/1) pasaron `memcheck`, `synccheck`, `racecheck` e `initcheck` con W explícita y paridad CPU/GPU. Los 16 logs y sus SHA-256 constan en [BATCH_SANITIZER_GATE.json](BATCH_SANITIZER_GATE.json). `racecheck` utiliza su resumen nativo `RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)`; las otras herramientas informaron `ERROR SUMMARY: 0 errors`. Todas las celdas tuvieron código de salida 0 y marcador de aserciones aprobado.
- `uv build --offline --out-dir dist`: sdist y wheel construidos. El wheel tiene sólo `iceemdan_cupy` y metadatos/licencias; el sdist incluye la referencia congelada con SHA-256 `d697bfb841ab2b099a1961a4575b671ba08867748e5cee63225f55a4be365b3a`. Metadatos: `iceemdan-gpu` 0.2.0, Apache-2.0, Python `>=3.13,<3.14`, NumPy y `cupy-cuda12x==14.2.0`.

Antes de la corrección, `racecheck` encontró cuatro avisos en `cupy_nonzero_kernel_incomplete_scan` para Colominas. La secuencia exacta de máscaras reprodujo tres avisos fuera del producto. `contracts.extrema` compacta ahora índices en GPU mediante prefijos `cumsum` y un kernel de dispersión; no se emplearon filtros ni supresiones del saneador. El mismo caso pasó después sin peligros. La [documentación de NVIDIA](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html) distingue el resumen propio de `racecheck`.

El informe histórico `results/stage4_full_cuda.json` de la fuente original sumaba 203 pruebas CUDA y 7 estáticas. No se repitió esa suite completa en esta candidata. Tampoco se ejecutaron el barrido de 500 casos, el benchmark completo ni saneamiento fuera de las cuatro señales especificadas. La ruta `graph_control=True` conserva pruebas funcionales anteriores, pero su saneamiento instrumentado sigue sin conclusión; permanece experimental. La paridad y el saneamiento actuales se limitan al hardware y los casos indicados.
