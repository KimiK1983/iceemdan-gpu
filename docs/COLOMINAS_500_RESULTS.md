# Colominas: 500 parejas CPU pública–GPU batch

Fecha: 2026-09-28. Señal Colominas de 1000 muestras; `epsilon=0.2`, `max_imf=-1`. CPU pública v2.0.0 serial frente a `batch_emd=True, graph_control=False`. Para cada pareja ordenada por I=(50,100,200,400,800) y luego semilla=0–99, se generó una sola matriz `W = numpy.default_rng(seed).normal(size=(I,1000))` y se entregó a ambas rutas. Las 500 filas CPU históricas sin W no se usaron como pares.

## Paridad

El barrido completó **500/500 PASS**, con 100 semillas únicas por tamaño, sin excepciones ni filas fallidas. `tools/audit_colominas_sweep.py` verificó orden, unicidad, hashes de W/fuente/protocolo/fila, finitud, forma, cinco métricas y diagnósticos discretos exactos. Todas las salidas tienen forma 2 × 1000, una etapa y parada natural `residue_is_imf`. La tolerancia aplicada a todas las matrices fue `rtol=1e-9, atol=1e-10`.

| I | Parejas | Máxima diferencia absoluta | Máxima norma L2 de diferencia |
|---:|---:|---:|---:|
| 50 | 100 | 4.44 × 10⁻¹⁶ | 2.76 × 10⁻¹⁵ |
| 100 | 100 | 4.44 × 10⁻¹⁶ | 2.37 × 10⁻¹⁵ |
| 200 | 100 | 4.44 × 10⁻¹⁶ | 2.23 × 10⁻¹⁵ |
| 400 | 100 | 4.44 × 10⁻¹⁶ | 1.96 × 10⁻¹⁵ |
| 800 | 100 | 4.44 × 10⁻¹⁶ | 1.92 × 10⁻¹⁵ |

Los recuentos de modos de ruido, modos ausentes, iteraciones de sifting, número de etapas, motivo de parada y terminación natural coincidieron exactamente en cada pareja. La mayor diferencia absoluta de cualquiera de las cinco métricas fue 2.78 × 10⁻¹⁷.

## Benchmark separado

Misma W para CPU y GPU por tamaño, semilla 0, una preparación y tres repeticiones cronometradas por ruta. Cada tiempo incluye construcción del modelo, transferencias, descomposición y sincronización explícita de GPU. Cada repetición verificó paridad funcional. Se muestra mediana y rango mínimo–máximo en segundos; la razón es mediana CPU / mediana GPU.

| I | CPU serial, mediana [mín–máx] | GPU batch, mediana [mín–máx] | Razón |
|---:|---:|---:|---:|
| 50 | 4.156 [3.265–4.336] | 1.318 [1.314–1.390] | 3.154 |
| 100 | 8.479 [8.412–8.834] | 1.475 [1.456–1.517] | 5.750 |
| 200 | 15.662 [14.535–17.688] | 1.483 [1.423–1.620] | 10.559 |
| 400 | 31.473 [29.502–35.773] | 1.800 [1.699–1.908] | 17.490 |
| 800 | 66.987 [60.305–83.845] | 1.913 [1.848–2.004] | 35.010 |

Estos tiempos son observaciones de tres repeticiones en este equipo, con una semilla; no establecen una velocidad mínima ni predicen otro hardware o señal. Los tiempos por fila del barrido no entraron en el benchmark.

## Entorno y reproducción

Windows 11, Intel Core Ultra 9 275HX, NVIDIA GeForce RTX 5080 Laptop GPU (16 303 MiB, driver 591.91), Python 3.13.14, NumPy 2.3.5, CuPy 14.2.0 y CUDA runtime 12.9. SHA-256 de la fuente CPU pública: `2de7564f9f01560ff1d3b1d87af12e88e34b3647152616dce5d2e90c926b695f`. SHA-256 del protocolo: `862269dfcc53a5c972acb60e73464bc9ab5b11c18d2cbb4046edb469e9d4a516`.

```powershell
# Usar el entorno Python 3.13.14 con NumPy 2.3.5 y CuPy 14.2.0.
# Ejecutar la siguiente línea cinco veces, secuencialmente; cada bloque añade 100 pares.
python tools/run_colominas_cupy_sweep.py --max-runs 100 --output results/colominas_500_pairs.jsonl
python tools/audit_colominas_sweep.py --input results/colominas_500_pairs.jsonl --output results/colominas_500_audit.json
python tools/benchmark_cpu_batch.py --sizes 50 100 200 400 800 --seed 0 --output results/benchmark_cpu_batch_5_sizes.json
```

Archivos locales ignorados por Git: `results/colominas_500_pairs.jsonl` (SHA-256 `5ca7e95561443a969ed9afef9d04cae45e36a653b31ab2c5795d0ef0bb17066e`), `results/colominas_500_audit.json` (SHA-256 `3082c6f263d287dcd409645cbcdd85a9b544fdc823e05452f77e88624eabc873`) y `results/benchmark_cpu_batch_5_sizes.json` (SHA-256 `040feca5068cea5321c630063c0d73b698154efc913a1186a30c90f0b99682c8`). Los cinco logs de bloque también quedan en `results/`. Los hashes de W del benchmark coinciden con las cinco parejas de semilla 0 del barrido. Las matrices W no se publican; se regeneran con la versión de NumPy indicada.
