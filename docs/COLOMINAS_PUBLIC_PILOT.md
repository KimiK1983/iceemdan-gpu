# Piloto Colominas: CPU pública frente a GPU batch

Fecha: 2026-09-28. CPU pública v2.0.0, SHA-256 del módulo `2de7564f9f01560ff1d3b1d87af12e88e34b3647152616dce5d2e90c926b695f`. Python 3.13.14, NumPy 2.3.5, CuPy 14.2.0, CUDA 12.9, RTX 5080 Laptop. Ruta GPU: `batch_emd=True, graph_control=False`; CPU serial. Señal Colominas de 1000 muestras, `epsilon=0.2`, `max_imf=-1`.

El script `tools/run_colominas_cupy_sweep.py` recorre tamaños 50, 100, 200, 400 y 800; dentro de cada tamaño recorre semillas 0–99. Para cada pareja genera una matriz `W = numpy.default_rng(seed).normal(size=(I, 1000))` y la entrega a ambas rutas. Compara todas las filas de componentes y residuo con `rtol=1e-9, atol=1e-10`, forma, finitud, cinco métricas, etapas, motivo de parada, terminación natural, recuentos de modos de ruido, modos ausentes e iteraciones de sifting. Cada fila registra hashes de W, fuente CPU, protocolo, identidad de baseline y contenido del registro. La reanudación exige un prefijo ordenado e identidades idénticas; los datos CPU históricos sin W quedan fuera de esta comparación emparejada.

| I | semilla | forma | diferencia máxima absoluta | norma de la diferencia | diagnósticos discretos |
|---:|---:|---:|---:|---:|---|
| 50 | 0 | 2 × 1000 | 4.44 × 10⁻¹⁶ | 2.49 × 10⁻¹⁵ | iguales |
| 800 | 0 | 2 × 1000 | 2.22 × 10⁻¹⁶ | 1.43 × 10⁻¹⁵ | iguales |

Ambos casos terminaron naturalmente tras una etapa con motivo `residue_is_imf`. El protocolo del piloto tiene SHA-256 `862269dfcc53a5c972acb60e73464bc9ab5b11c18d2cbb4046edb469e9d4a516`. Los registros completos están en `results/colominas_pilot_verified.jsonl` (ignorado por Git). Se probó la reanudación tras la primera pareja y de nuevo con 2/2 ya completas; las pruebas unitarias rechazan hashes, orden y estados alterados.

El benchmark separado `tools/benchmark_cpu_batch.py` incluye construcción del modelo, transferencias y sincronización explícita de GPU. Usa la misma W, una preparación por ruta y tres repeticiones cronometradas. En la prueba de humo I=50, las medianas fueron **3.459 s CPU** y **1.301 s GPU**, razón CPU/GPU **2.659**; datos crudos en `results/benchmark_cpu_batch_i50.json` (ignorado por Git). No se ejecutaron los otros cuatro tamaños del benchmark ni el barrido de 500 parejas, por lo que este resultado no estima su rendimiento o tasa de paridad.
