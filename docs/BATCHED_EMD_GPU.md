# EMD por lotes en CUDA

`ICEEMDAN(batch_emd=True, graph_control=False)` procesa realizaciones de igual longitud con el EMD cúbico incluido. La ruta de producción usa CuPy y kernels CUDA; SciPy y la referencia CPU sólo intervienen en pruebas. `graph_control=True` es opcional y experimental.

| Operación | Contrato comprobado en pruebas |
|---|---|
| Extremos y ceros | Mesetas, cruces de signo y casos terminales frente a la referencia CPU |
| Bordes | Reflexión con `nbsym=1,2,3,6` y nudos exactos |
| Envolventes | Spline de tres nudos y cúbica `not-a-knot` |
| IMF | Criterio de Rilling, iteraciones y EMD de varios modos |
| ICEEMDAN | Componentes con ruido W común, modos de ruido ausentes y estados de parada |

Los buffers de los kernels son contiguos y de tamaño fijo. Se exige `EMD.DTYPE='float64'`, el EMD cúbico incluido, sin `trace` ni backend externo. La salida tiene componentes en filas y el residuo en la última fila. La terminación natural, el motivo de parada, los modos de ruido disponibles y las iteraciones se informan en `diagnostics_`.

Antes de reservar el banco, `batched_emd` rechaza una capacidad imposible. Sus arrays principales simultáneos requieren al menos `(8K + 180) I N` bytes para `K` slots, `I` realizaciones y `N` muestras. Esa cota no incluye la entrada, otros temporales ni memoria de terceros; pasarla no garantiza que la descomposición quepa en VRAM. Por ejemplo, `N=1_000_000`, `I=100`, `K=12` exige al menos 25,70 GiB. Reducir `max_imf` modifica el número de componentes solicitado.

Para contrastar CPU y GPU, suministre la misma matriz `noise=W` a ambos modelos. Las pruebas `test_07_batched_extrema.py` y `test_09_numeric_contracts.py` comparan componentes con `rtol=1e-9, atol=1e-10` en casos de ICEEMDAN y exigen igualdad exacta en los estados discretos pertinentes. Los tests con `ICEEMDAN_TEST_MODE=cpu-surrogate` no ejecutan CUDA.

La validación local de CUDA se limita a Windows 11, Python 3.13.14, CuPy 14.2.0, runtime CUDA 12.9 y RTX 5080 Laptop. No se extrapola a otros entornos. Compute Sanitizer 2026.3 pasó una [matriz focal de cuatro señales y cuatro herramientas](BATCH_SANITIZER_GATE.json); no cubre todas las entradas. El grafo condicional carece de un saneamiento instrumentado concluyente y permanece experimental.
