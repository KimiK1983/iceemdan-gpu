# ICEEMDAN GPU

[![Licencia: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue)](LICENSE)

Implementación importable de **Improved Complete Ensemble Empirical Mode Decomposition with Adaptive Noise** (ICEEMDAN) para CUDA. Se instala como `iceemdan-gpu` y se importa como `iceemdan_cupy`. La ruta acelerada respaldada es `batch_emd=True, graph_control=False`; el control por grafo sigue siendo experimental.

[Instalación y uso](#instalación-y-uso) · [Reproducibilidad](#reproducibilidad) · [Registros públicos](#registros-públicos) · [Límites](#límites) · [English](README.md)

![Descomposición sintética original con GPU](assets/synthetic_decomposition.svg)

*Señal y descomposición sintéticas originales, generadas con cuatro realizaciones de ruido explícito mediante [`scripts/make_synthetic_figure.py`](scripts/make_synthetic_figure.py). No es una figura del artículo ni un registro clínico.*

## Instalación y uso

Requiere Python 3.13, NumPy 2, CuPy `cupy-cuda12x==14.2.0`, una GPU NVIDIA y un controlador compatibles. Desde un clon:

```bash
python -m pip install .
```

Cuando se publique y etiquete este repositorio, la instalación desde GitHub podrá usar `python -m pip install "git+https://github.com/KimiK1983/iceemdan-gpu.git@<tag>"`. En este punto de revisión no hay remoto ni etiqueta publicados; sustituya `<tag>` por la etiqueta real.

```python
import cupy as cp
from iceemdan_cupy import ICEEMDAN, to_numpy

t = cp.arange(1000, dtype=cp.float64) / 100
x = cp.sin(2 * cp.pi * 2 * t) + 0.4 * cp.sin(2 * cp.pi * 7 * t)
model = ICEEMDAN(trials=100, epsilon=0.2, seed=42,
                 batch_emd=True, graph_control=False)
parts = model(x, T=t)
imfs, residue = model.get_imfs_and_residue()
host_parts = to_numpy(parts)
```

La última fila siempre es el residuo. Extraer un componente no certifica que sea un modo estructural ni un IMF exacto. Las comparaciones CPU/GPU suministran la misma matriz explícita `noise=W`, pues sus generadores aleatorios internos difieren.

## Reproducibilidad

Quedan versionados los [500 pares CPU/GPU](evidence/colominas_500_pairs.jsonl), su [auditoría](evidence/colominas_500_audit.json) y el [benchmark separado](evidence/benchmark_cpu_batch_5_sizes.json). La auditoría aprobó **500/500** parejas ordenadas, 100 semillas por tamaño de ensamble (50, 100, 200, 400, 800). La máxima diferencia de componentes fue 4.44 × 10⁻¹⁶; los diagnósticos discretos coincidieron exactamente. Se evaluó una señal sintética de 1000 muestras y solo la rama ICEEMDAN de esta implementación, sin reproducir las extracciones MATLAB de los autores ni todos los métodos del artículo.

| Realizaciones | Mediana CPU serial (s) | Mediana GPU batch (s) | Razón CPU/GPU |
|---:|---:|---:|---:|
| 50 | 4.156 | 1.318 | 3.154 |
| 100 | 8.479 | 1.475 | 5.750 |
| 200 | 15.662 | 1.483 | 10.559 |
| 400 | 31.473 | 1.800 | 17.490 |
| 800 | 66.987 | 1.913 | 35.010 |

Cada tiempo es la mediana de tres ejecuciones completas tras una preparación por ruta en el equipo indicado. Consulte el [protocolo, comandos y límites](docs/COLOMINAS_500_RESULTS.md) y el [mapa de las Figuras 1–15](docs/PAPER_REPRODUCIBILITY.md).

## Registros públicos

Los scripts opcionales descargan archivos versionados de [Keele](https://zenodo.org/records/3921794) y [CUDB 1.0.0](https://physionet.org/content/cudb/1.0.0/) para analizar ventanas candidatas en GPU. El registro Keele indica uso no comercial y CUDB tiene condiciones propias de atribución. Revise esas condiciones antes de usar los datos. Las descargas van a `data/` y las métricas, matrices o gráficas opcionales a `outputs/`; ambas carpetas están excluidas de Git.

```bash
python -m scripts.fetch_public_data keele
python -m scripts.fetch_public_data cudb
python -m scripts.analyze_keele --plot-local
python -m scripts.analyze_cudb --plot-local
```

`--plot-local` requiere instalar Matplotlib aparte; no es dependencia de ejecución. Las ventanas son coincidencias candidatas, sin identidad confirmada respecto del artículo ni validación clínica. No se versiona ningún trazado biomédico ni matriz de paciente.

## Límites

La validación CUDA local usó Windows 11, Python 3.13.14, CuPy 14.2.0, CUDA runtime 12.9 y una RTX 5080 Laptop. Otras plataformas siguen sin verificar. La ruta batch comprueba una cota inferior de memoria, pero no garantiza que cualquier entrada quepa en VRAM. Consulte [límites batch](docs/BATCHED_EMD_GPU.md) y [estado de validación](docs/VALIDATION_STATUS.md). `graph_control=True` es experimental.

Prueba CUDA enfocada: `python -m pytest -q tests/test_07_batched_extrema.py tests/test_09_numeric_contracts.py`. El sustituto CPU solo verifica lógica Python. Copyright 2025 Javier F. Santamaria y colaboradores. Código bajo [Apache-2.0](LICENSE), con [avisos de terceros](THIRD_PARTY_NOTICES.md). Los derechos de los datos son independientes.
