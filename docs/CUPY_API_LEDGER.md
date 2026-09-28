# Documentación por operación CuPy

Versión objetivo: **14.2.0**. Publicación: 2026-08-20. Consulta: 25 de septiembre de 2026.

El registro contiene las APIs y operaciones consultadas para el código. Las entradas JSON indican si la consulta precedió a su uso; una se incorporó antes de documentarla y se verificó después. Incluye candidatos descartados; no todas las entradas se utilizan finalmente. Una página estable cacheada de una versión anterior se contrastó con fuente/docstring fijado a v14.2.0. Esto no constituye una prueba de ejecución CUDA.

`tools/check_api_coverage.py` comprueba las referencias estáticas del paquete. Los métodos y operadores sobre arrays se registran también; algunos sitios se cuentan conservadoramente aunque el operando sea metadato host.

## `cupy.asarray`

- Documentación consultada: https://docs.cupy.dev/en/stable/reference/generated/cupy.asarray.html
- Versión objetivo: 14.2.0; fecha: 2026-09-25.

blocking=False is asynchronous by default; use blocking=True for owned host input. Never build an array from nested device scalars; stack them.

## `cupy.array`

- Documentación consultada: https://docs.cupy.dev/en/stable/reference/generated/cupy.array.html
- Versión objetivo: 14.2.0; fecha: 2026-09-25.

blocking=False is asynchronous by default; use blocking=True for owned host input. Never build an array from nested device scalars; stack them.

## `cupy.empty`

- Documentación consultada: https://docs.cupy.dev/en/stable/reference/generated/cupy.empty.html
- Versión objetivo: 14.2.0; fecha: 2026-09-25.
- Fuente fijada de corroboración: https://github.com/cupy/cupy/blob/v14.2.0/cupy/_creation/basic.py

Explicit dtype. No uninitialized value is read. Tagged docstrings corroborate older cached stable pages.

## `cupy.zeros`

- Documentación consultada: https://docs.cupy.dev/en/stable/reference/generated/cupy.zeros.html
- Versión objetivo: 14.2.0; fecha: 2026-09-25.
- Fuente fijada de corroboración: https://github.com/cupy/cupy/blob/v14.2.0/cupy/_creation/basic.py

Explicit dtype. No uninitialized value is read. Tagged docstrings corroborate older cached stable pages.

## `cupy.ones`

- Documentación consultada: https://docs.cupy.dev/en/stable/reference/generated/cupy.ones.html
- Versión objetivo: 14.2.0; fecha: 2026-09-25.
- Fuente fijada de corroboración: https://github.com/cupy/cupy/blob/v14.2.0/cupy/_creation/basic.py

Explicit dtype. No uninitialized value is read. Tagged docstrings corroborate older cached stable pages.

## `cupy.zeros_like`

- Documentación consultada: https://docs.cupy.dev/en/stable/reference/generated/cupy.zeros_like.html
- Versión objetivo: 14.2.0; fecha: 2026-09-25.
- Fuente fijada de corroboración: https://github.com/cupy/cupy/blob/v14.2.0/cupy/_creation/basic.py

Explicit dtype. No uninitialized value is read. Tagged docstrings corroborate older cached stable pages.

## `cupy.full`

- Documentación consultada: https://docs.cupy.dev/en/stable/reference/generated/cupy.full.html
- Versión objetivo: 14.2.0; fecha: 2026-09-25.
- Fuente fijada de corroboración: https://github.com/cupy/cupy/blob/v14.2.0/cupy/_creation/basic.py

Explicit dtype. No uninitialized value is read. Tagged docstrings corroborate older cached stable pages.

## `cupy.arange`

- Documentación consultada: https://docs.cupy.dev/en/stable/reference/generated/cupy.arange.html
- Versión objetivo: 14.2.0; fecha: 2026-09-25.

Half-open bounds; sample indices and index arrays use explicit dtype.

## `cupy.concatenate`

- Documentación consultada: https://docs.cupy.dev/en/stable/reference/generated/cupy.concatenate.html
- Versión objetivo: 14.2.0; fecha: 2026-09-25.
- Fuente fijada de corroboración: https://github.com/cupy/cupy/blob/v14.2.0/cupy/_manipulation/join.py

Only join compatible device arrays, including explicit scalar-to-length-one arrays. Preserve axis and realization order.

## `cupy.stack`

- Documentación consultada: https://docs.cupy.dev/en/stable/reference/generated/cupy.stack.html
- Versión objetivo: 14.2.0; fecha: 2026-09-25.
- Fuente fijada de corroboración: https://github.com/cupy/cupy/blob/v14.2.0/cupy/_manipulation/join.py

Only join compatible device arrays, including explicit scalar-to-length-one arrays. Preserve axis and realization order.

## `cupy.vstack`

- Documentación consultada: https://docs.cupy.dev/en/stable/reference/generated/cupy.vstack.html
- Versión objetivo: 14.2.0; fecha: 2026-09-25.
- Fuente fijada de corroboración: https://github.com/cupy/cupy/blob/v14.2.0/cupy/_manipulation/join.py

Only join compatible device arrays, including explicit scalar-to-length-one arrays. Preserve axis and realization order.

## `cupy.diff`

- Documentación consultada: https://docs.cupy.dev/en/stable/reference/generated/cupy.diff.html
- Versión objetivo: 14.2.0; fecha: 2026-09-25.
- Fuente fijada de corroboración: https://github.com/cupy/cupy/blob/v14.2.0/cupy/_math/sumprod.py

Integer/float difference; boolean diff is inequality. Cast zero-run mask to int8 first.

## `cupy.cumsum`

- Documentación consultada: https://docs.cupy.dev/en/v14.2.0/reference/generated/cupy.cumsum.html
- Versión objetivo: 14.2.0; fecha: 2026-09-28.

Prefijos int64 para compactar una máscara booleana unidimensional. La longitud de salida se lee explícitamente antes del kernel de dispersión; evita el aviso reproducible de `cupy_nonzero_kernel_incomplete_scan` en CuPy 14.2.0 sin filtrar informes del saneador.

## `cupy.sort`

- Documentación consultada: https://docs.cupy.dev/en/stable/reference/generated/cupy.sort.html
- Versión objetivo: 14.2.0; fecha: 2026-09-25.

Stable sort; zeros/plateau indices preserve round-to-even convention.

## `cupy.rint`

- Documentación consultada: https://docs.cupy.dev/en/stable/reference/generated/cupy.rint.html
- Versión objetivo: 14.2.0; fecha: 2026-09-25.

Nearest integer; verify ties-to-even with executable vectors before GPU acceptance.

## `cupy.absolute`

- Documentación consultada: https://docs.cupy.dev/en/stable/reference/generated/cupy.absolute.html
- Versión objetivo: 14.2.0; fecha: 2026-09-25.
- Fuente fijada de corroboración: https://github.com/cupy/cupy/blob/v14.2.0/cupy/_math/misc.py

Elementwise absolute value; preserve arithmetic grouping.

## `cupy.max`

- Documentación consultada: https://docs.cupy.dev/en/stable/reference/generated/cupy.max.html
- Versión objetivo: 14.2.0; fecha: 2026-09-25.

Reductions return device scalars; .item is an explicit synchronization. std uses explicit ddof; even-length median must match NumPy.

## `cupy.mean`

- Documentación consultada: https://docs.cupy.dev/en/stable/reference/generated/cupy.mean.html
- Versión objetivo: 14.2.0; fecha: 2026-09-25.

Reductions return device scalars; .item is an explicit synchronization. std uses explicit ddof; even-length median must match NumPy.

## `cupy.std`

- Documentación consultada: https://docs.cupy.dev/en/stable/reference/generated/cupy.std.html
- Versión objetivo: 14.2.0; fecha: 2026-09-25.

Reductions return device scalars; .item is an explicit synchronization. std uses explicit ddof; even-length median must match NumPy.

## `cupy.median`

- Documentación consultada: https://docs.cupy.dev/en/stable/reference/generated/cupy.median.html
- Versión objetivo: 14.2.0; fecha: 2026-09-25.

Reductions return device scalars; .item is an explicit synchronization. std uses explicit ddof; even-length median must match NumPy.

## `cupy.sum`

- Documentación consultada: https://docs.cupy.dev/en/stable/reference/generated/cupy.sum.html
- Versión objetivo: 14.2.0; fecha: 2026-09-25.

Reductions return device scalars; .item is an explicit synchronization. std uses explicit ddof; even-length median must match NumPy.

## `cupy.ptp`

- Documentación consultada: https://docs.cupy.dev/en/stable/reference/generated/cupy.ptp.html
- Versión objetivo: 14.2.0; fecha: 2026-09-25.

Reductions return device scalars; .item is an explicit synchronization. std uses explicit ddof; even-length median must match NumPy.

## `cupy.all`

- Documentación consultada: https://docs.cupy.dev/en/stable/reference/generated/cupy.all.html
- Versión objetivo: 14.2.0; fecha: 2026-09-25.
- Fuente fijada de corroboración: https://github.com/cupy/cupy/blob/v14.2.0/cupy/_logic/truth.py

Return device scalar/array. Consume scalar intentionally for Python control, not NumPy evaluation.

## `cupy.any`

- Documentación consultada: https://docs.cupy.dev/en/stable/reference/generated/cupy.any.html
- Versión objetivo: 14.2.0; fecha: 2026-09-25.
- Fuente fijada de corroboración: https://github.com/cupy/cupy/blob/v14.2.0/cupy/_logic/truth.py

Return device scalar/array. Consume scalar intentionally for Python control, not NumPy evaluation.

## `cupy.isfinite`

- Documentación consultada: https://docs.cupy.dev/en/stable/reference/generated/cupy.isfinite.html
- Versión objetivo: 14.2.0; fecha: 2026-09-25.
- Fuente fijada de corroboración: https://github.com/cupy/cupy/blob/v14.2.0/cupy/_logic/content.py

True excludes infinity and NaN, device ufunc.

## `cupy.array_equal`

- Documentación consultada: https://docs.cupy.dev/en/stable/reference/generated/cupy.array_equal.html
- Versión objetivo: 14.2.0; fecha: 2026-09-25.
- Fuente fijada de corroboración: https://github.com/cupy/cupy/blob/v14.2.0/cupy/_logic/comparison.py

Exact equality used for stagnation; not a tolerant comparison.

## `cupy.where`

- Documentación consultada: https://docs.cupy.dev/en/stable/reference/generated/cupy.where.html
- Versión objetivo: 14.2.0; fecha: 2026-09-25.

Three-argument selection. Safe denominator before division; no unsupported divide(where=...).

## `cupy.gradient`

- Documentación consultada: https://docs.cupy.dev/en/stable/reference/generated/cupy.gradient.html
- Versión objetivo: 14.2.0; fecha: 2026-09-25.

Irregular coordinates supported; edge_order=1 to preserve reference.

## `cupy.linalg.solve`

- Documentación consultada: https://docs.cupy.dev/en/stable/reference/generated/cupy.linalg.solve.html
- Versión objetivo: 14.2.0; fecha: 2026-09-25.

Use cupyx.errstate(linalg="raise") to inspect solver failures; preserve the natural three-point system.

## `cupyx.errstate`

- Documentación consultada: https://docs.cupy.dev/en/stable/reference/generated/cupy.linalg.solve.html
- Versión objetivo: 14.2.0; fecha: 2026-09-25.
- Fuente fijada de corroboración: https://github.com/cupy/cupy/blob/v14.2.0/cupyx/_ufunc_config.py

Source docs TODO. Source reviewed: linalg ignore/raise supported; over/invalid/divide non-None raise NotImplementedError. Do not copy np.errstate(over=...).

## `cupy.asnumpy`

- Documentación consultada: https://docs.cupy.dev/en/stable/reference/generated/cupy.asnumpy.html
- Versión objetivo: 14.2.0; fecha: 2026-09-25.

Only explicit output/validation boundary. blocking=True. No call in numerical modules.

## `cupy.cuda.Device`

- Documentación consultada: https://docs.cupy.dev/en/stable/reference/generated/cupy.cuda.Device.html
- Versión objetivo: 14.2.0; fecha: 2026-09-25.

Context manager restores device; id, mem_info, compute_capability and synchronize documented.

## `cupy.cuda.runtime.getDeviceCount`

- Documentación consultada: https://docs.cupy.dev/en/stable/reference/generated/cupy.cuda.runtime.getDeviceCount.html
- Versión objetivo: 14.2.0; fecha: 2026-09-25.

Returns available device count; failures propagate as environment BLOCKED.

## `cupy.random.Philox4x3210`

- Documentación consultada: https://docs.cupy.dev/en/stable/reference/generated/cupy.random.Philox4x3210.html
- Versión objetivo: 14.2.0; fecha: 2026-09-25.

Explicit cuRAND generator, not default_rng (XORWOW). No claimed NumPy PCG64 sequence equivalence.

## `cupy.random.Generator`

- Documentación consultada: https://docs.cupy.dev/en/stable/reference/generated/cupy.random.Generator.html
- Versión objetivo: 14.2.0; fecha: 2026-09-25.
- Fuente fijada de corroboración: https://github.com/cupy/cupy/blob/v14.2.0/cupy/random/_generator_api.pyx

Generator methods accept device output dtype; no cross-version bitstream guarantee.

## `cupy.random.Generator.standard_normal`

- Documentación consultada: https://docs.cupy.dev/en/stable/reference/generated/cupy.random.Generator.html#cupy.random.Generator.standard_normal
- Versión objetivo: 14.2.0; fecha: 2026-09-25.
- Fuente fijada de corroboración: https://github.com/cupy/cupy/blob/v14.2.0/cupy/random/_generator_api.pyx

standard_normal(size,dtype=float64); uniform(low,high,size,dtype=float64). Native stream is separately versioned, request counter restored on failure.

## `cupy.random.Generator.uniform`

- Documentación consultada: https://docs.cupy.dev/en/stable/reference/generated/cupy.random.Generator.html#cupy.random.Generator.uniform
- Versión objetivo: 14.2.0; fecha: 2026-09-25.
- Fuente fijada de corroboración: https://github.com/cupy/cupy/blob/v14.2.0/cupy/random/_generator_api.pyx

standard_normal(size,dtype=float64); uniform(low,high,size,dtype=float64). Native stream is separately versioned, request counter restored on failure.

## `cupyx.scipy.interpolate.CubicSpline`

- Documentación consultada: https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.interpolate.CubicSpline.html
- Versión objetivo: 14.2.0; fecha: 2026-09-25.

Preserve exact interpolation choice. CubicSpline only with >3 supports. Akima classic, not MAKIMA. interp1d axis/default/error behavior checked individually.

## `cupyx.scipy.interpolate.PchipInterpolator`

- Documentación consultada: https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.interpolate.PchipInterpolator.html
- Versión objetivo: 14.2.0; fecha: 2026-09-25.

Preserve exact interpolation choice. CubicSpline only with >3 supports. Akima classic, not MAKIMA. interp1d axis/default/error behavior checked individually.

## `cupyx.scipy.interpolate.Akima1DInterpolator`

- Documentación consultada: https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.interpolate.Akima1DInterpolator.html
- Versión objetivo: 14.2.0; fecha: 2026-09-25.

Preserve exact interpolation choice. CubicSpline only with >3 supports. Akima classic, not MAKIMA. interp1d axis/default/error behavior checked individually.

## `cupyx.scipy.interpolate.CubicHermiteSpline`

- Documentación consultada: https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.interpolate.CubicHermiteSpline.html
- Versión objetivo: 14.2.0; fecha: 2026-09-25.

Preserve exact interpolation choice. CubicSpline only with >3 supports. Akima classic, not MAKIMA. interp1d axis/default/error behavior checked individually.

## `cupyx.scipy.interpolate.interp1d`

- Documentación consultada: https://docs.cupy.dev/en/stable/reference/generated/cupyx.scipy.interpolate.interp1d.html
- Versión objetivo: 14.2.0; fecha: 2026-09-25.

Preserve exact interpolation choice. CubicSpline only with >3 supports. Akima classic, not MAKIMA. interp1d axis/default/error behavior checked individually.

## `cupy.ndarray`

- Documentación consultada: https://docs.cupy.dev/en/stable/reference/generated/cupy.ndarray.html
- Versión objetivo: 14.2.0; fecha: 2026-09-25.

Device storage; metadata attributes shape, ndim, size, nbytes, dtype, device.id stay host. Basic and advanced indexing supported. Integer-array out-of-range wraps: generated indices must be in range.

## `cupy.ndarray.astype`

- Documentación consultada: https://docs.cupy.dev/en/stable/reference/generated/cupy.ndarray.html#cupy.ndarray.astype
- Versión objetivo: 14.2.0; fecha: 2026-09-25.

Read ndarray methods. item/conversion used only for explicit scalar control. copy owns result; reshape preserves order; no repeated-index scatter writes.

## `cupy.ndarray.copy`

- Documentación consultada: https://docs.cupy.dev/en/stable/reference/generated/cupy.ndarray.html#cupy.ndarray.copy
- Versión objetivo: 14.2.0; fecha: 2026-09-25.

Read ndarray methods. item/conversion used only for explicit scalar control. copy owns result; reshape preserves order; no repeated-index scatter writes.

## `cupy.ndarray.reshape`

- Documentación consultada: https://docs.cupy.dev/en/stable/reference/generated/cupy.ndarray.html#cupy.ndarray.reshape
- Versión objetivo: 14.2.0; fecha: 2026-09-25.

Read ndarray methods. item/conversion used only for explicit scalar control. copy owns result; reshape preserves order; no repeated-index scatter writes.

## `cupy.ndarray.sum`

- Documentación consultada: https://docs.cupy.dev/en/stable/reference/generated/cupy.ndarray.html#cupy.ndarray.sum
- Versión objetivo: 14.2.0; fecha: 2026-09-25.

Read ndarray methods. item/conversion used only for explicit scalar control. copy owns result; reshape preserves order; no repeated-index scatter writes.

## `cupy.ndarray.item`

- Documentación consultada: https://docs.cupy.dev/en/stable/reference/generated/cupy.ndarray.html#cupy.ndarray.item
- Versión objetivo: 14.2.0; fecha: 2026-09-25.

Read ndarray methods. item/conversion used only for explicit scalar control. copy owns result; reshape preserves order; no repeated-index scatter writes.

## `cupy.ndarray.__getitem__`

- Documentación consultada: https://docs.cupy.dev/en/stable/reference/generated/cupy.ndarray.html#cupy.ndarray.__getitem__
- Versión objetivo: 14.2.0; fecha: 2026-09-25.

Read ndarray methods. item/conversion used only for explicit scalar control. copy owns result; reshape preserves order; no repeated-index scatter writes.

## `cupy.ndarray.__setitem__`

- Documentación consultada: https://docs.cupy.dev/en/stable/reference/generated/cupy.ndarray.html#cupy.ndarray.__setitem__
- Versión objetivo: 14.2.0; fecha: 2026-09-25.

Read ndarray methods. item/conversion used only for explicit scalar control. copy owns result; reshape preserves order; no repeated-index scatter writes.

## `cupy.ndarray.__len__`

- Documentación consultada: https://docs.cupy.dev/en/stable/reference/generated/cupy.ndarray.html#cupy.ndarray.__len__
- Versión objetivo: 14.2.0; fecha: 2026-09-25.

Read ndarray methods. item/conversion used only for explicit scalar control. copy owns result; reshape preserves order; no repeated-index scatter writes.

## `cupy.ndarray.__iter__`

- Documentación consultada: https://docs.cupy.dev/en/stable/reference/generated/cupy.ndarray.html#cupy.ndarray.__iter__
- Versión objetivo: 14.2.0; fecha: 2026-09-25.

Read ndarray methods. item/conversion used only for explicit scalar control. copy owns result; reshape preserves order; no repeated-index scatter writes.

## `cupy.ndarray.__bool__`

- Documentación consultada: https://docs.cupy.dev/en/stable/reference/generated/cupy.ndarray.html#cupy.ndarray.__bool__
- Versión objetivo: 14.2.0; fecha: 2026-09-25.

Read ndarray methods. item/conversion used only for explicit scalar control. copy owns result; reshape preserves order; no repeated-index scatter writes.

## `cupy.ndarray.__float__`

- Documentación consultada: https://docs.cupy.dev/en/stable/reference/generated/cupy.ndarray.html#cupy.ndarray.__float__
- Versión objetivo: 14.2.0; fecha: 2026-09-25.

Read ndarray methods. item/conversion used only for explicit scalar control. copy owns result; reshape preserves order; no repeated-index scatter writes.

## `cupy.ndarray.__int__`

- Documentación consultada: https://docs.cupy.dev/en/stable/reference/generated/cupy.ndarray.html#cupy.ndarray.__int__
- Versión objetivo: 14.2.0; fecha: 2026-09-25.

Read ndarray methods. item/conversion used only for explicit scalar control. copy owns result; reshape preserves order; no repeated-index scatter writes.

## `cupy.ufunc`

- Documentación consultada: https://docs.cupy.dev/en/stable/reference/generated/cupy.ufunc.html
- Versión objetivo: 14.2.0; fecha: 2026-09-25.

Elementwise broadcasting for arrays/scalars; no assumed NumPy-only keyword arguments.

## `cupy.add`

- Documentación consultada: https://docs.cupy.dev/en/stable/reference/generated/cupy.add.html
- Versión objetivo: 14.2.0; fecha: 2026-09-25.
- Fuente fijada de corroboración: https://github.com/cupy/cupy/blob/v14.2.0/cupy/_math/arithmetic.py

Elementwise arithmetic, preserve written grouping; no fusion/fast-math. Tagged source corroborates stale documentation.

## `cupy.subtract`

- Documentación consultada: https://docs.cupy.dev/en/stable/reference/generated/cupy.subtract.html
- Versión objetivo: 14.2.0; fecha: 2026-09-25.
- Fuente fijada de corroboración: https://github.com/cupy/cupy/blob/v14.2.0/cupy/_math/arithmetic.py

Elementwise arithmetic, preserve written grouping; no fusion/fast-math. Tagged source corroborates stale documentation.

## `cupy.multiply`

- Documentación consultada: https://docs.cupy.dev/en/stable/reference/generated/cupy.multiply.html
- Versión objetivo: 14.2.0; fecha: 2026-09-25.
- Fuente fijada de corroboración: https://github.com/cupy/cupy/blob/v14.2.0/cupy/_math/arithmetic.py

Elementwise arithmetic, preserve written grouping; no fusion/fast-math. Tagged source corroborates stale documentation.

## `cupy.divide`

- Documentación consultada: https://docs.cupy.dev/en/stable/reference/generated/cupy.divide.html
- Versión objetivo: 14.2.0; fecha: 2026-09-25.
- Fuente fijada de corroboración: https://github.com/cupy/cupy/blob/v14.2.0/cupy/_math/arithmetic.py

Elementwise arithmetic, preserve written grouping; no fusion/fast-math. Tagged source corroborates stale documentation.

## `cupy.negative`

- Documentación consultada: https://docs.cupy.dev/en/stable/reference/generated/cupy.negative.html
- Versión objetivo: 14.2.0; fecha: 2026-09-25.
- Fuente fijada de corroboración: https://github.com/cupy/cupy/blob/v14.2.0/cupy/_math/arithmetic.py

Elementwise arithmetic, preserve written grouping; no fusion/fast-math. Tagged source corroborates stale documentation.

## `cupy.equal`

- Documentación consultada: https://docs.cupy.dev/en/stable/reference/generated/cupy.equal.html
- Versión objetivo: 14.2.0; fecha: 2026-09-25.
- Fuente fijada de corroboración: https://github.com/cupy/cupy/blob/v14.2.0/cupy/_logic/comparison.py

Strict comparisons including equality in boundary branches unchanged; no epsilon tie rule.

## `cupy.not_equal`

- Documentación consultada: https://docs.cupy.dev/en/stable/reference/generated/cupy.not_equal.html
- Versión objetivo: 14.2.0; fecha: 2026-09-25.
- Fuente fijada de corroboración: https://github.com/cupy/cupy/blob/v14.2.0/cupy/_logic/comparison.py

Strict comparisons including equality in boundary branches unchanged; no epsilon tie rule.

## `cupy.less`

- Documentación consultada: https://docs.cupy.dev/en/stable/reference/generated/cupy.less.html
- Versión objetivo: 14.2.0; fecha: 2026-09-25.
- Fuente fijada de corroboración: https://github.com/cupy/cupy/blob/v14.2.0/cupy/_logic/comparison.py

Strict comparisons including equality in boundary branches unchanged; no epsilon tie rule.

## `cupy.less_equal`

- Documentación consultada: https://docs.cupy.dev/en/stable/reference/generated/cupy.less_equal.html
- Versión objetivo: 14.2.0; fecha: 2026-09-25.
- Fuente fijada de corroboración: https://github.com/cupy/cupy/blob/v14.2.0/cupy/_logic/comparison.py

Strict comparisons including equality in boundary branches unchanged; no epsilon tie rule.

## `cupy.greater`

- Documentación consultada: https://docs.cupy.dev/en/stable/reference/generated/cupy.greater.html
- Versión objetivo: 14.2.0; fecha: 2026-09-25.
- Fuente fijada de corroboración: https://github.com/cupy/cupy/blob/v14.2.0/cupy/_logic/comparison.py

Strict comparisons including equality in boundary branches unchanged; no epsilon tie rule.

## `cupy.greater_equal`

- Documentación consultada: https://docs.cupy.dev/en/stable/reference/generated/cupy.greater_equal.html
- Versión objetivo: 14.2.0; fecha: 2026-09-25.
- Fuente fijada de corroboración: https://github.com/cupy/cupy/blob/v14.2.0/cupy/_logic/comparison.py

Strict comparisons including equality in boundary branches unchanged; no epsilon tie rule.

## `cupy.bitwise_and`

- Documentación consultada: https://docs.cupy.dev/en/stable/reference/generated/cupy.bitwise_and.html
- Versión objetivo: 14.2.0; fecha: 2026-09-25.
- Fuente fijada de corroboración: https://github.com/cupy/cupy/blob/v14.2.0/cupy/_binary/elementwise.py

Boolean masks use elementwise operators &, | (not Python and/or).

## `cupy.bitwise_or`

- Documentación consultada: https://docs.cupy.dev/en/stable/reference/generated/cupy.bitwise_or.html
- Versión objetivo: 14.2.0; fecha: 2026-09-25.
- Fuente fijada de corroboración: https://github.com/cupy/cupy/blob/v14.2.0/cupy/_binary/elementwise.py

Boolean masks use elementwise operators &, | (not Python and/or).
