"""CUDA conditional WHILE graph bridge for a CuPy stream (CUDA 12.9 ABI)."""

from __future__ import annotations

import ctypes as ct
import ctypes.util
import importlib.util
import sys
from importlib.resources import files

from numpy import uint64

from .runtime import cp

# ponytail: process-lifetime quarantine after cleanup failure; revisit only with context recovery.
_unsafe_owners: list[object] = []

_INITIAL_GATE = cp.RawKernel(
    r"""
    extern "C" __device__ void cudaGraphSetConditional(unsigned long long, unsigned int);
    extern "C" __global__ void initial_gate(
        unsigned long long setup, unsigned long long loop,
        const int* initial, int* reason) {
        int code = initial[0];
        reason[0] = code;
        cudaGraphSetConditional(setup, code == 0);
        cudaGraphSetConditional(loop, code == 0);
    }
    """,
    "initial_gate",
)


class _Conditional(ct.Structure):
    _fields_ = [
        ("handle", ct.c_uint64),
        ("kind", ct.c_int),
        ("size", ct.c_uint),
        ("body_graphs", ct.POINTER(ct.c_void_p)),
    ]


class _NodeParams(ct.Structure):
    # driver_types.h, CUDA 12.9: 16-byte header, 232-byte union, 8-byte tail.
    _fields_ = [
        ("type", ct.c_int),
        ("reserved0", ct.c_int * 3),
        ("payload", ct.c_byte * 232),
        ("reserved2", ct.c_int64),
    ]


assert ct.sizeof(_NodeParams) == 256


class CUDAWhileGraph:
    """Capture a warmed CuPy body into a device-controlled CUDA WHILE node.

    The caller retains all input and result arrays until the graph is closed.
    An optional preallocated arena serves CuPy allocations during capture;
    without it, callers must warm work on the same stream first.
    The body must set its conditional handle from a device kernel each pass.

    Parameters
    ----------
    stream : cupy.cuda.Stream
        Stream used for capture, launch and synchronization.
    initial_code, reason : cupy.ndarray, optional
        Device int32 scalars for the initial gate and termination reason.
        Supply both or neither.
    arena_bytes : int, default 0
        Bytes reserved for capture allocations; zero uses CuPy's allocator.
    """

    def __init__(self, stream, *, initial_code=None, reason=None, arena_bytes=0):
        if (initial_code is None) != (reason is None):
            raise ValueError("initial_code and reason must be supplied together.")
        if initial_code is not None and (
            initial_code.dtype.name != "int32"
            or initial_code.shape not in ((), (1,))
            or reason.dtype.name != "int32"
            or reason.shape not in ((), (1,))
        ):
            raise ValueError("initial_code and reason must be device int32 scalars.")
        library = None
        try:
            bundled_runtime = importlib.util.find_spec("nvidia.cuda_runtime") is not None
        except ModuleNotFoundError:
            bundled_runtime = False
        if bundled_runtime:
            suffix = (
                ("bin", "cudart64_12.dll")
                if sys.platform == "win32"
                else ("lib", "libcudart.so.12")
            )
            bundled = files("nvidia.cuda_runtime").joinpath(suffix[0]).joinpath(suffix[1])
            if bundled.is_file():
                library = str(bundled)
        if library is None:
            library = ctypes.util.find_library(
                "cudart64_12" if sys.platform == "win32" else "cudart"
            )
        if library is None:
            raise RuntimeError("CUDA runtime 12.x is required for conditional graphs.")
        runtime = ct.CDLL(library)
        runtime.cudaRuntimeGetVersion.argtypes = [ct.POINTER(ct.c_int)]
        runtime.cudaRuntimeGetVersion.restype = ct.c_int
        version = ct.c_int()
        if runtime.cudaRuntimeGetVersion(ct.byref(version)) != 0 or version.value != 12090:
            raise RuntimeError(
                "Conditional graph bridge requires the tested CUDA 12.9 runtime ABI."
            )
        self._runtime = runtime
        self._stream = stream
        self._graph = ct.c_void_p()
        self._executable = ct.c_void_p()
        self._body = None
        self._result = None
        self._setup_result = None
        self._setup_captured = False
        self._if_body = None
        self._owners = []
        self._arena = cp.cuda.alloc(arena_bytes) if arena_bytes else None
        self._arena_used = 0
        self._rng_state = None

        def api(name, args):
            fn = getattr(runtime, name)
            fn.argtypes = args
            fn.restype = ct.c_int
            return fn

        self._create = api("cudaGraphCreate", [ct.POINTER(ct.c_void_p), ct.c_uint])
        self._handle_create = api(
            "cudaGraphConditionalHandleCreate",
            [ct.POINTER(ct.c_uint64), ct.c_void_p, ct.c_uint, ct.c_uint],
        )
        self._add_node = api(
            "cudaGraphAddNode",
            [
                ct.POINTER(ct.c_void_p),
                ct.c_void_p,
                ct.c_void_p,
                ct.c_size_t,
                ct.POINTER(_NodeParams),
            ],
        )
        self._begin = api(
            "cudaStreamBeginCaptureToGraph",
            [ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.c_int],
        )
        self._end = api("cudaStreamEndCapture", [ct.c_void_p, ct.POINTER(ct.c_void_p)])
        self._instantiate = api(
            "cudaGraphInstantiate", [ct.POINTER(ct.c_void_p), ct.c_void_p, ct.c_uint64]
        )
        self._launch = api("cudaGraphLaunch", [ct.c_void_p, ct.c_void_p])
        self._exec_destroy = api("cudaGraphExecDestroy", [ct.c_void_p])
        self._destroy = api("cudaGraphDestroy", [ct.c_void_p])
        self._get_nodes = api(
            "cudaGraphGetNodes", [ct.c_void_p, ct.POINTER(ct.c_void_p), ct.POINTER(ct.c_size_t)]
        )
        self._add_dependencies = api(
            "cudaGraphAddDependencies",
            [ct.c_void_p, ct.POINTER(ct.c_void_p), ct.POINTER(ct.c_void_p), ct.c_size_t],
        )
        self._error_string = runtime.cudaGetErrorString
        self._error_string.argtypes = [ct.c_int]
        self._error_string.restype = ct.c_char_p

        self._check(self._create, ct.byref(self._graph), 0)
        try:
            handle = ct.c_uint64()
            self._check(self._handle_create, ct.byref(handle), self._graph, 1, 1)
            self.handle = handle.value
            params = _NodeParams()
            params.type = 0x0D
            conditional = _Conditional.from_buffer(params.payload)
            conditional.handle = self.handle
            conditional.kind = 1  # cudaGraphCondTypeWhile
            conditional.size = 1
            node = ct.c_void_p()
            self._check(self._add_node, ct.byref(node), self._graph, None, 0, ct.byref(params))
            self._body = conditional.body_graphs[0]
            if initial_code is not None:
                setup_handle = ct.c_uint64()
                self._check(self._handle_create, ct.byref(setup_handle), self._graph, 1, 1)
                params = _NodeParams()
                params.type = 0x0D
                conditional = _Conditional.from_buffer(params.payload)
                conditional.handle = setup_handle.value
                conditional.kind = 0  # cudaGraphCondTypeIf
                conditional.size = 1
                setup_node = ct.c_void_p()
                self._check(
                    self._add_node, ct.byref(setup_node), self._graph, None, 0, ct.byref(params)
                )
                self._if_body = conditional.body_graphs[0]
                _INITIAL_GATE.compile()
                self._capture_to(
                    self._graph,
                    lambda: _INITIAL_GATE(
                        (1,),
                        (1,),
                        (uint64(setup_handle.value), uint64(self.handle), initial_code, reason),
                    ),
                )
                count = ct.c_size_t()
                self._check(self._get_nodes, self._graph, None, ct.byref(count))
                nodes = (ct.c_void_p * count.value)()
                self._check(self._get_nodes, self._graph, nodes, ct.byref(count))
                prefix = [p for p in nodes if p not in (node.value, setup_node.value)]
                if len(prefix) != 1:
                    raise RuntimeError("Expected one initial device gate node.")
                first = ct.c_void_p(prefix[0])
                self._check(
                    self._add_dependencies,
                    self._graph,
                    ct.byref(first),
                    ct.byref(setup_node),
                    1,
                )
                self._check(
                    self._add_dependencies,
                    self._graph,
                    ct.byref(setup_node),
                    ct.byref(node),
                    1,
                )
        except BaseException:
            try:
                self.close()
            except BaseException:
                pass
            raise

    def _check(self, function, *args):
        code = function(*args)
        if code:
            detail = self._error_string(code).decode("utf-8", errors="replace")
            raise RuntimeError(f"{function.__name__}: {detail} ({code})")

    def prepare(self, work):
        """Warm captured work while retaining its allocations, then return them to the pool.

        Parameters
        ----------
        work : callable
            Zero-argument operation to warm on the capture stream.
        """
        owners = []
        allocator = cp.cuda.get_allocator()

        def retain(size):
            pointer = allocator(size)
            owners.append(pointer)
            return pointer

        try:
            with self._stream, cp.cuda.using_allocator(retain):
                result = work()
            self._stream.synchronize()
            del result
        except BaseException:
            _unsafe_owners.append(owners)
            raise
        owners.clear()

    def _capture_to(self, target, body):
        stream = ct.c_void_p(self._stream.ptr)
        self._check(self._begin, stream, target, None, None, 0, 0)
        allocator = cp.cuda.get_allocator()

        def retain(size):
            if self._arena is not None:
                pointer = self._arena_pointer(size)
            else:
                pointer = allocator(size)
            self._owners.append(pointer)
            return pointer

        def end_capture():
            captured = ct.c_void_p()
            code = self._end(stream, ct.byref(captured))
            if code or not captured.value or captured.value != target.value:
                # BeginCaptureToGraph transfers this body graph at EndCapture.
                # On error its prior handle may be invalid: quarantine it.
                _unsafe_owners.append((self._graph, self._owners))
                self._graph = ct.c_void_p()
                self._body = None
                self._owners = []
                if code:
                    detail = self._error_string(code).decode("utf-8", errors="replace")
                    raise RuntimeError(f"cudaStreamEndCapture: {detail} ({code})")
                raise RuntimeError("CUDA capture returned a null or different body graph.")
            return captured

        try:
            with self._stream, cp.cuda.using_allocator(retain):
                result = body()
        except BaseException:
            try:
                end_capture()
            except BaseException:
                pass
            raise
        end_capture()
        return result

    def _arena_pointer(self, size):
        assert self._arena is not None
        offset = (self._arena_used + 255) & ~255
        if offset + size > self._arena.mem.size:
            raise MemoryError("CUDA graph capture arena capacity exhausted.")
        pointer = cp.cuda.MemoryPointer(self._arena.mem, offset)
        self._arena_used = offset + size
        return pointer

    def rng_state_allocator(self, size):
        """Reuse Philox state after each ordered request in the capture stream.

        Parameters
        ----------
        size : int
            Requested state bytes.

        Returns
        -------
        cupy.cuda.MemoryPointer
            Arena pointer retained through graph lifetime.
        """
        if size == 16_384_000 and self._rng_state is not None:
            return self._rng_state
        pointer = self._arena_pointer(size)
        if size == 16_384_000:
            self._rng_state = pointer
        self._owners.append(pointer)
        return pointer

    def capture_setup(self, body):
        """Capture work that runs once before the conditional stage loop.

        Parameters
        ----------
        body : callable
            Zero-argument setup operation.

        Returns
        -------
        object
            Result retained until graph cleanup.
        """
        if self._if_body is None or self._setup_captured:
            raise RuntimeError("CUDA setup body is unavailable or already captured.")
        self._setup_result = self._capture_to(ct.c_void_p(self._if_body), body)
        self._setup_captured = True
        return self._setup_result

    def capture(self, body):
        """Capture body(handle) once; return arrays the body wishes to retain.

        Parameters
        ----------
        body : callable
            Stage operation accepting the device conditional handle.

        Returns
        -------
        object
            Result retained until graph cleanup.
        """
        if self._executable.value:
            raise RuntimeError("CUDA WHILE graph has already been captured.")
        if self._if_body is not None and not self._setup_captured:
            raise RuntimeError("CUDA setup body must be captured before the stage loop.")
        result = self._capture_to(ct.c_void_p(self._body), lambda: body(self.handle))
        self._check(self._instantiate, ct.byref(self._executable), self._graph, 0)
        self._result = result
        return result

    def launch(self):
        """Launch asynchronously; caller synchronizes before reading outputs."""
        if not self._executable.value:
            raise RuntimeError("CUDA WHILE graph has not been captured.")
        self._check(self._launch, self._executable, ct.c_void_p(self._stream.ptr))

    def close(self):
        error = None
        if self._executable.value:
            try:
                self._stream.synchronize()
            except BaseException as exc:
                error = exc
            try:
                self._check(self._exec_destroy, self._executable)
                self._executable = ct.c_void_p()
            except BaseException as exc:
                if error is None:
                    error = exc
        if self._graph.value:
            try:
                self._check(self._destroy, self._graph)
                self._graph = ct.c_void_p()
            except BaseException as exc:
                if error is None:
                    error = exc
        if error is not None:
            _unsafe_owners.append(
                (
                    getattr(self, "_owners", []),
                    self._result,
                    getattr(self, "_setup_result", None),
                    self._arena,
                )
            )
        self._owners = []
        self._result = None
        self._setup_result = None
        self._arena = None
        if error is not None:
            raise error

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if exc_type is None:
            self.close()
        else:
            try:
                self.close()
            except BaseException:
                pass
