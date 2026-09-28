"""Device noise with transactional request counters (Philox request v1)."""

from __future__ import annotations

import hashlib
import math
import secrets

from .contracts import integer, nonnegative
from .runtime import cp, scalar, synchronize


class NoiseStream:
    def __init__(self, mode="cupy_philox", seed=None):
        if mode != "cupy_philox":
            raise ValueError("Only rng_mode='cupy_philox' is supported; noise is generated on GPU.")
        self.mode = mode
        self.reset(seed)

    def reset(self, seed):
        if seed is not None:
            seed = integer("seed", seed, 0)
        self.base_seed = secrets.randbits(128) if seed is None else seed
        self.counter = 0

    def snapshot(self):
        return {
            "mode": self.mode,
            "seed": self.base_seed,
            "request": self.counter,
        }

    def restore(self, state):
        if state["mode"] != self.mode or state["seed"] != self.base_seed:
            raise ValueError("Snapshot belongs to a different random stream.")
        self.counter = state["request"]

    def draw(self, scale, size, kind="normal", *, defer_validation=False, _state_allocator=None):
        scale = nonnegative("scale", scale)
        if isinstance(size, (tuple, list)):
            shape = tuple(integer("size dimension", v, 0) for v in size)
        else:
            shape = integer("size", size, 0)
        if kind not in ("normal", "uniform"):
            raise ValueError("Invalid noise kind.")
        limit = scale * math.sqrt(3.0)
        if kind == "uniform" and not math.isfinite(limit):
            raise ValueError("Uniform noise scale is too large.")
        saved = self.snapshot()
        try:
            material = f"iceemdan-philox-request-v1:{self.base_seed}:{self.counter}".encode("ascii")
            derived = int.from_bytes(hashlib.sha256(material).digest()[:16], "little")
            if _state_allocator is None:
                bit_generator = cp.random.Philox4x3210(derived, size=256000)
            else:
                with cp.cuda.using_allocator(_state_allocator):
                    bit_generator = cp.random.Philox4x3210(derived, size=256000)
            generator = cp.random.Generator(bit_generator)
            if kind == "normal":
                out = generator.standard_normal(size=shape, dtype="float64") * scale
            else:
                out = generator.uniform(-limit, limit, size=shape, dtype="float64")
            if not defer_validation:
                if not scalar(cp.all(cp.isfinite(out))):
                    raise FloatingPointError("Noise generation overflowed.")
                synchronize()
            self.counter += 1
            return out
        except BaseException:
            self.restore(saved)
            raise
