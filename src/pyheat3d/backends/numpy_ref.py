"""Pure NumPy reference: the correctness oracle (CPU, any float dtype)."""

from __future__ import annotations

import numpy as np

from ..problem import Stencil
from .base import Stepper


def _shift_pairs(u: np.ndarray, periodic: bool) -> list[tuple[np.ndarray, np.ndarray]]:
    """(minus, plus) neighbour arrays per axis, ordered x, y, z.

    Dirichlet: views restricted to the interior. Periodic: rolled full arrays.
    """
    d = u.ndim
    pairs = []
    for ax in range(d - 1, -1, -1):  # x (last axis) first
        if periodic:
            pairs.append((np.roll(u, 1, axis=ax), np.roll(u, -1, axis=ax)))
        else:
            lo = [slice(1, -1)] * d
            hi = [slice(1, -1)] * d
            lo[ax] = slice(0, -2)
            hi[ax] = slice(2, None)
            pairs.append((u[tuple(lo)], u[tuple(hi)]))
    return pairs


def reference_step(
    u: np.ndarray, c0: float, c1: float, bc: str, b: np.ndarray | None = None
) -> np.ndarray:
    """One stencil update in the canonical summation order; returns a new array."""
    periodic = bc == "periodic"
    dt = u.dtype.type
    pairs = _shift_pairs(u, periodic)
    s = pairs[0][0] + pairs[0][1]
    s = s + (pairs[1][0] + pairs[1][1])
    if u.ndim == 3:
        s = s + (pairs[2][0] + pairs[2][1])
    out = u.copy()
    core = (slice(None),) * u.ndim if periodic else (slice(1, -1),) * u.ndim
    new = dt(c0) * u[core] + dt(c1) * s
    if b is not None:
        new = new + b[core]
    out[core] = new
    return out


class NumpyStepper(Stepper):
    """Vectorised slicing update with preallocated temporaries (allocation-free for Dirichlet)."""

    storage = "native"

    def __init__(self, u0: np.ndarray, stencil: Stencil, source: np.ndarray | None = None):
        super().__init__(u0, stencil, source)
        dtype = u0.dtype if u0.dtype in (np.float32, np.float64) else np.float32
        self.storage = np.dtype(dtype).name
        self.a = np.array(u0, dtype=dtype, copy=True)
        self.b = self.a.copy()
        self.src = None if source is None else np.asarray(source, dtype=dtype)
        self._c1 = np.dtype(dtype).type(self.c1)
        self._c0 = np.dtype(dtype).type(self.c0)
        if not self.periodic:
            core = tuple(s - 2 for s in self.shape)
            self.t1 = np.empty(core, dtype)
            self.t2 = np.empty(core, dtype)

    def _step_once(self) -> None:
        u, out = self.a, self.b
        if self.periodic:
            out[...] = reference_step(u, self.c0, self.c1, "periodic", self.src)
        else:
            d = u.ndim
            c = (slice(1, -1),) * d
            pairs = _shift_pairs(u, False)
            np.add(pairs[0][0], pairs[0][1], out=self.t1)
            np.add(pairs[1][0], pairs[1][1], out=self.t2)
            self.t1 += self.t2
            if d == 3:
                np.add(pairs[2][0], pairs[2][1], out=self.t2)
                self.t1 += self.t2
            self.t1 *= self._c1
            o = out[c]
            np.multiply(u[c], self._c0, out=o)
            o += self.t1
            if self.src is not None:
                o += self.src[c]
        self.a, self.b = out, u

    def get(self) -> np.ndarray:
        return self.a.copy()
