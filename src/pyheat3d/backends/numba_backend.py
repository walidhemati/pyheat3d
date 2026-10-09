"""Numba-CUDA backends: one thread per site, and a 3D z-marching variant (register queue)."""

from __future__ import annotations

import warnings

import numpy as np

# numba-cuda 0.30 still references ``np.row_stack``, removed in NumPy 2.5. Restore the alias
# (it was always identical to ``np.vstack``) before numba.cuda builds its typing registry.
if not hasattr(np, "row_stack"):  # pragma: no cover - depends on the installed NumPy
    setattr(np, "row_stack", np.vstack)  # noqa: B010

from numba import cuda  # noqa: E402
from numba.core.errors import NumbaPerformanceWarning  # noqa: E402

from ..problem import Stencil  # noqa: E402
from .base import Stepper  # noqa: E402

warnings.simplefilter("ignore", NumbaPerformanceWarning)

ZCH = 32


@cuda.jit(cache=False)
def heat2d(u, o, c0, c1, periodic):
    i, j = cuda.grid(2)
    ny, nx = u.shape
    if i >= nx or j >= ny:
        return
    if periodic:
        im = i - 1 if i > 0 else nx - 1
        ip = i + 1 if i < nx - 1 else 0
        jm = j - 1 if j > 0 else ny - 1
        jp = j + 1 if j < ny - 1 else 0
    else:
        if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
            return
        im, ip, jm, jp = i - 1, i + 1, j - 1, j + 1
    s = (u[j, im] + u[j, ip]) + (u[jm, i] + u[jp, i])
    o[j, i] = c0 * u[j, i] + c1 * s


@cuda.jit(cache=False)
def heat3d(u, o, c0, c1, periodic):
    i, j, k = cuda.grid(3)
    nz, ny, nx = u.shape
    if i >= nx or j >= ny or k >= nz:
        return
    if periodic:
        im = i - 1 if i > 0 else nx - 1
        ip = i + 1 if i < nx - 1 else 0
        jm = j - 1 if j > 0 else ny - 1
        jp = j + 1 if j < ny - 1 else 0
        km = k - 1 if k > 0 else nz - 1
        kp = k + 1 if k < nz - 1 else 0
    else:
        if i == 0 or i == nx - 1 or j == 0 or j == ny - 1 or k == 0 or k == nz - 1:
            return
        im, ip, jm, jp, km, kp = i - 1, i + 1, j - 1, j + 1, k - 1, k + 1
    s = ((u[k, j, im] + u[k, j, ip]) + (u[k, jm, i] + u[k, jp, i])) + (u[km, j, i] + u[kp, j, i])
    o[k, j, i] = c0 * u[k, j, i] + c1 * s


@cuda.jit(cache=False)
def heat3d_zmarch(u, o, c0, c1, periodic, zch):
    """Each thread owns an (x, y) column and marches ``zch`` planes with a register queue."""
    i, j, kc = cuda.grid(3)
    nz, ny, nx = u.shape
    if i >= nx or j >= ny:
        return
    k0 = kc * zch
    if k0 >= nz:
        return
    k1 = min(k0 + zch, nz)
    if periodic:
        im = i - 1 if i > 0 else nx - 1
        ip = i + 1 if i < nx - 1 else 0
        jm = j - 1 if j > 0 else ny - 1
        jp = j + 1 if j < ny - 1 else 0
        kb = k0 - 1 if k0 > 0 else nz - 1
    else:
        if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
            return
        im, ip, jm, jp = i - 1, i + 1, j - 1, j + 1
        k0 = max(k0, 1)
        k1 = min(k1, nz - 1)
        kb = k0 - 1
    below = u[kb, j, i]
    center = u[k0, j, i]
    for k in range(k0, k1):
        ka = k + 1
        if ka == nz:
            ka = 0
        above = u[ka, j, i]
        s = ((u[k, j, im] + u[k, j, ip]) + (u[k, jm, i] + u[k, jp, i])) + (below + above)
        o[k, j, i] = c0 * center + c1 * s
        below = center
        center = above


class NumbaStepper(Stepper):
    """Numba-CUDA variant ("naive" 2D/3D or "zmarch" 3D)."""

    kind = "naive"

    def __init__(self, u0: np.ndarray, stencil: Stencil, source: np.ndarray | None = None):
        super().__init__(u0, stencil, source)
        if source is not None:
            raise ValueError("Numba variants do not support a source term")
        if not cuda.is_available():
            raise RuntimeError("Numba-CUDA backend needs a CUDA GPU")
        host = np.ascontiguousarray(u0, dtype=np.float32)
        self.a = cuda.to_device(host)
        self.b = cuda.to_device(host)
        self.args = (np.float32(self.c0), np.float32(self.c1), bool(self.periodic))
        if self.dims == 2:
            ny, nx = self.shape
            tpb = (32, 8)
            grid = ((nx + 31) // 32, (ny + 7) // 8)
            self.kern = heat2d[grid, tpb]
        elif self.kind == "zmarch":
            nz, ny, nx = self.shape
            tpb = (32, 8, 1)
            grid = ((nx + 31) // 32, (ny + 7) // 8, (nz + ZCH - 1) // ZCH)
            self.kern = heat3d_zmarch[grid, tpb]
            self.args = (*self.args, ZCH)
        else:
            nz, ny, nx = self.shape
            tpb = (32, 4, 2)
            grid = ((nx + 31) // 32, (ny + 3) // 4, (nz + 1) // 2)
            self.kern = heat3d[grid, tpb]

    def _step_once(self) -> None:
        self.kern(self.a, self.b, *self.args)
        self.a, self.b = self.b, self.a

    def get(self) -> np.ndarray:
        return self.a.copy_to_host()

    def synchronize(self) -> None:
        cuda.synchronize()

    def close(self) -> None:
        del self.a, self.b


def make(kind: str) -> type[NumbaStepper]:
    """Create a NumbaStepper subclass for a kernel kind."""
    return type(f"Numba_{kind}", (NumbaStepper,), {"kind": kind})
