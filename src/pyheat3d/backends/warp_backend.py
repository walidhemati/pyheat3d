"""NVIDIA Warp backends: one thread per site, and a 3D z-marching variant (register queue)."""

from __future__ import annotations

import numpy as np
import warp as wp

from ..problem import Stencil
from .base import Stepper

wp.config.quiet = True

ZCH = 32  # planes per thread in the z-marching kernel


@wp.kernel
def heat2d_dirichlet(u: wp.array2d(dtype=wp.float32), o: wp.array2d(dtype=wp.float32),
                     c0: float, c1: float):
    j, i = wp.tid()
    ny = u.shape[0]
    nx = u.shape[1]
    if i == 0 or i >= nx - 1 or j == 0 or j >= ny - 1:
        return
    s = (u[j, i - 1] + u[j, i + 1]) + (u[j - 1, i] + u[j + 1, i])
    o[j, i] = c0 * u[j, i] + c1 * s


@wp.kernel
def heat2d_periodic(u: wp.array2d(dtype=wp.float32), o: wp.array2d(dtype=wp.float32),
                    c0: float, c1: float):
    j, i = wp.tid()
    ny = u.shape[0]
    nx = u.shape[1]
    im = (i + nx - 1) % nx
    ip = (i + 1) % nx
    jm = (j + ny - 1) % ny
    jp = (j + 1) % ny
    s = (u[j, im] + u[j, ip]) + (u[jm, i] + u[jp, i])
    o[j, i] = c0 * u[j, i] + c1 * s


@wp.kernel
def heat3d_dirichlet(u: wp.array3d(dtype=wp.float32), o: wp.array3d(dtype=wp.float32),
                     c0: float, c1: float):
    k, j, i = wp.tid()
    nz = u.shape[0]
    ny = u.shape[1]
    nx = u.shape[2]
    if i == 0 or i >= nx - 1 or j == 0 or j >= ny - 1 or k == 0 or k >= nz - 1:
        return
    s = ((u[k, j, i - 1] + u[k, j, i + 1]) + (u[k, j - 1, i] + u[k, j + 1, i])) + (
        u[k - 1, j, i] + u[k + 1, j, i]
    )
    o[k, j, i] = c0 * u[k, j, i] + c1 * s


@wp.kernel
def heat3d_periodic(u: wp.array3d(dtype=wp.float32), o: wp.array3d(dtype=wp.float32),
                    c0: float, c1: float):
    k, j, i = wp.tid()
    nz = u.shape[0]
    ny = u.shape[1]
    nx = u.shape[2]
    im = (i + nx - 1) % nx
    ip = (i + 1) % nx
    jm = (j + ny - 1) % ny
    jp = (j + 1) % ny
    km = (k + nz - 1) % nz
    kp = (k + 1) % nz
    s = ((u[k, j, im] + u[k, j, ip]) + (u[k, jm, i] + u[k, jp, i])) + (u[km, j, i] + u[kp, j, i])
    o[k, j, i] = c0 * u[k, j, i] + c1 * s


@wp.kernel
def heat3d_zmarch(u: wp.array3d(dtype=wp.float32), o: wp.array3d(dtype=wp.float32),
                  c0: float, c1: float, periodic: int, zch: int):
    """Each thread owns an (x, y) column and marches ``zch`` planes with a register queue."""
    kc, j, i = wp.tid()
    nz = u.shape[0]
    ny = u.shape[1]
    nx = u.shape[2]
    k0 = kc * zch
    k1 = wp.min(k0 + zch, nz)
    im = i - 1
    ip = i + 1
    jm = j - 1
    jp = j + 1
    kb = k0 - 1
    if periodic != 0:
        im = (i + nx - 1) % nx
        ip = (i + 1) % nx
        jm = (j + ny - 1) % ny
        jp = (j + 1) % ny
        kb = (k0 + nz - 1) % nz
    else:
        if i == 0 or i >= nx - 1 or j == 0 or j >= ny - 1:
            return
        k0 = wp.max(k0, 1)
        k1 = wp.min(k1, nz - 1)
        kb = k0 - 1
    below = u[kb, j, i]
    center = u[k0, j, i]
    for k in range(k0, k1):
        ka = k + 1
        if periodic != 0:
            ka = ka % nz
        above = u[ka, j, i]
        s = ((u[k, j, im] + u[k, j, ip]) + (u[k, jm, i] + u[k, jp, i])) + (below + above)
        o[k, j, i] = c0 * center + c1 * s
        below = center
        center = above


class WarpStepper(Stepper):
    """Warp variant; kernel launches are pre-recorded (``record_cmd``) to cut Python overhead."""

    kind = "naive"

    def __init__(self, u0: np.ndarray, stencil: Stencil, source: np.ndarray | None = None):
        super().__init__(u0, stencil, source)
        if source is not None:
            raise ValueError("Warp variants do not support a source term")
        wp.init()
        if not wp.is_cuda_available():
            raise RuntimeError("Warp backend needs a CUDA GPU")
        self.device = "cuda:0"
        host = np.ascontiguousarray(u0, dtype=np.float32)
        self.a = wp.array(host, dtype=wp.float32, device=self.device)
        self.b = wp.array(host, dtype=wp.float32, device=self.device)
        c0, c1 = self.c0, self.c1
        if self.kind == "zmarch":
            if self.dims != 3:
                raise ValueError("zmarch is 3D only")
            nz, ny, nx = self.shape
            dim = ((nz + ZCH - 1) // ZCH, ny, nx)
            kern, extra = heat3d_zmarch, [int(self.periodic), ZCH]
        else:
            dim = self.shape
            kern = {
                (2, False): heat2d_dirichlet, (2, True): heat2d_periodic,
                (3, False): heat3d_dirichlet, (3, True): heat3d_periodic,
            }[(self.dims, self.periodic)]
            extra = []
        self.cmds = [
            wp.launch(kern, dim=dim, inputs=[x, y, c0, c1, *extra], device=self.device,
                      record_cmd=True)
            for x, y in ((self.a, self.b), (self.b, self.a))
        ]
        self.parity = 0

    def _step_once(self) -> None:
        self.cmds[self.parity].launch()
        self.parity ^= 1

    def get(self) -> np.ndarray:
        cur = self.a if self.parity == 0 else self.b
        return cur.numpy().copy()

    def synchronize(self) -> None:
        wp.synchronize_device(self.device)

    def close(self) -> None:
        del self.cmds, self.a, self.b


def make(kind: str) -> type[WarpStepper]:
    """Create a WarpStepper subclass for a kernel kind ("naive" or "zmarch")."""
    return type(f"Warp_{kind}", (WarpStepper,), {"kind": kind})
