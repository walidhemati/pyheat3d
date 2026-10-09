"""Triton backends: tiled 2D kernel, tiled 3D kernel and a 3D z-marching kernel.

All kernels load the storage type (fp32, fp16 or bf16), compute in fp32 and store the storage
type, so the low-precision variants halve the DRAM traffic per update.
"""

from __future__ import annotations

import numpy as np
import torch
import triton
import triton.language as tl

from ..problem import Stencil
from .base import Stepper
from .torch_backends import _TORCH_DTYPES


@triton.jit
def _xy_indices(pid_x, pid_y, ny, nx, PERIODIC: tl.constexpr, BY: tl.constexpr,
                BX: tl.constexpr):
    """Tile coordinates, neighbour coordinates and the update mask of a (BY, BX) tile."""
    ix = (pid_x * BX + tl.arange(0, BX))[None, :]
    iy = (pid_y * BY + tl.arange(0, BY))[:, None]
    inb = (ix < nx) & (iy < ny)
    if PERIODIC:
        xm = tl.where(ix == 0, nx - 1, ix - 1)
        xp = tl.where(ix == nx - 1, 0, ix + 1)
        ym = tl.where(iy == 0, ny - 1, iy - 1)
        yp = tl.where(iy == ny - 1, 0, iy + 1)
        mask = inb
    else:
        xm = ix - 1
        xp = ix + 1
        ym = iy - 1
        yp = iy + 1
        mask = inb & (ix > 0) & (ix < nx - 1) & (iy > 0) & (iy < ny - 1)
    return ix, iy, xm, xp, ym, yp, mask


@triton.jit
def _ld(ptr, off, mask):
    return tl.load(ptr + off, mask=mask, other=0.0).to(tl.float32)


@triton.jit
def heat2d_tile(u, out, ny, nx, c0, c1, PERIODIC: tl.constexpr, BY: tl.constexpr,
                BX: tl.constexpr):
    """One program updates a (BY, BX) tile; neighbour tiles come through L1/L2."""
    ix, iy, xm, xp, ym, yp, mask = _xy_indices(tl.program_id(0), tl.program_id(1), ny, nx,
                                               PERIODIC, BY, BX)
    row = iy * nx
    c = _ld(u, row + ix, mask)
    sx = _ld(u, row + xm, mask) + _ld(u, row + xp, mask)
    sy = _ld(u, ym * nx + ix, mask) + _ld(u, yp * nx + ix, mask)
    r = c0 * c + c1 * (sx + sy)
    tl.store(out + row + ix, r.to(out.dtype.element_ty), mask=mask)


@triton.jit
def heat3d_tile(u, out, nz, ny, nx, c0, c1, PERIODIC: tl.constexpr, BY: tl.constexpr,
                BX: tl.constexpr):
    """One program per (xy tile, z plane); all seven operands are loaded from memory."""
    z = tl.program_id(2)
    ix, iy, xm, xp, ym, yp, mask = _xy_indices(tl.program_id(0), tl.program_id(1), ny, nx,
                                               PERIODIC, BY, BX)
    if PERIODIC:
        zm = tl.where(z == 0, nz - 1, z - 1)
        zp = tl.where(z == nz - 1, 0, z + 1)
    else:
        mask = mask & (z > 0) & (z < nz - 1)
        zm = z - 1
        zp = z + 1
    plane = ny * nx
    row = iy * nx
    base = z * plane
    c = _ld(u, base + row + ix, mask)
    sx = _ld(u, base + row + xm, mask) + _ld(u, base + row + xp, mask)
    sy = _ld(u, base + ym * nx + ix, mask) + _ld(u, base + yp * nx + ix, mask)
    sz = _ld(u, zm * plane + row + ix, mask) + _ld(u, zp * plane + row + ix, mask)
    r = c0 * c + c1 * ((sx + sy) + sz)
    tl.store(out + base + row + ix, r.to(out.dtype.element_ty), mask=mask)


@triton.jit
def heat3d_zmarch(u, out, nz, ny, nx, c0, c1, PERIODIC: tl.constexpr, BY: tl.constexpr,
                  BX: tl.constexpr, ZCH: tl.constexpr):
    """2.5D blocking: a program owns an xy tile and marches ZCH planes along z.

    The z-neighbours live in a register queue (below, center, above), so each plane is read
    from DRAM once per sweep; x/y neighbours are re-read from L1.
    """
    ix, iy, xm, xp, ym, yp, mask = _xy_indices(tl.program_id(0), tl.program_id(1), ny, nx,
                                               PERIODIC, BY, BX)
    plane = ny * nx
    row = iy * nx
    z0 = tl.program_id(2) * ZCH
    z1 = tl.minimum(z0 + ZCH, nz)
    if PERIODIC:
        zb = tl.where(z0 == 0, nz - 1, z0 - 1)
    else:
        z0 = tl.maximum(z0, 1)
        z1 = tl.minimum(z1, nz - 1)
        zb = z0 - 1
    below = _ld(u, zb * plane + row + ix, mask)
    center = _ld(u, z0 * plane + row + ix, mask)
    for z in range(z0, z1):
        za = z + 1
        if PERIODIC:
            za = tl.where(za == nz, 0, za)
        above = _ld(u, za * plane + row + ix, mask)
        base = z * plane
        sx = _ld(u, base + row + xm, mask) + _ld(u, base + row + xp, mask)
        sy = _ld(u, base + ym * nx + ix, mask) + _ld(u, base + yp * nx + ix, mask)
        r = c0 * center + c1 * ((sx + sy) + (below + above))
        tl.store(out + base + row + ix, r.to(out.dtype.element_ty), mask=mask)
        below = center
        center = above


class TritonStepper(Stepper):
    """Triton variant; ``kind`` is "tile" (2D/3D) or "zmarch" (3D)."""

    kind = "tile"
    storage = "float32"
    BY2, BX2, WARPS2 = 8, 128, 4
    BY3, BX3, WARPS3 = 4, 64, 4
    BYZ, BXZ, WARPSZ, ZCH = 4, 64, 4, 8

    def __init__(self, u0: np.ndarray, stencil: Stencil, source: np.ndarray | None = None):
        super().__init__(u0, stencil, source)
        if source is not None:
            raise ValueError("Triton variants do not support a source term")
        if not torch.cuda.is_available():
            raise RuntimeError("Triton backend needs a CUDA GPU")
        dt = _TORCH_DTYPES[self.storage]
        self.a = torch.as_tensor(np.ascontiguousarray(u0, dtype=np.float32)).cuda().to(dt)
        self.b = self.a.clone()
        ny, nx = self.shape[-2:]
        if self.dims == 2:
            self.grid = (triton.cdiv(nx, self.BX2), triton.cdiv(ny, self.BY2))
        elif self.kind == "zmarch":
            nz = self.shape[0]
            self.grid = (triton.cdiv(nx, self.BXZ), triton.cdiv(ny, self.BYZ),
                         triton.cdiv(nz, self.ZCH))
        else:
            self.grid = (triton.cdiv(nx, self.BX3), triton.cdiv(ny, self.BY3), self.shape[0])

    def _step_once(self) -> None:
        u, o = self.a, self.b
        p = self.periodic
        if self.dims == 2:
            heat2d_tile[self.grid](u, o, *self.shape, self.c0, self.c1, PERIODIC=p,
                                   BY=self.BY2, BX=self.BX2, num_warps=self.WARPS2)
        elif self.kind == "zmarch":
            heat3d_zmarch[self.grid](u, o, *self.shape, self.c0, self.c1, PERIODIC=p,
                                     BY=self.BYZ, BX=self.BXZ, ZCH=self.ZCH,
                                     num_warps=self.WARPSZ)
        else:
            heat3d_tile[self.grid](u, o, *self.shape, self.c0, self.c1, PERIODIC=p,
                                   BY=self.BY3, BX=self.BX3, num_warps=self.WARPS3)
        self.a, self.b = o, u

    def get(self) -> np.ndarray:
        return self.a.float().cpu().numpy()

    def synchronize(self) -> None:
        torch.cuda.synchronize()

    def close(self) -> None:
        del self.a, self.b
        torch.cuda.empty_cache()


def make(kind: str, storage: str) -> type[TritonStepper]:
    """Create a TritonStepper subclass for a kernel kind and a storage precision."""
    return type(f"Triton_{kind}_{storage}", (TritonStepper,), {"kind": kind, "storage": storage})
