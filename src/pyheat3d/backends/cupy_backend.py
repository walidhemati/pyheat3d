"""Hand-written CUDA C reference kernels (``stencil.cu``) compiled at runtime by NVRTC via CuPy.

Variants: ``naive`` (one thread per site), ``zmarch`` (3D 2.5D blocking with shared-memory
plane + register queue), ``ymarch`` (2D register-queue marching) and ``tb<K>`` (temporal
blocking, K fused time steps per DRAM pass), each with fp32 / fp16 / bf16 storage.
"""

from __future__ import annotations

import ctypes
import glob
import os
import site
import warnings
from functools import cache
from importlib import resources

import numpy as np

from ..problem import Stencil
from .base import Stepper


def _preload_nvrtc() -> None:
    """Make the pip-installed NVRTC (``nvidia-cuda-nvrtc-cu12``) visible to CuPy.

    CuPy dlopens ``libnvrtc.so.12`` by name; when the CUDA toolkit is not installed system-wide,
    the copy shipped as a wheel is not on the loader path. Loading it with RTLD_GLOBAL first
    makes the subsequent dlopen resolve to it.
    """
    roots = list(site.getsitepackages()) + [site.getusersitepackages()]
    for root in roots:
        for path in glob.glob(os.path.join(root, "nvidia", "cuda_nvrtc", "lib", "libnvrtc.so.*")):
            if path.endswith(".so.12") or path.endswith(".so.13"):
                try:
                    ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                    return
                except OSError:  # pragma: no cover
                    continue


_preload_nvrtc()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # CuPy warns when several cupy-cuda* wheels coexist
    import cupy as cp

_STORAGE_CODE = {"float32": 0, "float16": 1, "bfloat16": 2}

#: default launch / tiling parameters (chosen by a small sweep, see bench/tune.py)
NAIVE2D_BLOCK = (32, 8, 1)
NAIVE3D_BLOCK = (32, 4, 2)
#: sites per thread in the naive kernels, per storage type
NAIVE_VEC = {"float32": 1, "float16": 2, "bfloat16": 2}
ZMARCH = {"BX": 64, "BY": 4, "ZCH": 16}
# temporal blocking 3D, per K: block 32 x TYT threads, R rows per thread (tile 32 x TYT*R)
TB3D = {
    2: {"TYT": 4, "R": 4, "ZCH": 64},
    3: {"TYT": 8, "R": 4, "ZCH": 64},
    4: {"TYT": 8, "R": 8, "ZCH": 128},
}
TB3D_DEFAULT = {"TYT": 8, "R": 8, "ZCH": 128}
# temporal blocking 2D (warp-synchronous), per K: R columns per lane (strip 32*R), WARPS per block
TB2D = {
    1: {"R": 2, "WARPS": 2, "ZCH": 256},
    2: {"R": 2, "WARPS": 2, "ZCH": 256},
    3: {"R": 2, "WARPS": 2, "ZCH": 256},
    4: {"R": 2, "WARPS": 2, "ZCH": 256},
    8: {"R": 2, "WARPS": 2, "ZCH": 256},
}
TB2D_DEFAULT = {"R": 2, "WARPS": 2, "ZCH": 256}


#: aim for at least this many resident threads when choosing the march-chunk length, so small
#: grids still fill the 40 SMs (shorter chunks cost (ZCH + 2K) / ZCH redundant planes)
TARGET_THREADS = 48 * 1024


def chunk_length(n_march: int, columns_threads: int, zch_max: int, k: int) -> int:
    """Power-of-two march-chunk length giving >= TARGET_THREADS threads (>= 4K, <= zch_max)."""
    want_chunks = max(1, -(-TARGET_THREADS // max(columns_threads, 1)))
    z = -(-n_march // want_chunks)
    z = 1 << max(0, (z - 1).bit_length())  # round up to a power of two
    return int(max(min(z, zch_max), 4 * k, 8))


@cache
def _source() -> str:
    return resources.files("pyheat3d.backends").joinpath("stencil.cu").read_text()


@cache
def _module(storage: str, periodic: bool, **macros: int) -> cp.RawModule:
    opts = ["--std=c++14", f"-DSTORAGE={_STORAGE_CODE[storage]}", f"-DPERIODIC={int(periodic)}"]
    opts += [f"-D{k}={v}" for k, v in sorted(macros.items())]
    return cp.RawModule(code=_source(), options=tuple(opts))


def kernel(name: str, storage: str, periodic: bool, **macros: int):
    """Fetch (compiling and caching on first use) a kernel for a macro configuration."""
    return _module(storage, periodic, **macros).get_function(name)


def to_bf16_bits(x: np.ndarray) -> np.ndarray:
    """float32 -> bfloat16 bit pattern (uint16), round-to-nearest-even."""
    b = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    b = (b + 0x7FFF + ((b >> 16) & 1)) >> 16
    return b.astype(np.uint16)


def from_bf16_bits(b: np.ndarray) -> np.ndarray:
    """bfloat16 bit pattern (uint16) -> float32."""
    return (b.astype(np.uint32) << 16).view(np.float32)


class CupyStepper(Stepper):
    """CUDA C variant. ``kind`` in {"naive", "zmarch", "tb"}; ``K`` = fused steps for "tb"."""

    kind = "naive"
    K = 1
    storage = "float32"

    def __init__(self, u0: np.ndarray, stencil: Stencil, source: np.ndarray | None = None):
        super().__init__(u0, stencil, source)
        if source is not None:
            raise ValueError("CUDA C variants do not support a source term")
        if cp.cuda.runtime.getDeviceCount() < 1:  # pragma: no cover
            raise RuntimeError("no CUDA device")
        self.steps_per_pass = self.K if self.kind == "tb" else 1
        host = np.ascontiguousarray(u0, dtype=np.float32)
        if self.storage == "bfloat16":
            dev = cp.asarray(to_bf16_bits(host))
        else:
            dev = cp.asarray(host.astype(np.dtype(self.storage)))
        self.a = dev
        self.b = dev.copy()
        self.c = (np.float32(self.c0), np.float32(self.c1))
        self.dimargs = tuple(np.int32(s) for s in self.shape)
        self._setup()

    # -- launch configuration -----------------------------------------------------------------
    def _setup(self) -> None:
        st, p = self.storage, self.periodic
        sh = self.shape
        nx, ny = sh[-1], sh[-2]
        vec = NAIVE_VEC[st]
        if self.dims == 2:
            bx, by, _ = NAIVE2D_BLOCK
            self.naive = (kernel("naive2d", st, p, VEC=vec),
                          ((nx + bx * vec - 1) // (bx * vec), (ny + by - 1) // by), (bx, by))
        else:
            bx, by, bz = NAIVE3D_BLOCK
            self.naive = (kernel("naive3d", st, p, VEC=vec),
                          ((nx + bx * vec - 1) // (bx * vec), (ny + by - 1) // by,
                           (sh[0] + bz - 1) // bz), (bx, by, bz))
        self.main = self.naive
        if self.kind == "zmarch":
            if self.dims != 3:
                raise ValueError("zmarch is 3D only (use ymarch/tb in 2D)")
            m = ZMARCH
            self.main = (kernel("zmarch3d", st, p, **m),
                         ((nx + m["BX"] - 1) // m["BX"], (ny + m["BY"] - 1) // m["BY"],
                          (sh[0] + m["ZCH"] - 1) // m["ZCH"]), (m["BX"], m["BY"]))
        elif self.kind == "tb":
            K = self.K
            if self.dims == 3:
                m = dict(TB3D.get(K, TB3D_DEFAULT), K=K)
                ox, oy = 32 - 2 * K, m["TYT"] * m["R"] - 2 * K
                cols = ((nx + ox - 1) // ox) * ((ny + oy - 1) // oy) * 32 * m["TYT"]
                m["ZCH"] = chunk_length(sh[0], cols, m["ZCH"], K)
                self.main = (kernel("tb3d", st, p, **m),
                             ((nx + ox - 1) // ox, (ny + oy - 1) // oy,
                              (sh[0] + m["ZCH"] - 1) // m["ZCH"]), (32, m["TYT"]))
            else:
                m = dict(TB2D.get(K, TB2D_DEFAULT), K=K)
                ox = 32 * m["R"] - 2 * K
                strips = (nx + ox - 1) // ox
                m["ZCH"] = chunk_length(ny, strips * 32, m["ZCH"], K)
                self.main = (kernel("tb2d", st, p, **m),
                             ((strips + m["WARPS"] - 1) // m["WARPS"],
                              (ny + m["ZCH"] - 1) // m["ZCH"]), (32, m["WARPS"]))
            self.tile = m

    def _launch(self, cfg) -> None:
        kern, grid, block = cfg
        kern(grid, block, (self.a, self.b, *self.dimargs, *self.c))
        self.a, self.b = self.b, self.a

    def step(self, n: int = 1) -> None:
        k = self.steps_per_pass
        for _ in range(n // k):
            self._launch(self.main)
        for _ in range(n % k):
            self._launch(self.naive)
        self.steps_done += n

    def get(self) -> np.ndarray:
        h = cp.asnumpy(self.a)
        if self.storage == "bfloat16":
            return from_bf16_bits(h)
        return h.astype(np.float32)

    def synchronize(self) -> None:
        cp.cuda.runtime.deviceSynchronize()

    def close(self) -> None:
        del self.a, self.b
        cp.get_default_memory_pool().free_all_blocks()


def make(kind: str, storage: str = "float32", K: int = 1) -> type[CupyStepper]:
    """Create a CupyStepper subclass for a kernel kind, storage precision and K."""
    return type(f"Cupy_{kind}{K}_{storage}", (CupyStepper,),
                {"kind": kind, "storage": storage, "K": K})
