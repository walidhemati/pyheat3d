"""Registry of every backend variant (the optimisation ladder).

Variants are keyed ``"<backend>/<variant>"``. Factories import their backend lazily, so listing
the registry never requires torch/triton/warp/numba/cupy to be installed.
"""

from __future__ import annotations

import ctypes
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from .backends.base import Stepper
from .problem import Stencil

BACKENDS = ("numpy", "torch-eager", "torch-compile", "triton", "warp", "numba", "cupy-cuda-c")

#: shared colours across the Python-HPC suite
COLORS = {
    "numpy": "#7f7f7f",
    "torch-eager": "#9467bd",
    "torch-compile": "#8c564b",
    "triton": "#1f77b4",
    "warp": "#2ca02c",
    "numba": "#ff7f0e",
    "cupy-cuda-c": "#d62728",
}


@dataclass(frozen=True)
class Variant:
    """One rung of the ladder."""

    backend: str
    name: str
    factory: Callable[[], type[Stepper]] = field(repr=False, compare=False)
    dims: tuple[int, ...] = (2, 3)
    storage: str = "float32"
    steps_per_pass: int = 1
    gpu: bool = True
    source: bool = False
    rung: int = 0
    description: str = ""

    @property
    def key(self) -> str:
        return f"{self.backend}/{self.name}"

    @property
    def bytes_per_value(self) -> int:
        return {"float32": 4, "float16": 2, "bfloat16": 2, "float64": 8}.get(self.storage, 4)


def _numpy():
    from .backends.numpy_ref import NumpyStepper

    return NumpyStepper


def _torch_eager():
    from .backends.torch_backends import TorchEagerStepper

    return TorchEagerStepper


def _torch_compile():
    from .backends.torch_backends import TorchCompileStepper

    return TorchCompileStepper


def _triton(kind: str, storage: str):
    def f():
        from .backends import triton_backend

        return triton_backend.make(kind, storage)

    return f


def _warp(kind: str):
    def f():
        from .backends import warp_backend

        return warp_backend.make(kind)

    return f


def _numba(kind: str):
    def f():
        from .backends import numba_backend

        return numba_backend.make(kind)

    return f


def _cupy(kind: str, storage: str = "float32", K: int = 1):
    def f():
        from .backends import cupy_backend

        return cupy_backend.make(kind, storage, K)

    return f


_SHORT = {"float32": "", "float16": "-fp16", "bfloat16": "-bf16"}


def _build() -> dict[str, Variant]:
    v: list[Variant] = [
        Variant("numpy", "slicing", _numpy, gpu=False, source=True, rung=0, storage="native",
                description="NumPy vectorised slicing on the CPU (reference oracle)"),
        Variant("torch-eager", "slicing", _torch_eager, source=True, rung=1,
                description="NumPy-style slicing on the GPU; one kernel per operator"),
        Variant("torch-compile", "inductor", _torch_compile, source=True, rung=2,
                description="torch.compile/Inductor fuses the expression into one kernel"),
    ]
    for st in ("float32", "float16", "bfloat16"):
        v.append(Variant("triton", "tile" + _SHORT[st], _triton("tile", st), storage=st,
                         rung=3, description=f"Triton tiled kernel, {st} storage"))
    for st in ("float32", "float16", "bfloat16"):
        v.append(Variant("triton", "zmarch" + _SHORT[st], _triton("zmarch", st), dims=(3,),
                         storage=st, rung=4,
                         description=f"Triton 2.5D z-marching (register queue), {st} storage"))
    v += [
        Variant("warp", "naive", _warp("naive"), rung=3,
                description="Warp kernel, one thread per site"),
        Variant("warp", "zmarch", _warp("zmarch"), dims=(3,), rung=4,
                description="Warp kernel, per-thread z column with register queue"),
        Variant("numba", "naive", _numba("naive"), rung=3,
                description="Numba-CUDA kernel, one thread per site"),
        Variant("numba", "zmarch", _numba("zmarch"), dims=(3,), rung=4,
                description="Numba-CUDA kernel, per-thread z column with register queue"),
    ]
    for st in ("float32", "float16", "bfloat16"):
        v.append(Variant("cupy-cuda-c", "naive" + _SHORT[st], _cupy("naive", st), storage=st,
                         rung=3, description=f"CUDA C, one thread per site, {st} storage"))
    v += [
        Variant("cupy-cuda-c", "zmarch", _cupy("zmarch"), dims=(3,), rung=4,
                description="CUDA C 2.5D blocking: shared-memory plane + register queue"),
        Variant("cupy-cuda-c", "ymarch", _cupy("tb", K=1), dims=(2,), rung=4,
                description="CUDA C 2D y-marching with register queue"),
    ]
    for K in (2, 3, 4):
        v.append(Variant("cupy-cuda-c", f"tb{K}", _cupy("tb", K=K), dims=(2, 3),
                         steps_per_pass=K, rung=5,
                         description=f"CUDA C temporal blocking, {K} steps per DRAM pass"))
    v.append(Variant("cupy-cuda-c", "tb8", _cupy("tb", K=8), dims=(2,), steps_per_pass=8,
                     rung=5, description="CUDA C temporal blocking, 8 steps per DRAM pass"))
    v.append(Variant("cupy-cuda-c", "tb4-fp16", _cupy("tb", "float16", 4), dims=(2, 3),
                     storage="float16", steps_per_pass=4, rung=6,
                     description="CUDA C temporal blocking (K=4) with fp16 storage"))
    return {x.key: x for x in v}


VARIANTS: dict[str, Variant] = _build()


@lru_cache(maxsize=1)
def cuda_available() -> bool:
    """True if the CUDA driver reports at least one device (library-independent check)."""
    for name in ("libcuda.so.1", "libcuda.so", "nvcuda.dll"):
        try:
            lib = ctypes.CDLL(name)
        except OSError:
            continue
        if lib.cuInit(0) != 0:
            return False
        n = ctypes.c_int(0)
        return lib.cuDeviceGetCount(ctypes.byref(n)) == 0 and n.value > 0
    return False


@lru_cache(maxsize=None)
def availability(key: str) -> tuple[bool, str]:
    """``(ok, reason)``: whether a variant can run here."""
    var = VARIANTS[key]
    if var.gpu and not cuda_available() and var.backend != "torch-eager":
        return False, "no CUDA GPU"
    try:
        var.factory()
    except Exception as e:  # missing package, broken install, ...
        return False, f"{type(e).__name__}: {e}".splitlines()[0][:120]
    return True, ""


def available_variants(dims: int | None = None) -> list[str]:
    """Keys of variants that can run on this machine (optionally for one dimensionality)."""
    return [k for k, v in VARIANTS.items()
            if (dims is None or dims in v.dims) and availability(k)[0]]


def make_stepper(key: str, u0: np.ndarray, stencil: Stencil,
                 source: np.ndarray | None = None) -> Stepper:
    """Instantiate a variant on an initial state."""
    var = VARIANTS[key]
    if u0.ndim not in var.dims:
        raise ValueError(f"{key} supports dims {var.dims}, got {u0.ndim}")
    if source is not None and not var.source:
        raise ValueError(f"{key} does not support a source term")
    cls = var.factory()
    st = cls(u0, stencil, source)
    st.steps_per_pass = var.steps_per_pass
    return st
