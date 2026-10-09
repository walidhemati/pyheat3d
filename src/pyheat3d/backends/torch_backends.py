"""PyTorch backends: eager slicing (many small kernels) and ``torch.compile`` (one fused kernel)."""

from __future__ import annotations

import numpy as np
import torch

from ..problem import Stencil
from .base import Stepper

_TORCH_DTYPES = {
    "float32": torch.float32,
    "float64": torch.float64,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def device() -> torch.device:
    """CUDA device if present, else CPU (the torch backends also run on CPU for CI)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _pairs(u: torch.Tensor, periodic: bool) -> list[tuple[torch.Tensor, torch.Tensor]]:
    d = u.dim()
    pairs = []
    for ax in range(d - 1, -1, -1):
        if periodic:
            pairs.append((torch.roll(u, 1, dims=ax), torch.roll(u, -1, dims=ax)))
        else:
            lo = [slice(1, -1)] * d
            hi = [slice(1, -1)] * d
            lo[ax] = slice(0, -2)
            hi[ax] = slice(2, None)
            pairs.append((u[tuple(lo)], u[tuple(hi)]))
    return pairs


def functional_step(
    u: torch.Tensor, out: torch.Tensor, c0: float, c1: float, periodic: bool,
    b: torch.Tensor | None = None,
) -> None:
    """Out-of-place update written as one expression (the form handed to ``torch.compile``)."""
    p = _pairs(u, periodic)
    s = (p[0][0] + p[0][1]) + (p[1][0] + p[1][1])
    if u.dim() == 3:
        s = s + (p[2][0] + p[2][1])
    core = (slice(None),) * u.dim() if periodic else (slice(1, -1),) * u.dim()
    new = c0 * u[core] + c1 * s
    if b is not None:
        new = new + b[core]
    out[core] = new


class _TorchStepper(Stepper):
    storage = "float32"

    def __init__(self, u0: np.ndarray, stencil: Stencil, source: np.ndarray | None = None):
        super().__init__(u0, stencil, source)
        dt = torch.float64 if u0.dtype == np.float64 else _TORCH_DTYPES[self.storage]
        self.storage = str(dt).replace("torch.", "")
        self.dev = device()
        self.a = torch.as_tensor(np.ascontiguousarray(u0)).to(device=self.dev, dtype=dt)
        self.b = self.a.clone()
        self.src = (
            None if source is None else torch.as_tensor(source).to(device=self.dev, dtype=dt)
        )

    def get(self) -> np.ndarray:
        x = self.a
        if x.dtype not in (torch.float32, torch.float64):
            x = x.float()
        return x.cpu().numpy()

    def synchronize(self) -> None:
        if self.dev.type == "cuda":
            torch.cuda.synchronize()

    def close(self) -> None:
        del self.a, self.b
        if self.dev.type == "cuda":
            torch.cuda.empty_cache()


class TorchEagerStepper(_TorchStepper):
    """Rung 1 on the GPU: NumPy-style slicing; each operator is its own kernel/memory pass."""

    def __init__(self, u0: np.ndarray, stencil: Stencil, source: np.ndarray | None = None):
        super().__init__(u0, stencil, source)
        if not self.periodic:
            core = tuple(s - 2 for s in self.shape)
            self.t1 = torch.empty(core, device=self.dev, dtype=self.a.dtype)
            self.t2 = torch.empty_like(self.t1)

    def _step_once(self) -> None:
        u, out = self.a, self.b
        if self.periodic:
            functional_step(u, out, self.c0, self.c1, True, self.src)
        else:
            c = (slice(1, -1),) * u.dim()
            p = _pairs(u, False)
            torch.add(p[0][0], p[0][1], out=self.t1)
            torch.add(p[1][0], p[1][1], out=self.t2)
            self.t1.add_(self.t2)
            if u.dim() == 3:
                torch.add(p[2][0], p[2][1], out=self.t2)
                self.t1.add_(self.t2)
            o = out[c]
            torch.mul(u[c], self.c0, out=self.t2)
            torch.add(self.t2, self.t1, alpha=self.c1, out=self.t2)
            if self.src is not None:
                self.t2.add_(self.src[c])
            o.copy_(self.t2)
        self.a, self.b = out, u


_COMPILED: dict[tuple, object] = {}

# dynamo specialises on shapes and on the float coefficients; every (shape, c0, c1) combination
# is one graph. Allow enough of them for test sweeps (benchmarks use one per problem size).
for _name in ("recompile_limit", "cache_size_limit"):
    if hasattr(torch._dynamo.config, _name):
        setattr(torch._dynamo.config, _name, 256)


class TorchCompileStepper(_TorchStepper):
    """Rung 2: the same expression traced by ``torch.compile``/Inductor into one fused kernel."""

    def __init__(self, u0: np.ndarray, stencil: Stencil, source: np.ndarray | None = None):
        super().__init__(u0, stencil, source)
        key = (self.periodic, self.has_source)
        if key not in _COMPILED:
            _COMPILED[key] = torch.compile(functional_step, fullgraph=True, dynamic=False)
        self.fn = _COMPILED[key]

    def _step_once(self) -> None:
        self.fn(self.a, self.b, self.c0, self.c1, self.periodic, self.src)
        self.a, self.b = self.b, self.a
