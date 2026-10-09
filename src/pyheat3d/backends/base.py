"""Common interface implemented by every backend variant."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..problem import Stencil


class Stepper(ABC):
    """Owns a ping-pong pair of device buffers and advances the stencil update.

    Subclasses implement :meth:`_step_once` (or override :meth:`step`), :meth:`get` and,
    for asynchronous devices, :meth:`synchronize`.

    Parameters
    ----------
    u0:
        Initial state, shape ``(ny, nx)`` or ``(nz, ny, nx)``. For Dirichlet problems the
        outermost layer holds the (fixed) boundary values.
    stencil:
        Coefficients ``c0, c1`` and boundary condition.
    source:
        Optional additive term ``b`` (same shape as ``u0``), only for variants that declare
        ``source=True`` in the registry.
    """

    #: storage precision on the device ("float32", "float16", "bfloat16", "float64")
    storage: str = "float32"
    #: time steps fused per memory pass (temporal blocking factor)
    steps_per_pass: int = 1

    def __init__(self, u0: np.ndarray, stencil: Stencil, source: np.ndarray | None = None):
        if u0.ndim not in (2, 3):
            raise ValueError("u0 must be 2D or 3D")
        if min(u0.shape) < 3:
            raise ValueError("every axis needs at least 3 points")
        self.shape: tuple[int, ...] = tuple(int(s) for s in u0.shape)
        self.dims = u0.ndim
        self.stencil = stencil
        self.c0 = float(stencil.c0)
        self.c1 = float(stencil.c1)
        self.bc = stencil.bc
        self.periodic = stencil.bc == "periodic"
        self.has_source = source is not None
        self.steps_done = 0

    @property
    def ncells(self) -> int:
        """Number of lattice sites updated per step (interior only for Dirichlet)."""
        if self.periodic:
            return int(np.prod(self.shape))
        return int(np.prod([s - 2 for s in self.shape]))

    def step(self, n: int = 1) -> None:
        """Advance ``n`` time steps / Jacobi sweeps."""
        for _ in range(n):
            self._step_once()
        self.steps_done += n

    def _step_once(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    @abstractmethod
    def get(self) -> np.ndarray:
        """Return the current state as a host NumPy array (float32 or float64)."""

    def synchronize(self) -> None:  # noqa: B027 - no-op for synchronous (CPU) backends
        """Block until all queued device work has finished."""

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release device memory (optional)."""
