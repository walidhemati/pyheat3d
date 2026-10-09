"""Analytic and discrete-exact reference solutions used for validation.

Grids are the unit square/cube. Dirichlet grids have ``n`` points per axis *including* the two
boundary points (``h = 1/(n-1)``); periodic grids have ``n`` points per period (``h = 1/n``).
"""

from __future__ import annotations

import math

import numpy as np


def grid_spacing(n: int, bc: str) -> float:
    """Mesh width of an ``n``-point axis on the unit interval."""
    return 1.0 / (n - 1) if bc == "dirichlet" else 1.0 / n


def axis(n: int, bc: str) -> np.ndarray:
    """Coordinates of an ``n``-point axis on the unit interval (float64)."""
    return np.arange(n, dtype=np.float64) * grid_spacing(n, bc)


def mesh(n: int, dims: int, bc: str) -> tuple[np.ndarray, ...]:
    """Coordinate arrays in array order: ``(y, x)`` for 2D, ``(z, y, x)`` for 3D."""
    a = axis(n, bc)
    return tuple(np.meshgrid(*([a] * dims), indexing="ij"))


def wavenumber(m: int, bc: str) -> float:
    """Wavenumber of the m-th sine mode: ``m pi`` (Dirichlet) or ``2 pi m`` (periodic)."""
    return m * math.pi if bc == "dirichlet" else 2.0 * math.pi * m


def fourier_mode(n: int, dims: int, bc: str, m: int = 1) -> np.ndarray:
    """Initial condition ``prod_i sin(k x_i)``; zero on Dirichlet boundaries."""
    k = wavenumber(m, bc)
    s = np.sin(k * axis(n, bc))
    if bc == "dirichlet":
        s[0] = 0.0
        s[-1] = 0.0
    out = s
    for _ in range(dims - 1):
        out = np.multiply.outer(out, s)
    return out


def decay_continuous(t: float, dims: int, bc: str, m: int = 1, alpha: float = 1.0) -> float:
    """Amplitude of the mode at time ``t`` for the PDE ``u_t = alpha lap(u)``."""
    k = wavenumber(m, bc)
    return math.exp(-alpha * dims * k * k * t)


def decay_semidiscrete(
    t: float, n: int, dims: int, bc: str, m: int = 1, alpha: float = 1.0
) -> float:
    """Amplitude for the semi-discrete (exact-in-time) 2nd-order central scheme."""
    k = wavenumber(m, bc)
    h = grid_spacing(n, bc)
    lam = dims * 4.0 * math.sin(0.5 * k * h) ** 2 / (h * h)
    return math.exp(-alpha * lam * t)


def ftcs_amplification(r: float, n: int, dims: int, bc: str, m: int = 1) -> float:
    """Per-step amplification factor of FTCS for the mode (the fully discrete exact solution)."""
    k = wavenumber(m, bc)
    h = grid_spacing(n, bc)
    return 1.0 - 4.0 * dims * r * math.sin(0.5 * k * h) ** 2


def harmonic_sinh(n: int, dims: int) -> np.ndarray:
    """Smooth harmonic function on the unit square/cube used for the Laplace test.

    2D: ``sin(pi x) sinh(pi y) / sinh(pi)``;
    3D: ``sin(pi x) sin(pi y) sinh(sqrt(2) pi z) / sinh(sqrt(2) pi)``.
    Its 5/7-point discrete solution differs from it by O(h^2).
    """
    pts = mesh(n, dims, "dirichlet")
    if dims == 2:
        y, x = pts
        return np.sin(math.pi * x) * np.sinh(math.pi * y) / math.sinh(math.pi)
    z, y, x = pts
    q = math.sqrt(2.0) * math.pi
    return np.sin(math.pi * x) * np.sin(math.pi * y) * np.sinh(q * z) / math.sinh(q)


def harmonic_quadratic(n: int, dims: int) -> np.ndarray:
    """Quadratic harmonic function (``x^2 - y^2`` or ``x^2 + y^2 - 2 z^2``).

    The 5/7-point Laplacian is exact on quadratics, so the converged Jacobi solution equals it
    up to round-off: a discretisation-error-free steady-state test.
    """
    pts = mesh(n, dims, "dirichlet")
    if dims == 2:
        y, x = pts
        return x * x - y * y
    z, y, x = pts
    return x * x + y * y - 2.0 * z * z


def poisson_manufactured(n: int, dims: int) -> tuple[np.ndarray, np.ndarray]:
    """Manufactured Poisson problem ``lap(u) = f`` with ``u = prod sin(pi x_i)``.

    Returns ``(u_exact, f)``; ``u`` vanishes on the boundary.
    """
    u = fourier_mode(n, dims, "dirichlet", 1)
    f = -dims * math.pi**2 * u
    return u, f


def interior(a: np.ndarray) -> np.ndarray:
    """Interior view (drop one boundary layer on every axis)."""
    return a[(slice(1, -1),) * a.ndim]
