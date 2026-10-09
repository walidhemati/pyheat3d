"""Problem definition: the generic constant-coefficient stencil update and its two modes.

Every backend implements exactly one update, applied to a 2D array ``u[y, x]`` or a 3D array
``u[z, y, x]`` (C order, x contiguous)::

    u'[c] = c0 * u[c] + c1 * S[c] (+ b[c])
    S     = ((u[x-1] + u[x+1]) + (u[y-1] + u[y+1])) + (u[z-1] + u[z+1])

The summation order is fixed so that every backend computes the same rounding sequence, up to
fused multiply-add contraction.

* **Heat mode** (explicit FTCS for ``u_t = alpha * lap(u)``): with the mesh Fourier number
  ``r = alpha * dt / h**2`` the coefficients are ``c0 = 1 - 2 d r`` and ``c1 = r``.
  The scheme is stable for ``r <= 1 / (2 d)``.
* **Jacobi mode** (Laplace / Poisson ``lap(u) = f``): ``c0 = 0``, ``c1 = 1 / (2 d)`` and the
  optional source ``b = -h**2 f / (2 d)``.

Boundary conditions:

* ``"dirichlet"``: the outermost layer of the array holds fixed boundary values. Only interior
  cells are updated; both ping-pong buffers carry the same boundary values.
* ``"periodic"``: every cell is updated, neighbours wrap around.
"""

from __future__ import annotations

from dataclasses import dataclass

BCS = ("dirichlet", "periodic")
MODES = ("heat", "jacobi")


@dataclass(frozen=True)
class Stencil:
    """Coefficients and boundary condition of the update ``u' = c0 u + c1 S``."""

    c0: float
    c1: float
    bc: str = "dirichlet"

    def __post_init__(self) -> None:
        if self.bc not in BCS:
            raise ValueError(f"bc must be one of {BCS}, got {self.bc!r}")


def stability_limit(dims: int) -> float:
    """Largest stable mesh Fourier number ``r = alpha dt / h^2`` of FTCS in ``dims`` dimensions."""
    return 1.0 / (2.0 * dims)


def heat_stencil(dims: int, r: float, bc: str = "dirichlet", check: bool = True) -> Stencil:
    """FTCS heat-equation stencil for mesh Fourier number ``r``.

    Raises ``ValueError`` if ``r`` violates the von Neumann limit ``r <= 1/(2 d)``.
    """
    if dims not in (2, 3):
        raise ValueError("dims must be 2 or 3")
    if check and not (0.0 < r <= stability_limit(dims) + 1e-15):
        raise ValueError(
            f"r = {r:g} violates the FTCS stability limit r <= 1/(2d) = {stability_limit(dims):g}"
        )
    return Stencil(c0=1.0 - 2.0 * dims * r, c1=r, bc=bc)


def jacobi_stencil(dims: int, bc: str = "dirichlet") -> Stencil:
    """Jacobi iteration stencil for the Laplace equation (source added separately)."""
    if dims not in (2, 3):
        raise ValueError("dims must be 2 or 3")
    return Stencil(c0=0.0, c1=1.0 / (2.0 * dims), bc=bc)


def dt_from_r(r: float, h: float, alpha: float = 1.0) -> float:
    """Time step corresponding to mesh Fourier number ``r``."""
    return r * h * h / alpha
