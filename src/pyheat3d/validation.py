"""Validation studies: analytic decay of a Fourier mode, Laplace steady states, Poisson MMS,
and cross-backend agreement with the NumPy reference.

All errors are relative max-norm errors ``max|u - u_ref| / max|u_ref|`` unless stated.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from . import analytic as an
from .backends.numpy_ref import reference_step
from .problem import heat_stencil, jacobi_stencil
from .registry import VARIANTS, availability, cuda_available, make_stepper


def _rel(u: np.ndarray, ref: np.ndarray) -> float:
    return float(np.max(np.abs(u - ref)) / max(np.max(np.abs(ref)), 1e-300))


def fp64_backend() -> str:
    """Fastest available float64-capable variant (torch-eager on a GPU, else NumPy)."""
    if cuda_available() and availability("torch-eager/slicing")[0]:
        return "torch-eager/slicing"
    return "numpy/slicing"


def slope(x: list[float], y: list[float]) -> float:
    """Least-squares slope of log(y) against log(x)."""
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def fourier_run(key: str, n: int, dims: int, bc: str, r: float, t_final: float,
                dtype: type = np.float32, alpha: float = 1.0) -> dict[str, Any]:
    """Advance the m=1 sine mode to ``t_final`` and compare with three references.

    The time step is adjusted so that an integer number of steps lands exactly on
    ``t_final`` (the effective ``r`` is reported).
    """
    h = an.grid_spacing(n, bc)
    dt_target = r * h * h / alpha
    steps = max(1, int(math.ceil(t_final / dt_target)))
    dt = t_final / steps
    r_eff = alpha * dt / (h * h)
    u0 = an.fourier_mode(n, dims, bc).astype(dtype)
    st = make_stepper(key, u0, heat_stencil(dims, r_eff, bc))
    st.step(steps)
    st.synchronize()
    u = st.get().astype(np.float64)
    st.close()
    shape0 = an.fourier_mode(n, dims, bc)
    amp_c = an.decay_continuous(t_final, dims, bc, alpha=alpha)
    amp_s = an.decay_semidiscrete(t_final, n, dims, bc, alpha=alpha)
    amp_d = an.ftcs_amplification(r_eff, n, dims, bc) ** steps
    return {
        "variant": key, "n": n, "dims": dims, "bc": bc, "h": h, "dt": dt, "r": r_eff,
        "steps": steps, "t_final": t_final,
        "err_exact": _rel(u, amp_c * shape0),
        "err_semidiscrete": _rel(u, amp_s * shape0),
        "err_discrete": _rel(u, amp_d * shape0),
    }


def spatial_convergence(dims: int, ns: list[int], r: float = 0.1, t_final: float = 0.02,
                        key: str | None = None, bc: str = "dirichlet") -> dict[str, Any]:
    """Error vs the PDE solution at fixed ``r`` (so ``dt ~ h^2``): expect slope 2 in h."""
    key = key or fp64_backend()
    runs = [fourier_run(key, n, dims, bc, r, t_final, np.float64) for n in ns]
    return {"dims": dims, "bc": bc, "variant": key, "r": r, "t_final": t_final, "runs": runs,
            "order": slope([x["h"] for x in runs], [x["err_exact"] for x in runs])}


def temporal_convergence(dims: int, n: int, steps_list: list[int], t_final: float = 0.02,
                         key: str | None = None, bc: str = "dirichlet") -> dict[str, Any]:
    """Error vs the semi-discrete (exact-in-time) solution on a fixed grid: expect slope 1."""
    key = key or fp64_backend()
    h = an.grid_spacing(n, bc)
    runs = []
    for s in steps_list:
        r = t_final / s / (h * h)
        runs.append(fourier_run(key, n, dims, bc, r, t_final, np.float64))
    return {"dims": dims, "bc": bc, "variant": key, "n": n, "t_final": t_final, "runs": runs,
            "order": slope([x["dt"] for x in runs], [x["err_semidiscrete"] for x in runs])}


def jacobi_iterations(n: int, tol: float) -> int:
    """Sweeps for Jacobi (spectral radius cos(pi h)) to damp an O(1) error to ``tol``."""
    h = 1.0 / (n - 1)
    rho = math.cos(math.pi * h)
    return int(math.ceil(math.log(tol) / math.log(rho)))


def solve_laplace(key: str, exact: np.ndarray, tol: float, dtype: type = np.float64,
                  source: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    """Jacobi from a zero interior with the boundary values of ``exact``."""
    dims = exact.ndim
    n = exact.shape[0]
    u0 = exact.astype(dtype).copy()
    u0[(slice(1, -1),) * dims] = 0.0
    iters = jacobi_iterations(n, tol)
    iters += iters % 2
    st = make_stepper(key, u0, jacobi_stencil(dims), source)
    st.step(iters)
    st.synchronize()
    u = st.get().astype(np.float64)
    st.close()
    return u, iters


def laplace_convergence(dims: int, ns: list[int], key: str | None = None,
                        tol: float = 1e-10) -> dict[str, Any]:
    """Converged Jacobi solution vs the smooth harmonic function: expect slope 2 in h."""
    key = key or fp64_backend()
    runs = []
    for n in ns:
        ex = an.harmonic_sinh(n, dims)
        u, it = solve_laplace(key, ex, tol)
        runs.append({"n": n, "h": 1.0 / (n - 1), "iterations": it, "err": _rel(u, ex)})
    return {"dims": dims, "variant": key, "tol": tol, "runs": runs,
            "order": slope([x["h"] for x in runs], [x["err"] for x in runs])}


def laplace_quadratic(dims: int, n: int, key: str | None = None,
                      tol: float = 1e-12) -> dict[str, Any]:
    """The 5/7-point stencil is exact on quadratics: converged error is round-off only."""
    key = key or fp64_backend()
    ex = an.harmonic_quadratic(n, dims)
    u, it = solve_laplace(key, ex, tol)
    return {"dims": dims, "n": n, "variant": key, "iterations": it, "err": _rel(u, ex)}


def poisson_convergence(dims: int, ns: list[int], key: str | None = None,
                        tol: float = 1e-10) -> dict[str, Any]:
    """Jacobi for ``lap(u) = f`` with a manufactured solution: expect slope 2 in h."""
    key = key or fp64_backend()
    runs = []
    for n in ns:
        ex, f = an.poisson_manufactured(n, dims)
        h = 1.0 / (n - 1)
        b = -h * h * f / (2 * dims)
        u, it = solve_laplace(key, ex, tol, source=b)
        runs.append({"n": n, "h": h, "iterations": it, "err": _rel(u, ex)})
    return {"dims": dims, "variant": key, "runs": runs,
            "order": slope([x["h"] for x in runs], [x["err"] for x in runs])}


#: max relative deviation from the reference. fp32 variants are compared with the plain fp32
#: reference (differences come only from FMA contraction); fp16/bf16 variants with a reference
#: that applies the same storage rounding after every DRAM pass (a few storage ulps allowed).
TOLERANCE = {"float32": 2e-6, "float16": 4e-3, "bfloat16": 3e-2, "native": 0.0,
             "float64": 1e-12}


def round_storage(u: np.ndarray, storage: str) -> np.ndarray:
    """Round a float32 array to the storage precision and back to float32."""
    if storage == "float16":
        return u.astype(np.float16).astype(np.float32)
    if storage == "bfloat16":
        b = np.ascontiguousarray(u, dtype=np.float32).view(np.uint32).astype(np.uint64)
        b = ((b + 0x7FFF + ((b >> 16) & 1)) >> 16).astype(np.uint32) << 16
        return b.view(np.float32)
    return u


def emulated_reference(u0: np.ndarray, c0: float, c1: float, bc: str, steps: int,
                       storage: str = "float32", k: int = 1) -> np.ndarray:
    """NumPy fp32 reference that mimics the storage rounding of a variant.

    Values are rounded to ``storage`` after every pass of ``k`` fused steps (temporal blocking
    keeps intermediate levels in fp32 registers); leftover steps are rounded one by one.
    """
    u = round_storage(np.asarray(u0, dtype=np.float32), storage)
    passes, rem = divmod(steps, k)
    for _ in range(passes):
        for _ in range(k):
            u = reference_step(u, c0, c1, bc)
        u = round_storage(u, storage)
    for _ in range(rem):
        u = round_storage(reference_step(u, c0, c1, bc), storage)
    return u


def cross_backend(keys: list[str] | None = None, steps: int = 24,
                  shapes: dict[int, tuple[int, ...]] | None = None,
                  seed: int = 0) -> list[dict[str, Any]]:
    """Every available variant vs the NumPy fp32 reference on a random field (both BCs).

    Low-precision variants are compared with :func:`emulated_reference`. The relative max
    error must stay below the storage-type tolerance in :data:`TOLERANCE`.
    """
    shapes = shapes or {2: (131, 517), 3: (45, 70, 97)}
    rng = np.random.default_rng(seed)
    out = []
    for dims, shape in shapes.items():
        u0 = rng.random(shape, dtype=np.float32)
        for bc in ("dirichlet", "periodic"):
            st = heat_stencil(dims, 0.9 / (2 * dims), bc)
            refs: dict[tuple[str, int], np.ndarray] = {}
            for key, var in VARIANTS.items():
                if dims not in var.dims or (keys is not None and key not in keys):
                    continue
                if not availability(key)[0]:
                    continue
                rk = ("float32", 1) if var.storage in ("float32", "native") else (
                    var.storage, var.steps_per_pass)
                if rk not in refs:
                    refs[rk] = emulated_reference(u0, st.c0, st.c1, bc, steps, *rk)
                ref = refs[rk]
                s = make_stepper(key, u0, st)
                s.step(steps)
                s.synchronize()
                u = s.get()
                s.close()
                err = _rel(u.astype(np.float64), ref.astype(np.float64))
                out.append({"variant": key, "dims": dims, "bc": bc, "shape": list(shape),
                            "steps": steps, "storage": var.storage, "rel_err": err,
                            "bitwise": bool(np.array_equal(u, ref)),
                            "tolerance": TOLERANCE[var.storage],
                            "pass": err <= TOLERANCE[var.storage]})
    return out


def fourier_all_backends(n2: int = 257, n3: int = 65, r: float = 0.1,
                         t_final: float = 0.01) -> list[dict[str, Any]]:
    """Every available variant on the decaying mode (fp32 or its storage type)."""
    out = []
    for dims, n in ((2, n2), (3, n3)):
        for key, var in VARIANTS.items():
            if dims not in var.dims or not availability(key)[0]:
                continue
            out.append(fourier_run(key, n, dims, "dirichlet", r, t_final, np.float32))
    return out


def run_all(quick: bool = False) -> dict[str, Any]:
    """Every study; ``quick`` uses grids small enough for a CPU-only CI run."""
    res: dict[str, Any] = {"quick": quick, "fp64_variant": fp64_backend()}
    if quick:
        res["spatial"] = [spatial_convergence(2, [9, 17, 33]), spatial_convergence(3, [9, 17])]
        res["temporal"] = [temporal_convergence(2, 17, [64, 128, 256])]
        res["laplace"] = [laplace_convergence(2, [9, 17, 33], tol=1e-9)]
        res["laplace_quadratic"] = [laplace_quadratic(2, 17)]
        res["poisson"] = [poisson_convergence(2, [9, 17, 33], tol=1e-9)]
        res["cross_backend"] = cross_backend(shapes={2: (33, 47), 3: (13, 17, 19)}, steps=8)
        res["fourier_backends"] = fourier_all_backends(n2=33, n3=17)
    else:
        res["spatial"] = [
            spatial_convergence(2, [17, 33, 65, 129, 257]),
            spatial_convergence(3, [9, 17, 33, 65, 129]),
            # periodic modes have twice the wavenumber: start at finer grids (asymptotic range)
            spatial_convergence(2, [32, 64, 128, 256, 512], bc="periodic", r=0.05,
                                t_final=0.005),
            spatial_convergence(3, [16, 32, 64, 128], bc="periodic", r=0.05, t_final=0.005),
        ]
        res["temporal"] = [
            temporal_convergence(2, 65, [1024, 2048, 4096, 8192, 16384]),
            temporal_convergence(3, 33, [256, 512, 1024, 2048, 4096]),
        ]
        res["laplace"] = [laplace_convergence(2, [9, 17, 33, 65, 129]),
                          laplace_convergence(3, [9, 17, 33, 65])]
        res["laplace_quadratic"] = [laplace_quadratic(2, 65), laplace_quadratic(3, 33)]
        res["poisson"] = [poisson_convergence(2, [9, 17, 33, 65]),
                          poisson_convergence(3, [9, 17, 33])]
        res["cross_backend"] = cross_backend()
        res["fourier_backends"] = fourier_all_backends()
    return res
