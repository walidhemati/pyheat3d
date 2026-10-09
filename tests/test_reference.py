"""NumPy reference (runs on CPU in CI)."""

import numpy as np
import pytest

from pyheat3d import analytic as an
from pyheat3d import validation as val
from pyheat3d.backends.numpy_ref import reference_step
from pyheat3d.problem import heat_stencil
from pyheat3d.registry import make_stepper


@pytest.mark.parametrize("dims", [2, 3])
@pytest.mark.parametrize("bc", ["dirichlet", "periodic"])
def test_stepper_matches_reference_step_bitwise(dims, bc):
    rng = np.random.default_rng(1)
    u0 = rng.random((11, 13, 9)[:dims]).astype(np.float32)
    st = heat_stencil(dims, 0.1, bc)
    ref = u0
    for _ in range(5):
        ref = reference_step(ref, st.c0, st.c1, bc)
    s = make_stepper("numpy/slicing", u0, st)
    s.step(5)
    assert np.array_equal(s.get(), ref)


@pytest.mark.parametrize("dims", [2, 3])
@pytest.mark.parametrize("bc", ["dirichlet", "periodic"])
def test_fully_discrete_mode_is_exact(dims, bc):
    """FTCS maps the sine mode onto itself times g per step: match to fp64 round-off."""
    n = 16 if bc == "periodic" else 17
    run = val.fourier_run("numpy/slicing", n, dims, bc, 0.1, 0.01, np.float64)
    assert run["err_discrete"] < 1e-12


def test_dirichlet_boundary_untouched():
    rng = np.random.default_rng(2)
    u0 = rng.random((10, 12, 14))
    s = make_stepper("numpy/slicing", u0, heat_stencil(3, 0.15))
    s.step(7)
    u = s.get()
    for ax in range(3):
        for idx in (0, -1):
            np.testing.assert_array_equal(np.take(u, idx, ax), np.take(u0, idx, ax))


def test_periodic_heat_conserves_mean():
    rng = np.random.default_rng(3)
    u0 = rng.random((32, 24))
    s = make_stepper("numpy/slicing", u0, heat_stencil(2, 0.2, "periodic"))
    s.step(50)
    assert s.get().mean() == pytest.approx(u0.mean(), rel=1e-13)


def test_maximum_principle():
    rng = np.random.default_rng(4)
    u0 = rng.random((20, 20, 20))
    s = make_stepper("numpy/slicing", u0, heat_stencil(3, 1 / 6))
    s.step(20)
    u = s.get()
    assert u.max() <= u0.max() + 1e-15 and u.min() >= u0.min() - 1e-15


def test_spatial_order_two():
    res = val.spatial_convergence(2, [9, 17, 33], key="numpy/slicing")
    assert 1.85 < res["order"] < 2.15


def test_temporal_order_one():
    res = val.temporal_convergence(2, 17, [64, 128, 256], key="numpy/slicing")
    assert 0.95 < res["order"] < 1.05


def test_laplace_order_two_and_quadratic_exact():
    res = val.laplace_convergence(2, [9, 17, 33], key="numpy/slicing", tol=1e-9)
    assert 1.85 < res["order"] < 2.15
    q = val.laplace_quadratic(2, 17, key="numpy/slicing")
    assert q["err"] < 1e-10


def test_poisson_order_two():
    res = val.poisson_convergence(2, [9, 17, 33], key="numpy/slicing", tol=1e-9)
    assert 1.85 < res["order"] < 2.15


def test_harmonic_sinh_boundary():
    u = an.harmonic_sinh(9, 2)
    assert np.abs(u[0]).max() < 1e-15  # y = 0
    assert u[-1].max() == pytest.approx(1.0, abs=1e-12)  # sin(pi x) at y = 1
