"""Every GPU variant against the NumPy reference (auto-skipped without CUDA)."""

import numpy as np
import pytest

from pyheat3d import validation as val
from pyheat3d.backends.numpy_ref import reference_step
from pyheat3d.problem import heat_stencil, jacobi_stencil
from pyheat3d.registry import VARIANTS, availability, make_stepper

GPU_KEYS = [k for k, v in VARIANTS.items() if v.gpu]
SHAPES = {2: (131, 517), 3: (45, 70, 97)}


def _need(key, dims):
    if dims not in VARIANTS[key].dims:
        pytest.skip(f"{key} has no {dims}D kernel")
    ok, why = availability(key)
    if not ok:
        pytest.skip(why)


def _ref(u0, st, steps):
    ref = u0.copy()
    for _ in range(steps):
        ref = reference_step(ref, st.c0, st.c1, st.bc)
    return ref


@pytest.mark.gpu
@pytest.mark.parametrize("bc", ["dirichlet", "periodic"])
@pytest.mark.parametrize("dims", [2, 3])
@pytest.mark.parametrize("key", GPU_KEYS)
def test_matches_numpy(key, dims, bc):
    _need(key, dims)
    rng = np.random.default_rng(7)
    u0 = rng.random(SHAPES[dims], dtype=np.float32)
    st = heat_stencil(dims, 0.9 / (2 * dims), bc)
    steps = 2 * 3 * 4 * 8 + 5  # covers every temporal-blocking K plus a remainder
    var = VARIANTS[key]
    ref = val.emulated_reference(u0, st.c0, st.c1, bc, steps, var.storage,
                                 var.steps_per_pass if var.storage != "float32" else 1)
    s = make_stepper(key, u0, st)
    s.step(steps)
    s.synchronize()
    u = s.get()
    s.close()
    err = np.abs(u.astype(np.float64) - ref).max() / np.abs(ref).max()
    assert err <= val.TOLERANCE[VARIANTS[key].storage], err


@pytest.mark.gpu
@pytest.mark.parametrize("dims", [2, 3])
@pytest.mark.parametrize("key", GPU_KEYS)
def test_jacobi_mode(key, dims):
    _need(key, dims)
    if VARIANTS[key].storage != "float32":
        pytest.skip("low-precision storage")
    rng = np.random.default_rng(8)
    u0 = rng.random(SHAPES[dims], dtype=np.float32)
    st = jacobi_stencil(dims)
    ref = _ref(u0, st, 24)
    s = make_stepper(key, u0, st)
    s.step(24)
    s.synchronize()
    err = np.abs(s.get().astype(np.float64) - ref).max()
    assert err < 2e-6


@pytest.mark.gpu
@pytest.mark.parametrize("key", [k for k in GPU_KEYS if VARIANTS[k].storage == "float32"])
def test_fourier_mode_fp32(key):
    """fp32 variants reproduce the fully discrete solution to ~fp32 round-off."""
    dims = 3 if 3 in VARIANTS[key].dims else 2
    _need(key, dims)
    run = val.fourier_run(key, 33, dims, "dirichlet", 0.1, 0.005)
    assert run["err_discrete"] < 1e-5


@pytest.mark.gpu
def test_fp64_torch_convergence():
    res = val.spatial_convergence(3, [9, 17, 33], key="torch-eager/slicing")
    assert 1.9 < res["order"] < 2.1


@pytest.mark.gpu
def test_small_grids_smaller_than_one_tile():
    rng = np.random.default_rng(9)
    for dims, shape in ((2, (5, 7)), (3, (5, 6, 7))):
        u0 = rng.random(shape, dtype=np.float32)
        for bc in ("dirichlet", "periodic"):
            st = heat_stencil(dims, 0.1, bc)
            for key in GPU_KEYS:
                if dims not in VARIANTS[key].dims or not availability(key)[0]:
                    continue
                s = make_stepper(key, u0, st)
                s.step(9)
                s.synchronize()
                var = VARIANTS[key]
                ref = val.emulated_reference(u0, st.c0, st.c1, bc, 9, var.storage,
                                             var.steps_per_pass)
                err = np.abs(s.get() - ref).max()
                assert err <= max(val.TOLERANCE[var.storage], 2e-6), (key, bc, err)
