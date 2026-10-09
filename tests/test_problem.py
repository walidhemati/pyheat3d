import math

import numpy as np
import pytest

from pyheat3d import analytic as an
from pyheat3d import traffic
from pyheat3d.problem import heat_stencil, jacobi_stencil, stability_limit


def test_stability_limit():
    assert stability_limit(2) == 0.25
    assert stability_limit(3) == pytest.approx(1 / 6)


def test_heat_coefficients_and_limit():
    st = heat_stencil(3, 0.1)
    assert st.c0 == pytest.approx(0.4) and st.c1 == 0.1
    heat_stencil(3, 1 / 6)  # exactly at the limit is allowed
    with pytest.raises(ValueError):
        heat_stencil(3, 0.17)
    with pytest.raises(ValueError):
        heat_stencil(2, 0.26, "periodic")
    with pytest.raises(ValueError):
        heat_stencil(3, 0.1, "neumann")


def test_jacobi_coefficients():
    assert jacobi_stencil(2).c1 == 0.25
    assert jacobi_stencil(3).c0 == 0.0


def test_fourier_mode_is_zero_on_dirichlet_boundary():
    u = an.fourier_mode(17, 3, "dirichlet")
    assert u.shape == (17, 17, 17)
    assert np.abs(u[0]).max() == 0 and np.abs(u[:, :, -1]).max() == 0
    assert u.max() == pytest.approx(1.0)


def test_decay_rates_agree_for_small_h():
    t = 0.01
    c = an.decay_continuous(t, 3, "dirichlet")
    s = an.decay_semidiscrete(t, 1025, 3, "dirichlet")
    assert s == pytest.approx(c, rel=1e-5)
    g = an.ftcs_amplification(0.1, 1025, 3, "dirichlet")
    h = 1 / 1024
    steps = round(t / (0.1 * h * h))
    assert g**steps == pytest.approx(c, rel=1e-4)


def test_harmonic_functions_are_discretely_harmonic():
    q = an.harmonic_quadratic(9, 3)
    lap = (q[1:-1, 1:-1, :-2] + q[1:-1, 1:-1, 2:] + q[1:-1, :-2, 1:-1] + q[1:-1, 2:, 1:-1]
           + q[:-2, 1:-1, 1:-1] + q[2:, 1:-1, 1:-1] - 6 * q[1:-1, 1:-1, 1:-1])
    assert np.abs(lap).max() < 1e-14


def test_traffic_model():
    assert traffic.minimal_bytes(4) == 8
    assert traffic.minimal_bytes(2) == 4
    assert traffic.minimal_bytes(4, 4) == 2
    assert traffic.flops_per_update(3) == 8
    assert traffic.roofline_mlups(361.0, 8) == pytest.approx(45125.0)
    # temporal blocking: halo over-fetch makes the model exceed the minimal 8/K
    tb = traffic.tb_model_bytes(3, 4, 4, 32, 64, 128)
    assert traffic.minimal_bytes(4, 4) < tb < traffic.minimal_bytes(4, 1)
    assert math.isclose(traffic.tb_model_bytes(2, 1, 4, 10**6, 1, 10**6), 8, rel_tol=1e-5)
