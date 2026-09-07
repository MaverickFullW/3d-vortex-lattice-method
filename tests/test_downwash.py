import numpy as np
import pytest

from src.geometry import swept_tapered_horseshoe_geometry, swept_tapered_twisted_horseshoe_geometry
from src.solver import horseshoe_induced_velocity, build_horseshoe_influence_matrix, build_rhs, solve_circulation
from src.vortex import horseshoe_velocity


def test_zero_circulation_and_shapes():
    geometry = swept_tapered_horseshoe_geometry(4, 1, 0.6, 20, 2, 4)
    velocity, w = horseshoe_induced_velocity(*geometry, np.zeros((2, 4)), 100)
    assert velocity.shape == (2, 4, 3)
    assert w.shape == (2, 4)
    np.testing.assert_array_equal(velocity, np.zeros((2, 4, 3)))
    np.testing.assert_array_equal(w, np.zeros((2, 4)))


def test_direct_sum_all_components_and_panel_order():
    A, B, P, _, _ = swept_tapered_twisted_horseshoe_geometry(4, 1, 0.6, 20, 10, 2, 2)
    gamma = np.array([[0.1, -0.2], [0.3, 0.4]])
    velocity, w = horseshoe_induced_velocity(A, B, P, gamma, 100)
    expected = np.zeros_like(P)
    for i, j in np.ndindex(2, 2):
        for p, q in np.ndindex(2, 2):
            expected[i, j] += horseshoe_velocity(
                A[p, q], B[p, q], P[i, j], gamma=gamma[p, q], wake_length=100
            )
    np.testing.assert_allclose(velocity, expected, rtol=1e-14, atol=1e-15)
    np.testing.assert_array_equal(w, velocity[..., 2])
    assert np.any(abs(expected[..., :2]) > 1e-8)


def test_solved_flat_wing_symmetry_and_downward_sign():
    geometry = swept_tapered_horseshoe_geometry(5, 1, 1, 45, 2, 8)
    matrix = build_horseshoe_influence_matrix(*geometry, 100)
    rhs = build_rhs(1, 3, 16)
    gamma = solve_circulation(matrix, rhs).reshape(2, 8, order="C")
    velocity, w = horseshoe_induced_velocity(*geometry, gamma, 100)
    np.testing.assert_allclose(w, w[:, ::-1], rtol=0, atol=1e-14)
    np.testing.assert_allclose(w.ravel(order="C"), matrix @ gamma.ravel(order="C"), atol=1e-14)
    np.testing.assert_allclose(w, -np.sin(np.deg2rad(3)), rtol=0, atol=1e-14)
    assert np.all(w < 0)  # Global z is upward; induced downward flow is negative.
    np.testing.assert_array_equal(velocity[..., 2], w)


@pytest.mark.parametrize("kind,message", [
    ("mismatch", "matching shape"), ("dimensions", "Geometry arrays must have shape"),
    ("gamma_shape", "Gamma must have shape"), ("gamma_nan", "Gamma must be finite"),
    ("geometry_inf", "Geometry arrays must be finite"),
])
def test_invalid_inputs(kind, message):
    geometry = list(swept_tapered_horseshoe_geometry(4, 1, 1, 0, 2, 2))
    gamma = np.ones((2, 2))
    if kind == "mismatch":
        geometry[0] = geometry[0][:1]
    elif kind == "dimensions":
        geometry = [p.reshape(-1, 3) for p in geometry]
    elif kind == "gamma_shape":
        gamma = gamma.ravel()
    elif kind == "gamma_nan":
        gamma[0, 0] = np.nan
    else:
        geometry[2][0, 0, 0] = np.inf
    with pytest.raises(ValueError, match=message):
        horseshoe_induced_velocity(*geometry, gamma, 100)


@pytest.mark.parametrize("wake", [0, -1, np.nan, np.inf, [100]])
def test_invalid_wake(wake):
    geometry = swept_tapered_horseshoe_geometry(4, 1, 1, 0, 2, 2)
    with pytest.raises(ValueError, match="wake_length"):
        horseshoe_induced_velocity(*geometry, np.ones((2, 2)), wake)
