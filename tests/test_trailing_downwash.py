import numpy as np
import pytest

from src.geometry import rectangular_horseshoe_geometry
from src.solver import (trailing_vortex_downwash, horseshoe_induced_velocity,
                        build_horseshoe_influence_matrix, build_rhs, solve_circulation)
from src.vortex import vortex_segment_velocity


def test_zero_and_shape():
    geometry = rectangular_horseshoe_geometry(2, 1, 2, 3)
    w = trailing_vortex_downwash(*geometry, np.zeros((2, 3)), 100)
    assert w.shape == (2, 3)
    np.testing.assert_array_equal(w, np.zeros((2, 3)))


def test_explicit_legs_and_exclusion_of_bound_segment():
    A, B, P = rectangular_horseshoe_geometry(2, 1, 2, 2)
    gamma = np.array([[0.1, 0.2], [-0.3, 0.4]])
    expected = np.zeros((2, 2))
    bound = np.zeros((2, 2))
    for i, j in np.ndindex(2, 2):
        for p, q in np.ndindex(2, 2):
            far_a, far_b = A[p, q].copy(), B[p, q].copy()
            far_a[0] += 100
            far_b[0] += 100
            expected[i, j] += (
                vortex_segment_velocity(far_a, A[p, q], P[i, j], gamma[p, q])[2]
                + vortex_segment_velocity(B[p, q], far_b, P[i, j], gamma[p, q])[2]
            )
            bound[i, j] += vortex_segment_velocity(A[p, q], B[p, q], P[i, j], gamma[p, q])[2]
    actual = trailing_vortex_downwash(A, B, P, gamma, 100)
    _, full = horseshoe_induced_velocity(A, B, P, gamma, 100)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)
    np.testing.assert_allclose(full - actual, bound, rtol=0, atol=1e-14)
    assert np.max(abs(full - actual)) > 1e-3


def test_positive_lift_symmetry_and_negative_downwash():
    geometry = rectangular_horseshoe_geometry(2, 1, 1, 8)
    matrix = build_horseshoe_influence_matrix(*geometry, 100)
    gamma = solve_circulation(matrix, build_rhs(1, 5, 8)).reshape(1, 8)
    w = trailing_vortex_downwash(*geometry, gamma, 100)
    assert np.all(gamma > 0)
    np.testing.assert_allclose(w, w[:, ::-1], rtol=0, atol=1e-14)
    assert np.all(w < 0)


@pytest.mark.parametrize("kind", ["shape", "nan", "wake", "geometry"])
def test_invalid_inputs(kind):
    geometry = list(rectangular_horseshoe_geometry(2, 1, 1, 2))
    gamma = np.ones((1, 2))
    wake = 100
    if kind == "shape":
        gamma = gamma.ravel()
    elif kind == "nan":
        gamma[0, 0] = np.nan
    elif kind == "wake":
        wake = -1
    else:
        geometry[0] = geometry[0][:, :1]
    with pytest.raises(ValueError):
        trailing_vortex_downwash(*geometry, gamma, wake)
