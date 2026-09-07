import numpy as np
import pytest

from src.geometry import rectangular_horseshoe_geometry
from src.solver import (
    build_horseshoe_influence_matrix,
    build_rhs,
    mcbain_total_lift,
    solve_circulation,
    spanwise_lift_distribution,
)


def test_single_chordwise_row():
    gamma = np.array([[0.5, 1.0, -0.25, 2.0]])
    circulation, lift = spanwise_lift_distribution(gamma, 1.25, 4.0)
    np.testing.assert_array_equal(circulation, gamma[0])
    np.testing.assert_array_equal(lift, 1.25 * 4.0 * gamma[0])


def test_multiple_rows_hand_calculation_preserves_spanwise_order():
    gamma = np.array([[1.0, 2.0, 3.0], [4.0, -2.0, 6.0]])
    original = gamma.copy()
    circulation, lift = spanwise_lift_distribution(gamma, 2.0, 3.0)
    np.testing.assert_array_equal(circulation, [5.0, 0.0, 9.0])
    np.testing.assert_array_equal(lift, [30.0, 0.0, 54.0])
    np.testing.assert_array_equal(gamma, original)


def test_symmetric_circulation():
    circulation, lift = spanwise_lift_distribution(
        [[1, 2, 3, 3, 2, 1], [2, 4, 5, 5, 4, 2]], 1.225, 10.0
    )
    np.testing.assert_array_equal(circulation, circulation[::-1])
    np.testing.assert_array_equal(lift, lift[::-1])


@pytest.mark.parametrize("shape", [(1, 8), (3, 8), (4, 1)])
def test_zero_and_output_shapes(shape):
    circulation, lift = spanwise_lift_distribution(np.zeros(shape), 1.225, 10.0)
    assert circulation.shape == lift.shape == (shape[1],)
    np.testing.assert_array_equal(circulation, np.zeros(shape[1]))
    np.testing.assert_array_equal(lift, np.zeros(shape[1]))


@pytest.mark.parametrize("gamma", [1.0, [1, 2], np.ones((2, 3, 1)),
                                       [[np.nan]], [[np.inf]], [[-np.inf]]])
def test_invalid_gamma(gamma):
    with pytest.raises(ValueError, match="Gamma must be a finite 2D array"):
        spanwise_lift_distribution(gamma, 1.0, 1.0)


@pytest.mark.parametrize("name", ["rho", "V_inf"])
@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf, -np.inf, [1], [1, 2]])
def test_invalid_positive_scalars(name, value):
    args = dict(Gamma=[[1.0]], rho=1.0, V_inf=1.0)
    args[name] = value
    with pytest.raises(ValueError, match=name):
        spanwise_lift_distribution(**args)


def test_rectangular_diagnostic_lift_matches_mcbain():
    A, B, controls = rectangular_horseshoe_geometry(2.0, 1.0, 1, 8)
    matrix = build_horseshoe_influence_matrix(A, B, controls, 100.0)
    gamma = solve_circulation(matrix, build_rhs(1.0, 5.0, 8)).reshape(1, 8, order="C")
    _, lift = spanwise_lift_distribution(gamma, 1.0, 1.0)
    integrated = np.sum(lift * np.abs(B[0, :, 1] - A[0, :, 1]))
    expected = mcbain_total_lift(gamma, A, B, 1.0, 1.0)
    np.testing.assert_allclose(integrated, expected, rtol=1e-14, atol=1e-15)
