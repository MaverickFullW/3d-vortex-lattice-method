import numpy as np
import pytest

from src.solver import induced_drag_from_downwash, induced_drag_coefficient


@pytest.mark.parametrize("zero", ["gamma", "downwash"])
def test_zero_drag(zero):
    gamma, w = np.ones((2, 3)), -np.ones((2, 3))
    if zero == "gamma":
        gamma[:] = 0
    else:
        w[:] = 0
    panels, total = induced_drag_from_downwash(gamma, w, np.ones((2, 3)), 1)
    np.testing.assert_array_equal(panels, np.zeros((2, 3)))
    assert total == 0


def test_hand_calculated_drag_and_sum():
    panels, total = induced_drag_from_downwash(
        [[1, 2], [3, 4]], [[-0.5, -0.25], [-0.125, -0.5]], [[1, 2], [2, 0.5]], 2
    )
    np.testing.assert_array_equal(panels, [[1, 2], [1.5, 2]])
    assert np.all(panels > 0)
    assert total == np.sum(panels.ravel(order="C")) == 6.5


def test_symmetric_drag():
    panels, _ = induced_drag_from_downwash([[1, 2, 2, 1]], [[-2, -1, -1, -2]],
                                          [[0.25, 0.25, 0.25, 0.25]], 1)
    np.testing.assert_array_equal(panels, panels[:, ::-1])


def test_coefficient_hand_calculation():
    assert induced_drag_coefficient(6, rho=2, V_inf=3, S_ref=4) == pytest.approx(1 / 6)


@pytest.mark.parametrize("name", ["rho", "V_inf", "S_ref"])
@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf, [1]])
def test_invalid_positive_scalars(name, value):
    args = dict(rho=1, V_inf=1, S_ref=1)
    args[name] = value
    with pytest.raises(ValueError, match=name):
        induced_drag_coefficient(1, **args)
    if name == "rho":
        with pytest.raises(ValueError, match="rho"):
            induced_drag_from_downwash([[1]], [[-1]], [[1]], value)


@pytest.mark.parametrize("index", [0, 1, 2])
@pytest.mark.parametrize("bad", [np.array([1, 2]), np.ones((2, 1)), np.array([[np.nan, 1]]), np.array([[np.inf, 1]])])
def test_invalid_panel_arrays(index, bad):
    args = [np.ones((1, 2)), -np.ones((1, 2)), np.ones((1, 2))]
    args[index] = bad
    with pytest.raises(ValueError):
        induced_drag_from_downwash(*args, rho=1)


@pytest.mark.parametrize("width", [0, -1])
def test_nonpositive_width(width):
    with pytest.raises(ValueError, match="panel_widths"):
        induced_drag_from_downwash([[1]], [[-1]], [[width]], 1)


@pytest.mark.parametrize("drag", [np.nan, np.inf, [1]])
def test_invalid_total_drag(drag):
    with pytest.raises(ValueError, match="Di"):
        induced_drag_coefficient(drag, 1, 1, 1)
