import numpy as np
import pytest

from src.solver import induced_angle_from_downwash


@pytest.mark.parametrize("shape", [(8,), (2, 4), (2, 3, 4)])
def test_zero_downwash_and_shape(shape):
    actual = induced_angle_from_downwash(np.zeros(shape), 2.0)
    assert actual.shape == shape
    np.testing.assert_array_equal(actual, np.zeros(shape))


def test_known_values_in_radians_and_sign():
    downwash = np.array([[-0.2, -0.1], [0, 0.4]])
    actual = induced_angle_from_downwash(downwash, 2.0)
    np.testing.assert_allclose(actual, [[0.1, 0.05], [0, -0.2]], rtol=1e-15, atol=0)
    assert np.all(actual[0] > 0)
    np.testing.assert_array_equal(downwash, [[-0.2, -0.1], [0, 0.4]])


@pytest.mark.parametrize("speed", [0, -1, np.nan, np.inf, -np.inf, [1, 2]])
def test_invalid_freestream(speed):
    with pytest.raises(ValueError, match="V_inf must be a finite, strictly positive scalar"):
        induced_angle_from_downwash(np.zeros((1, 2)), speed)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_nonfinite_downwash(value):
    with pytest.raises(ValueError, match="downwash must contain finite values"):
        induced_angle_from_downwash([[0, value]], 1)
