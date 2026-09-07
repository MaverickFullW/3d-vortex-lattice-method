import numpy as np
import pytest

from src.solver import span_efficiency


def test_hand_calculation_and_scalar():
    result = span_efficiency(0.5, 0.02, 8)
    assert isinstance(result, float)
    assert result == pytest.approx(25 / (16 * np.pi))


def test_increasing_drag_decreases_efficiency():
    assert span_efficiency(0.5, 0.04, 8) == pytest.approx(span_efficiency(0.5, 0.02, 8) / 2)


def test_no_clamping_and_lift_sign():
    assert span_efficiency(1, 1 / (12 * np.pi), 6) == pytest.approx(2)
    assert span_efficiency(-1, 1 / (12 * np.pi), 6) == pytest.approx(2)
    assert span_efficiency(0, 0.02, 8) == 0


@pytest.mark.parametrize("name", ["CDi", "aspect_ratio"])
@pytest.mark.parametrize("value", [0, -1, np.nan, np.inf, -np.inf, [1]])
def test_invalid_positive_inputs(name, value):
    args = dict(CL=0.5, CDi=0.02, aspect_ratio=8)
    args[name] = value
    with pytest.raises(ValueError, match=name):
        span_efficiency(**args)


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, [0.5]])
def test_invalid_lift(value):
    with pytest.raises(ValueError, match="CL must be a finite scalar"):
        span_efficiency(value, 0.02, 8)
