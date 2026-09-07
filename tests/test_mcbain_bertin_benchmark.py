import numpy as np

from scripts.run_mcbain_bertin_benchmark import run_benchmark


def test_mcbain_bertin_lift_curve_slope():
    result = run_benchmark()

    # Allow 0.1% for the finite wake and the four-decimal reference value.
    np.testing.assert_allclose(result["CLa"], 3.4442, rtol=1e-3, atol=0.0)
    assert result["Gamma"].shape == (1, 8)
    np.testing.assert_allclose(
        result["Gamma"], result["Gamma"][:, ::-1], rtol=0.0, atol=1e-14
    )
    assert result["residual_norm"] < 1e-14
