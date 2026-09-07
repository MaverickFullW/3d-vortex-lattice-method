import numpy as np

from scripts.run_mcbain_campbell_benchmark import run_benchmark


def test_mcbain_campbell_lift_curve_slope():
    result = run_benchmark()

    np.testing.assert_allclose(result["root_chord"], 5.0 / 24.0, atol=1e-14)
    np.testing.assert_allclose(result["tip_chord"], 0.125, atol=1e-14)
    np.testing.assert_allclose(result["reference_area"], 1.0 / 6.0, atol=1e-14)
    np.testing.assert_allclose(
        result["leading_edge_sweep_deg"], 46.169139327907, rtol=0.0, atol=1e-12
    )
    assert result["reference_CLa"] == 3.5633
    # Four-decimal rounding allows 5e-5; leave 1e-5 for the 100-span wake.
    np.testing.assert_allclose(result["CLa"], 3.5633, rtol=0.0, atol=6e-5)
    np.testing.assert_allclose(
        result["CL"], 3.5633 * np.deg2rad(1.0),
        rtol=0.0, atol=6e-5 * np.deg2rad(1.0),
    )
    assert result["Gamma"].shape == (1, 20)
    np.testing.assert_allclose(
        result["Gamma"], result["Gamma"][:, ::-1], rtol=0.0, atol=1e-14
    )
    assert result["residual_norm"] < 1e-14
