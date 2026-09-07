import numpy as np

from scripts.run_mcbain_bertin_3x8_diagnostic import run_benchmark


def test_mcbain_bertin_3x8_chordwise_refinement():
    result = run_benchmark()

    assert result["Gamma"].shape == (3, 8)
    np.testing.assert_allclose(
        result["Gamma"], result["Gamma"][:, ::-1], rtol=0.0, atol=1e-14
    )
    assert result["reference_CLa"] == 3.4369
    # Same 0.1% allowance as the 1x8 benchmark for the finite wake and
    # the four-decimal McBain reference.
    np.testing.assert_allclose(result["CLa"], 3.4369, rtol=1e-3, atol=0.0)
    np.testing.assert_allclose(
        result["CL"], 3.4369 * np.deg2rad(1.0), rtol=1e-3, atol=0.0
    )
    assert result["residual_norm"] < 1e-14

