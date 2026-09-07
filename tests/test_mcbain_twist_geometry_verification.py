import numpy as np
import pytest

from scripts.run_mcbain_twist_geometry_verification import run_verification


@pytest.fixture(scope="module")
def verification():
    return run_verification()


def test_mcbain_twist_sections(verification):
    np.testing.assert_allclose(verification["root_chord"], 2 / 9, rtol=0, atol=1e-14)
    np.testing.assert_allclose(verification["tip_chord"], 1 / 9, rtol=0, atol=1e-14)
    # Both tips, both mid-semispans, and the root recovered from actual corners.
    np.testing.assert_allclose(
        verification["station_twist_deg"][[0, 5, 10, 15, 20]],
        [7.5, 3.75, 0, 3.75, 7.5], rtol=0, atol=1e-13,
    )
    np.testing.assert_allclose(verification["leading_edge_sweep_deg"], 15,
                               rtol=0, atol=1e-13)
    np.testing.assert_allclose(verification["station_chords"][[0, 5, 10, 15, 20]],
                               [1 / 9, 1 / 6, 2 / 9, 1 / 6, 1 / 9], atol=1e-14)


@pytest.mark.parametrize("name", ["corner", "A", "B", "control", "normal"])
def test_mcbain_twist_reference_coordinates(verification, name):
    shape = (4, 20, 4, 3) if name == "corner" else (4, 20, 3)
    assert verification["generated"][name].shape == shape
    np.testing.assert_allclose(verification["generated"][name],
                               verification["reference"][name], rtol=0, atol=1e-14)
    assert verification["errors"][name] < 1e-14


def test_mcbain_twist_symmetry_and_normals(verification):
    g = verification["generated"]
    reflection = np.array([1, -1, 1])
    for name, other in [("A", "B"), ("B", "A"), ("control", "control"),
                        ("normal", "normal")]:
        np.testing.assert_allclose(g[name], g[other][:, ::-1] * reflection,
                                   rtol=0, atol=1e-14)
    np.testing.assert_allclose(g["corner"],
        g["corner"][:, ::-1][:, :, [3, 2, 1, 0]] * reflection, rtol=0, atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(g["normal"], axis=-1), 1,
                               rtol=0, atol=1e-14)
    assert np.all(g["normal"][..., 2] > 0)
