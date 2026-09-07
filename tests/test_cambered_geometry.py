import numpy as np
import pytest

from src.geometry import (
    swept_tapered_twisted_cambered_horseshoe_geometry as cambered_geometry,
    swept_tapered_twisted_horseshoe_geometry as twisted_geometry,
)


def camber(xi):
    return 0.1 * xi * (1 - xi)


def test_analytical_mean_line_and_panel_normals():
    A, B, P, normals, corners = cambered_geometry(2, 1, 1, 0, 0, 4, 2, camber)
    stations = np.concatenate((corners[:, 0, 0], corners[-1:, 0, 1]))
    np.testing.assert_allclose(stations[:, 0], [0, 0.25, 0.5, 0.75, 1], atol=1e-15)
    np.testing.assert_allclose(stations[:, 2], [0, 0.01875, 0.025, 0.01875, 0], atol=1e-15)
    for points, offset in ((A, 0.25), (B, 0.25), (P, 0.75)):
        xi = (np.arange(4) + offset) / 4
        for j in range(2):
            np.testing.assert_allclose(points[:, j, 0], xi, atol=1e-15)
            np.testing.assert_allclose(points[:, j, 2], camber(xi), atol=1e-15)
    # Secant slopes of the geometric panels, not analytical slopes at controls.
    slopes = np.array([0.075, 0.025, -0.025, -0.075])
    expected = np.column_stack((-slopes, np.zeros(4), np.ones(4)))
    expected /= np.linalg.norm(expected, axis=1)[:, None]
    for j in range(2):
        np.testing.assert_allclose(normals[:, j], expected, atol=1e-14)


@pytest.mark.parametrize("n_chord,n_span,twist", [(1, 2, 0), (3, 5, -7), (4, 20, 7.5)])
def test_zero_camber_matches_twisted_geometry(n_chord, n_span, twist):
    args = (6, 2, 0.5, 15, twist, n_chord, n_span)
    for actual, expected in zip(cambered_geometry(*args, camber=lambda xi: 0),
                                twisted_geometry(*args)):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)


@pytest.mark.parametrize("n_span", [4, 5])
def test_cambered_symmetry_shapes_and_unit_normals(n_span):
    A, B, P, normals, corners = cambered_geometry(6, 2, 0.5, 30, 7.5, 3, n_span, camber)
    reflection = np.array([1, -1, 1])
    for points, opposite in ((A, B), (B, A), (P, P), (normals, normals)):
        assert points.shape == (3, n_span, 3)
        np.testing.assert_allclose(points, opposite[:, ::-1] * reflection, rtol=0, atol=1e-14)
        assert np.all(np.isfinite(points))
    assert corners.shape == (3, n_span, 4, 3)
    np.testing.assert_allclose(corners, corners[:, ::-1][:, :, [3, 2, 1, 0]] * reflection,
                               rtol=0, atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=-1), 1, atol=1e-14)


def test_camber_twist_rotation_and_chord_scaling():
    A, B, P, _, corners = cambered_geometry(2, 2, 0.5, 45, 30, 1, 2, camber)
    # Right tip: c=1, theta=30 degrees, x_LE=1, quarter-chord eta=0.01875.
    expected = [1 + 0.25 * np.sqrt(3) / 2 + 0.01875 / 2,
                1, -0.25 / 2 + 0.01875 * np.sqrt(3) / 2]
    np.testing.assert_allclose(B[0, 1], expected, atol=1e-14)
    # Root: c=2, zero twist. Mid-semispan control: c=1.5, theta=15 degrees.
    np.testing.assert_allclose(A[0, 1], [0.5, 0, 0.0375], atol=1e-14)
    theta = np.deg2rad(15)
    np.testing.assert_allclose(P[0, 1],
        [0.5 + 1.5 * (0.75 * np.cos(theta) + 0.01875 * np.sin(theta)), 0.5,
         1.5 * (-0.75 * np.sin(theta) + 0.01875 * np.cos(theta))], atol=1e-14)
    np.testing.assert_allclose(corners[0, 1, 2], [1 + np.sqrt(3) / 2, 1, -0.5], atol=1e-14)


@pytest.mark.parametrize("bad_camber", [None, lambda xi: np.nan, lambda xi: [0, 0]])
def test_invalid_camber(bad_camber):
    with pytest.raises(ValueError, match="camber"):
        cambered_geometry(2, 1, 1, 0, 0, 2, 2, bad_camber)
