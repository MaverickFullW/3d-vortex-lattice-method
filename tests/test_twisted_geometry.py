import numpy as np
import pytest

from src.geometry import (
    swept_tapered_horseshoe_geometry,
    swept_tapered_twisted_horseshoe_geometry,
)


@pytest.mark.parametrize("n_chord,n_span", [(1, 2), (2, 8), (3, 5)])
def test_zero_twist_matches_existing_geometry(n_chord, n_span):
    actual = swept_tapered_twisted_horseshoe_geometry(
        6, 2, 0.6, 35, 0, n_chord, n_span
    )
    expected = swept_tapered_horseshoe_geometry(6, 2, 0.6, 35, n_chord, n_span)
    for points, reference in zip(actual[:3], expected):
        np.testing.assert_allclose(points, reference, rtol=0, atol=1e-14)
    np.testing.assert_allclose(actual[3][..., :2], 0, atol=1e-14)
    np.testing.assert_allclose(actual[3][..., 2], 1, atol=1e-14)


@pytest.mark.parametrize("tip_twist", [-12, 12])
def test_twist_station_angles_and_chord_lengths(tip_twist):
    *_, corners = swept_tapered_twisted_horseshoe_geometry(
        8, 2, 0.5, 30, tip_twist, 1, 4
    )
    # Recover actual section angles and lengths from geometric corner edges.
    left = corners[0, :, 1] - corners[0, :, 0]
    right = corners[0, :, 2] - corners[0, :, 3]
    edges = np.concatenate((left, right[-1:]))
    angles = np.rad2deg(np.arctan2(-edges[:, 2], edges[:, 0]))
    np.testing.assert_allclose(angles, np.array([1, 0.5, 0, 0.5, 1]) * tip_twist,
                               rtol=0, atol=1e-13)
    np.testing.assert_allclose(np.linalg.norm(edges, axis=1), [1, 1.5, 2, 1.5, 1])
    # Leading edge is the fixed rotation axis.
    np.testing.assert_allclose(corners[0, :, 0, 2], 0, atol=1e-14)


@pytest.mark.parametrize("n_span", [4, 5])
@pytest.mark.parametrize("twist", [-10, 10])
def test_twisted_symmetry_unit_normals_and_order(n_span, twist):
    A, B, P, normals, corners = swept_tapered_twisted_horseshoe_geometry(
        6, 2, 0.6, 30, twist, 3, n_span
    )
    reflection = np.array([1, -1, 1])
    for points, mirrored in ((A, B), (B, A), (P, P), (normals, normals)):
        assert points.shape == (3, n_span, 3)
        np.testing.assert_allclose(points, mirrored[:, ::-1] * reflection,
                                   rtol=0, atol=1e-14)
        for i in range(3):
            for j in range(n_span):
                np.testing.assert_array_equal(
                    points.reshape(-1, 3, order="C")[i * n_span + j], points[i, j]
                )
    assert corners.shape == (3, n_span, 4, 3)
    np.testing.assert_allclose(corners, corners[:, ::-1][:, :, [3, 2, 1, 0]] * reflection,
                               rtol=0, atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(normals, axis=-1), 1, atol=1e-14)
    assert np.all(normals[..., 0] * twist > 0)
    assert np.all(normals[..., 2] > 0)
    # Normals must be perpendicular to the actual quad's center tangents.
    p0, p1, p2, p3 = (corners[:, :, k] for k in range(4))
    for tangent in ((p1 - p0) + (p2 - p3), (p3 - p0) + (p2 - p1)):
        np.testing.assert_allclose(np.sum(normals * tangent, axis=-1), 0, atol=1e-14)


def test_twisted_panel_hand_calculation():
    A, B, P, normals, corners = swept_tapered_twisted_horseshoe_geometry(
        2, 1, 1, 0, 60, 1, 2
    )
    s = np.sqrt(3)
    # Right panel: root angle 0, tip angle 60 degrees.
    np.testing.assert_allclose(corners[0, 1],
        [[0, 0, 0], [1, 0, 0], [0.5, 1, -s / 2], [0, 1, 0]], atol=1e-14)
    np.testing.assert_allclose(A[0, 1], [0.25, 0, 0], atol=1e-14)
    np.testing.assert_allclose(B[0, 1], [0.125, 1, -s / 8], atol=1e-14)
    # Control is at 3/4 chord, y=1/2, with local twist 30 degrees.
    np.testing.assert_allclose(P[0, 1], [3 * s / 8, 0.5, -3 / 8], atol=1e-14)
    # Center tangents: (3/4,0,-sqrt(3)/4), (-1/4,1,-sqrt(3)/4).
    # Cross product: (sqrt(3)/4,sqrt(3)/4,3/4).
    np.testing.assert_allclose(normals[0, 1], np.array([1, 1, s]) / np.sqrt(5),
                               rtol=0, atol=1e-14)
