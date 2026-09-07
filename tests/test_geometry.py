import numpy as np
import pytest

from src.geometry import (
    rectangular_horseshoe_geometry,
    rectangular_vortex_ring_geometry,
    spanwise_edges,
    swept_rectangular_horseshoe_geometry,
    swept_tapered_horseshoe_geometry,
)


def test_spanwise_edges_uniform_rectangular_wing():
    span = 10.0
    n_span = 4
    expected = np.array([-5.0, -2.5, 0.0, 2.5, 5.0])

    result = spanwise_edges(span, n_span)

    np.testing.assert_allclose(result, expected)


def test_rectangular_horseshoe_geometry_reference_case():
    A_points, B_points, control_points = rectangular_horseshoe_geometry(
        span=2.0,
        chord=1.0,
        n_chord=2,
        n_span=2,
    )

    expected_shape = (2, 2, 3)
    assert A_points.shape == expected_shape
    assert B_points.shape == expected_shape
    assert control_points.shape == expected_shape

    expected_A = np.array(
        [
            [[0.125, -1.0, 0.0], [0.125, 0.0, 0.0]],
            [[0.625, -1.0, 0.0], [0.625, 0.0, 0.0]],
        ]
    )
    expected_B = np.array(
        [
            [[0.125, 0.0, 0.0], [0.125, 1.0, 0.0]],
            [[0.625, 0.0, 0.0], [0.625, 1.0, 0.0]],
        ]
    )
    expected_control = np.array(
        [
            [[0.375, -0.5, 0.0], [0.375, 0.5, 0.0]],
            [[0.875, -0.5, 0.0], [0.875, 0.5, 0.0]],
        ]
    )

    np.testing.assert_allclose(A_points, expected_A)
    np.testing.assert_allclose(B_points, expected_B)
    np.testing.assert_allclose(control_points, expected_control)


@pytest.mark.parametrize("n_chord,n_span", [(1, 2), (2, 2), (3, 5)])
def test_swept_horseshoe_zero_sweep_matches_rectangular(n_chord, n_span):
    expected = rectangular_horseshoe_geometry(2.0, 1.0, n_chord, n_span)
    actual = swept_rectangular_horseshoe_geometry(
        2.0, 1.0, 0.0, n_chord, n_span
    )

    for points, expected_points in zip(actual, expected):
        assert points.shape == (n_chord, n_span, 3)
        np.testing.assert_allclose(points, expected_points, rtol=0.0, atol=1e-14)


@pytest.mark.parametrize("n_span", [4, 5])
def test_swept_horseshoe_geometry_mirror_symmetry(n_span):
    A, B, P = swept_rectangular_horseshoe_geometry(6.0, 2.0, 30.0, 3, n_span)
    reflection = np.array([1.0, -1.0, 1.0])

    # Reflection swaps left/right endpoints and reverses spanwise panel order.
    np.testing.assert_allclose(A, B[:, ::-1] * reflection, atol=1e-14)
    np.testing.assert_allclose(B, A[:, ::-1] * reflection, atol=1e-14)
    np.testing.assert_allclose(P, P[:, ::-1] * reflection, atol=1e-14)
    for points in (A, B, P):
        assert points.shape == (3, n_span, 3)
        np.testing.assert_array_equal(points[:, :, 2], 0.0)


def test_swept_horseshoe_geometry_45_degree_reference_case():
    A, B, P = swept_rectangular_horseshoe_geometry(2.0, 1.0, 45.0, 1, 2)

    np.testing.assert_allclose(A, [[[1.25, -1.0, 0.0], [0.25, 0.0, 0.0]]])
    np.testing.assert_allclose(B, [[[0.25, 0.0, 0.0], [1.25, 1.0, 0.0]]])
    np.testing.assert_allclose(P, [[[1.25, -0.5, 0.0], [1.25, 0.5, 0.0]]])


def test_swept_horseshoe_geometry_chordwise_panel_order():
    A, B, P = swept_rectangular_horseshoe_geometry(2.0, 1.0, 45.0, 2, 2)

    np.testing.assert_allclose(A.reshape(4, 3), [
        [1.125, -1.0, 0.0], [0.125, 0.0, 0.0],
        [1.625, -1.0, 0.0], [0.625, 0.0, 0.0],
    ])
    np.testing.assert_allclose(B.reshape(4, 3), [
        [0.125, 0.0, 0.0], [1.125, 1.0, 0.0],
        [0.625, 0.0, 0.0], [1.625, 1.0, 0.0],
    ])
    np.testing.assert_allclose(P.reshape(4, 3), [
        [0.875, -0.5, 0.0], [0.875, 0.5, 0.0],
        [1.375, -0.5, 0.0], [1.375, 0.5, 0.0],
    ])


def test_rectangular_vortex_ring_geometry_reference_case():
    span = 2.0
    chord = 1.0
    n_chord = 2
    n_span = 2

    A_points, B_points, C_points, D_points, control_points = (
        rectangular_vortex_ring_geometry(span, chord, n_chord, n_span)
    )

    expected_shape = (2, 2, 3)
    assert A_points.shape == expected_shape
    assert B_points.shape == expected_shape
    assert C_points.shape == expected_shape
    assert D_points.shape == expected_shape
    assert control_points.shape == expected_shape

    np.testing.assert_allclose(A_points[0, 0], [0.125, -1.0, 0.0])
    np.testing.assert_allclose(B_points[0, 0], [0.125, 0.0, 0.0])
    np.testing.assert_allclose(C_points[0, 0], [0.625, 0.0, 0.0])
    np.testing.assert_allclose(D_points[0, 0], [0.625, -1.0, 0.0])
    np.testing.assert_allclose(control_points[0, 0], [0.375, -0.5, 0.0])

    np.testing.assert_allclose(A_points[0, 1], [0.125, 0.0, 0.0])
    np.testing.assert_allclose(B_points[0, 1], [0.125, 1.0, 0.0])
    np.testing.assert_allclose(C_points[0, 1], [0.625, 1.0, 0.0])
    np.testing.assert_allclose(D_points[0, 1], [0.625, 0.0, 0.0])
    np.testing.assert_allclose(control_points[0, 1], [0.375, 0.5, 0.0])

    np.testing.assert_allclose(A_points[1, 0], [0.625, -1.0, 0.0])
    np.testing.assert_allclose(B_points[1, 0], [0.625, 0.0, 0.0])
    np.testing.assert_allclose(C_points[1, 0], [1.125, 0.0, 0.0])
    np.testing.assert_allclose(D_points[1, 0], [1.125, -1.0, 0.0])
    np.testing.assert_allclose(control_points[1, 0], [0.875, -0.5, 0.0])

    np.testing.assert_allclose(A_points[1, 1], [0.625, 0.0, 0.0])
    np.testing.assert_allclose(B_points[1, 1], [0.625, 1.0, 0.0])
    np.testing.assert_allclose(C_points[1, 1], [1.125, 1.0, 0.0])
    np.testing.assert_allclose(D_points[1, 1], [1.125, 0.0, 0.0])
    np.testing.assert_allclose(control_points[1, 1], [0.875, 0.5, 0.0])


@pytest.mark.parametrize("n_chord,n_span", [(1, 2), (2, 8), (3, 5)])
@pytest.mark.parametrize("sweep", [0.0, 30.0, 45.0])
def test_taper_one_matches_swept_rectangular(n_chord, n_span, sweep):
    expected = swept_rectangular_horseshoe_geometry(
        5.0, 2.0, sweep, n_chord, n_span
    )
    actual = swept_tapered_horseshoe_geometry(
        5.0, 2.0, 1.0, sweep, n_chord, n_span
    )
    for points, reference in zip(actual, expected):
        assert points.shape == (n_chord, n_span, 3)
        np.testing.assert_allclose(points, reference, rtol=0.0, atol=1e-14)


def test_tapered_root_tip_and_linear_chord():
    A, B, P = swept_tapered_horseshoe_geometry(8.0, 4.0, 0.25, 30.0, 2, 8)
    # With two chordwise panels, bound locations differ by half the chord.
    # Recover chord independently of the leading-edge offset.
    for endpoints, expected in (
        (A, [1.0, 1.75, 2.5, 3.25, 4.0, 3.25, 2.5, 1.75]),
        (B, [1.75, 2.5, 3.25, 4.0, 3.25, 2.5, 1.75, 1.0]),
    ):
        chords = 2.0 * (endpoints[1, :, 0] - endpoints[0, :, 0])
        np.testing.assert_allclose(chords, expected, rtol=0.0, atol=1e-14)
    midpoint_chords = 2.0 * (P[1, :, 0] - P[0, :, 0])
    np.testing.assert_allclose(
        midpoint_chords, [1.375, 2.125, 2.875, 3.625, 3.625, 2.875, 2.125, 1.375],
        rtol=0.0, atol=1e-14,
    )


@pytest.mark.parametrize("n_span", [4, 5])
def test_tapered_geometry_mirror_symmetry(n_span):
    A, B, P = swept_tapered_horseshoe_geometry(6.0, 2.0, 0.4, 30.0, 3, n_span)
    reflection = np.array([1.0, -1.0, 1.0])
    np.testing.assert_allclose(A, B[:, ::-1] * reflection, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(B, A[:, ::-1] * reflection, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(P, P[:, ::-1] * reflection, rtol=0.0, atol=1e-14)
    for points in (A, B, P):
        assert points.shape == (3, n_span, 3)
        np.testing.assert_array_equal(points[:, :, 2], 0.0)


def test_tapered_zero_sweep_has_straight_leading_edge():
    A, B, P = swept_tapered_horseshoe_geometry(4.0, 2.0, 0.5, 0.0, 2, 4)
    for points in (A, B):
        chords = 2.0 * (points[1, :, 0] - points[0, :, 0])
        leading_edge = points[0, :, 0] - chords / 8.0
        np.testing.assert_allclose(leading_edge, 0.0, rtol=0.0, atol=1e-14)
        assert np.ptp(chords) > 0.0
    np.testing.assert_allclose(A[0, :, 0], [0.125, 0.1875, 0.25, 0.1875])
    np.testing.assert_allclose(P[0, :, 0], [0.46875, 0.65625, 0.65625, 0.46875])


def test_tapered_geometry_hand_calculated_panel_positions_and_order():
    A, B, P = swept_tapered_horseshoe_geometry(2.0, 2.0, 0.5, 45.0, 2, 2)
    # Chords: root 2, tips 1, midpoints 1.5. Leading-edge x: 0, 1, 0.5.
    # Bound chord fractions: 1/8, 5/8; control fractions: 3/8, 7/8.
    # Flat rows explicitly follow k = i * n_span + j.
    for points, expected in (
        (A, [[1.125, -1, 0], [0.25, 0, 0],
             [1.625, -1, 0], [1.25, 0, 0]]),
        (B, [[0.25, 0, 0], [1.125, 1, 0],
             [1.25, 0, 0], [1.625, 1, 0]]),
        (P, [[1.0625, -0.5, 0], [1.0625, 0.5, 0],
             [1.8125, -0.5, 0], [1.8125, 0.5, 0]]),
    ):
        assert points.shape == (2, 2, 3)
        np.testing.assert_allclose(
            points.reshape(4, 3, order="C"), expected, rtol=0.0, atol=1e-14
        )
