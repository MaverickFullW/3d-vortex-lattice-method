import numpy as np

from src.vortex import (
    horseshoe_velocity,
    trailing_edge_vortex_velocity,
    vortex_ring_velocity,
    vortex_segment_velocity,
)

def test_finite_vortex_segment_known_solution():
    A = np.array([0.0, -1.0, 0.0])
    B = np.array([0.0, 1.0, 0.0])
    P = np.array([1.0, 0.0, 0.0])

    gamma = 1.0

    velocity = vortex_segment_velocity(A, B, P, gamma)

    expected = np.array([
        0.0,
        0.0,
        -np.sqrt(2.0) / (4.0 * np.pi),
    ])

    np.testing.assert_allclose(velocity, expected)

def test_velocity_scales_linearly_with_gamma():
    A = np.array([0.0, -1.0, 0.0])
    B = np.array([0.0, 1.0, 0.0])
    P = np.array([1.0, 0.0, 0.0])

    velocity_1 = vortex_segment_velocity(A, B, P, 1.0)
    velocity_2 = vortex_segment_velocity(A, B, P, 2.0)

    np.testing.assert_allclose(velocity_2, 2.0 * velocity_1)

def test_reversing_segment_reverses_velocity():
    A = np.array([0.0, -1.0, 0.0])
    B = np.array([0.0, 1.0, 0.0])
    P = np.array([1.0, 0.0, 0.0])

    velocity_ab = vortex_segment_velocity(A, B, P, 1.0)
    velocity_ba = vortex_segment_velocity(B, A, P, 1.0)

    np.testing.assert_allclose(velocity_ba, -velocity_ab)

def test_point_on_vortex_line_returns_zero():
    A = np.array([0.0, -1.0, 0.0])
    B = np.array([0.0, 1.0, 0.0])
    P = np.array([0.0, 0.0, 0.0])

    velocity = vortex_segment_velocity(A, B, P, 1.0)

    np.testing.assert_allclose(velocity, np.zeros(3))

def test_horseshoe_velocity_is_finite():
    A = np.array([0.0, -1.0, 0.0])
    B = np.array([0.0, 1.0, 0.0])
    P = np.array([1.0, 0.0, 1.0])
    gamma = 1.0
    wake_length = 100.0

    velocity = horseshoe_velocity(A, B, P, gamma, wake_length)

    assert np.all(np.isfinite(velocity))


def test_vortex_ring_velocity_scales_linearly_with_gamma():
    A = np.array([0.0, -1.0, 0.0])
    B = np.array([1.0, -1.0, 0.0])
    C = np.array([1.0, 1.0, 0.0])
    D = np.array([0.0, 1.0, 0.0])
    P = np.array([0.5, 0.0, 1.0])

    velocity_1 = vortex_ring_velocity(A, B, C, D, P, 1.0)
    velocity_2 = vortex_ring_velocity(A, B, C, D, P, 2.0)

    np.testing.assert_allclose(velocity_2, 2.0 * velocity_1)


def test_reversing_vortex_ring_reverses_velocity():
    A = np.array([0.0, -1.0, 0.0])
    B = np.array([1.0, -1.0, 0.0])
    C = np.array([1.0, 1.0, 0.0])
    D = np.array([0.0, 1.0, 0.0])
    P = np.array([0.5, 0.0, 1.0])

    velocity_abcd = vortex_ring_velocity(A, B, C, D, P, 1.0)
    velocity_adcb = vortex_ring_velocity(A, D, C, B, P, 1.0)

    np.testing.assert_allclose(velocity_adcb, -velocity_abcd)


def test_trailing_edge_vortex_velocity_scales_linearly_with_gamma():
    A = np.array([0.0, -1.0, 0.0])
    B = np.array([0.0, 1.0, 0.0])
    C = np.array([1.0, 1.0, 0.0])
    D = np.array([1.0, -1.0, 0.0])
    P = np.array([0.5, 0.0, 1.0])
    wake_length = 10.0

    velocity_1 = trailing_edge_vortex_velocity(
        A, B, C, D, P, 1.0, wake_length
    )
    velocity_2 = trailing_edge_vortex_velocity(
        A, B, C, D, P, 2.0, wake_length
    )

    np.testing.assert_allclose(velocity_2, 2.0 * velocity_1)


def test_reversing_trailing_edge_vortex_path_reverses_velocity():
    A = np.array([0.0, -1.0, 0.0])
    B = np.array([0.0, 1.0, 0.0])
    C = np.array([1.0, 1.0, 0.0])
    D = np.array([1.0, -1.0, 0.0])
    P = np.array([0.5, 0.0, 1.0])
    gamma = 1.0
    wake_length = 10.0
    C_far = C + np.array([wake_length, 0.0, 0.0])
    D_far = D + np.array([wake_length, 0.0, 0.0])

    forward_velocity = trailing_edge_vortex_velocity(
        A, B, C, D, P, gamma, wake_length
    )
    reverse_velocity = (
        vortex_segment_velocity(A, D, P, gamma)
        + vortex_segment_velocity(D, D_far, P, gamma)
        + vortex_segment_velocity(D_far, C_far, P, gamma)
        + vortex_segment_velocity(C_far, C, P, gamma)
        + vortex_segment_velocity(C, B, P, gamma)
        + vortex_segment_velocity(B, A, P, gamma)
    )

    np.testing.assert_allclose(
        reverse_velocity,
        -forward_velocity,
        rtol=1e-12,
        atol=1e-12,
    )


def test_trailing_edge_vortex_velocity_is_finite():
    A = np.array([0.0, -1.0, 0.0])
    B = np.array([0.0, 1.0, 0.0])
    C = np.array([1.0, 1.0, 0.0])
    D = np.array([1.0, -1.0, 0.0])
    P = np.array([0.5, 0.0, 1.0])

    velocity = trailing_edge_vortex_velocity(
        A, B, C, D, P, 1.0, 10.0
    )

    assert np.all(np.isfinite(velocity))
