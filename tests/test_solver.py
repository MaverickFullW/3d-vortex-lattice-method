import numpy as np
import pytest

from src.geometry import rectangular_horseshoe_geometry
from src.geometry import rectangular_vortex_ring_geometry
from src.geometry import rectangular_wing_panels
from src.solver import build_rhs
from src.solver import build_influence_matrix
from src.solver import build_horseshoe_influence_matrix
from src.solver import build_vortex_ring_influence_matrix
from src.solver import effective_chordwise_circulation
from src.solver import lift_coefficient
from src.solver import lift_distribution
from src.solver import mcbain_total_lift
from src.solver import mcbain_lift_coefficient
from src.solver import solve_circulation
from src.solver import total_lift
from src.solver import vortex_ring_lift_distribution
from src.vortex import trailing_edge_vortex_velocity
from src.vortex import vortex_ring_velocity
from src.vortex import horseshoe_velocity


def test_horseshoe_influence_matrix_shape_and_finite_values():
    geometry = rectangular_horseshoe_geometry(2.0, 1.0, 2, 2)

    matrix = build_horseshoe_influence_matrix(*geometry, wake_length=10.0)

    assert matrix.shape == (4, 4)
    assert np.all(np.isfinite(matrix))


def test_horseshoe_influence_matrix_flattening_and_direct_velocity():
    A_points, B_points, control_points = rectangular_horseshoe_geometry(
        2.0, 1.0, 2, 2
    )
    wake_length = 10.0
    matrix = build_horseshoe_influence_matrix(
        A_points, B_points, control_points, wake_length
    )
    panel_indices = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for row, (i_r, j_r) in enumerate(panel_indices):
        for col, (i_s, j_s) in enumerate(panel_indices):
            velocity = horseshoe_velocity(
                A_points[i_s, j_s],
                B_points[i_s, j_s],
                control_points[i_r, j_r],
                gamma=1.0,
                wake_length=wake_length,
            )
            np.testing.assert_allclose(matrix[row, col], velocity[2])

    # Row 1 is control panel (0, 1); column 2 is source panel (1, 0).
    direct = horseshoe_velocity(
        A_points[1, 0], B_points[1, 0], control_points[0, 1],
        gamma=1.0, wake_length=wake_length,
    )
    np.testing.assert_allclose(matrix[1, 2], direct[2])


@pytest.mark.parametrize("array_index", [0, 1, 2])
def test_horseshoe_influence_matrix_rejects_mismatched_shapes(array_index):
    geometry = list(rectangular_horseshoe_geometry(2.0, 1.0, 2, 2))
    geometry[array_index] = geometry[array_index][:1]

    with pytest.raises(ValueError, match="matching shape"):
        build_horseshoe_influence_matrix(*geometry, wake_length=10.0)


@pytest.mark.parametrize("shape", [(4, 3), (2, 2, 2)])
def test_horseshoe_influence_matrix_rejects_invalid_dimensions(shape):
    geometry = [np.zeros(shape) for _ in range(3)]

    with pytest.raises(ValueError, match="must have shape"):
        build_horseshoe_influence_matrix(*geometry, wake_length=10.0)


def test_mcbain_2x2_circulation_symmetry_and_residual():
    geometry = rectangular_horseshoe_geometry(
        span=2.0, chord=1.0, n_chord=2, n_span=2
    )
    matrix = build_horseshoe_influence_matrix(*geometry, wake_length=100.0)
    rhs = build_rhs(V_inf=1.0, alpha_deg=5.0, n_points=4)

    gamma = solve_circulation(matrix, rhs)
    gamma_panels = gamma.reshape(2, 2, order="C")
    residual = matrix @ gamma - rhs

    assert np.all(np.isfinite(gamma))
    np.testing.assert_allclose(
        gamma_panels[:, 0], gamma_panels[:, 1], rtol=0.0, atol=1e-14
    )
    np.testing.assert_allclose(residual, 0.0, rtol=0.0, atol=1e-14)
    assert np.linalg.norm(residual) < 1e-14


def test_mcbain_2x2_total_lift_and_coefficient():
    A_points, B_points, control_points = rectangular_horseshoe_geometry(
        span=2.0, chord=1.0, n_chord=2, n_span=2
    )
    matrix = build_horseshoe_influence_matrix(
        A_points, B_points, control_points, wake_length=100.0
    )
    gamma = solve_circulation(matrix, build_rhs(1.0, 5.0, 4))

    lift = mcbain_total_lift(gamma, A_points, B_points, rho=1.0, V_inf=1.0)
    lift_2d = mcbain_total_lift(
        gamma.reshape(2, 2), A_points, B_points, rho=1.0, V_inf=1.0
    )
    CL = mcbain_lift_coefficient(lift, rho=1.0, V_inf=1.0, reference_area=2.0)

    assert np.isscalar(lift)
    np.testing.assert_allclose(lift, 0.29010214275, rtol=0.0, atol=1e-11)
    np.testing.assert_allclose(lift_2d, lift, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(CL, 0.29010214275, rtol=0.0, atol=1e-11)


def test_mcbain_total_lift_weights_spanwise_lengths_in_panel_order():
    A_points = np.zeros((2, 2, 3))
    B_points = np.array([
        [[3.0, 1.0, 4.0], [3.0, -2.0, 4.0]],
        [[3.0, 3.0, 4.0], [3.0, -4.0, 4.0]],
    ])
    gamma = np.array([1.0, 2.0, -3.0, 4.0])

    lift = mcbain_total_lift(gamma, A_points, B_points, rho=2.0, V_inf=3.0)

    # Only |dy| contributes: 2 * 3 * (1 + 4 - 9 + 16).
    np.testing.assert_allclose(lift, 72.0)


def test_mcbain_lift_coefficient_uses_explicit_reference_area():
    CL = mcbain_lift_coefficient(24.0, rho=2.0, V_inf=4.0, reference_area=3.0)

    np.testing.assert_allclose(CL, 0.5)


@pytest.mark.parametrize("shape", [(3,), (2, 1), (1, 4), (2, 2, 1)])
def test_mcbain_total_lift_rejects_mismatched_gamma(shape):
    A_points, B_points, _ = rectangular_horseshoe_geometry(2.0, 1.0, 2, 2)

    with pytest.raises(ValueError, match="gamma must have shape.*matching"):
        mcbain_total_lift(np.zeros(shape), A_points, B_points, 1.0, 1.0)


def test_mcbain_total_lift_rejects_mismatched_geometry():
    A_points, B_points, _ = rectangular_horseshoe_geometry(2.0, 1.0, 2, 2)

    with pytest.raises(ValueError, match="matching shape"):
        mcbain_total_lift(np.ones(4), A_points, B_points[:1], 1.0, 1.0)


@pytest.mark.parametrize("shape", [(4, 3), (2, 2, 2)])
def test_mcbain_total_lift_rejects_invalid_geometry_shape(shape):
    with pytest.raises(ValueError, match="Geometry arrays must have shape"):
        mcbain_total_lift(np.ones(4), np.zeros(shape), np.zeros(shape), 1.0, 1.0)


def test_influence_matrix_basic_properties():
    span = 10.0
    chord = 1.0
    n_span = 4
    wake_length = 100.0

    A_points, B_points, control_points = rectangular_wing_panels(
        span,
        chord,
        n_span,
    )

    matrix = build_influence_matrix(
        A_points, B_points, control_points, wake_length
    )

    assert matrix.shape == (4, 4)
    assert np.all(np.isfinite(matrix))


def test_influence_matrix_mirror_symmetry():
    span = 10.0
    chord = 1.0
    n_span = 4
    wake_length = 100.0

    A_points, B_points, control_points = rectangular_wing_panels(
        span,
        chord,
        n_span,
    )

    matrix = build_influence_matrix(
        A_points, B_points, control_points, wake_length
    )

    np.testing.assert_allclose(
        matrix,
        matrix[::-1, ::-1],
    )


def test_build_rhs():
    V_inf = 10.0
    alpha_deg = 5.0
    n_points = 4

    result = build_rhs(V_inf, alpha_deg, n_points)
    expected_value = -10.0 * np.sin(np.deg2rad(5.0))
    expected = np.full(4, expected_value)

    np.testing.assert_allclose(result, expected)


def test_solve_circulation_known_system():
    matrix = np.array([
        [2.0, 0.0],
        [0.0, 4.0],
    ])
    rhs = np.array([6.0, 8.0])

    result = solve_circulation(matrix, rhs)
    expected = np.array([3.0, 2.0])

    np.testing.assert_allclose(result, expected)


def test_lift_distribution():
    gamma = np.array([1.0, 2.0, 3.0])
    rho = 1.225
    V_inf = 10.0

    result = lift_distribution(gamma, rho, V_inf)
    expected = np.array([12.25, 24.50, 36.75])

    np.testing.assert_allclose(result, expected)


def test_total_lift():
    lift_per_span = np.array([10.0, 20.0, 20.0, 10.0])
    span = 2.0

    result = total_lift(lift_per_span, span)

    np.testing.assert_allclose(result, 30.0)


def test_lift_coefficient():
    lift = 100.0
    rho = 1.0
    V_inf = 10.0
    span = 2.0
    chord = 1.0

    result = lift_coefficient(lift, rho, V_inf, span, chord)

    np.testing.assert_allclose(result, 1.0)


def test_solved_circulation_satisfies_linear_system():
    span = 10.0
    chord = 1.0
    n_span = 20
    wake_length = 100.0
    V_inf = 10.0
    alpha_deg = 5.0

    A_points, B_points, control_points = rectangular_wing_panels(
        span,
        chord,
        n_span,
    )
    matrix = build_influence_matrix(
        A_points, B_points, control_points, wake_length
    )
    rhs = build_rhs(V_inf, alpha_deg, n_span)
    gamma = solve_circulation(matrix, rhs)

    residual = matrix @ gamma - rhs

    np.testing.assert_allclose(residual, 0.0, rtol=0.0, atol=1e-12)


def test_vortex_ring_influence_matrix_shape_and_finite_values():
    geometry = rectangular_vortex_ring_geometry(2.0, 1.0, 2, 2)

    matrix = build_vortex_ring_influence_matrix(*geometry, 10.0)

    assert matrix.shape == (4, 4)
    assert np.all(np.isfinite(matrix))


def test_vortex_ring_influence_matrix_flattening_and_source_types():
    A_points, B_points, C_points, D_points, control_points = (
        rectangular_vortex_ring_geometry(2.0, 1.0, 2, 2)
    )
    wake_length = 10.0
    matrix = build_vortex_ring_influence_matrix(
        A_points,
        B_points,
        C_points,
        D_points,
        control_points,
        wake_length,
    )

    panel_indices = [(0, 0), (0, 1), (1, 0), (1, 1)]

    for row, (i_r, j_r) in enumerate(panel_indices):
        P = control_points[i_r, j_r]

        for col, (i_s, j_s) in enumerate(panel_indices):
            if i_s == 0:
                velocity = vortex_ring_velocity(
                    A_points[i_s, j_s],
                    B_points[i_s, j_s],
                    C_points[i_s, j_s],
                    D_points[i_s, j_s],
                    P,
                    1.0,
                )
            else:
                velocity = trailing_edge_vortex_velocity(
                    A_points[i_s, j_s],
                    B_points[i_s, j_s],
                    C_points[i_s, j_s],
                    D_points[i_s, j_s],
                    P,
                    1.0,
                    wake_length,
                )

            np.testing.assert_allclose(matrix[row, col], velocity[2])


def test_vortex_ring_influence_matrix_superposition():
    n_chord = 2
    n_span = 2
    wake_length = 10.0
    A_points, B_points, C_points, D_points, control_points = (
        rectangular_vortex_ring_geometry(2.0, 1.0, n_chord, n_span)
    )
    matrix = build_vortex_ring_influence_matrix(
        A_points,
        B_points,
        C_points,
        D_points,
        control_points,
        wake_length,
    )
    gamma = np.array([1.0, 2.0, 3.0, 4.0])

    matrix_result = matrix @ gamma
    explicit_result = np.zeros(4)

    for i_r in range(n_chord):
        for j_r in range(n_span):
            row = i_r * n_span + j_r
            P = control_points[i_r, j_r]

            for i_s in range(n_chord):
                for j_s in range(n_span):
                    col = i_s * n_span + j_s

                    if i_s < n_chord - 1:
                        velocity = vortex_ring_velocity(
                            A_points[i_s, j_s],
                            B_points[i_s, j_s],
                            C_points[i_s, j_s],
                            D_points[i_s, j_s],
                            P,
                            1.0,
                        )
                    else:
                        velocity = trailing_edge_vortex_velocity(
                            A_points[i_s, j_s],
                            B_points[i_s, j_s],
                            C_points[i_s, j_s],
                            D_points[i_s, j_s],
                            P,
                            1.0,
                            wake_length,
                        )

                    explicit_result[row] += velocity[2] * gamma[col]

    np.testing.assert_allclose(
        matrix_result,
        explicit_result,
        rtol=1e-12,
        atol=1e-12,
    )


def test_solve_circulation_with_vortex_ring_influence_matrix():
    span = 6.0
    chord = 1.0
    n_chord = 2
    n_span = 4
    wake_length = 100.0
    V_inf = 10.0
    alpha_deg = 5.0

    A_points, B_points, C_points, D_points, control_points = (
        rectangular_vortex_ring_geometry(span, chord, n_chord, n_span)
    )
    matrix = build_vortex_ring_influence_matrix(
        A_points,
        B_points,
        C_points,
        D_points,
        control_points,
        wake_length,
    )
    n_points = n_chord * n_span
    rhs = build_rhs(V_inf, alpha_deg, n_points)
    gamma = solve_circulation(matrix, rhs)

    residual = matrix @ gamma - rhs

    assert gamma.shape == (n_chord * n_span,)
    assert np.all(np.isfinite(gamma))
    assert np.all(np.isfinite(residual))
    np.testing.assert_allclose(
        residual,
        np.zeros_like(residual),
        rtol=1e-12,
        atol=1e-12,
    )


def test_effective_chordwise_circulation_known_matrix():
    gamma_matrix = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [1.5, 3.0, 4.5, 6.0],
    ])
    expected = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [0.5, 1.0, 1.5, 2.0],
    ])

    result = effective_chordwise_circulation(gamma_matrix)

    np.testing.assert_allclose(result, expected)


def test_effective_chordwise_circulation_telescopes_to_last_row():
    gamma_matrix = np.array([
        [1.0, 2.0, 2.0, 1.0],
        [1.5, 3.0, 3.0, 1.5],
        [2.0, 4.0, 4.0, 2.0],
    ])

    delta_gamma = effective_chordwise_circulation(gamma_matrix)

    np.testing.assert_allclose(np.sum(delta_gamma, axis=0), gamma_matrix[-1])


def test_vortex_ring_lift_distribution_uses_last_gamma_row():
    gamma_matrix = np.array([
        [1.0, 2.0, 3.0, 4.0],
        [2.0, 3.0, 4.0, 5.0],
        [3.0, 4.0, 5.0, 6.0],
    ])
    rho = 1.225
    V_inf = 10.0

    result = vortex_ring_lift_distribution(gamma_matrix, rho, V_inf)
    expected = rho * V_inf * gamma_matrix[-1]

    np.testing.assert_allclose(result, expected)


def test_vortex_ring_lift_distribution_is_spanwise_symmetric():
    gamma_matrix = np.array([
        [1.0, 2.0, 2.0, 1.0],
        [2.0, 4.0, 4.0, 2.0],
    ])

    result = vortex_ring_lift_distribution(gamma_matrix, 1.225, 10.0)

    np.testing.assert_allclose(result, result[::-1])
