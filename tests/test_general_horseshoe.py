import numpy as np
import pytest

from src.geometry import swept_tapered_horseshoe_geometry, swept_tapered_twisted_horseshoe_geometry
from src.solver import build_horseshoe_influence_matrix, build_general_rhs, build_rhs, solve_circulation
from src.vortex import horseshoe_velocity


def test_flat_matrix_equivalence():
    geometry = swept_tapered_horseshoe_geometry(6, 2, 0.6, 30, 2, 4)
    normals = np.zeros_like(geometry[2])
    normals[..., 2] = 1
    legacy = build_horseshoe_influence_matrix(*geometry, 100)
    general = build_horseshoe_influence_matrix(*geometry, 100, normals=normals)
    np.testing.assert_array_equal(general, legacy)


@pytest.mark.parametrize("alpha", [-10, 0, 1, 15])
def test_flat_rhs_equivalence(alpha):
    normals = np.zeros((2, 4, 3))
    normals[..., 2] = 1
    angle = np.deg2rad(alpha)
    velocity = [2 * np.cos(angle), 0, 2 * np.sin(angle)]
    np.testing.assert_array_equal(build_general_rhs(velocity, normals), build_rhs(2, alpha, 8))


def test_twisted_projection_and_solve():
    A, B, P, normals, _ = swept_tapered_twisted_horseshoe_geometry(
        6, 2, 0.6, 30, -5, 2, 4
    )
    matrix = build_horseshoe_influence_matrix(A, B, P, 100, normals=normals)
    velocity = np.array([np.cos(np.deg2rad(3)), 0.02, np.sin(np.deg2rad(3))])
    rhs = build_general_rhs(velocity, normals)
    assert matrix.shape == (8, 8)
    assert rhs.shape == (8,)
    assert np.all(np.isfinite(matrix))
    assert np.all(np.isfinite(rhs))
    # Explicit grid indexing catches source/receiver normal mixups and order errors.
    for i in range(2):
        for j in range(4):
            row = i * 4 + j
            np.testing.assert_allclose(rhs[row], -np.dot(velocity, normals[i, j]), atol=1e-15)
            for p in range(2):
                for q in range(4):
                    induced = horseshoe_velocity(A[p, q], B[p, q], P[i, j], gamma=1, wake_length=100)
                    np.testing.assert_allclose(matrix[row, p * 4 + q], np.dot(induced, normals[i, j]), atol=1e-15)
    gamma = solve_circulation(matrix, rhs)
    assert np.all(np.isfinite(gamma))
    assert np.linalg.norm(matrix @ gamma - rhs) < 1e-14


@pytest.mark.parametrize("normals,message", [
    (np.ones((8, 3)), "shape"),
    (np.ones((2, 4, 2)), "shape"),
    (np.full((2, 4, 3), np.nan), "finite"),
    (np.full((2, 4, 3), np.inf), "finite"),
    (np.zeros((2, 4, 3)), "nonzero"),
    (np.full((2, 4, 3), 2.0), "unit magnitude"),
])
def test_invalid_normals_rejected_by_both_apis(normals, message):
    geometry = swept_tapered_horseshoe_geometry(6, 2, 0.6, 30, 2, 4)
    with pytest.raises(ValueError, match=message):
        build_horseshoe_influence_matrix(*geometry, 100, normals=normals)
    with pytest.raises(ValueError, match=message):
        build_general_rhs([1, 0, 0], normals)


def test_normal_grid_must_match_even_with_equal_panel_count():
    geometry = swept_tapered_horseshoe_geometry(6, 2, 0.6, 30, 2, 4)
    normals = np.zeros((4, 2, 3))
    normals[..., 2] = 1
    with pytest.raises(ValueError, match="same panel-grid shape"):
        build_horseshoe_influence_matrix(*geometry, 100, normals=normals)


@pytest.mark.parametrize("velocity", [[1, 0], [[1, 0, 0]], [1, np.nan, 0], [1, 0, np.inf]])
def test_invalid_freestream(velocity):
    with pytest.raises(ValueError, match="Freestream velocity vector"):
        build_general_rhs(velocity, [[[0, 0, 1]]])
