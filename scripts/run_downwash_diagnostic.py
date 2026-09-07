"""Inspect total induced velocity at flat-wing VLM control points."""

from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.geometry import rectangular_horseshoe_geometry
from src.solver import (
    build_horseshoe_influence_matrix,
    build_rhs,
    solve_circulation,
    horseshoe_induced_velocity,
)


def main():
    span, chord = 2.0, 1.0
    n_chord, n_span = 1, 8
    V_inf, alpha_deg, wake_length = 1.0, 5.0, 100.0
    A, B, controls = rectangular_horseshoe_geometry(span, chord, n_chord, n_span)
    matrix = build_horseshoe_influence_matrix(A, B, controls, wake_length)
    rhs = build_rhs(V_inf, alpha_deg, n_chord * n_span)
    gamma_flat = solve_circulation(matrix, rhs)
    gamma = gamma_flat.reshape(n_chord, n_span, order="C")
    velocity, w = horseshoe_induced_velocity(A, B, controls, gamma, wake_length)

    print("Rectangular wing control-point downwash diagnostic (1x8, alpha=5 deg)")
    print("panel         y           Gamma          V_ind_x          V_ind_y                w")
    for j in range(n_span):
        print(f"{j:5d} {controls[0, j, 1]:9.5f} {gamma[0, j]:15.12f}"
              f" {velocity[0, j, 0]:16.12f} {velocity[0, j, 1]:16.12f} {w[0, j]:16.12f}")
    print(f"Maximum circulation symmetry error = {np.max(abs(gamma - gamma[:, ::-1])):.12e}")
    print(f"Maximum downwash symmetry error = {np.max(abs(w - w[:, ::-1])):.12e}")
    print(f"Minimum downwash = {w.min():.12f}")
    print(f"Maximum downwash = {w.max():.12f}")
    print(f"Linear-system residual norm = {np.linalg.norm(matrix @ gamma_flat - rhs):.12e}")
    print("At flat-wing control points, no penetration enforces w = -V_inf*sin(alpha).")


if __name__ == "__main__":
    main()
