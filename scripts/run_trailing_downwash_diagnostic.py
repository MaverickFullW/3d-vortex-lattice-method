"""Inspect trailing-leg induced velocity at flat-wing VLM control points."""

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
    trailing_vortex_downwash,
    induced_angle_from_downwash,
    induced_drag_from_downwash,
    induced_drag_coefficient,
    mcbain_total_lift,
    mcbain_lift_coefficient,
    span_efficiency,
    spanwise_lift_distribution,
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
    w = trailing_vortex_downwash(A, B, controls, gamma, wake_length)
    induced_angle = induced_angle_from_downwash(w, V_inf)
    induced_angle_deg = np.rad2deg(induced_angle)
    rho = 1.0
    panel_drag, Di = induced_drag_from_downwash(gamma, w, abs(B[..., 1] - A[..., 1]), rho)
    S_ref = span * chord
    aspect_ratio = span**2 / S_ref
    CDi = induced_drag_coefficient(Di, rho, V_inf, S_ref)
    lift = mcbain_total_lift(gamma, A, B, rho, V_inf)
    CL = mcbain_lift_coefficient(lift, rho, V_inf, S_ref)
    efficiency = span_efficiency(CL, CDi, aspect_ratio)

    print("Rectangular wing trailing-only downwash diagnostic (1x8, alpha=5 deg)")
    print("panel         y           Gamma       trailing w      angle [deg]         delta Di")
    for j in range(n_span):
        print(f"{j:5d} {controls[0, j, 1]:9.5f} {gamma[0, j]:15.12f}"
              f" {w[0, j]:16.12f} {induced_angle_deg[0, j]:16.12f} {panel_drag[0, j]:16.12f}")
    print(f"Total Di = {Di:.12f}")
    print(f"CL = {CL:.12f}")
    print(f"Aspect ratio = {aspect_ratio:.12f}")
    print(f"CDi = {CDi:.12f}")
    print(f"Span efficiency e = {efficiency:.12f}")
    print(f"Maximum panel-drag symmetry error = {np.max(abs(panel_drag - panel_drag[:, ::-1])):.12e}")
    print(f"Linear-system residual norm = {np.linalg.norm(matrix @ gamma_flat - rhs):.12e}")

    Gamma_span, L_prime = spanwise_lift_distribution(gamma, rho, V_inf)
    delta_y = np.abs(B[0, :, 1] - A[0, :, 1])
    L_from_distribution = float(np.sum(L_prime * delta_y))
    absolute_difference = abs(L_from_distribution - lift)
    relative_difference = absolute_difference / abs(lift)
    print("\nSpanwise lift distribution")
    print("panel         y      Gamma_span          L_prime")
    for j in range(n_span):
        print(f"{j:5d} {controls[0, j, 1]:9.5f} {Gamma_span[j]:15.12f} {L_prime[j]:16.12f}")
    print(f"L from existing lift calculation = {lift:.12f}")
    print(f"L from spanwise distribution = {L_from_distribution:.12f}")
    print(f"absolute lift difference = {absolute_difference:.12e}")
    print(f"relative lift difference = {relative_difference:.12e}")
    print(f"CL = {CL:.12f}")
    print(f"maximum L_prime symmetry error = {np.max(abs(L_prime - L_prime[::-1])):.12e}")
    print(f"linear-system residual norm = {np.linalg.norm(matrix @ gamma_flat - rhs):.12e}")



if __name__ == "__main__":
    main()

