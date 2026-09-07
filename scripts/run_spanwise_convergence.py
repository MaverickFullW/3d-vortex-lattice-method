"""Study spanwise mesh convergence of the flat rectangular McBain wing."""

from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.geometry import rectangular_horseshoe_geometry
from src.solver import (
    build_horseshoe_influence_matrix,
    build_rhs,
    induced_drag_coefficient,
    induced_drag_from_downwash,
    mcbain_lift_coefficient,
    mcbain_total_lift,
    solve_circulation,
    span_efficiency,
    trailing_vortex_downwash,
)


def main():
    span, chord = 2.0, 1.0
    n_chord = 1
    rho, V_inf = 1.0, 1.0
    alpha_deg, wake_length = 5.0, 100.0
    S_ref = span * chord
    aspect_ratio = span**2 / S_ref
    meshes = [4, 8, 16, 32, 64, 128]
    results = []

    for n_span in meshes:
        total_panels = n_chord * n_span
        A, B, controls = rectangular_horseshoe_geometry(span, chord, n_chord, n_span)
        matrix = build_horseshoe_influence_matrix(A, B, controls, wake_length)
        rhs = build_rhs(V_inf, alpha_deg, total_panels)
        gamma_flat = solve_circulation(matrix, rhs)
        gamma = gamma_flat.reshape(n_chord, n_span, order="C")
        downwash = trailing_vortex_downwash(A, B, controls, gamma, wake_length)
        panel_widths = np.abs(B[..., 1] - A[..., 1])
        _, Di = induced_drag_from_downwash(gamma, downwash, panel_widths, rho)
        CDi = induced_drag_coefficient(Di, rho, V_inf, S_ref)
        lift = mcbain_total_lift(gamma, A, B, rho, V_inf)
        CL = mcbain_lift_coefficient(lift, rho, V_inf, S_ref)
        efficiency = span_efficiency(CL, CDi, aspect_ratio)
        residual = np.linalg.norm(matrix @ gamma_flat - rhs)
        results.append((n_span, total_panels, np.array([CL, CDi, efficiency]), residual))

    reference = results[-1][2]
    print("Spanwise convergence: rectangular McBain wing (n_chord=1, alpha=5 deg)")
    print("Relative differences = 100 * abs(value - reference) / abs(reference); "
          f"reference: n_span={results[-1][0]}")
    print(f"{'n_span':>6} {'total panels':>12} {'CL':>14} {'rel CL [%]':>12}"
          f" {'CDi':>14} {'rel CDi [%]':>12} {'e':>14} {'rel e [%]':>12} {'residual norm':>18}")
    for n_span, total_panels, values, residual in results:
        relative = 100.0 * np.abs(values - reference) / np.abs(reference)
        CL, CDi, efficiency = values
        print(f"{n_span:6d} {total_panels:12d} {CL:14.12f} {relative[0]:12.8f}"
              f" {CDi:14.12f} {relative[1]:12.8f} {efficiency:14.12f}"
              f" {relative[2]:12.8f} {residual:18.12e}")

    finest_changes = np.abs(reference - results[-2][2])
    print()
    for name, difference in zip(("CL", "CDi", "e"), finest_changes):
        print(f"|{name}_{results[-1][0]} - {name}_{results[-2][0]}| = {difference:.12e}")


if __name__ == "__main__":
    main()
