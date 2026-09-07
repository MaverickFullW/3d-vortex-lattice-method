"""Campbell tapered swept-wing benchmark from McBain, Chapter 14."""

from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.geometry import swept_tapered_horseshoe_geometry
from src.solver import (
    build_horseshoe_influence_matrix,
    solve_circulation,
    mcbain_total_lift,
    mcbain_lift_coefficient,
)


def run_benchmark():
    AR = 6.0
    taper_ratio = 0.6
    quarter_chord_sweep_deg = 45.0
    n_chord = 1
    n_span = 20
    span = 1.0
    rho = 1.0
    V_inf = 1.0
    alpha_rad = np.deg2rad(1.0)
    wake_length = 100.0  # Existing project convention; 100 spans here.

    root_chord = 2.0 * span / (AR * (1.0 + taper_ratio))
    tip_chord = taper_ratio * root_chord
    reference_area = span * (root_chord + tip_chord) / 2.0
    leading_edge_sweep_deg = np.rad2deg(np.arctan(
        (1.0 - taper_ratio) / ((1.0 + taper_ratio) * AR)
        + np.tan(np.deg2rad(quarter_chord_sweep_deg))
    ))
    A_points, B_points, control_points = swept_tapered_horseshoe_geometry(
        span, root_chord, taper_ratio, leading_edge_sweep_deg, n_chord, n_span
    )
    A = build_horseshoe_influence_matrix(
        A_points, B_points, control_points, wake_length
    )
    b = np.full(n_chord * n_span, -V_inf * alpha_rad)
    gamma = solve_circulation(A, b)
    lift = mcbain_total_lift(gamma, A_points, B_points, rho, V_inf)
    CL = mcbain_lift_coefficient(lift, rho, V_inf, reference_area)
    CLa = CL / alpha_rad
    reference_CLa = 3.5633
    absolute_error = abs(CLa - reference_CLa)

    return {
        "root_chord": root_chord,
        "tip_chord": tip_chord,
        "leading_edge_sweep_deg": leading_edge_sweep_deg,
        "reference_area": reference_area,
        # Preserve k = i * n_span + j.
        "Gamma": gamma.reshape(n_chord, n_span, order="C"),
        "CL": CL,
        "CLa": CLa,
        "reference_CLa": reference_CLa,
        "absolute_error": absolute_error,
        "relative_percentage_error": 100.0 * absolute_error / abs(reference_CLa),
        "residual_norm": np.linalg.norm(A @ gamma - b),
    }


def main():
    result = run_benchmark()
    print("McBain Campbell benchmark (1x20, AR = 6, taper_ratio = 0.6)")
    print(f"Root chord = {result['root_chord']:.12f}")
    print(f"Tip chord = {result['tip_chord']:.12f}")
    print(f"Leading-edge sweep [deg] = {result['leading_edge_sweep_deg']:.12f}")
    print(f"Gamma shape = {result['Gamma'].shape}")
    print(f"CL = {result['CL']:.12f}")
    print(f"CLa [1/rad] = {result['CLa']:.12f}")
    print(f"McBain reference CLa [1/rad] = {result['reference_CLa']:.4f}")
    print(f"Absolute error = {result['absolute_error']:.12e}")
    print(f"Relative percentage error = {result['relative_percentage_error']:.8f}%")
    print(f"Residual norm ||A @ gamma - b|| = {result['residual_norm']:.12e}")


if __name__ == "__main__":
    main()
