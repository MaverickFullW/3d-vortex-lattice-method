"""Bertin swept-wing benchmark from McBain, Theory of Lift, Chapter 14."""

from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.geometry import swept_rectangular_horseshoe_geometry
from src.solver import (
    build_horseshoe_influence_matrix,
    solve_circulation,
    mcbain_total_lift,
    mcbain_lift_coefficient,
)


def run_benchmark():
    AR = 5.0
    taper_ratio = 1.0
    sweep_deg = 45.0
    n_chord = 1
    n_span = 8
    chord = 1.0
    span = AR * chord
    reference_area = span * chord
    rho = 1.0
    V_inf = 1.0
    alpha_deg = 1.0
    wake_length = 100.0  # Existing project finite-wake convention.

    # The rectangular geometry implements taper_ratio = 1.0.
    A_points, B_points, control_points = swept_rectangular_horseshoe_geometry(
        span, chord, sweep_deg, n_chord, n_span
    )
    A = build_horseshoe_influence_matrix(
        A_points, B_points, control_points, wake_length
    )
    alpha_rad = np.deg2rad(alpha_deg)
    rhs = np.full(n_chord * n_span, -V_inf * alpha_rad)
    Gamma = solve_circulation(A, rhs)
    lift = mcbain_total_lift(Gamma, A_points, B_points, rho, V_inf)
    CL = mcbain_lift_coefficient(lift, rho, V_inf, reference_area)
    CLa = CL / alpha_rad
    reference_CLa = 3.4442
    absolute_error = abs(CLa - reference_CLa)

    return {
        "taper_ratio": taper_ratio,
        "Gamma": Gamma.reshape(n_chord, n_span, order="C"),
        "CL": CL,
        "CLa": CLa,
        "reference_CLa": reference_CLa,
        "absolute_error": absolute_error,
        "relative_percentage_error": 100.0 * absolute_error / abs(reference_CLa),
        "residual_norm": np.linalg.norm(A @ Gamma - rhs),
    }


def main():
    result = run_benchmark()
    print(f"Bertin benchmark (taper_ratio = {result['taper_ratio']})")
    print("Gamma reshaped as (n_chord, n_span):")
    with np.printoptions(precision=12, linewidth=120):
        print(result["Gamma"])
    print(f"CL = {result['CL']:.12f}")
    print(f"CLa [1/rad] = {result['CLa']:.12f}")
    print(f"McBain reference CLa [1/rad] = {result['reference_CLa']:.4f}")
    print(f"Absolute error = {result['absolute_error']:.12e}")
    print(f"Relative percentage error = {result['relative_percentage_error']:.8f}%")
    print(f"Residual norm ||A @ Gamma - rhs|| = {result['residual_norm']:.12e}")


if __name__ == "__main__":
    main()
