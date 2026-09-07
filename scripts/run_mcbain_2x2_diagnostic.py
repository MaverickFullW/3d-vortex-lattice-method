"""Print the McBain 2x2 horseshoe system and its numerical checks."""

from pathlib import Path
import sys

import numpy as np

# Support direct execution with python scripts/run_mcbain_2x2_diagnostic.py.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.geometry import rectangular_horseshoe_geometry
from src.solver import (
    build_horseshoe_influence_matrix,
    build_rhs,
    solve_circulation,
)


def main():
    span = 2.0
    chord = 1.0
    n_chord = 2
    n_span = 2
    V_inf = 1.0
    alpha_deg = 5.0
    wake_length = 100.0  # Same finite wake as run_rectangular_wing.py.

    geometry = rectangular_horseshoe_geometry(span, chord, n_chord, n_span)
    A = build_horseshoe_influence_matrix(*geometry, wake_length)
    b = build_rhs(V_inf, alpha_deg, n_chord * n_span)
    Gamma = solve_circulation(A, b)
    gamma_panels = Gamma.reshape(n_chord, n_span, order="C")
    residual = A @ Gamma - b

    with np.printoptions(precision=16):
        print(f"McBain 2x2 diagnostic (wake_length = {wake_length})")
        print("Influence matrix A:")
        print(A)
        print("RHS vector b:")
        print(b)
        print("Flattened circulation Gamma (k = i * n_span + j):")
        print(Gamma)
        print("Gamma reshaped as (n_chord, n_span):")
        print(gamma_panels)
        for i in range(n_chord):
            difference = abs(gamma_panels[i, 0] - gamma_panels[i, 1])
            print(f"abs(Gamma[{i},0] - Gamma[{i},1]) = {difference:.16e}")
        print("Residual A @ Gamma - b:")
        print(residual)
        print(f"||A @ Gamma - b|| = {np.linalg.norm(residual):.16e}")


if __name__ == "__main__":
    main()
