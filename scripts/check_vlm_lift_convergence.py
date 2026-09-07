import numpy as np

from src.geometry import rectangular_vortex_ring_geometry
from src.solver import (
    build_rhs,
    build_vortex_ring_influence_matrix,
    lift_coefficient,
    solve_circulation,
    total_lift,
    vortex_ring_lift_distribution,
)


meshes = [
    (1, 4),
    (2, 4),
    (2, 8),
    (4, 8),
    (4, 16),
    (8, 16),
    (8, 32),
]

span = 6.0
chord = 1.0
V_inf = 10.0
alpha_deg = 5.0
rho = 1.0
wake_length = 100.0

alpha_rad = np.deg2rad(alpha_deg)

print(f"{'Nx':>3} {'Ny':>3} {'Unknowns':>8} {'CL':>12} "
      f"{'CL/alpha_rad':>14} {'Residual norm':>14}")

for n_chord, n_span in meshes:
    geometry = rectangular_vortex_ring_geometry(
        span,
        chord,
        n_chord,
        n_span,
    )
    matrix = build_vortex_ring_influence_matrix(*geometry, wake_length)
    n_points = n_chord * n_span
    rhs = build_rhs(V_inf, alpha_deg, n_points)
    gamma = solve_circulation(matrix, rhs)
    gamma_matrix = gamma.reshape(n_chord, n_span)
    lift_per_span = vortex_ring_lift_distribution(
        gamma_matrix,
        rho,
        V_inf,
    )
    lift = total_lift(lift_per_span, span)
    CL = lift_coefficient(lift, rho, V_inf, span, chord)
    residual_norm = np.linalg.norm(matrix @ gamma - rhs)

    print(
        f"{n_chord:3d} {n_span:3d} {n_points:8d} {CL:12.8f} "
        f"{CL / alpha_rad:14.8f} {residual_norm:14.6e}"
    )

mcbain_CL_alpha = 4.530424981
mcbain_CL = mcbain_CL_alpha * alpha_rad

print()
print("McBain lifting-line reference CL/alpha = 4.530424981 1/rad")
print(f"McBain lifting-line reference CL at 5 degrees = {mcbain_CL:.9f}")
