import numpy as np

from src.geometry import rectangular_wing_panels
from src.solver import (
    build_influence_matrix,
    build_rhs,
    lift_coefficient,
    lift_distribution,
    solve_circulation,
    total_lift,
)


span = 6.0
chord = 1.0
n_span = 160
wake_length = 100.0
V_inf = 10.0
alpha_deg = 5.0
rho = 1.225

A_points, B_points, control_points = rectangular_wing_panels(
    span,
    chord,
    n_span,
)

matrix = build_influence_matrix(
    A_points,
    B_points,
    control_points,
    wake_length,
)
rhs = build_rhs(V_inf, alpha_deg, n_span)
circulation = solve_circulation(matrix, rhs)
lift_per_span = lift_distribution(circulation, rho, V_inf)
lift = total_lift(lift_per_span, span)
CL = lift_coefficient(lift, rho, V_inf, span, chord)

aspect_ratio = span / chord
alpha_rad = np.deg2rad(alpha_deg)
CL_alpha = CL / alpha_rad

print("aspect ratio:", aspect_ratio)
print("CL:", CL)
print("CL / alpha [1/rad]:", CL_alpha)
