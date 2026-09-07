import numpy as np

from src.vortex import (
    horseshoe_velocity,
    vortex_segment_velocity,
    trailing_edge_vortex_velocity,
    vortex_ring_velocity,
)


def build_influence_matrix(
    A_points,
    B_points,
    control_points,
    wake_length,
):
    N = len(control_points)
    matrix = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            velocity = horseshoe_velocity(
                A_points[j],
                B_points[j],
                control_points[i],
                gamma=1.0,
                wake_length=wake_length,
            )
            matrix[i, j] = velocity[2]

    return matrix


def build_horseshoe_influence_matrix(
    A_points,
    B_points,
    control_points,
    wake_length,
    normals=None,
):
    """Build McBain influence coefficients with C-order k = i * n_span + j.

    Optional unit normals have shape (n_chord, n_span, 3). Each induced
    velocity is projected onto the receiving panel's normal. Omitting normals
    preserves the legacy flat-wing z-component calculation exactly.
    """
    if not (A_points.shape == B_points.shape == control_points.shape):
        raise ValueError("Geometry arrays must have matching shape.")
    if control_points.ndim != 3 or control_points.shape[2] != 3:
        raise ValueError("Geometry arrays must have shape (n_chord, n_span, 3).")

    n_chord, n_span, _ = control_points.shape
    n_panels = n_chord * n_span
    A_flat = A_points.reshape(n_panels, 3, order="C")
    B_flat = B_points.reshape(n_panels, 3, order="C")
    control_flat = control_points.reshape(n_panels, 3, order="C")
    normal_flat = None
    if normals is not None:
        normals = _validated_panel_normals(normals)
        if normals.shape != control_points.shape:
            raise ValueError("Normals must have the same panel-grid shape as control points.")
        normal_flat = normals.reshape(n_panels, 3, order="C")
    matrix = np.zeros((n_panels, n_panels))

    for m in range(n_panels):
        for n in range(n_panels):
            velocity = horseshoe_velocity(
                A_flat[n],
                B_flat[n],
                control_flat[m],
                gamma=1.0,
                wake_length=wake_length,
            )
            matrix[m, n] = (
                velocity[2] if normal_flat is None else np.dot(velocity, normal_flat[m])
            )

    return matrix


def horseshoe_induced_velocity(A_points, B_points, control_points, Gamma, wake_length):
    """Return total induced velocity and global-z downwash at control points.

    Geometry has shape (n_chord, n_span, 3); Gamma must have shape
    (n_chord, n_span). Outputs have these respective shapes. All panels use
    C-order indexing k = i*n_span + j, including each panel's own horseshoe.
    Axes are downstream x, spanwise y, upward z: downward velocity means
    negative downwash. Downwash is global velocity[..., 2], not a projection
    onto local normals. The freestream is not included.
    """
    A_points = np.asarray(A_points, dtype=float)
    B_points = np.asarray(B_points, dtype=float)
    control_points = np.asarray(control_points, dtype=float)
    Gamma = np.asarray(Gamma, dtype=float)
    if not (A_points.shape == B_points.shape == control_points.shape):
        raise ValueError("Geometry arrays must have matching shape.")
    if control_points.ndim != 3 or control_points.shape[-1] != 3:
        raise ValueError("Geometry arrays must have shape (n_chord, n_span, 3).")
    if Gamma.shape != control_points.shape[:2]:
        raise ValueError("Gamma must have shape (n_chord, n_span) matching the geometry.")
    if not all(np.all(np.isfinite(points)) for points in (A_points, B_points, control_points)):
        raise ValueError("Geometry arrays must be finite.")
    if not np.all(np.isfinite(Gamma)):
        raise ValueError("Gamma must be finite.")
    wake_length = np.asarray(wake_length, dtype=float)
    if wake_length.shape != () or not np.isfinite(wake_length) or wake_length <= 0:
        raise ValueError("wake_length must be a finite positive scalar.")
    A_flat = A_points.reshape(-1, 3, order="C")
    B_flat = B_points.reshape(-1, 3, order="C")
    control_flat = control_points.reshape(-1, 3, order="C")
    gamma_flat = Gamma.ravel(order="C")
    velocity = np.zeros_like(control_flat)
    for i, point in enumerate(control_flat):
        for j, gamma in enumerate(gamma_flat):
            if gamma == 0.0:
                continue
            velocity[i] += gamma * horseshoe_velocity(
                A_flat[j], B_flat[j], point, gamma=1.0, wake_length=float(wake_length)
            )
    velocity = velocity.reshape(control_points.shape, order="C")
    return velocity, velocity[..., 2].copy()


def trailing_vortex_downwash(A_points, B_points, control_points, Gamma, wake_length):
    """Return global-z velocity from only A_far->A and B->B_far wake legs.

    Geometry shape is (n_chord, n_span, 3); Gamma and output shape are
    (n_chord, n_span), using C-order k = i*n_span + j. The finite wake
    extends downstream along +x. With z upward, downward flow is negative.
    Neither the bound segment A->B nor a far-wake closing segment is included.
    """
    A_points = np.asarray(A_points, dtype=float)
    B_points = np.asarray(B_points, dtype=float)
    control_points = np.asarray(control_points, dtype=float)
    Gamma = np.asarray(Gamma, dtype=float)
    if not (A_points.shape == B_points.shape == control_points.shape):
        raise ValueError("Geometry arrays must have matching shape.")
    if control_points.ndim != 3 or control_points.shape[-1] != 3:
        raise ValueError("Geometry arrays must have shape (n_chord, n_span, 3).")
    if Gamma.shape != control_points.shape[:2]:
        raise ValueError("Gamma must have shape (n_chord, n_span) matching the geometry.")
    if not all(np.all(np.isfinite(points)) for points in (A_points, B_points, control_points)):
        raise ValueError("Geometry arrays must be finite.")
    if not np.all(np.isfinite(Gamma)):
        raise ValueError("Gamma must be finite.")
    wake_length = np.asarray(wake_length, dtype=float)
    if wake_length.shape != () or not np.isfinite(wake_length) or wake_length <= 0:
        raise ValueError("wake_length must be a finite positive scalar.")
    A_flat = A_points.reshape(-1, 3, order="C")
    B_flat = B_points.reshape(-1, 3, order="C")
    control_flat = control_points.reshape(-1, 3, order="C")
    gamma_flat = Gamma.ravel(order="C")
    velocity = np.zeros_like(control_flat)
    offset = np.array([float(wake_length), 0.0, 0.0])
    for i, point in enumerate(control_flat):
        for j, gamma in enumerate(gamma_flat):
            if gamma == 0.0:
                continue
            velocity[i] += gamma * (
                vortex_segment_velocity(A_flat[j] + offset, A_flat[j], point, gamma=1.0)
                + vortex_segment_velocity(B_flat[j], B_flat[j] + offset, point, gamma=1.0)
            )
    velocity = velocity.reshape(control_points.shape, order="C")
    return velocity[..., 2].copy()


def induced_angle_from_downwash(downwash, V_inf):
    """Return small-angle induced incidence -downwash/V_inf in radians.

    Preserve the downwash array shape. Global z is upward, so negative
    downwash produces a positive induced angle. V_inf is a positive scalar.
    """
    downwash = np.asarray(downwash, dtype=float)
    V_inf = np.asarray(V_inf, dtype=float)
    if not np.all(np.isfinite(downwash)):
        raise ValueError("downwash must contain finite values.")
    if V_inf.shape != () or not np.isfinite(V_inf) or V_inf <= 0:
        raise ValueError("V_inf must be a finite, strictly positive scalar.")
    return -downwash / V_inf


def _positive_finite_scalar(value, name):
    value = np.asarray(value, dtype=float)
    if value.shape != () or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite, strictly positive scalar.")
    return float(value)


def induced_drag_from_downwash(Gamma, downwash, panel_widths, rho):
    """Return panel and total near-field induced drag -rho*w*Gamma*delta_y.

    Supply trailing-vortex-only global-z downwash (negative downward).
    Gamma, downwash and positive spanwise widths must all have the same
    (n_chord, n_span) shape. Widths are abs(B_y - A_y), not segment lengths.
    Panel indexing and total summation use C-order k = i*n_span + j.
    """
    Gamma = np.asarray(Gamma, dtype=float)
    downwash = np.asarray(downwash, dtype=float)
    panel_widths = np.asarray(panel_widths, dtype=float)
    if Gamma.ndim != 2 or not (Gamma.shape == downwash.shape == panel_widths.shape):
        raise ValueError("Gamma, downwash and panel_widths must have matching (n_chord, n_span) shapes.")
    for name, values in (("Gamma", Gamma), ("downwash", downwash), ("panel_widths", panel_widths)):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} must contain finite values.")
    if np.any(panel_widths <= 0):
        raise ValueError("panel_widths must be strictly positive.")
    rho = _positive_finite_scalar(rho, "rho")
    panel_drag = -rho * downwash * Gamma * panel_widths
    return panel_drag, float(np.sum(panel_drag.ravel(order="C")))


def induced_drag_coefficient(Di, rho, V_inf, S_ref):
    """Return CDi = Di / (0.5*rho*V_inf**2*S_ref)."""
    Di = np.asarray(Di, dtype=float)
    if Di.shape != () or not np.isfinite(Di):
        raise ValueError("Di must be a finite scalar.")
    rho = _positive_finite_scalar(rho, "rho")
    V_inf = _positive_finite_scalar(V_inf, "V_inf")
    S_ref = _positive_finite_scalar(S_ref, "S_ref")
    return float(Di) / (0.5 * rho * V_inf**2 * S_ref)


def span_efficiency(CL, CDi, aspect_ratio):
    """Return scalar CL**2/(pi*aspect_ratio*CDi), without clamping to one."""
    CL = np.asarray(CL, dtype=float)
    if CL.shape != () or not np.isfinite(CL):
        raise ValueError("CL must be a finite scalar.")
    CDi = _positive_finite_scalar(CDi, "CDi")
    aspect_ratio = _positive_finite_scalar(aspect_ratio, "aspect_ratio")
    return float(CL)**2 / (np.pi * aspect_ratio * CDi)


def build_vortex_ring_influence_matrix(
    A_points,
    B_points,
    C_points,
    D_points,
    control_points,
    wake_length,
):
    n_chord, n_span, _ = control_points.shape
    n_panels = n_chord * n_span
    matrix = np.zeros((n_panels, n_panels))

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
                            gamma=1.0,
                        )
                    else:
                        velocity = trailing_edge_vortex_velocity(
                            A_points[i_s, j_s],
                            B_points[i_s, j_s],
                            C_points[i_s, j_s],
                            D_points[i_s, j_s],
                            P,
                            gamma=1.0,
                            wake_length=wake_length,
                        )

                    matrix[row, col] = velocity[2]

    return matrix


def build_rhs(V_inf, alpha_deg, n_points):
    alpha = np.deg2rad(alpha_deg)
    b_value = -V_inf * np.sin(alpha)

    return np.full(n_points, b_value)


def _validated_panel_normals(normals):
    """Validate unit normals without changing their magnitude or orientation."""
    normals = np.asarray(normals, dtype=float)
    if normals.ndim != 3 or normals.shape[-1] != 3:
        raise ValueError("Normals must have shape (n_chord, n_span, 3).")
    if not np.all(np.isfinite(normals)):
        raise ValueError("Normals must be finite.")
    magnitude = np.linalg.norm(normals, axis=-1)
    if np.any(magnitude == 0):
        raise ValueError("Normals must have nonzero magnitude.")
    if not np.allclose(magnitude, 1.0, rtol=1e-7, atol=1e-12):
        raise ValueError("Normals must have unit magnitude; no renormalization is performed.")
    return normals


def build_general_rhs(V_inf_vector, normals):
    """Return -V_inf_vector dot normal in C-order (k = i*n_span + j).

    Coordinates are downstream x, spanwise y, vertical z. The legacy flat-wing
    AoA convention uses [V*cos(alpha), 0, V*sin(alpha)] (alpha in radians).
    Normals must be finite unit vectors with shape (n_chord, n_span, 3).
    """
    velocity = np.asarray(V_inf_vector, dtype=float)
    if velocity.shape != (3,):
        raise ValueError("Freestream velocity vector must have shape (3,).")
    if not np.all(np.isfinite(velocity)):
        raise ValueError("Freestream velocity vector must be finite.")
    normals = _validated_panel_normals(normals)
    return -(normals.reshape(-1, 3, order="C") @ velocity)


def solve_circulation(matrix, rhs):
    circulation = np.linalg.solve(matrix, rhs)

    return circulation


def lift_distribution(gamma, rho, V_inf):
    lift_per_span = rho * V_inf * gamma

    return lift_per_span


def spanwise_lift_distribution(Gamma, rho, V_inf):
    """Return (Gamma_span, L_prime), each with shape (n_span,).

    Gamma has shape (n_chord, n_span): i indexes chordwise rows and j
    indexes spanwise panels, consistent with C-order k = i*n_span + j.
    Sum over i without reordering j; Kutta-Joukowski gives lift per unit
    span L_prime = rho * V_inf * Gamma_span.
    """
    Gamma = np.asarray(Gamma, dtype=float)
    if Gamma.ndim != 2 or not np.all(np.isfinite(Gamma)):
        raise ValueError("Gamma must be a finite 2D array with shape (n_chord, n_span).")
    rho = _positive_finite_scalar(rho, "rho")
    V_inf = _positive_finite_scalar(V_inf, "V_inf")
    Gamma_span = np.sum(Gamma, axis=0)
    return Gamma_span, rho * V_inf * Gamma_span


def total_lift(lift_per_span, span):
    n_span = len(lift_per_span)
    dy = span / n_span
    lift = np.sum(lift_per_span * dy)

    return lift


def lift_coefficient(lift, rho, V_inf, span, chord):
    S = span * chord
    q_inf = 0.5 * rho * V_inf**2
    CL = lift / (q_inf * S)

    return CL


def mcbain_total_lift(gamma, A_points, B_points, rho, V_inf):
    """Sum lift from all bound segments, with k = i * n_span + j."""
    A_points = np.asarray(A_points)
    B_points = np.asarray(B_points)
    gamma = np.asarray(gamma)
    if A_points.shape != B_points.shape:
        raise ValueError("A_points and B_points must have matching shape.")
    if A_points.ndim != 3 or A_points.shape[2] != 3:
        raise ValueError("Geometry arrays must have shape (n_chord, n_span, 3).")

    n_chord, n_span, _ = A_points.shape
    if gamma.shape not in ((n_chord * n_span,), (n_chord, n_span)):
        raise ValueError(
            "gamma must have shape (n_panels,) or (n_chord, n_span) "
            "matching the geometry."
        )

    lengths = np.abs(B_points[:, :, 1] - A_points[:, :, 1])
    return float(
        rho * V_inf * np.sum(gamma.ravel(order="C") * lengths.ravel(order="C"))
    )


def mcbain_lift_coefficient(lift, rho, V_inf, reference_area):
    """Compute McBain CL using an explicitly supplied reference area."""
    q_inf = 0.5 * rho * V_inf**2
    return lift / (q_inf * reference_area)


def effective_chordwise_circulation(gamma_matrix):
    delta_gamma = np.empty_like(gamma_matrix)
    delta_gamma[0] = gamma_matrix[0]
    delta_gamma[1:] = gamma_matrix[1:] - gamma_matrix[:-1]

    return delta_gamma


def vortex_ring_lift_distribution(gamma_matrix, rho, V_inf):
    lift_per_span = rho * V_inf * gamma_matrix[-1]

    return lift_per_span
