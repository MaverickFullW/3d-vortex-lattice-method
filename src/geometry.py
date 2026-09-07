import numpy as np


def spanwise_edges(span, n_span):
    return np.linspace(-span / 2.0, span / 2.0, n_span + 1)


def rectangular_wing_panels(span, chord, n_span):
    edges = spanwise_edges(span, n_span)

    A_points = np.zeros((n_span, 3))
    B_points = np.zeros((n_span, 3))
    control_points = np.zeros((n_span, 3))

    for i in range(n_span):
        y_left = edges[i]
        y_right = edges[i + 1]

        A_points[i] = [chord / 4.0, y_left, 0.0]
        B_points[i] = [chord / 4.0, y_right, 0.0]
        control_points[i] = [
            3.0 * chord / 4.0,
            (y_left + y_right) / 2.0,
            0.0,
        ]

    return A_points, B_points, control_points


def rectangular_horseshoe_geometry(span, chord, n_chord, n_span):
    dx = chord / n_chord
    y_grid = spanwise_edges(span, n_span)

    shape = (n_chord, n_span, 3)
    A_points = np.zeros(shape)
    B_points = np.zeros(shape)
    control_points = np.zeros(shape)

    for i in range(n_chord):
        x_leading = i * dx
        x_bound = x_leading + 0.25 * dx
        x_control = x_leading + 0.75 * dx

        for j in range(n_span):
            y_left = y_grid[j]
            y_right = y_grid[j + 1]

            A_points[i, j] = [x_bound, y_left, 0.0]
            B_points[i, j] = [x_bound, y_right, 0.0]
            control_points[i, j] = [
                x_control,
                (y_left + y_right) / 2.0,
                0.0,
            ]

    return A_points, B_points, control_points


def swept_rectangular_horseshoe_geometry(span, chord, sweep_deg, n_chord, n_span):
    """Build flat McBain panels with x_LE = |y| tan(sweep).

    Arrays use (chordwise, spanwise, xyz), so k = i * n_span + j.
    """
    y_grid = spanwise_edges(span, n_span)
    xi_grid = np.linspace(0.0, 1.0, n_chord + 1)
    sweep_tangent = np.tan(np.deg2rad(sweep_deg))

    shape = (n_chord, n_span, 3)
    A_points = np.zeros(shape)
    B_points = np.zeros(shape)
    control_points = np.zeros(shape)

    for i in range(n_chord):
        xi_left = xi_grid[i]
        dxi = xi_grid[i + 1] - xi_left
        xi_bound = xi_left + 0.25 * dxi
        xi_control = xi_left + 0.75 * dxi

        for j in range(n_span):
            y_left = y_grid[j]
            y_right = y_grid[j + 1]
            y_mid = 0.5 * (y_left + y_right)
            A_points[i, j] = [
                abs(y_left) * sweep_tangent + xi_bound * chord, y_left, 0.0
            ]
            B_points[i, j] = [
                abs(y_right) * sweep_tangent + xi_bound * chord, y_right, 0.0
            ]
            control_points[i, j] = [
                abs(y_mid) * sweep_tangent + xi_control * chord, y_mid, 0.0
            ]

    return A_points, B_points, control_points


def swept_tapered_horseshoe_geometry(
    span, root_chord, taper_ratio, leading_edge_sweep_deg, n_chord, n_span
):
    """Build flat McBain panels with linear taper and x_LE = |y| tan(sweep).

    Panel edges follow constant fractions of the local chord. Bound endpoints
    lie at quarter-panel chord at each spanwise edge; controls lie at
    three-quarter-panel chord at the spanwise midpoint. As in the rectangular
    implementation, midpoint geometry is evaluated locally, including when an
    odd spanwise panel count places a panel across the centerline.

    Return arrays shaped (n_chord, n_span, 3), with C-order index
    k = i * n_span + j. Sweep is the leading-edge angle in degrees.
    """
    tip_chord = taper_ratio * root_chord
    y_grid = spanwise_edges(span, n_span)
    xi_grid = np.linspace(0.0, 1.0, n_chord + 1)
    sweep_tangent = np.tan(np.deg2rad(leading_edge_sweep_deg))

    def local_chord(y):
        return root_chord - (root_chord - tip_chord) * 2.0 * abs(y) / span

    shape = (n_chord, n_span, 3)
    A_points = np.zeros(shape)
    B_points = np.zeros(shape)
    control_points = np.zeros(shape)

    for i in range(n_chord):
        xi_left = xi_grid[i]
        dxi = xi_grid[i + 1] - xi_left
        xi_bound = xi_left + 0.25 * dxi
        xi_control = xi_left + 0.75 * dxi

        for j in range(n_span):
            y_left = y_grid[j]
            y_right = y_grid[j + 1]
            y_mid = 0.5 * (y_left + y_right)
            A_points[i, j] = [
                abs(y_left) * sweep_tangent + xi_bound * local_chord(y_left),
                y_left, 0.0,
            ]
            B_points[i, j] = [
                abs(y_right) * sweep_tangent + xi_bound * local_chord(y_right),
                y_right, 0.0,
            ]
            control_points[i, j] = [
                abs(y_mid) * sweep_tangent + xi_control * local_chord(y_mid),
                y_mid, 0.0,
            ]

    return A_points, B_points, control_points


def swept_tapered_twisted_horseshoe_geometry(
    span, root_chord, taper_ratio, leading_edge_sweep_deg, tip_twist_deg,
    n_chord, n_span,
):
    """Return (A, B, controls, normals, corners) for a twisted flat planform.

    Positive twist rotates downstream chord toward negative z about the local
    leading edge: (dx, 0, 0) -> (dx*cos(theta), 0, -dx*sin(theta)).
    theta = 2*abs(y)/span * tip_twist_deg; angles are in degrees.

    The first four arrays have shape (n_chord, n_span, 3). Corners have
    shape (n_chord, n_span, 4, 3), ordered upstream-left, downstream-left,
    downstream-right, upstream-right (left means lower y). C-order indexing
    remains k = i*n_span + j.

    Twisted quads can be nonplanar. Normals are the normalized cross product
    of chordwise and spanwise tangents at the bilinear quad's center (averaged
    opposite edges), giving +z for zero twist and mirror-symmetric normals.
    Controls retain the existing local midpoint placement, rotated by the
    local twist; they need not lie on the bilinear surface of a coarse quad.
    """
    A, B, controls = swept_tapered_horseshoe_geometry(
        span, root_chord, taper_ratio, leading_edge_sweep_deg, n_chord, n_span
    )
    sweep_tangent = np.tan(np.deg2rad(leading_edge_sweep_deg))

    def rotate(points):
        leading_x = abs(points[..., 1]) * sweep_tangent
        angle = np.deg2rad(tip_twist_deg) * 2.0 * abs(points[..., 1]) / span
        dx = points[..., 0] - leading_x
        result = points.copy()
        result[..., 0] = leading_x + dx * np.cos(angle)
        result[..., 2] = -dx * np.sin(angle)
        return result

    y = spanwise_edges(span, n_span)
    chord = root_chord - (root_chord - taper_ratio * root_chord) * 2 * abs(y) / span
    xi = np.linspace(0.0, 1.0, n_chord + 1)
    stations = np.zeros((n_chord + 1, n_span + 1, 3))
    stations[..., 0] = abs(y) * sweep_tangent + xi[:, None] * chord
    stations[..., 1] = y
    stations = rotate(stations)
    corners = np.stack((stations[:-1, :-1], stations[1:, :-1],
                        stations[1:, 1:], stations[:-1, 1:]), axis=2)
    p0, p1, p2, p3 = (corners[:, :, k] for k in range(4))
    chordwise = 0.5 * ((p1 - p0) + (p2 - p3))
    spanwise = 0.5 * ((p3 - p0) + (p2 - p1))
    normals = np.cross(chordwise, spanwise)
    magnitude = np.linalg.norm(normals, axis=-1, keepdims=True)
    if np.any(magnitude == 0.0):
        raise ValueError("Degenerate panel geometry has no unit normal.")
    normals /= magnitude
    return rotate(A), rotate(B), rotate(controls), normals, corners


def swept_tapered_twisted_cambered_horseshoe_geometry(
    span, root_chord, taper_ratio, leading_edge_sweep_deg, tip_twist_deg,
    n_chord, n_span, camber,
):
    """Return (A, B, controls, normals, corners) for a cambered mean surface.

    camber(xi) accepts a scalar chord fraction in [0, 1] and returns finite
    dimensionless mean-line height eta = z_c/c, identical at every span station.
    With local chord c and twist theta, the local coordinates are
    x = c*(xi*cos(theta) + eta*sin(theta)) and
    z = c*(-xi*sin(theta) + eta*cos(theta)). Leading-edge sweep offsets x.
    Positive twist thus rotates the downstream chord toward negative z.

    Quarter-panel bound endpoints and three-quarter-panel controls are sampled
    directly on the mean surface; controls use the local spanwise midpoint.
    Normals use averaged opposite corner edges (bilinear center tangents),
    just as in the uncambered twisted geometry, not analytical camber slopes.
    The first four outputs have shape (n_chord, n_span, 3); corners have shape
    (n_chord, n_span, 4, 3), ordered upstream-left, downstream-left,
    downstream-right, upstream-right. C-order index is k = i*n_span + j.
    """
    if not callable(camber):
        raise ValueError("camber must be a callable returning scalar eta = z_c/c.")
    y_edges = spanwise_edges(span, n_span)
    xi_edges = np.linspace(0.0, 1.0, n_chord + 1)
    sweep_tangent = np.tan(np.deg2rad(leading_edge_sweep_deg))

    def section_points(xi, y):
        eta = np.asarray(camber(float(xi)), dtype=float)
        if eta.shape != () or not np.isfinite(eta):
            raise ValueError("camber(xi) must return a finite scalar.")
        fraction = 2.0 * abs(y) / span
        chord = root_chord - (root_chord - taper_ratio * root_chord) * fraction
        theta = np.deg2rad(tip_twist_deg) * fraction
        return np.stack((
            abs(y) * sweep_tangent + chord * (xi * np.cos(theta) + eta * np.sin(theta)),
            y,
            chord * (-xi * np.sin(theta) + eta * np.cos(theta)),
        ), axis=-1)

    stations = np.array([section_points(xi, y_edges) for xi in xi_edges])
    corners = np.stack((stations[:-1, :-1], stations[1:, :-1],
                        stations[1:, 1:], stations[:-1, 1:]), axis=2)
    shape = (n_chord, n_span, 3)
    A, B, controls = (np.empty(shape) for _ in range(3))
    for i in range(n_chord):
        dxi = xi_edges[i + 1] - xi_edges[i]
        bound_xi = xi_edges[i] + 0.25 * dxi
        control_xi = xi_edges[i] + 0.75 * dxi
        A[i] = section_points(bound_xi, y_edges[:-1])
        B[i] = section_points(bound_xi, y_edges[1:])
        controls[i] = section_points(control_xi, 0.5 * (y_edges[:-1] + y_edges[1:]))
    p0, p1, p2, p3 = (corners[:, :, k] for k in range(4))
    chordwise = 0.5 * ((p1 - p0) + (p2 - p3))
    spanwise = 0.5 * ((p3 - p0) + (p2 - p1))
    normals = np.cross(chordwise, spanwise)
    magnitude = np.linalg.norm(normals, axis=-1, keepdims=True)
    if np.any(magnitude == 0.0):
        raise ValueError("Degenerate panel geometry has no unit normal.")
    normals /= magnitude
    return A, B, controls, normals, corners


def rectangular_vortex_ring_geometry(span, chord, n_chord, n_span):
    dx = chord / n_chord
    y_grid = np.linspace(-span / 2.0, span / 2.0, n_span + 1)
    x_vortex = dx * (np.arange(n_chord + 1) + 0.25)

    shape = (n_chord, n_span, 3)
    A_points = np.zeros(shape)
    B_points = np.zeros(shape)
    C_points = np.zeros(shape)
    D_points = np.zeros(shape)
    control_points = np.zeros(shape)

    for i in range(n_chord):
        for j in range(n_span):
            x_forward = x_vortex[i]
            x_aft = x_vortex[i + 1]
            y_left = y_grid[j]
            y_right = y_grid[j + 1]

            A_points[i, j] = [x_forward, y_left, 0.0]
            B_points[i, j] = [x_forward, y_right, 0.0]
            C_points[i, j] = [x_aft, y_right, 0.0]
            D_points[i, j] = [x_aft, y_left, 0.0]
            control_points[i, j] = [
                (x_forward + x_aft) / 2.0,
                (y_left + y_right) / 2.0,
                0.0,
            ]

    return A_points, B_points, C_points, D_points, control_points
