import numpy as np

def vortex_segment_velocity(A, B, P, gamma):
    r1 = P - A
    r2 = P - B
    r0 = B - A

    cross = np.cross(r1, r2)

    r1_norm = np.linalg.norm(r1)
    r2_norm = np.linalg.norm(r2)
    cross_norm_sq = np.dot(cross, cross)
    if cross_norm_sq < 1e-12:
        return np.zeros(3)

    direction_term = r1 / r1_norm - r2 / r2_norm
    strength_term = np.dot(r0, direction_term)

    velocity = (
        gamma / (4.0 * np.pi)
        * cross / cross_norm_sq
        * strength_term
    )

    return velocity

def horseshoe_velocity(A, B, P, gamma, wake_length):
    A_far = A + np.array([wake_length, 0.0, 0.0])
    B_far = B + np.array([wake_length, 0.0, 0.0])

    v_left = vortex_segment_velocity(A_far, A, P, gamma)
    v_bound = vortex_segment_velocity(A, B, P, gamma)
    v_right = vortex_segment_velocity(B, B_far, P, gamma)

    velocity = v_left + v_bound + v_right

    return velocity


def vortex_ring_velocity(A, B, C, D, P, gamma):
    """Induced velocity along the closed path A -> B -> C -> D -> A."""
    return (
        vortex_segment_velocity(A, B, P, gamma)
        + vortex_segment_velocity(B, C, P, gamma)
        + vortex_segment_velocity(C, D, P, gamma)
        + vortex_segment_velocity(D, A, P, gamma)
    )


def trailing_edge_vortex_velocity(A, B, C, D, P, gamma, wake_length):
    """Extend the ring's aft edge along +x, closing it at the far wake."""
    C_far = C + np.array([wake_length, 0.0, 0.0])
    D_far = D + np.array([wake_length, 0.0, 0.0])

    # A -> B -> C -> C_far -> D_far -> D -> A.
    return (
        vortex_segment_velocity(A, B, P, gamma)
        + vortex_segment_velocity(B, C, P, gamma)
        + vortex_segment_velocity(C, C_far, P, gamma)
        + vortex_segment_velocity(C_far, D_far, P, gamma)
        + vortex_segment_velocity(D_far, D, P, gamma)
        + vortex_segment_velocity(D, A, P, gamma)
    )
