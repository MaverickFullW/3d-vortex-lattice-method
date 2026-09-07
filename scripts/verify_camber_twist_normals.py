"""Independent corner-based normal verification; geometry only."""

from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.geometry import swept_tapered_twisted_cambered_horseshoe_geometry


def reference_normal(corners):
    """Cross the two quad diagonals, oriented for positive z in the flat limit.

    Corners are upstream-left, downstream-left, downstream-right,
    upstream-right. The diagonals connect points on the bilinear surface
    through its center; their cross gives the center tangent-plane normal,
    including for nonplanar quads. This differs from the implementation's
    averaged-edge construction and uses only corner coordinates.
    """
    upstream_left, downstream_left, downstream_right, upstream_right = corners
    diagonal_forward = downstream_right - upstream_left
    diagonal_backward = upstream_right - downstream_left
    normal = np.cross(diagonal_forward, diagonal_backward)
    return normal / np.linalg.norm(normal)


def main():
    # Four chordwise panels and two panels per semispan. The selected outer
    # panel touches the 30-degree tip; twist varies across its spanwise width.
    i, j = 1, 3
    *_, normals, corners = swept_tapered_twisted_cambered_horseshoe_geometry(
        span=2.0, root_chord=1.0, taper_ratio=1.0,
        leading_edge_sweep_deg=0.0, tip_twist_deg=30.0,
        n_chord=4, n_span=4, camber=lambda xi: 0.1 * xi * (1 - xi),
    )
    panel = corners[i, j]
    actual = normals[i, j]
    reference = reference_normal(panel)
    difference = abs(actual - reference)
    dot = np.dot(actual, reference)
    with np.printoptions(precision=15):
        print(f"Selected panel indices (chordwise, spanwise): ({i}, {j})")
        print("Outer edge at y=1 has local twist 30 deg; inner edge has 15 deg.")
        print("Corners (upstream-left, downstream-left, downstream-right, upstream-right):")
        print(panel)
        print("Geometry normal:", actual)
        print("Independent reference normal:", reference)
        print(f"Norm of geometry normal: {np.linalg.norm(actual):.16f}")
        print(f"Norm of reference normal: {np.linalg.norm(reference):.16f}")
        print("Absolute component-wise difference:", difference)
        print(f"Maximum absolute difference: {difference.max():.12e}")
        print(f"Dot product: {dot:.16f}")

    *_, flat_normals, flat_corners = swept_tapered_twisted_cambered_horseshoe_geometry(
        span=2.0, root_chord=1.0, taper_ratio=1.0,
        leading_edge_sweep_deg=0.0, tip_twist_deg=0.0,
        n_chord=4, n_span=4, camber=lambda xi: 0.0,
    )
    flat_reference = reference_normal(flat_corners[i, j])
    print("Flat geometry normal:", flat_normals[i, j])
    print("Flat reference normal:", flat_reference)
    np.testing.assert_allclose(actual, reference, rtol=0, atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(actual), 1, rtol=0, atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(reference), 1, rtol=0, atol=1e-14)
    np.testing.assert_allclose(flat_normals, np.broadcast_to([0, 0, 1], flat_normals.shape),
                               rtol=0, atol=1e-14)
    np.testing.assert_allclose(flat_reference, [0, 0, 1], rtol=0, atol=1e-14)
    assert dot > 0, "Cambered/twisted normal orientation is flipped."
    print("PASS: normals agree, have unit length, and retain the expected orientation.")


if __name__ == "__main__":
    main()
