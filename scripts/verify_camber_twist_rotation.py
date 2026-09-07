"""Check rigid rotation of a cambered tip section; geometry only."""

from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.geometry import swept_tapered_twisted_cambered_horseshoe_geometry


def main():
    # Full span 2 places the positive tip at y=1, with local twist 30 degrees.
    *_, corners = swept_tapered_twisted_cambered_horseshoe_geometry(
        span=2.0, root_chord=1.0, taper_ratio=1.0,
        leading_edge_sweep_deg=0.0, tip_twist_deg=30.0,
        n_chord=4, n_span=2, camber=lambda xi: 0.1 * xi * (1 - xi),
    )
    # Upstream-right corner of each chordwise panel, then trailing-edge tip.
    tip = np.concatenate((corners[:, -1, 3], corners[-1:, -1, 2]))
    actual = tip[:, [0, 2]]

    # Independent analytical section calculation; no geometry helpers.
    xi = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    eta = 0.1 * xi * (1 - xi)
    theta = np.pi / 6
    expected = np.column_stack((xi * np.cos(theta) + eta * np.sin(theta),
                                -xi * np.sin(theta) + eta * np.cos(theta)))
    errors = np.linalg.norm(actual - expected, axis=1)
    print("30-degree tip section; absolute coordinate error = Euclidean x-z distance")
    print("  xi      eta    x_expected   z_expected   x_geometry   z_geometry    abs_error")
    for t, h, e, a, error in zip(xi, eta, expected, actual, errors):
        print(f"{t:5.2f}  {h:7.5f}  {e[0]:12.9f} {e[1]:12.9f}"
              f" {a[0]:12.9f} {a[1]:12.9f}  {error:.3e}")

    chord = np.column_stack((xi * np.cos(theta), -xi * np.sin(theta)))
    positive_normal = np.array([np.sin(theta), np.cos(theta)])
    chord_direction = np.array([np.cos(theta), -np.sin(theta)])
    displacement = actual - chord
    signed_camber = displacement @ positive_normal
    print(f"\nZero-camber chord at xi=0.5: x={chord[2, 0]:.9f}, z={chord[2, 1]:.9f}")
    print(f"Signed normal camber offset at xi=0.5: {signed_camber[2]:.9f}")
    print(f"Maximum coordinate error: {errors.max():.3e}")
    np.testing.assert_allclose(tip[:, 1], 1.0, rtol=0, atol=1e-14)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-14)
    np.testing.assert_allclose(signed_camber, eta, rtol=0, atol=1e-14)
    np.testing.assert_allclose(displacement @ chord_direction, 0, rtol=0, atol=1e-14)
    assert np.all(signed_camber[1:-1] > 0)
    # Inverse rotation recovers the original mean line and its curvature sign.
    np.testing.assert_allclose(actual @ chord_direction, xi, rtol=0, atol=1e-14)
    np.testing.assert_allclose(np.diff(signed_camber, n=2), np.diff(eta, n=2),
                               rtol=0, atol=1e-14)
    print("PASS: rigid rotation preserves positive camber and curvature sign.")


if __name__ == "__main__":
    main()
