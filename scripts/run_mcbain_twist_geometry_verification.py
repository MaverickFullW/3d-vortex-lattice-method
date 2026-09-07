"""Verify the supplied McBain meshwing equations; no aerodynamic validation."""

from pathlib import Path
import sys

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.geometry import swept_tapered_twisted_horseshoe_geometry


def run_verification():
    span, AR, taper = 1.0, 6.0, 0.5
    sweep_deg, tip_twist_deg = 15.0, 7.5
    n_chord, n_span = 4, 20
    root_chord = 2 * span / (AR * (1 + taper))

    # Independent analytic reference: no project geometry helpers are used.
    def point(xi, y):
        fraction = 2 * abs(y) / span
        chord = root_chord * (1 - (1 - taper) * fraction)
        angle = np.deg2rad(tip_twist_deg * fraction)
        return np.array([
            abs(y) * np.tan(np.deg2rad(sweep_deg)) + xi * chord * np.cos(angle),
            y, -xi * chord * np.sin(angle),
        ])

    ys = np.linspace(-span / 2, span / 2, n_span + 1)
    lattice = np.array([[point(i / n_chord, y) for y in ys]
                        for i in range(n_chord + 1)])
    reference = {name: np.empty((n_chord, n_span, 3))
                 for name in ("A", "B", "control", "normal")}
    reference["corner"] = np.empty((n_chord, n_span, 4, 3))
    for i in range(n_chord):
        for j in range(n_span):
            corners = np.array([lattice[i, j], lattice[i + 1, j],
                                lattice[i + 1, j + 1], lattice[i, j + 1]])
            reference["corner"][i, j] = corners
            reference["A"][i, j] = point((i + 0.25) / n_chord, ys[j])
            reference["B"][i, j] = point((i + 0.25) / n_chord, ys[j + 1])
            reference["control"][i, j] = point((i + 0.75) / n_chord,
                                               (ys[j] + ys[j + 1]) / 2)
            # Crossed diagonals independently give the same direction as
            # bilinear center tangents for a possibly nonplanar quadrilateral.
            normal = np.cross(corners[2] - corners[0], corners[3] - corners[1])
            reference["normal"][i, j] = normal / np.linalg.norm(normal)

    generated = dict(zip(("A", "B", "control", "normal", "corner"),
        swept_tapered_twisted_horseshoe_geometry(
            span, root_chord, taper, sweep_deg, tip_twist_deg, n_chord, n_span
        )))
    errors = {name: float(np.max(abs(generated[name] - reference[name])))
              for name in reference}
    # Recover physical section data from the generated corners, not inputs.
    corners = generated["corner"]
    leading = np.concatenate((corners[0, :, 0], corners[0, -1:, 3]))
    trailing = np.concatenate((corners[-1, :, 1], corners[-1, -1:, 2]))
    chords = trailing - leading
    return {
        "root_chord": np.linalg.norm(chords[10]),
        "tip_chord": np.linalg.norm(chords[-1]),
        "station_chords": np.linalg.norm(chords, axis=1),
        "station_twist_deg": np.rad2deg(np.arctan2(-chords[:, 2], chords[:, 0])),
        "leading_edge_sweep_deg": np.rad2deg(np.arctan2(
            leading[-1, 0] - leading[10, 0], leading[-1, 1] - leading[10, 1])),
        "errors": errors,
        "generated": generated,
        "reference": reference,
    }


def main():
    result = run_verification()
    print("McBain meshwing geometry verification (4x20; no dihedral or camber)")
    print(f"Root chord = {result['root_chord']:.12f}")
    print(f"Tip chord = {result['tip_chord']:.12f}")
    print(f"Tip twist [deg] = {result['station_twist_deg'][-1]:.12f}")
    print(f"Leading-edge sweep [deg] = {result['leading_edge_sweep_deg']:.12f}")
    for name, error in result["errors"].items():
        print(f"Maximum {name} error = {error:.12e}")


if __name__ == "__main__":
    main()
