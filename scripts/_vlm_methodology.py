"""Shared presentation utilities for the five independent methodology scripts."""

import argparse
from pathlib import Path

import matplotlib
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from src.geometry import swept_tapered_twisted_cambered_horseshoe_geometry


def final_geometry():
    """Return the exact final 6x60 showcase geometry, without an aerodynamic solve."""
    return swept_tapered_twisted_cambered_horseshoe_geometry(
        span=10.0, root_chord=1.6, taper_ratio=0.4,
        leading_edge_sweep_deg=20.0, tip_twist_deg=-3.0,
        n_chord=6, n_span=60, camber=lambda xi: 0.04 * xi * (1 - xi))


def setup(description, argv):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--interactive", action="store_true",
                        help="Rotate/zoom/pan in Matplotlib; report camera on mouse release; do not save.")
    args = parser.parse_args(argv)
    matplotlib.use("TkAgg" if args.interactive else "Agg")
    import matplotlib.pyplot as plt
    return args, plt


STYLE = {"font.size": 10, "axes.labelsize": 11, "axes.titlesize": 15,
         "xtick.labelsize": 9, "ytick.labelsize": 9, "figure.facecolor": "white"}


def finish(fig, ax, plt, interactive, filename):
    def camera(event=None):
        if event is None or event.name == "close_event" or event.inaxes is ax:
            print(f"Current camera:\nelev = {ax.elev}\nazim = {ax.azim}\nroll = {ax.roll}", flush=True)
    if interactive:
        camera()
        fig.canvas.mpl_connect("button_release_event", camera)
        fig.canvas.mpl_connect("close_event", camera)
        plt.show(block=True)
    else:
        output = Path(__file__).resolve().parents[1] / "figures" / filename
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.18,
                    bbox_extra_artists=(*fig.texts, *fig.legends,
                                        ax.xaxis.label, ax.yaxis.label, ax.zaxis.label))
        print(f"Saved: {output}")
    plt.close(fig)


def triad(ax, lower, upper):
    origin = np.array([lower[0] + 0.1, lower[1] + 0.25, lower[2] + 0.05])
    length = 0.5
    for direction, label in zip(np.eye(3), ("x", "y", "z")):
        ax.quiver(*origin, *(direction * length), color="0.25", linewidth=0.8,
                  arrow_length_ratio=0.2, normalize=False, zorder=10)
        ax.text(*(origin + direction * length * 1.15), label, fontsize=9, zorder=11)


def wing_figure(kind, argv=None):
    titles = {"geometry": "Wing Geometry", "panels": "VLM Panel Discretization",
              "horseshoes": "Horseshoe Vortex Discretization",
              "wake": "3D Vortex Lattice and Prescribed Wake"}
    filenames = {"geometry": "wing_geometry.png", "panels": "vlm_panel_discretization.png",
                 "horseshoes": "horseshoe_vortex_discretization.png",
                 "wake": "vlm_lattice_and_wake.png"}
    args, plt = setup(titles[kind], argv)
    A, B, controls, normals, corners = final_geometry()
    vortex_layers = kind in ("horseshoes", "wake")
    # These are visible portions of the solver's A+[100,0,0], B+[100,0,0]
    # prescribed legs (see src.vortex.horseshoe_velocity), not a new wake model.
    solver_wake_length = 100.0
    displayed_wake_length = 1.6 if kind == "horseshoes" else 4.0
    offset = np.array([min(displayed_wake_length, solver_wake_length), 0.0, 0.0])
    points = corners.reshape(-1, 3)
    if vortex_layers:
        points = np.concatenate((points, (A + offset).reshape(-1, 3), (B + offset).reshape(-1, 3)))
    lower, upper = points.min(axis=0) - [0.35, 0.35, 0.25], points.max(axis=0) + [0.35, 0.35, 0.35]
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(12, 7))
        ax = fig.add_subplot(111, projection="3d", proj_type="persp", computed_zorder=False)
        ax.add_collection3d(Poly3DCollection(
            corners.reshape(-1, 4, 3), facecolors="#eef1f3",
            edgecolors="0.6" if kind == "geometry" else "0.3",
            linewidths=0.16 if kind == "geometry" else 0.3, zorder=2, clip_on=False))
        if vortex_layers:
            for a, b in zip(A.reshape(-1, 3), B.reshape(-1, 3)):
                ax.plot(*np.stack((a, b)).T, color="#b72435", linewidth=0.42, zorder=6, clip_on=False)
                for start in (a, b):
                    ax.plot(*np.stack((start, start + offset)).T, color="#2864a0",
                            linewidth=0.22, alpha=0.4, zorder=3, clip_on=False)
            ax.scatter(*controls.reshape(-1, 3).T, color="#00896c", s=2.0,
                       linewidths=0, depthshade=False, zorder=7, clip_on=False)
            handles = [Line2D([], [], color="#b72435", lw=0.8, label="Bound vortex"),
                       Line2D([], [], color="#2864a0", lw=0.6, label="Trailing vortex"),
                       Line2D([], [], color="#00896c", marker="o", ls="", markersize=3, label="Control point")]
            fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.93), ncol=3, frameon=False)
            fig.text(0.5, 0.035, f"Straight prescribed +x wake; first {displayed_wake_length:g} m shown of 100 m solver legs.",
                     ha="center", fontsize=9, color="0.4")
            print(f"All {A.shape[0] * A.shape[1]} bound vortices and both straight legs shown; display length {displayed_wake_length} m.")
        if kind == "panels":
            nx, ny = controls.shape[:2]
            fig.text(0.78, 0.86, f"$N_x = {nx}$\n$N_y = {ny}$\nPanels = {nx * ny}", fontsize=10,
                     bbox=dict(facecolor="white", edgecolor="0.8", boxstyle="round,pad=0.4", linewidth=0.6))
        ax.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]), zlim=(lower[2], upper[2]),
               xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")
        ax.set_box_aspect(upper - lower, zoom=1.15)
        ax.view_init(elev=52 if not vortex_layers else 40, azim=-115 if not vortex_layers else -135)
        ax.set_zticks([0])
        ax.xaxis.labelpad, ax.yaxis.labelpad, ax.zaxis.labelpad = 12, 16, 10
        ax.grid(False)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.fill = False
            axis.pane.set_edgecolor("0.93")
        triad(ax, lower, upper)
        fig.suptitle(titles[kind], fontsize=15)
        ax.set_position([0.1, 0.14, 0.8, 0.7])
        ax.set_anchor("C")
        finish(fig, ax, plt, args.interactive, filenames[kind])


def convention_figure(argv=None):
    args, plt = setup("VLM Panel Convention", argv)
    A, B, controls, normals, corners = final_geometry()
    i, j = 2, 44
    panel, a, b, cp = corners[i, j], A[i, j], B[i, j], controls[i, j]
    # Translate only: all four corners, bound endpoints and control point retain
    # their exact relative 3D positions. No flat-panel reconstruction is used.
    origin = panel[0]
    panel, a, b, cp = panel - origin, a - origin, b - origin, cp - origin
    le, te = 0.5 * (panel[0] + panel[3]), 0.5 * (panel[1] + panel[2])
    chord_length = np.linalg.norm(te - le)
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection="3d", proj_type="persp", computed_zorder=False)
        ax.add_collection3d(Poly3DCollection([panel], facecolors="#eef1f3", edgecolors="0.25", linewidths=0.9, zorder=2))
        ax.plot(*np.stack((a, b)).T, color="#b72435", linewidth=2, zorder=5, label="Bound vortex")
        ax.scatter(*cp, color="#00896c", s=25, depthshade=False, zorder=7)
        ax.plot(*np.stack((le, te)).T, color="0.45", linewidth=0.8, linestyle="--", zorder=4)
        ax.text(*le, "Leading edge\n(panel)", ha="right", va="top", fontsize=10, zorder=10)
        ax.text(*te, "Trailing edge\n(panel)", ha="left", va="top", fontsize=10, zorder=10)
        ax.text(*a, "Bound vortex\n1/4 chord", ha="right", va="bottom", fontsize=10, color="#9d1f2d", zorder=10)
        ax.text(*cp, "Control point\n3/4 chord", ha="left", va="bottom", fontsize=10, color="#006e56", zorder=10)
        # Dimension line offset in y is annotation only, anchored to actual edges.
        shift = np.array([0, -0.06, 0])
        dim_start, dim_end = le + shift, te + shift
        ax.quiver(*dim_start, *(dim_end - dim_start), color="0.3", linewidth=0.8, arrow_length_ratio=0.08)
        ax.quiver(*dim_end, *(dim_start - dim_end), color="0.3", linewidth=0.8, arrow_length_ratio=0.08)
        ax.text(*(0.5 * (dim_start + dim_end)), "Panel chord", ha="center", va="top", fontsize=10)
        lower = panel.min(axis=0) - [0.06, 0.09, 0.03]
        upper = panel.max(axis=0) + [0.08, 0.06, 0.04]
        ax.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]), zlim=(lower[2], upper[2]),
               xlabel="Local x [m]", ylabel="Local y [m]", zlabel="Local z [m]")
        ax.set_box_aspect(upper - lower)
        ax.view_init(elev=65, azim=-70)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.grid(False)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.fill = False
            axis.pane.set_edgecolor("0.94")
        fig.suptitle("VLM Panel Convention", fontsize=15)
        fig.text(0.5, 0.06, f"Actual panel (i={i}, j={j}); fractions refer to local panel chord. Zero-thickness camber surface.",
                 ha="center", fontsize=9, color="0.4")
        ax.set_position([0.12, 0.15, 0.76, 0.7])
        finish(fig, ax, plt, args.interactive, "vlm_panel_convention.png")
