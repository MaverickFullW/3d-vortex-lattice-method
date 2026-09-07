"""Shared case and presentation for the final geometry methodology figures."""

from pathlib import Path

import numpy as np
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

from scripts._vlm_methodology import setup, STYLE
from src.geometry import swept_tapered_twisted_cambered_horseshoe_geometry


SPAN = 10.0
ROOT_CHORD = 1.6
TAPER_RATIO = 0.4
SWEEP_DEG = 20.0
TIP_TWIST_DEG = -3.0
CAMBER_PARAMETER = 0.04
NX, NY = 6, 60
WAKE_LENGTH = 100.0
CAMERA = dict(elev=35.566289331059416, azim=-136.3648226977656,
              roll=4.5291959271707745)
BOX_ASPECT = np.array([1.43639384, 2.24604157, 0.10062595])


def finish(fig, ax, plt, interactive, filename):
    def camera(event=None):
        print(f"Camera:\nelev = {ax.elev}\nazim = {ax.azim}\nroll = {ax.roll}\n"
              f"box_aspect = {ax.get_box_aspect().tolist()}", flush=True)
    camera()
    if interactive:
        fig.canvas.mpl_connect("button_release_event", camera)
        plt.show(block=True)
    else:
        output = Path(__file__).resolve().parents[1] / "figures" / filename
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.08,
                    bbox_extra_artists=(*fig.texts, *fig.legends,
                                        ax.xaxis.label, ax.yaxis.label, ax.zaxis.label))
        print(f"Saved: {output}")
    plt.close(fig)


def geometry():
    return swept_tapered_twisted_cambered_horseshoe_geometry(
        SPAN, ROOT_CHORD, TAPER_RATIO, SWEEP_DEG, TIP_TWIST_DEG, NX, NY,
        camber=lambda xi: CAMBER_PARAMETER * xi * (1 - xi))


def draw_3d(lattice=False, argv=None):
    title = "Horseshoe-Vortex Discretization" if lattice else "3D Wing Geometry and Surface Discretization"
    args, plt = setup(title, argv)
    a, b, controls, _, corners = geometry()
    # Same visible straight +x wake portion and physical limits as the showcase.
    # Only the rendered portion is shortened; the solver uses 100 m legs.
    visible_wake = 2.5 * ROOT_CHORD
    starts = np.stack((a, b), axis=2).reshape(-1, 3)
    ends = starts + [visible_wake, 0, 0]
    points = corners.reshape(-1, 3)
    padding = np.array([0.2, 0.3, 0.22])
    lower, upper = points.min(0) - padding, points.max(0) + padding
    upper[0] = max(upper[0], ends[:, 0].max() + padding[0])
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(10, 5.8))
        ax = fig.add_subplot(111, projection="3d", computed_zorder=False)
        ax.add_collection3d(Poly3DCollection(
            corners.reshape(-1, 4, 3), facecolors="#eff0f1",
            edgecolors="0.45", linewidths=0.22, zorder=2))
        if lattice:
            ax.add_collection3d(Line3DCollection(
                np.stack((a, b), axis=2).reshape(-1, 2, 3),
                colors="#be3944", linewidths=0.3, alpha=0.75, zorder=4))
            ax.add_collection3d(Line3DCollection(
                np.stack((starts, ends), axis=1), colors="#6593b6",
                linewidths=0.2, alpha=0.32, zorder=3))
            ax.scatter(*controls.reshape(-1, 3).T, s=1.5, color="#208769",
                       linewidths=0, depthshade=False, zorder=5)
            fig.legend(handles=[
                Line2D([], [], color="#be3944", lw=0.65, label="Bound vortex"),
                Line2D([], [], color="#6593b6", lw=0.5, label="Trailing vortex"),
                Line2D([], [], color="#208769", marker="o", ls="", markersize=2.5,
                       label="Control point")], loc="upper center", ncol=3,
                       bbox_to_anchor=(0.5, 0.925), frameon=False, fontsize=8)
            print(f"Prescribed +x wake: {visible_wake:g} m visible of {WAKE_LENGTH:g} m solver legs.")
        ax.set(xlim=(lower[0], upper[0]), ylim=(lower[1], upper[1]),
               zlim=(lower[2], upper[2]), xlabel="x [m]", ylabel="y [m]", zlabel="z [m]")
        # Matplotlib normalizes box-aspect inputs. Restore the requested absolute
        # aspect norm through its public zoom argument, preserving all ratios.
        ax.set_box_aspect(BOX_ASPECT.copy())
        zoom = np.linalg.norm(BOX_ASPECT) / np.linalg.norm(ax.get_box_aspect())
        ax.set_box_aspect(BOX_ASPECT.copy(), zoom=zoom)
        np.testing.assert_allclose(ax.get_box_aspect(), BOX_ASPECT, atol=1e-12)
        ax.view_init(**CAMERA)
        ax.set_xticks([0, 2, 4, 6])
        ax.set_yticks([-5, -2.5, 0, 2.5, 5])
        ax.set_zticks([0])
        ax.xaxis.labelpad, ax.yaxis.labelpad, ax.zaxis.labelpad = 7, 9, 5
        ax.grid(False)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.fill = False
            axis.pane.set_edgecolor("0.93")
        ax.set_position([0.10, 0.06, 0.80, 0.84])
        fig.suptitle(title, fontsize=13, y=0.98)
        if not lattice:
            # Sample xi=1/4 with the existing mean-surface geometry function.
            # This auxiliary one-row sampling defines a geometric reference
            # only; the actual aerodynamic panel mesh above remains NX x NY.
            qa, qb, *_ = swept_tapered_twisted_cambered_horseshoe_geometry(
                SPAN, ROOT_CHORD, TAPER_RATIO, SWEEP_DEG, TIP_TWIST_DEG, 1, NY,
                camber=lambda xi: CAMBER_PARAMETER * xi * (1 - xi))
            edges = (np.vstack((corners[0, :, 0], corners[0, -1, 3])),
                     np.vstack((corners[-1, :, 1], corners[-1, -1, 2])),
                     np.vstack((qa[0], qb[0, -1])))
            handles = []
            for edge, style, label in zip(edges, ("-", "--", ":"),
                                          ("Leading Edge", "Trailing Edge", "Quarter-chord")):
                ax.plot(*edge.T, color="0.3", linewidth=0.65, linestyle=style, zorder=4)
                handles.append(Line2D([], [], color="0.3", lw=0.65, ls=style, label=label))
            fig.legend(handles=handles, loc="upper center", ncol=3,
                       bbox_to_anchor=(0.5, 0.925), frameon=False, fontsize=8)
        finish(fig, ax, plt, args.interactive,
               "final_horseshoe_lattice.png" if lattice else "final_wing_geometry.png")


def draw_planform(argv=None):
    args, plt = setup("Wing Planform Geometry", argv)
    from matplotlib.patches import Polygon, Arc

    _, _, controls, _, corners = geometry()
    # Extract the perimeter directly from the solver's quadrilateral corners.
    le = np.vstack((corners[0, :, 0], corners[0, -1, 3]))
    te = np.vstack((corners[-1, :, 1], corners[-1, -1, 2]))
    outline = np.vstack((le, te[::-1]))
    tip_chord = ROOT_CHORD * TAPER_RATIO
    area = SPAN * (ROOT_CHORD + tip_chord) / 2
    aspect_ratio = SPAN**2 / area
    np.testing.assert_allclose(tip_chord / ROOT_CHORD, TAPER_RATIO)
    np.testing.assert_allclose(aspect_ratio, SPAN**2 / area)
    np.testing.assert_allclose(np.ptp(le[:, 1]), SPAN)
    root = np.argmin(abs(le[:, 1]))
    direction = le[-1] - le[root]
    sweep = np.degrees(np.arctan2(direction[0], direction[1]))
    np.testing.assert_allclose(sweep, SWEEP_DEG)
    nx, ny = controls.shape[:2]
    rows = [
        ("Wing span", r"$b$", f"{SPAN:.2f}", "m"),
        ("Root chord", r"$c_r$", f"{ROOT_CHORD:.2f}", "m"),
        ("Tip chord", r"$c_t$", f"{tip_chord:.2f}", "m"),
        ("Taper ratio", r"$\lambda$", f"{TAPER_RATIO:.4f}", "–"),
        ("Leading-edge sweep", r"$\Lambda_{\mathrm{LE}}$", f"{sweep:.2f}", "deg"),
        ("Reference area", r"$S_{\mathrm{ref}}$", f"{area:.2f}", "m²"),
        ("Aspect ratio", r"$AR$", f"{aspect_ratio:.4f}", "–"),
        ("Tip twist", r"$\theta_{\mathrm{tip}}$", f"{TIP_TWIST_DEG:.2f}", "deg"),
        ("Camber coefficient", r"$k$", f"{CAMBER_PARAMETER:g}", "–"),
        ("Chordwise panels", r"$N_x$", str(nx), "–"),
        ("Spanwise panels", r"$N_y$", str(ny), "–"),
        ("Total panels", r"$N$", str(nx * ny), "–")]
    for label, symbol, value, unit in rows:
        print(f"{label}: {value} {unit}")
    with plt.rc_context(STYLE):
        fig = plt.figure(figsize=(15, 5.5))
        ax = fig.add_axes([0.015, 0.15, 0.635, 0.73])
        table_ax = fig.add_axes([0.655, 0.18, 0.335, 0.66])
        # Horizontal drawing coordinate is physical y; vertical is physical x.
        ax.add_patch(Polygon(outline[:, [1, 0]], closed=True, facecolor="#eff0f1",
                             edgecolor="0.25", linewidth=0.85))
        qa, qb, *_ = swept_tapered_twisted_cambered_horseshoe_geometry(
            SPAN, ROOT_CHORD, TAPER_RATIO, SWEEP_DEG, TIP_TWIST_DEG, 1, NY,
            camber=lambda xi: CAMBER_PARAMETER * xi * (1 - xi))
        quarter = np.vstack((qa[0], qb[0, -1]))
        ax.plot(quarter[:, 1], quarter[:, 0], color="0.55", lw=0.6, ls=":")
        ax.axvline(0, color="0.65", lw=0.6, ls=(0, (6, 3, 1, 3)))

        def dimension(p, q, label, text_offset=(0, 0)):
            ax.annotate("", xy=q, xytext=p,
                        arrowprops=dict(arrowstyle="<->", lw=0.65, color="0.3",
                                        shrinkA=0, shrinkB=0, mutation_scale=8))
            midpoint = (np.array(p) + q) / 2 + text_offset
            ax.text(*midpoint, label, ha="center", va="center", fontsize=10,
                    bbox=dict(facecolor="white", edgecolor="none", pad=1.4))

        # Span dimension with extensions from the actual tip leading edges.
        dim_x = -0.85
        for tip in (le[0], le[-1]):
            ax.plot([tip[1], tip[1]], [tip[0], dim_x - 0.12], color="0.5", lw=0.55)
        dimension((-SPAN / 2, dim_x), (SPAN / 2, dim_x), rf"$b={SPAN:.2f}\ \mathrm{{m}}$")
        # Root dimension is outside the planform; extension lines originate
        # at the actual root leading and trailing edges.
        root_dim_y = -SPAN / 2 - 0.5
        for p in (le[root], te[root]):
            ax.plot([p[1], root_dim_y - 0.1], [p[0], p[0]], color="0.65", lw=0.5)
        dimension((root_dim_y, le[root, 0]), (root_dim_y, te[root, 0]), "")
        ax.text(root_dim_y - 0.23, ROOT_CHORD / 2,
                rf"$c_r={ROOT_CHORD:.2f}\ \mathrm{{m}}$", rotation=90,
                ha="center", va="center", fontsize=10)
        tip_y = SPAN / 2 + 0.55
        for p in (le[-1], te[-1]):
            ax.plot([p[1], tip_y + 0.1], [p[0], p[0]], color="0.5", lw=0.55)
        dimension((tip_y, le[-1, 0]), (tip_y, te[-1, 0]),
                  rf"$c_t={tip_chord:.2f}\ \mathrm{{m}}$", text_offset=(0.65, 0))
        # Arc is tied to the actual root-to-tip LE direction and spanwise datum.
        ax.plot([0, 2.2], [le[root, 0], le[root, 0]], color="0.5", lw=0.6, ls="--")
        radius = 1.85
        ax.add_patch(Arc((0, le[root, 0]), 2 * radius, 2 * radius,
                         theta1=0, theta2=sweep, color="0.3", lw=0.7))
        for angle, reverse in ((0, 1), (sweep, -1)):
            t, t0 = np.deg2rad([angle, angle + reverse * 3])
            ax.annotate("", xy=(radius * np.cos(t), radius * np.sin(t)),
                        xytext=(radius * np.cos(t0), radius * np.sin(t0)),
                        arrowprops=dict(arrowstyle="->", lw=0.6, mutation_scale=7))
        ax.text(2.15, 0.28, rf"$\Lambda_{{LE}}={sweep:.2f}^\circ$", fontsize=10)
        ax.annotate("Leading Edge", xy=(le[12, 1], le[12, 0]), xytext=(-3.2, 0.25),
                    fontsize=9, ha="center", arrowprops=dict(arrowstyle="-", lw=0.6))
        ax.annotate("Trailing Edge", xy=(te[12, 1], te[12, 0]), xytext=(-2.8, 3.0),
                    fontsize=9, ha="center", arrowprops=dict(arrowstyle="-", lw=0.6))
        for endpoint, label in (((-4.25, 3.1), "$y$"), ((-4.8, 3.65), "$x$")):
            ax.annotate("", xy=endpoint, xytext=(-4.8, 3.1),
                        arrowprops=dict(arrowstyle="->", lw=0.7, mutation_scale=8))
            ax.text(endpoint[0] + 0.1, endpoint[1], label, fontsize=10)
        ax.set(xlim=(-6.15, 6.95), ylim=(3.9, -1.2), aspect="equal")
        ax.axis("off")
        table_ax.axis("off")
        table_ax.set(xlim=(0, 1), ylim=(0, 1))
        from matplotlib import font_manager

        available_fonts = {font.name for font in font_manager.fontManager.ttflist}
        table_font = next(name for name in (
            "STIX Two Text", "STIX", "STIXGeneral", "Times New Roman",
            "Liberation Serif", "DejaVu Serif") if name in available_fonts)
        text_style = dict(fontfamily=table_font, math_fontfamily="stix",
                          fontsize=10, color="0.15", va="center")
        # Column widths: 48%, 18%, 20%, 14%; values share a right edge.
        x_parameter, x_symbol, x_value, x_unit = 0.015, 0.57, 0.845, 0.93
        positions = (x_parameter, x_symbol, x_value, x_unit)
        alignments = ("left", "center", "right", "center")
        table_ax.text(x_parameter, 1.075, "Wing Geometric Parameters",
                      ha="left", **(text_style | dict(fontsize=11, fontweight="bold")))
        for y_rule, width in ((1.0, 0.8), (0.907, 0.55), (0.065, 0.8)):
            table_ax.plot([0, 1], [y_rule, y_rule], color="0.2", lw=width,
                          clip_on=False)
        table_texts = []
        for row_index, row in enumerate([
                ("Parameter", "Symbol", "Value", "Unit"), *rows]):
            y_text = 0.955 if row_index == 0 else 0.855 - (row_index - 1) * 0.067
            for x_text, alignment, value in zip(positions, alignments, row):
                table_texts.append(table_ax.text(
                    x_text, y_text, value, ha=alignment,
                    fontweight="bold" if row_index == 0 else "normal", **text_style))
        table_ax.text(x_parameter, -0.10,
                      r"Camber: $\frac{z_c}{c}=k\,\xi(1-\xi)$",
                      ha="left", **text_style)
        fig.suptitle("Wing Planform Geometry", fontsize=13, y=0.98)
        fig.text(0.02, 0.035, r"Top-view projection of the twisted lifting surface. Chord dimensions denote true section chords; $S_{ref}$ denotes the untwisted reference planform area.",
                 fontsize=8.5, color="0.4")
        fig.canvas.draw()
        # Check the manually positioned text at the actual rendering size.
        renderer = fig.canvas.get_renderer()
        bounds = table_ax.get_window_extent(renderer)
        text_bounds_list = [text.get_window_extent(renderer) for text in table_texts]
        for index, text_bounds in enumerate(text_bounds_list):
            if not (bounds.x0 <= text_bounds.x0 <= text_bounds.x1 <= bounds.x1
                    and bounds.y0 <= text_bounds.y0 <= text_bounds.y1 <= bounds.y1):
                raise ValueError(f"Table text exceeds axes: {table_texts[index].get_text()}")
            if any(text_bounds.overlaps(other) for other in text_bounds_list[:index]):
                raise ValueError("Parameter-table text overlaps")
        if args.interactive:
            plt.show()
        else:
            output = Path(__file__).resolve().parents[1] / "figures" / "final_wing_planform.png"
            output.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output, dpi=300, bbox_inches="tight", pad_inches=0.08)
            print(f"Saved: {output}")
        plt.close(fig)
