"""Sutherland-Hodgman polygon clipping for Computer Graphics Lab Assignment 06.

Roll number: 208
a = second-last digit = 0
b = last digit = 8

The polygon is clipped one window boundary at a time in this exact order:
left, right, bottom, top. Matplotlib is used only after clipping has been
computed, to draw the original and clipped vertex lists.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as PolygonPatch, Rectangle


# ---------------------------------------------------------------------------
# Input data
# ---------------------------------------------------------------------------
ROLL_NUMBER = "208"
A = int(ROLL_NUMBER[-2])
B = int(ROLL_NUMBER[-1])

# Rectangular clipping window: (xmin, ymin, xmax, ymax)
WINDOW = (50.0, 50.0, 200.0, 150.0)

POLYGONS = {
    "Q1": [(80.0, 70.0), (150.0, 70.0), (115.0, 130.0)],
    "Q2": [(20.0, 20.0), (250.0, 20.0), (135.0, 200.0)],
    "Q3": [
        (140.0 + A, 20.0 + B),
        (230.0 + A, 20.0 + B),
        (230.0 + A, 110.0 + B),
        (140.0 + A, 110.0 + B),
    ],
}


# ---------------------------------------------------------------------------
# Sutherland-Hodgman clipping: clip against ONE boundary per call
# ---------------------------------------------------------------------------
def clip_against_boundary(
    vertices: list[tuple[float, float]],
    boundary: str,
    window: tuple[float, float, float, float] = WINDOW,
) -> list[tuple[float, float]]:
    """Return vertices clipped against one boundary of the rectangle.

    The four cases are handled for each directed edge from previous to
    current vertex:
      1. inside -> inside:  append current
      2. outside -> inside: append intersection, then current
      3. inside -> outside: append intersection
      4. outside -> outside: append nothing

    Boundary points count as inside. The routine is implemented directly;
    no ready-made clipping function is used.
    """
    if not vertices:
        return []

    xmin, ymin, xmax, ymax = window
    boundary_values = {
        "left": xmin,
        "right": xmax,
        "bottom": ymin,
        "top": ymax,
    }
    if boundary not in boundary_values:
        raise ValueError("boundary must be 'left', 'right', 'bottom', or 'top'")

    edge_value = boundary_values[boundary]

    def is_inside(point: tuple[float, float]) -> bool:
        x, y = point
        if boundary == "left":
            return x >= edge_value
        if boundary == "right":
            return x <= edge_value
        if boundary == "bottom":
            return y >= edge_value
        return y <= edge_value  # top

    def intersection(
        start: tuple[float, float], end: tuple[float, float]
    ) -> tuple[float, float]:
        x1, y1 = start
        x2, y2 = end

        if boundary in ("left", "right"):
            # Vertical clipping edge: solve for y at x = edge_value.
            if x2 == x1:
                return (edge_value, y1)
            t = (edge_value - x1) / (x2 - x1)
            y = y1 + t * (y2 - y1)
            return (edge_value, y)

        # Horizontal clipping edge: solve for x at y = edge_value.
        if y2 == y1:
            return (x1, edge_value)
        t = (edge_value - y1) / (y2 - y1)
        x = x1 + t * (x2 - x1)
        return (x, edge_value)

    output: list[tuple[float, float]] = []
    previous = vertices[-1]
    previous_inside = is_inside(previous)

    for current in vertices:
        current_inside = is_inside(current)

        if previous_inside and current_inside:
            # Case 1: both endpoints are inside.
            output.append(current)
        elif not previous_inside and current_inside:
            # Case 2: entering - add intersection and the current vertex.
            output.append(intersection(previous, current))
            output.append(current)
        elif previous_inside and not current_inside:
            # Case 3: leaving - add only the intersection.
            output.append(intersection(previous, current))
        # Case 4: both endpoints outside - add nothing.

        previous = current
        previous_inside = current_inside

    return remove_duplicate_neighbors(output)


def remove_duplicate_neighbors(
    vertices: list[tuple[float, float]], tolerance: float = 1e-9
) -> list[tuple[float, float]]:
    """Remove adjacent duplicate points, including a repeated closing point."""
    if not vertices:
        return []

    cleaned: list[tuple[float, float]] = []
    for point in vertices:
        if not cleaned or (
            abs(point[0] - cleaned[-1][0]) > tolerance
            or abs(point[1] - cleaned[-1][1]) > tolerance
        ):
            cleaned.append(point)

    if len(cleaned) > 1 and (
        abs(cleaned[0][0] - cleaned[-1][0]) <= tolerance
        and abs(cleaned[0][1] - cleaned[-1][1]) <= tolerance
    ):
        cleaned.pop()

    return cleaned


def clip_polygon(
    vertices: list[tuple[float, float]],
    window: tuple[float, float, float, float] = WINDOW,
) -> dict[str, list[tuple[float, float]]]:
    """Clip a polygon in the required left, right, bottom, top order.

    The returned dictionary includes the input and the result after every
    individual clipping stage, so the intermediate vertex lists are visible.
    """
    stages = {"input": list(vertices)}
    current = list(vertices)
    for boundary in ("left", "right", "bottom", "top"):
        current = clip_against_boundary(current, boundary, window)
        stages[boundary] = current
    stages["final"] = current
    return stages


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------
def format_vertices(vertices: Iterable[tuple[float, float]]) -> str:
    points = list(vertices)
    if not points:
        return "(empty)"
    return "  " + "  ".join(f"({x:.3f}, {y:.3f})" for x, y in points)


def make_report(all_stages: dict[str, dict[str, list[tuple[float, float]]]]) -> str:
    lines = [
        f"SUTHERLAND-HODGMAN POLYGON CLIPPING - ROLL NUMBER {ROLL_NUMBER}",
        f"a = {A}, b = {B}",
        f"Q3 = {format_vertices(POLYGONS['Q3'])}",
        f"Clipping window: lower-left (50, 50), upper-right (200, 150)",
        "Boundary order: left -> right -> bottom -> top",
        "",
    ]

    stage_order = ("input", "left", "right", "bottom", "top", "final")
    for polygon_name, stages in all_stages.items():
        lines.append(f"{polygon_name}")
        for stage in stage_order:
            vertices = stages[stage]
            label = "Input" if stage == "input" else (
                "Final clipped polygon" if stage == "final" else f"After {stage} boundary"
            )
            lines.append(f"{label} ({len(vertices)} vertices):")
            lines.append(format_vertices(vertices))
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Plotting only: all clipped coordinates are computed above, before drawing.
# ---------------------------------------------------------------------------
def draw_polygon(
    ax: plt.Axes,
    name: str,
    original: list[tuple[float, float]],
    clipped: list[tuple[float, float]],
    window: tuple[float, float, float, float] = WINDOW,
) -> None:
    xmin, ymin, xmax, ymax = window

    # Clipping window.
    ax.add_patch(
        Rectangle(
            (xmin, ymin), xmax - xmin, ymax - ymin,
            facecolor="#dcecff", edgecolor="#243b64",
            linewidth=2.2, alpha=0.45, label="Clipping window", zorder=1,
        )
    )

    # Original input polygon as a dashed outline.
    original_closed = original + [original[0]]
    ax.plot(
        [p[0] for p in original_closed],
        [p[1] for p in original_closed],
        linestyle=(0, (5, 3)), linewidth=2.0,
        color="#e58a24", marker="o", markersize=5,
        markerfacecolor="white", markeredgewidth=1.4,
        label="Original polygon", zorder=3,
    )

    # Clipped polygon as a solid filled shape, using computed vertices only.
    if len(clipped) >= 3:
        ax.add_patch(
            PolygonPatch(
                clipped, closed=True,
                facecolor="#42b883", edgecolor="#147d59",
                linewidth=2.0, alpha=0.72,
                label="Clipped polygon", zorder=4,
            )
        )
        clipped_closed = clipped + [clipped[0]]
        ax.plot(
            [p[0] for p in clipped_closed],
            [p[1] for p in clipped_closed],
            color="#147d59", linewidth=1.5, zorder=5,
        )
        ax.scatter(
            [p[0] for p in clipped], [p[1] for p in clipped],
            s=34, color="#0b5d43", edgecolors="white", linewidths=0.8,
            zorder=6,
        )

    # Label only the clipped vertices so the original dashed outline stays
    # visually clear, especially where original and clipped vertices overlap.
    for index, (x, y) in enumerate(clipped):
        ax.annotate(
            f"C{index + 1}", (x, y), xytext=(5, -12),
            textcoords="offset points", fontsize=8, color="#07543c",
            fontweight="bold",
        )

    ax.set_title(f"{name}: Sutherland-Hodgman Clipping", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0, 270)
    ax.set_ylim(0, 220)
    ax.set_xticks(range(0, 271, 25))
    ax.set_yticks(range(0, 221, 25))
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.55)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.95)


def save_figures(
    all_stages: dict[str, dict[str, list[tuple[float, float]]]],
    output_dir: Path,
) -> None:
    """Save one PNG per polygon and a combined comparison figure."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, original in POLYGONS.items():
        clipped = all_stages[name]["final"]
        fig, ax = plt.subplots(figsize=(8.2, 6.8), constrained_layout=True)
        fig.patch.set_facecolor("#f7f9fc")
        ax.set_facecolor("#ffffff")
        draw_polygon(ax, name, original, clipped)
        fig.savefig(output_dir / f"{name}_clipping.png", dpi=240, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6.6), constrained_layout=True)
    fig.patch.set_facecolor("#f7f9fc")
    for ax, (name, original) in zip(axes, POLYGONS.items()):
        ax.set_facecolor("#ffffff")
        draw_polygon(ax, name, original, all_stages[name]["final"])
    fig.suptitle(
        "Assignment 06 | Polygon Clipping Results | Roll No. 208",
        fontsize=16, fontweight="bold", y=1.02,
    )
    fig.savefig(output_dir / "all_polygons_summary.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    output_dir = Path(__file__).resolve().parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_stages = {
        name: clip_polygon(vertices)
        for name, vertices in POLYGONS.items()
    }

    report = make_report(all_stages)
    print(report)

    report_path = Path(__file__).resolve().parent / "results.txt"
    report_path.write_text(report + "\n", encoding="utf-8")

    save_figures(all_stages, output_dir)
    print(f"\nFigures saved in: {output_dir}")
    print(f"Stage-by-stage output saved in: {report_path}")


if __name__ == "__main__":
    main()
