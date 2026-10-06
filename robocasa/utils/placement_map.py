"""
2D placement map rendering for the occupancy-grid strategy.

Reusable drawing functions consumed by ``SimToolExecutor.save_placement_map``
and ``experiments/visualize_grid.py``.

Object rendering – current approach: **colored labels** (option 1).
Each object gets a colored text label placed at its position, using the
same overlap-avoidance logic as fixture labels.
"""

from __future__ import annotations

import matplotlib.patches as patches
from matplotlib.colors import ListedColormap
import numpy as np
import textwrap

from robocasa.utils.placement import (
    get_fixture_aabb,
    is_ground_obstacle,
)


def _format_fixture_label(name: str, clean: bool = True, wrap_width: int = 24) -> str:
    """Render a readable fixture label from an internal fixture id.

    When *clean* is True, preserves the full fixture id but converts
    underscores to spaces for readability.

    When *clean* is False, returns the full raw fixture id (only wrapped).
    """
    label = str(name).strip()
    if clean:
        label = label.replace("_", " ")
    if len(label) <= wrap_width:
        return label
    return textwrap.fill(
        label,
        width=wrap_width,
        break_long_words=False,
        break_on_hyphens=False,
    )


def _aabb_overlaps(box_a, box_b, margin: float = 0.0) -> bool:
    """Return whether two axis-aligned boxes overlap."""
    return not (
        box_a[2] + margin < box_b[0]
        or box_b[2] + margin < box_a[0]
        or box_a[3] + margin < box_b[1]
        or box_b[3] + margin < box_a[1]
    )


def _aabb_overlap_area(box_a, box_b, margin: float = 0.0) -> float:
    """Return the overlap area between two AABBs (with optional margin)."""
    ox = max(0.0, min(box_a[2], box_b[2]) - max(box_a[0], box_b[0]) + margin)
    oy = max(0.0, min(box_a[3], box_b[3]) - max(box_a[1], box_b[1]) + margin)
    return ox * oy


def _estimate_label_extent(label: str, label_fontsize: int) -> tuple[float, float]:
    """Approximate label size in world coordinates for collision avoidance."""
    lines = label.splitlines() or [label]
    font_scale = max(label_fontsize / 6.0, 0.75)
    width = max(len(line) for line in lines) * 0.028 * font_scale + 0.08
    height = len(lines) * 0.11 * font_scale + 0.04
    return width, height


def _label_candidate_positions(fmin, fmax):
    """Return candidate anchor points for a fixture label (two rings + center)."""
    cx, cy = (fmin[0] + fmax[0]) / 2, (fmin[1] + fmax[1]) / 2
    candidates = [(cx, cy)]
    for scale in (1.0, 2.0):
        dx = max((fmax[0] - fmin[0]) * 0.5 + 0.12 * scale, 0.18 * scale)
        dy = max((fmax[1] - fmin[1]) * 0.5 + 0.12 * scale, 0.18 * scale)
        candidates.extend([
            (cx, fmax[1] + dy),
            (fmax[0] + dx, cy),
            (fmin[0] - dx, cy),
            (cx, fmin[1] - dy),
            (fmax[0] + dx, fmax[1] + dy),
            (fmin[0] - dx, fmax[1] + dy),
            (fmax[0] + dx, fmin[1] - dy),
            (fmin[0] - dx, fmin[1] - dy),
        ])
    return candidates


def _pick_label_position(
    fmin,
    fmax,
    label: str,
    placed_label_boxes,
    label_fontsize: int,
):
    """Pick a label anchor with minimal overlap against other labels."""
    cx, cy = (fmin[0] + fmax[0]) / 2, (fmin[1] + fmax[1]) / 2
    label_width, label_height = _estimate_label_extent(label, label_fontsize)
    best_position = (cx, cy)
    best_box = (
        cx - label_width / 2,
        cy - label_height / 2,
        cx + label_width / 2,
        cy + label_height / 2,
    )
    best_score = float("inf")

    for x, y in _label_candidate_positions(fmin, fmax):
        candidate_box = (
            x - label_width / 2,
            y - label_height / 2,
            x + label_width / 2,
            y + label_height / 2,
        )
        label_overlap_area = sum(
            _aabb_overlap_area(candidate_box, other_box, margin=0.03)
            for other_box in placed_label_boxes
        )
        distance_penalty = float(np.linalg.norm(np.array([x - cx, y - cy])))
        score = (
            500.0 * label_overlap_area
            + distance_penalty
        )
        if score < best_score:
            best_score = score
            best_position = (x, y)
            best_box = candidate_box

    return best_position, best_box


def _should_skip_fixture_label(name: str) -> bool:
    """Return True for fixture labels that should not be rendered."""
    name_lower = str(name).lower()
    if "floor" in name_lower or "wall" in name_lower or name_lower.startswith("stack_"):
        return True

    # Skip counter stack helper labels while keeping regular counter labels.
    if "counter" in name_lower and "stack" in name_lower:
        return True

    return False


def _draw_fixtures(ax, fixtures, label_fontsize=7, clean_labels=True):
    """Draw fixture AABBs on the axes and return placed label boxes."""
    fixture_entries = []
    for name, fxtr in fixtures.items():
        aabb = get_fixture_aabb(fxtr)
        if aabb is None:
            continue
        fixture_entries.append((name, fxtr, *aabb))

    # Draw all fixture rectangles first.
    for name, fxtr, fmin, fmax in fixture_entries:
        obstacle = is_ground_obstacle(name, fxtr)
        color = "red" if obstacle else "green"
        alpha = 0.35 if obstacle else 0.15
        rect = patches.Rectangle(
            (fmin[0], fmin[1]),
            fmax[0] - fmin[0], fmax[1] - fmin[1],
            linewidth=1.0, edgecolor=color, facecolor=color, alpha=alpha,
        )
        ax.add_patch(rect)

    # Filter to only fixtures that need labels, then place labels.
    label_entries = [e for e in fixture_entries if not _should_skip_fixture_label(e[0])]
    placed_label_boxes = []

    for name, fxtr, fmin, fmax in sorted(
        label_entries,
        key=lambda entry: (
            -float((entry[3][0] - entry[2][0]) * (entry[3][1] - entry[2][1])),
            entry[0],
        ),
    ):

        label = _format_fixture_label(name, clean=clean_labels)
        cx, cy = (fmin[0] + fmax[0]) / 2, (fmin[1] + fmax[1]) / 2
        (label_x, label_y), label_box = _pick_label_position(
            fmin,
            fmax,
            label,
            placed_label_boxes,
            label_fontsize,
        )
        if not np.allclose([label_x, label_y], [cx, cy]):
            ax.plot(
                [cx, label_x],
                [cy, label_y],
                color="#444444",
                linewidth=0.4,
                alpha=0.45,
                zorder=4,
            )
        ax.text(
            label_x,
            label_y,
            label,
            fontsize=label_fontsize,
            ha="center",
            va="center",
            color="black",
            clip_on=True,
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "#777777",
                "linewidth": 0.35,
                "alpha": 0.85,
            },
            zorder=5,
        )
        placed_label_boxes.append(label_box)

    return placed_label_boxes


def _draw_robots(ax, runner, placed_label_boxes=None):
    """Draw robot positions with readable agent labels."""
    if placed_label_boxes is None:
        placed_label_boxes = []

    colors = ["blue", "orange", "purple", "brown"]
    label_offsets = [
        np.array([0.18, 0.18]),
        np.array([0.18, -0.18]),
        np.array([-0.18, 0.18]),
        np.array([-0.18, -0.18]),
    ]
    for ridx, color in enumerate(colors):
        if ridx >= runner._num_robots:
            break
        pos = runner._get_robot_position(ridx)[:2]
        ax.plot(pos[0], pos[1], "o", color=color, markersize=8, zorder=5)
        label = f"agent {ridx}"
        label_width, label_height = _estimate_label_extent(label, label_fontsize=7)
        best_position = pos + label_offsets[ridx % len(label_offsets)]
        best_box = (
            best_position[0] - label_width / 2,
            best_position[1] - label_height / 2,
            best_position[0] + label_width / 2,
            best_position[1] + label_height / 2,
        )
        best_score = float("inf")
        for offset in label_offsets:
            candidate = pos + offset
            candidate_box = (
                candidate[0] - label_width / 2,
                candidate[1] - label_height / 2,
                candidate[0] + label_width / 2,
                candidate[1] + label_height / 2,
            )
            overlap_count = sum(
                _aabb_overlaps(candidate_box, other_box, margin=0.03)
                for other_box in placed_label_boxes
            )
            score = 100.0 * overlap_count + float(np.linalg.norm(offset))
            if score < best_score:
                best_score = score
                best_position = candidate
                best_box = candidate_box

        ax.text(
            best_position[0],
            best_position[1],
            label,
            fontsize=7,
            ha="center",
            va="center",
            color=color,
            fontweight="bold",
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": color,
                "linewidth": 0.5,
                "alpha": 0.9,
            },
            zorder=6,
        )
        placed_label_boxes.append(best_box)


def _get_object_positions(runner):
    """Return list of (display_name, x, y) for all task objects in the scene.

    Objects live in ``env.objects`` / ``env.obj_body_id``; their world
    positions come from the MuJoCo simulation state.  Display names use the
    object type (e.g. "mug") from ep_meta instead of the raw env key
    (e.g. "obj").
    """
    entries = []
    env = runner.env
    if not hasattr(env, "objects") or not env.objects:
        return entries

    # Build env_key → human-readable type from ep_meta object_cfgs
    obj_type_map = {}
    if hasattr(env, "get_ep_meta"):
        for cfg in env.get_ep_meta().get("object_cfgs", []):
            name = cfg.get("name", "")
            cat = (cfg.get("info") or {}).get("cat", "")
            if name and cat:
                obj_type_map[name] = cat

    for obj_name in env.objects:
        body_id = env.obj_body_id.get(obj_name)
        if body_id is None:
            continue
        pos = env.sim.data.body_xpos[body_id]
        display_name = obj_type_map.get(obj_name, obj_name)
        entries.append((display_name, float(pos[0]), float(pos[1])))
    return entries


_OBJECT_COLORS = [
    "#dd55ff", "#ff8800", "#00aadd", "#dd2255", "#44bb44", "#8866cc",
]


def _draw_objects(ax, runner, placed_label_boxes=None):
    """Draw colored labels at each object's position, avoiding collisions.

    Labels are placed using the same overlap-avoidance logic as fixture
    labels so they don't collide with fixtures, robots, or each other.
    """
    if placed_label_boxes is None:
        placed_label_boxes = []

    entries = _get_object_positions(runner)
    if not entries:
        return

    label_fontsize = 7
    for idx, (obj_name, x, y) in enumerate(entries):
        color = _OBJECT_COLORS[idx % len(_OBJECT_COLORS)]
        label = str(obj_name)
        # Draw marker dot at object position
        ax.plot(x, y, "D", color=color, markersize=5, zorder=7)
        fmin = np.array([x, y])
        fmax = np.array([x, y])
        (label_x, label_y), label_box = _pick_label_position(
            fmin, fmax, label, placed_label_boxes, label_fontsize,
        )
        ax.text(
            label_x, label_y, label,
            fontsize=label_fontsize, ha="center", va="center",
            color=color, fontweight="bold", zorder=8,
            bbox={
                "boxstyle": "round,pad=0.15",
                "facecolor": "white",
                "edgecolor": color,
                "linewidth": 0.5,
                "alpha": 0.88,
            },
        )
        placed_label_boxes.append(label_box)


_GRID_CELL_COLORS = ("#f0f0f0", "#d9923b", "#444444")


def _draw_grid_cells_legacy(ax, grid):
    """Draw one rectangle per cell, preserving the historical map pixels."""

    occupied_count = 0
    enclosed_count = 0
    free_count = 0
    for r in range(grid._rows):
        for c in range(grid._cols):
            xy = grid._grid_to_world(r, c)
            half = grid.cell_size / 2
            if grid._grid[r, c]:
                color = _GRID_CELL_COLORS[2]
                occupied_count += 1
            elif not grid.is_standable(xy):
                color = _GRID_CELL_COLORS[1]
                enclosed_count += 1
            else:
                color = _GRID_CELL_COLORS[0]
                free_count += 1
            rect = patches.Rectangle(
                (xy[0] - half, xy[1] - half),
                grid.cell_size,
                grid.cell_size,
                linewidth=0.3,
                edgecolor="#cccccc",
                facecolor=color,
            )
            ax.add_patch(rect)
    return occupied_count, enclosed_count, free_count


def _draw_grid_cells_raster(ax, grid):
    """Draw all cell classes as one raster artist for live-sim throughput."""

    cell_classes = np.zeros((grid._rows, grid._cols), dtype=np.uint8)
    occupied = np.asarray(grid._grid, dtype=bool)
    cell_classes[occupied] = 2
    occupied_count = int(np.count_nonzero(occupied))
    enclosed_count = 0
    for r, c in np.argwhere(~occupied):
        if not grid.is_standable(grid._grid_to_world(int(r), int(c))):
            cell_classes[r, c] = 1
            enclosed_count += 1
    free_count = int(grid._rows * grid._cols - occupied_count - enclosed_count)

    x_min = grid._origin[0]
    x_max = grid._origin[0] + grid._cols * grid.cell_size
    y_min = grid._origin[1]
    y_max = grid._origin[1] + grid._rows * grid.cell_size
    ax.imshow(
        cell_classes,
        origin="lower",
        extent=(x_min, x_max, y_min, y_max),
        cmap=ListedColormap(_GRID_CELL_COLORS),
        interpolation="nearest",
        vmin=-0.5,
        vmax=2.5,
        zorder=0,
    )
    return occupied_count, enclosed_count, free_count


def draw_grid_map(ax, runner, clean_labels=True, grid_renderer="legacy"):
    """Draw the occupancy grid on axes.

    *clean_labels*: when True (default), fixture labels are shortened
    (strip ``_group``, ``_main``, dedupe). Set False for raw fixture ids.
    *grid_renderer*: ``legacy`` preserves per-cell patches; ``raster``
    draws the same cell classes as one image layer.
    """

    grid = runner._occupancy_grid
    if grid_renderer == "legacy":
        counts = _draw_grid_cells_legacy(ax, grid)
    elif grid_renderer == "raster":
        counts = _draw_grid_cells_raster(ax, grid)
    else:
        raise ValueError(
            "grid_renderer must be either 'legacy' or 'raster', "
            f"got {grid_renderer!r}."
        )
    occupied_count, enclosed_count, free_count = counts

    placed_label_boxes = _draw_fixtures(
        ax, runner._fixtures, clean_labels=clean_labels
    )
    _draw_robots(ax, runner, placed_label_boxes=placed_label_boxes)
    _draw_objects(ax, runner, placed_label_boxes=placed_label_boxes)

    x_min = grid._origin[0]
    x_max = grid._origin[0] + grid._cols * grid.cell_size
    y_min = grid._origin[1]
    y_max = grid._origin[1] + grid._rows * grid.cell_size
    ax.set_xlim(x_min - 0.2, x_max + 0.2)
    ax.set_ylim(y_min - 0.2, y_max + 0.2)
    ax.set_aspect("equal")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_title(
        f"Grid — {grid._rows}x{grid._cols} @ {grid.cell_size}m | "
        f"occ={occupied_count}, enclosed={enclosed_count}, free={free_count}"
    )
