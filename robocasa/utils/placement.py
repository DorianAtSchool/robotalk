"""Shared robot-base placement helpers used by the occupancy-grid strategy."""

from __future__ import annotations

import numpy as np

from robocasa.models.fixtures.fixture import Fixture


# ======================================================================
# Shared helpers
# ======================================================================

# Fixture name substrings that should NOT be treated as ground obstacles.
# "floor" — walkable area, not an obstacle.
# NOTE: "backing" was previously skipped but this caused walls to be invisible
# to the placement system, allowing candidates behind walls.  Walls are now
# treated as obstacles; only "floor" is skipped.
_SKIP_NAME_PATTERNS = ("floor",)

# Minimum z of the fixture's lowest ext_site point to be "above ground."
_ABOVE_GROUND_Z_THRESHOLD = 0.60

# Default standoff from fixture face edge.
_FACE_STANDOFF = 0.40

# Front-working poses for enclosing fixtures should stay close to the center of
# the front face. Candidates are expanded gradually only along that face.
_FRONT_WORKING_LATERAL_LIMITS = (0.05, 0.12, 0.20, 0.24)
MAX_FRONT_WORKING_LATERAL_OFFSET = _FRONT_WORKING_LATERAL_LIMITS[-1]
_FRONT_WORKING_SIDE_CLEARANCE_RATIO = 0.25
_FRONT_WORKING_SIDE_CLEARANCE_MIN = 0.10
_FRONT_WORKING_SIDE_CLEARANCE_MAX = 0.18
EXCLUSIVE_WORKSPACE_CORRIDOR_WIDTH = 0.465


def is_ground_obstacle(name: str, fxtr: Fixture) -> bool:
    """Return True if *fxtr* is a ground-level physical obstacle."""
    name_lower = name.lower()
    for pat in _SKIP_NAME_PATTERNS:
        if pat in name_lower:
            return False
    try:
        ext = fxtr.get_ext_sites(all_points=True, relative=False)
        min_z = min(float(np.asarray(p, dtype=float)[2]) for p in ext)
        if min_z > _ABOVE_GROUND_Z_THRESHOLD:
            return False
    except Exception:
        pass
    return True


def get_fixture_aabb(fxtr: Fixture) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (min_xy, max_xy) for the fixture's 2D AABB, or None."""
    try:
        ext = fxtr.get_ext_sites(all_points=True, relative=False)
        pts = np.array([np.asarray(p, dtype=float)[:2] for p in ext])
        return pts.min(axis=0), pts.max(axis=0)
    except Exception:
        return None


_FACE_NORMALS = {
    "neg_y": np.array([0.0, -1.0]),
    "pos_y": np.array([0.0, 1.0]),
    "neg_x": np.array([-1.0, 0.0]),
    "pos_x": np.array([1.0, 0.0]),
}


def _get_rot_based_face_order(fixture: Fixture) -> list[str]:
    """Return face keys ordered front -> sides -> back based on fixture.rot."""
    all_faces = ["neg_y", "pos_y", "neg_x", "pos_x"]
    if not hasattr(fixture, "rot") or fixture.rot is None:
        return all_faces

    rot = float(fixture.rot)
    front_dir = np.array([-np.cos(rot), -np.sin(rot)])
    return sorted(all_faces, key=lambda f: -float(np.dot(_FACE_NORMALS[f], front_dir)))


def infer_front_face_from_target(
    fmin: np.ndarray,
    fmax: np.ndarray,
    target_xy: np.ndarray,
) -> str:
    """Infer the front face from a world-space target on / near the active face."""
    target = np.asarray(target_xy, dtype=float)[:2]
    all_faces = ["neg_y", "pos_y", "neg_x", "pos_x"]

    def _projected_distance(face_key: str) -> float:
        projected = get_face_target_point(
            face_key,
            fmin,
            fmax,
            target_xy=target,
            standoff=0.0,
            side_clearance=0.0,
        )
        return float(np.linalg.norm(projected - target))

    return min(all_faces, key=_projected_distance)


def get_face_order(
    fixture: Fixture,
    front_target_xy: np.ndarray | None = None,
) -> list[str]:
    """Return face keys ordered front -> sides -> back.

    If *front_target_xy* is provided, infer the front face from that target and
    only use fixture rotation to order the remaining faces.
    """
    base_order = _get_rot_based_face_order(fixture)
    if front_target_xy is None:
        return base_order

    aabb = get_fixture_aabb(fixture)
    if aabb is None:
        return base_order

    inferred_front = infer_front_face_from_target(aabb[0], aabb[1], front_target_xy)
    return [inferred_front, *[face for face in base_order if face != inferred_front]]


def get_face_center(
    face_key: str,
    fmin: np.ndarray,
    fmax: np.ndarray,
    standoff: float = _FACE_STANDOFF,
) -> np.ndarray:
    """Return the center point of a fixture face at the given standoff."""
    center = np.array([(fmin[0] + fmax[0]) / 2, (fmin[1] + fmax[1]) / 2], dtype=float)
    if face_key == "neg_y":
        center[1] = fmin[1] - standoff
    elif face_key == "pos_y":
        center[1] = fmax[1] + standoff
    elif face_key == "neg_x":
        center[0] = fmin[0] - standoff
    elif face_key == "pos_x":
        center[0] = fmax[0] + standoff
    else:
        raise KeyError(f"Unknown face key: {face_key}")
    return center


def get_face_lateral_axis_and_span(
    face_key: str,
    fmin: np.ndarray,
    fmax: np.ndarray,
) -> tuple[int, float, float]:
    """Return the lateral axis and min/max span for a face."""
    if face_key in {"neg_y", "pos_y"}:
        return 0, float(fmin[0]), float(fmax[0])
    if face_key in {"neg_x", "pos_x"}:
        return 1, float(fmin[1]), float(fmax[1])
    raise KeyError(f"Unknown face key: {face_key}")


def get_front_working_side_clearance(
    face_key: str,
    fmin: np.ndarray,
    fmax: np.ndarray,
) -> float:
    """Return the side exclusion margin for a face's working band."""
    _, lateral_min, lateral_max = get_face_lateral_axis_and_span(face_key, fmin, fmax)
    span = max(0.0, lateral_max - lateral_min)
    if span <= 0.0:
        return 0.0
    clearance = float(np.clip(
        span * _FRONT_WORKING_SIDE_CLEARANCE_RATIO,
        _FRONT_WORKING_SIDE_CLEARANCE_MIN,
        _FRONT_WORKING_SIDE_CLEARANCE_MAX,
    ))
    return min(clearance, max(0.0, span / 2.0 - 1e-3))


def get_face_working_lateral_bounds(
    face_key: str,
    fmin: np.ndarray,
    fmax: np.ndarray,
    side_clearance: float | None = None,
) -> tuple[int, float, float]:
    """Return the safe lateral interval for a face's working band."""
    lateral_axis, lateral_min, lateral_max = get_face_lateral_axis_and_span(face_key, fmin, fmax)
    clearance = (
        get_front_working_side_clearance(face_key, fmin, fmax)
        if side_clearance is None else
        max(0.0, float(side_clearance))
    )
    if lateral_max - lateral_min <= 0.0:
        return lateral_axis, lateral_min, lateral_max

    safe_min = lateral_min + clearance
    safe_max = lateral_max - clearance
    if safe_min > safe_max:
        center = (lateral_min + lateral_max) / 2.0
        safe_min = center
        safe_max = center
    return lateral_axis, float(safe_min), float(safe_max)


def is_within_face_working_band(
    face_key: str,
    pos_xy: np.ndarray,
    fmin: np.ndarray,
    fmax: np.ndarray,
    side_clearance: float | None = None,
    margin: float = 1e-6,
) -> bool:
    """Return True if *pos_xy* lies within the face's safe working band."""
    lateral_axis, safe_min, safe_max = get_face_working_lateral_bounds(
        face_key,
        fmin,
        fmax,
        side_clearance=side_clearance,
    )
    pos = np.asarray(pos_xy, dtype=float)[:2]
    return bool(safe_min - margin <= pos[lateral_axis] <= safe_max + margin)


def get_face_target_point(
    face_key: str,
    fmin: np.ndarray,
    fmax: np.ndarray,
    target_xy: np.ndarray | None = None,
    standoff: float = _FACE_STANDOFF,
    side_clearance: float | None = None,
) -> np.ndarray:
    """Project *target_xy* onto the face's working line at the given standoff."""
    face_target = get_face_center(face_key, fmin, fmax, standoff=standoff)
    lateral_axis, safe_min, safe_max = (
        get_face_working_lateral_bounds(face_key, fmin, fmax, side_clearance)
        if side_clearance is not None else
        get_face_lateral_axis_and_span(face_key, fmin, fmax)
    )
    if target_xy is None:
        face_target[lateral_axis] = float((safe_min + safe_max) / 2.0)
        return face_target

    target = np.asarray(target_xy, dtype=float)[:2]
    face_target[lateral_axis] = float(np.clip(target[lateral_axis], safe_min, safe_max))
    return face_target


def prepend_face_target_candidate(
    candidates: list[tuple[np.ndarray, float]],
    face_key: str,
    fmin: np.ndarray,
    fmax: np.ndarray,
    target_xy: np.ndarray | None = None,
    standoff: float = _FACE_STANDOFF,
    side_clearance: float | None = None,
) -> list[tuple[np.ndarray, float]]:
    """Return candidates with the exact projected face-target candidate first."""
    target_pos = get_face_target_point(
        face_key,
        fmin,
        fmax,
        target_xy=target_xy,
        standoff=standoff,
        side_clearance=side_clearance,
    )
    if candidates:
        target_yaw = candidates[0][1]
        if any(np.allclose(pos, target_pos, atol=1e-6) for pos, _ in candidates):
            return candidates
    else:
        target_yaw = {
            "neg_y": np.pi / 2,
            "pos_y": -np.pi / 2,
            "neg_x": 0.0,
            "pos_x": np.pi,
        }[face_key]
    return [(target_pos, target_yaw), *candidates]


def prepend_face_center_candidate(
    candidates: list[tuple[np.ndarray, float]],
    face_key: str,
    fmin: np.ndarray,
    fmax: np.ndarray,
    standoff: float = _FACE_STANDOFF,
) -> list[tuple[np.ndarray, float]]:
    """Return candidates with the exact face-center candidate placed first."""
    return prepend_face_target_candidate(
        candidates,
        face_key,
        fmin,
        fmax,
        target_xy=None,
        standoff=standoff,
    )


def get_front_alignment_metrics(
    fixture: Fixture,
    pos_xy: np.ndarray,
    span_margin: float = 0.05,
    target_xy: np.ndarray | None = None,
    front_face: str | None = None,
) -> dict[str, float | bool | str] | None:
    """Return front-face alignment metrics for *pos_xy* relative to *fixture*."""
    aabb = get_fixture_aabb(fixture)
    if aabb is None:
        return None

    fmin, fmax = aabb
    pos = np.asarray(pos_xy, dtype=float)[:2]
    if front_face is None:
        front_face = get_face_order(fixture, front_target_xy=target_xy)[0]
    side_clearance = get_front_working_side_clearance(front_face, fmin, fmax)
    front_target = get_face_target_point(
        front_face,
        fmin,
        fmax,
        target_xy=target_xy,
        standoff=0.0,
        side_clearance=side_clearance,
    )

    lateral_axis, safe_min, safe_max = get_face_working_lateral_bounds(
        front_face,
        fmin,
        fmax,
        side_clearance=side_clearance,
    )
    centerline = float(front_target[lateral_axis])
    lateral_offset = abs(float(pos[lateral_axis] - centerline))
    within_span = bool(safe_min - span_margin <= pos[lateral_axis] <= safe_max + span_margin)

    if front_face in {"neg_y", "pos_y"}:
        front_gap = float(fmin[1] - pos[1]) if front_face == "neg_y" else float(pos[1] - fmax[1])
    else:
        front_gap = float(fmin[0] - pos[0]) if front_face == "neg_x" else float(pos[0] - fmax[0])

    return {
        "front_face": front_face,
        "lateral_offset": lateral_offset,
        "within_span": within_span,
        "on_front_face": front_gap > 0.0,
        "front_gap": front_gap,
    }


def is_in_front_workspace_corridor(
    fixture: Fixture,
    pos_xy: np.ndarray,
    max_depth: float = 0.75,
    margin: float = 1e-6,
    front_face: str | None = None,
    corridor_width: float | None = None,
) -> bool:
    """Whether *pos_xy* is in the fixture-width corridor before its front.

    Unlike a radial/AABB dilation, this reserves no space beside or behind the
    fixture.  The lateral interval is the fixture's full AABB width on its
    inferred front face; ``max_depth`` only limits how far that rectangle
    extends outward into the room.
    """
    aabb = get_fixture_aabb(fixture)
    if aabb is None:
        return False
    fmin, fmax = aabb
    pos = np.asarray(pos_xy, dtype=float)[:2]
    front_face = front_face or get_face_order(fixture)[0]
    lateral_axis, lateral_min, lateral_max = get_face_lateral_axis_and_span(
        front_face, fmin, fmax
    )
    if corridor_width is not None:
        center = (lateral_min + lateral_max) / 2.0
        half_width = float(corridor_width) / 2.0
        lateral_min = center - half_width
        lateral_max = center + half_width
    if not lateral_min - margin <= pos[lateral_axis] <= lateral_max + margin:
        return False
    if front_face == "neg_y":
        gap = float(fmin[1] - pos[1])
    elif front_face == "pos_y":
        gap = float(pos[1] - fmax[1])
    elif front_face == "neg_x":
        gap = float(fmin[0] - pos[0])
    else:
        gap = float(pos[0] - fmax[0])
    return bool(-margin < gap <= float(max_depth) + margin)


def front_lateral_offset(
    face_key: str,
    pos_xy: np.ndarray,
    front_target: np.ndarray,
) -> float:
    """Return lateral offset from *pos_xy* to a front-face working target."""
    lateral_axis = 0 if face_key in {"neg_y", "pos_y"} else 1
    pos = np.asarray(pos_xy, dtype=float)[:2]
    target = np.asarray(front_target, dtype=float)[:2]
    return abs(float(pos[lateral_axis] - target[lateral_axis]))
