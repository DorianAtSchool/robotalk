"""Build scene-agnostic grounding specs and resolve them for one scene."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from robotalk.tools.object_type_families import object_type_matches
from robotalk.tasks.specs import load_task_spec

GROUNDING_MAP_VERSION = 2

def build_grounding_map_for_task(
    composite_task: str,
    initial_state: dict[str, Any],
) -> dict[str, Any]:
    """Builds one scene-agnostic grounding map for a supported task."""

    task_spec = load_task_spec(composite_task)
    normalized_initial_state = _normalize_initial_state_symbols(
        composite_task, initial_state
    )
    grounding_spec = dict(task_spec.grounding)
    symbols = deepcopy(dict(grounding_spec.get("symbols", {})))

    for symbol_name, symbol_spec in symbols.items():
        if not isinstance(symbol_spec, dict):
            continue
        if symbol_spec.get("entity_type") == "object":
            object_state = normalized_initial_state["objects"][symbol_name]
            symbol_spec.setdefault("object_type", object_state["object_type"])
            symbol_spec.setdefault("symbolic_location", object_state.get("location"))
        elif symbol_spec.get("entity_type") == "fixture":
            fixture_state = normalized_initial_state["fixtures"][symbol_name]
            symbol_spec.setdefault("fixture_type", fixture_state["fixture_type"])

    return _grounding_map(
        composite_task,
        symbols=symbols,
    )


def build_resolved_grounding_payload(
    trajectory: dict[str, Any],
    scene: dict[str, Any],
    *,
    scene_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Builds one persisted grounding payload for a trajectory and scene."""

    grounding_map = trajectory.get("grounding_map")
    if not isinstance(grounding_map, dict):
        grounding_map = build_grounding_map_for_task(
            trajectory["composite_task"],
            trajectory["initial_state"],
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "trajectory_id": trajectory.get("trajectory_id"),
        "composite_task": trajectory.get("composite_task"),
        "scene_config": deepcopy(scene_config) if scene_config is not None else None,
        "grounding_map": deepcopy(grounding_map),
        "scene_summary": build_scene_summary(scene),
        "resolved_grounding": resolve_grounding_map(grounding_map, scene),
    }


def resolve_grounding_map(
    grounding_map: dict[str, Any],
    scene: dict[str, Any],
) -> dict[str, Any]:
    """Resolves a scene-agnostic grounding map into concrete scene IDs."""

    symbols = grounding_map.get("symbols", {})
    if not isinstance(symbols, dict):
        raise ValueError("grounding_map.symbols must be a dict.")

    resolved_symbols: dict[str, dict[str, Any]] = {}
    unresolved_symbols: list[dict[str, Any]] = []

    object_symbols = {
        symbol: spec
        for symbol, spec in symbols.items()
        if isinstance(spec, dict) and spec.get("entity_type") == "object"
    }
    fixture_symbols = {
        symbol: spec
        for symbol, spec in symbols.items()
        if isinstance(spec, dict) and spec.get("entity_type") == "fixture"
    }

    for symbol, spec in object_symbols.items():
        resolution = _resolve_symbol(symbol, spec, scene, resolved_symbols)
        if resolution.get("resolved_id") is None:
            unresolved_symbols.append(resolution)
            continue
        resolved_symbols[symbol] = resolution

    pending_fixture_symbols = dict(fixture_symbols)
    while pending_fixture_symbols:
        resolved_in_pass = False
        for symbol in list(pending_fixture_symbols):
            resolution = _resolve_symbol(
                symbol,
                pending_fixture_symbols[symbol],
                scene,
                resolved_symbols,
            )
            if resolution.get("resolved_id") is None:
                dependency_symbol = resolution.get("dependency_symbol")
                if (
                    isinstance(dependency_symbol, str)
                    and dependency_symbol not in resolved_symbols
                    and dependency_symbol in pending_fixture_symbols
                ):
                    continue
                unresolved_symbols.append(resolution)
                pending_fixture_symbols.pop(symbol)
                continue
            resolved_symbols[symbol] = resolution
            pending_fixture_symbols.pop(symbol)
            resolved_in_pass = True

        if resolved_in_pass:
            continue

        for symbol, spec in pending_fixture_symbols.items():
            unresolved_symbols.append(
                _resolve_symbol(symbol, spec, scene, resolved_symbols)
            )
        break

    return {
        "version": grounding_map.get("version", GROUNDING_MAP_VERSION),
        "map_kind": grounding_map.get("map_kind", "scene_agnostic"),
        "composite_task": grounding_map.get("composite_task"),
        "is_complete": len(unresolved_symbols) == 0,
        "resolved_symbols": resolved_symbols,
        "unresolved_symbols": unresolved_symbols,
    }


def build_scene_summary(scene: dict[str, Any]) -> dict[str, Any]:
    """Builds a compact serializable scene summary for grounding outputs."""

    fixtures = scene.get("fixtures", {})
    objects = scene.get("objects", {})
    return {
        "fixtures": {
            fixture_id: {
                "fixture_type": fixture_info.get("fixture_type"),
                "can_place_objects": bool(fixture_info.get("can_place_objects", False)),
            }
            for fixture_id, fixture_info in fixtures.items()
            if isinstance(fixture_info, dict)
        },
        "objects": {
            object_id: {
                "object_type": object_info.get("object_type"),
                "location": object_info.get("location"),
            }
            for object_id, object_info in objects.items()
            if isinstance(object_info, dict)
        },
    }


def _normalize_initial_state_symbols(
    composite_task: str,
    initial_state: dict[str, Any],
) -> dict[str, Any]:
    """Normalizes legacy task symbols so grounding maps use canonical role names."""

    task_spec = load_task_spec(composite_task)
    grounding_spec = dict(task_spec.grounding)
    legacy_aliases = dict(grounding_spec.get("legacy_symbol_aliases", {}))
    if not legacy_aliases:
        return deepcopy(initial_state)
    return _rename_symbol_aliases(initial_state, legacy_aliases)


def _rename_symbol_aliases(
    value: Any,
    aliases: dict[str, str],
) -> Any:
    """Recursively rewrites legacy symbolic IDs inside one structured payload."""

    if isinstance(value, dict):
        return {
            aliases.get(key, key): _rename_symbol_aliases(item, aliases)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_rename_symbol_aliases(item, aliases) for item in value]
    if isinstance(value, str):
        return aliases.get(value, value)
    return deepcopy(value)


def _grounding_map(
    composite_task: str,
    *,
    symbols: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Builds the top-level scene-agnostic grounding-map payload."""

    return {
        "version": GROUNDING_MAP_VERSION,
        "map_kind": "scene_agnostic",
        "composite_task": composite_task,
        "symbols": symbols,
    }


def _resolve_symbol(
    symbol: str,
    spec: dict[str, Any],
    scene: dict[str, Any],
    resolved_symbols: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Resolves one symbol into a concrete scene identifier."""

    resolver = spec.get("resolver")
    if resolver == "object_by_type":
        return _resolve_object_by_type(symbol, spec, scene)
    if resolver == "source_fixture_for_object":
        return _resolve_fixture_for_object(
            symbol,
            spec,
            scene,
            resolved_symbols,
            reason_prefix="Using the source fixture for object symbol",
        )
    if resolver == "support_fixture_for_object":
        return _resolve_fixture_for_object(
            symbol,
            spec,
            scene,
            resolved_symbols,
            reason_prefix="Using the supporting fixture for object symbol",
        )
    if resolver == "unique_fixture_type":
        return _resolve_unique_fixture_type(symbol, spec, scene)
    if resolver == "nearest_placeable_surface_to_fixture":
        return _resolve_nearest_placeable_surface(symbol, spec, scene, resolved_symbols)
    return _unresolved_symbol(
        symbol,
        spec,
        reason=f"Unsupported resolver {resolver!r}.",
    )


def _resolve_object_by_type(
    symbol: str,
    spec: dict[str, Any],
    scene: dict[str, Any],
) -> dict[str, Any]:
    """Resolves one symbolic object to a concrete scene object."""

    object_type = spec.get("object_type")
    if not isinstance(object_type, str) or not object_type:
        return _unresolved_symbol(symbol, spec, reason="Missing object_type.")

    candidates = _object_candidates_for_type(
        scene,
        object_type,
        preferred_fixture_types=spec.get("preferred_fixture_types"),
    )
    if len(candidates) == 1:
        reason = f"Matched unique object with type {object_type!r}."
        if spec.get("preferred_fixture_types"):
            reason = (
                f"Matched unique object with type {object_type!r} after preferred "
                "fixture-type filtering."
            )
        confidence = 0.95 if spec.get("preferred_fixture_types") else 1.0
        return _resolved_symbol(
            symbol,
            spec,
            candidates[0],
            confidence=confidence,
            candidates=candidates,
            reason=reason,
        )
    if not candidates:
        return _unresolved_symbol(
            symbol,
            spec,
            reason=f"No scene object matched type {object_type!r}.",
            candidates=[],
        )
    return _unresolved_symbol(
        symbol,
        spec,
        reason=f"Multiple scene objects matched type {object_type!r}.",
        candidates=candidates,
    )


def _resolve_fixture_for_object(
    symbol: str,
    spec: dict[str, Any],
    scene: dict[str, Any],
    resolved_symbols: dict[str, dict[str, Any]],
    *,
    reason_prefix: str,
) -> dict[str, Any]:
    """Resolves one fixture symbol from a previously resolved object symbol."""

    object_symbol = spec.get("object_symbol")
    if not isinstance(object_symbol, str) or not object_symbol:
        return _unresolved_symbol(symbol, spec, reason="Missing object_symbol.")
    object_resolution = resolved_symbols.get(object_symbol)
    if object_resolution is None:
        return _unresolved_symbol(
            symbol,
            spec,
            reason=f"Object symbol {object_symbol!r} is not resolved yet.",
            dependency_symbol=object_symbol,
        )

    object_id = object_resolution["resolved_id"]
    object_info = scene.get("objects", {}).get(object_id)
    if not isinstance(object_info, dict):
        return _unresolved_symbol(
            symbol,
            spec,
            reason=f"Resolved object {object_id!r} is missing from the scene.",
        )

    fixture_id = object_info.get("location")
    if not isinstance(fixture_id, str):
        return _unresolved_symbol(
            symbol,
            spec,
            reason=f"Resolved object {object_id!r} has no fixture location.",
        )

    fixtures = scene.get("fixtures", {})
    fixture_info = fixtures.get(fixture_id)
    if not isinstance(fixture_info, dict):
        return _unresolved_symbol(
            symbol,
            spec,
            reason=f"Resolved object location {fixture_id!r} is not a known fixture.",
        )

    preferred_fixture_types = tuple(spec.get("preferred_fixture_types") or ())
    fixture_type = fixture_info.get("fixture_type")
    if preferred_fixture_types and fixture_type not in preferred_fixture_types:
        return _unresolved_symbol(
            symbol,
            spec,
            reason=(
                f"Fixture {fixture_id!r} has type {fixture_type!r}, which is outside "
                f"preferred types {sorted(preferred_fixture_types)!r}."
            ),
        )

    return _resolved_symbol(
        symbol,
        spec,
        fixture_id,
        confidence=1.0,
        candidates=[fixture_id],
        reason=f"{reason_prefix} {object_symbol!r}.",
    )


def _resolve_unique_fixture_type(
    symbol: str,
    spec: dict[str, Any],
    scene: dict[str, Any],
) -> dict[str, Any]:
    """Resolves one fixture symbol by unique fixture type."""

    fixture_type = spec.get("fixture_type")
    if not isinstance(fixture_type, str) or not fixture_type:
        return _unresolved_symbol(symbol, spec, reason="Missing fixture_type.")

    candidates = _fixture_candidates_for_types(scene, (fixture_type,))
    if len(candidates) == 1:
        return _resolved_symbol(
            symbol,
            spec,
            candidates[0],
            confidence=1.0,
            candidates=candidates,
            reason=f"Matched unique fixture with type {fixture_type!r}.",
        )
    if not candidates:
        return _unresolved_symbol(
            symbol,
            spec,
            reason=f"No scene fixture matched type {fixture_type!r}.",
            candidates=[],
        )
    return _unresolved_symbol(
        symbol,
        spec,
        reason=f"Multiple scene fixtures matched type {fixture_type!r}.",
        candidates=candidates,
    )


def _resolve_nearest_placeable_surface(
    symbol: str,
    spec: dict[str, Any],
    scene: dict[str, Any],
    resolved_symbols: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Resolves one fixture symbol to the nearest placeable surface."""

    anchor_fixture_symbol = spec.get("anchor_fixture_symbol")
    if not isinstance(anchor_fixture_symbol, str) or not anchor_fixture_symbol:
        return _unresolved_symbol(
            symbol,
            spec,
            reason="Missing anchor_fixture_symbol.",
        )

    anchor_resolution = resolved_symbols.get(anchor_fixture_symbol)
    if anchor_resolution is None:
        return _unresolved_symbol(
            symbol,
            spec,
            reason=f"Anchor fixture symbol {anchor_fixture_symbol!r} is not resolved yet.",
            dependency_symbol=anchor_fixture_symbol,
        )

    fixtures = scene.get("fixtures", {})
    anchor_fixture_id = anchor_resolution["resolved_id"]
    anchor_fixture = fixtures.get(anchor_fixture_id)
    if not isinstance(anchor_fixture, dict):
        return _unresolved_symbol(
            symbol,
            spec,
            reason=f"Resolved anchor fixture {anchor_fixture_id!r} is missing.",
        )

    preferred_fixture_types = tuple(spec.get("preferred_fixture_types") or ())
    candidates = _fixture_candidates_for_types(
        scene,
        preferred_fixture_types,
        require_placeable=True,
    )
    if not candidates:
        return _unresolved_symbol(
            symbol,
            spec,
            reason="No placeable candidate fixtures matched the preferred types.",
            candidates=[],
        )

    parent_fixture_id = anchor_fixture.get("parent_fixture")
    if isinstance(parent_fixture_id, str):
        parent_fixture = fixtures.get(parent_fixture_id)
        if isinstance(parent_fixture, dict) and bool(
            parent_fixture.get("can_place_objects", False)
        ) and (
            not preferred_fixture_types
            or parent_fixture.get("fixture_type") in preferred_fixture_types
        ):
            return _resolved_symbol(
                symbol,
                spec,
                parent_fixture_id,
                confidence=1.0,
                candidates=[parent_fixture_id],
                reason=f"Resolved via parent_fixture of {anchor_fixture_symbol!r}.",
            )

    if bool(anchor_fixture.get("can_place_objects", False)) and (
        not preferred_fixture_types
        or anchor_fixture.get("fixture_type") in preferred_fixture_types
    ):
        return _resolved_symbol(
            symbol,
            spec,
            anchor_fixture_id,
            confidence=1.0,
            candidates=[anchor_fixture_id],
            reason=f"Anchor fixture {anchor_fixture_symbol!r} is already placeable.",
        )

    anchor_xy = _xy_position(anchor_fixture.get("position"))
    if anchor_xy is None:
        return _unresolved_symbol(
            symbol,
            spec,
            reason=f"Anchor fixture {anchor_fixture_id!r} has no valid position.",
            candidates=candidates,
        )

    ranked_candidates = sorted(
        (
            (_distance_to_fixture(scene, fixture_id, anchor_xy), fixture_id)
            for fixture_id in candidates
        ),
        key=lambda item: item[0],
    )
    resolved_id = ranked_candidates[0][1]
    if len(ranked_candidates) > 1:
        best_distance = ranked_candidates[0][0]
        tied_candidates = [
            fixture_id
            for distance, fixture_id in ranked_candidates
            if distance <= best_distance + 0.15
        ]
        if len(tied_candidates) > 1:
            return _unresolved_symbol(
                symbol,
                spec,
                reason=(
                    f"Multiple placeable surfaces are similarly near {anchor_fixture_symbol!r}; "
                    "requires explicit grounding."
                ),
                candidates=tied_candidates,
            )
    return _resolved_symbol(
        symbol,
        spec,
        resolved_id,
        confidence=0.95,
        candidates=candidates,
        reason=f"Selected nearest placeable surface to {anchor_fixture_symbol!r}.",
    )


def _object_candidates_for_type(
    scene: dict[str, Any],
    object_type: str,
    *,
    preferred_fixture_types: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    """Returns scene object candidates that match one symbolic object entry."""

    object_type = str(object_type).lower()

    preferred_fixture_types = tuple(preferred_fixture_types or ())
    fixtures = scene.get("fixtures", {})
    candidates: list[str] = []
    for object_id, object_info in scene.get("objects", {}).items():
        if not isinstance(object_info, dict):
            continue
        actual_type = str(object_info.get("object_type", "")).lower()
        if not object_type_matches(object_type, actual_type):
            continue
        if preferred_fixture_types:
            location_id = object_info.get("location")
            fixture_type = fixtures.get(location_id, {}).get("fixture_type")
            if fixture_type not in preferred_fixture_types:
                continue
        candidates.append(object_id)
    return sorted(candidates)


def _fixture_candidates_for_types(
    scene: dict[str, Any],
    fixture_types: tuple[str, ...],
    *,
    require_placeable: bool = False,
) -> list[str]:
    """Returns scene fixture candidates that match one symbolic fixture entry."""

    candidates: list[str] = []
    for fixture_id, fixture_info in scene.get("fixtures", {}).items():
        if not isinstance(fixture_info, dict):
            continue
        if fixture_types and fixture_info.get("fixture_type") not in fixture_types:
            continue
        if require_placeable and not bool(fixture_info.get("can_place_objects", False)):
            continue
        candidates.append(fixture_id)
    return sorted(candidates)


def _distance_to_fixture(
    scene: dict[str, Any],
    fixture_id: str,
    anchor_xy: tuple[float, float],
) -> float:
    """Returns the 2-D distance between one fixture and an anchor point."""

    fixture_info = scene.get("fixtures", {}).get(fixture_id, {})
    fixture_xy = _xy_position(fixture_info.get("position"))
    if fixture_xy is None:
        return float("inf")
    dx = fixture_xy[0] - anchor_xy[0]
    dy = fixture_xy[1] - anchor_xy[1]
    return (dx * dx + dy * dy) ** 0.5


def _xy_position(position: Any) -> tuple[float, float] | None:
    """Returns the 2-D XY position tuple for one fixture position payload."""

    if not isinstance(position, (list, tuple)) or len(position) < 2:
        return None
    try:
        return (float(position[0]), float(position[1]))
    except (TypeError, ValueError):
        return None


def _resolved_symbol(
    symbol: str,
    spec: dict[str, Any],
    resolved_id: str,
    *,
    confidence: float,
    candidates: list[str],
    reason: str,
) -> dict[str, Any]:
    """Builds one successful symbol-resolution payload."""

    return {
        "symbol": symbol,
        "entity_type": spec.get("entity_type"),
        "resolver": spec.get("resolver"),
        "role": spec.get("role"),
        "resolved_id": resolved_id,
        "confidence": confidence,
        "candidates": candidates,
        "reason": reason,
    }


def _unresolved_symbol(
    symbol: str,
    spec: dict[str, Any],
    *,
    reason: str,
    candidates: list[str] | None = None,
    dependency_symbol: str | None = None,
) -> dict[str, Any]:
    """Builds one unresolved symbol-resolution payload."""

    payload = {
        "symbol": symbol,
        "entity_type": spec.get("entity_type"),
        "resolver": spec.get("resolver"),
        "role": spec.get("role"),
        "resolved_id": None,
        "reason": reason,
    }
    if candidates is not None:
        payload["candidates"] = candidates
    if dependency_symbol is not None:
        payload["dependency_symbol"] = dependency_symbol
    return payload
