from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re
from typing import Any, Iterable

import yaml


_LAYOUT_ROOT = (
    Path(__file__).resolve().parents[1]
    / "models"
    / "assets"
    / "scenes"
    / "kitchen_layouts"
)

# Limit pre-build fixture pruning to surface clutter / countertop appliances.
# Structural fixtures stay intact unless a task explicitly manages them.
REMOVABLE_FIXTURE_TYPES = frozenset(
    {
        "blender",
        "coffee_machine",
        "digital_scale",
        "dish_rack",
        "electric_kettle",
        "flower_vase",
        "fruit_bowl",
        "glass_cup",
        "jar",
        "jar_lid",
        "knife_block",
        "oil_bottle",
        "paper_towel",
        "pepper_shaker",
        "plant",
        "salt_shaker",
        "soap_dispenser",
        "stand_mixer",
        "tiered_basket",
        "toaster",
        "toaster_oven",
        "turmeric",
        "cinnamon",
        "paprika",
        "utensil_holder",
        "utensil_set",
        "vinegar_bottle",
    }
)

_AUTO_GENERATED_SUFFIXES = ("_container", "_auxiliary")
_GENERIC_OBJECT_NAME_RE = re.compile(r"^(obj(?:ect)?(?:[_-]?\d+)?|item(?:[_-]?\d+)?)$")


def build_trajectory_pruning_config(
    trajectory: dict[str, Any],
    *,
    layout: int | None = None,
) -> dict[str, Any]:
    """Return fixture-disable updates and trajectory object requirements."""

    initial_state = trajectory.get("initial_state") or {}
    grounding_map = trajectory.get("grounding_map") or {}

    required_object_names, required_object_types = (
        _extract_required_object_requirements(
            initial_state=initial_state,
            grounding_map=grounding_map,
        )
    )
    required_fixture_types = _extract_required_fixture_types(
        initial_state=initial_state,
        grounding_map=grounding_map,
    )

    update_fxtr_cfg_dict = {}
    if layout is not None:
        update_fxtr_cfg_dict = _build_fixture_disable_updates(
            layout=layout,
            required_fixture_types=required_fixture_types,
        )

    return {
        "update_fxtr_cfg_dict": update_fxtr_cfg_dict,
        "trajectory_object_names": tuple(sorted(required_object_names)),
        "trajectory_object_types": tuple(sorted(required_object_types)),
        "trajectory_object_specs": _extract_required_object_specs(initial_state),
        "trajectory_fixture_types": tuple(sorted(required_fixture_types)),
    }


def filter_object_cfgs_for_trajectory(
    object_cfgs: Iterable[dict[str, Any]],
    *,
    required_object_names: Iterable[str] | None = None,
    required_object_types: Iterable[str] | None = None,
    required_object_specs: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Keep only task objects that are referenced by the trajectory."""

    required_specs = normalize_required_object_specs(
        required_object_specs,
        required_object_names=required_object_names,
        required_object_types=required_object_types,
    )
    if not required_specs:
        return [deepcopy(cfg) for cfg in object_cfgs]

    cfg_copies = [deepcopy(cfg) for cfg in object_cfgs]
    matches = resolve_trajectory_object_cfg_matches(
        cfg_copies,
        required_object_specs=required_specs,
    )
    keep_names = set(matches.values())
    keep_names.update(
        _expand_keep_names_with_implicit_support_cfgs(
            cfg_copies,
            initial_keep_names=keep_names,
        )
    )

    filtered: list[dict[str, Any]] = []
    for cfg_copy in cfg_copies:
        if str(cfg_copy.get("name", "")).strip() in keep_names:
            filtered.append(cfg_copy)
    return filtered


def _expand_keep_names_with_implicit_support_cfgs(
    object_cfgs: list[dict[str, Any]],
    *,
    initial_keep_names: set[str],
) -> set[str]:
    """Keep receptacle/support cfgs referenced via placement.try_to_place_in.

    Example: if kept cfg `steak` has `try_to_place_in="plate"`, keep the cfg
    that declares type `plate` as well, even if it is not explicitly named in
    the trajectory spec.
    """

    cfg_by_name: dict[str, dict[str, Any]] = {
        str(cfg.get("name", "")).strip(): cfg
        for cfg in object_cfgs
        if str(cfg.get("name", "")).strip()
    }
    keep_names = set(initial_keep_names)
    changed = True
    while changed:
        changed = False
        for cfg_name in tuple(sorted(keep_names)):
            cfg = cfg_by_name.get(cfg_name)
            if not isinstance(cfg, dict):
                continue
            placement = cfg.get("placement")
            if not isinstance(placement, dict):
                continue
            support_type = placement.get("try_to_place_in")
            if not isinstance(support_type, str) or not support_type.strip():
                continue
            support_type = support_type.strip()
            for candidate_name, candidate_cfg in cfg_by_name.items():
                if candidate_name in keep_names:
                    continue
                if _object_cfg_declares_type(candidate_cfg, support_type):
                    keep_names.add(candidate_name)
                    changed = True
                    break
    return keep_names


def _object_cfg_declares_type(object_cfg: dict[str, Any], object_type: str) -> bool:
    """Return whether cfg can instantiate the requested semantic object type."""

    normalized_type = str(object_type).strip()
    if not normalized_type:
        return False

    candidate_types: set[str] = set()
    cfg_name = str(object_cfg.get("name", "")).strip()
    if cfg_name:
        candidate_types.add(cfg_name)
    obj_groups = object_cfg.get("obj_groups")
    if isinstance(obj_groups, str):
        candidate_types.add(obj_groups)
    elif isinstance(obj_groups, (list, tuple, set)):
        candidate_types.update(str(group) for group in obj_groups if group)
    return normalized_type in candidate_types


def resolve_trajectory_object_cfg_matches(
    object_cfgs: Iterable[dict[str, Any]],
    *,
    required_object_specs: dict[str, dict[str, Any]],
) -> dict[str, str]:
    """Resolve symbolic trajectory object names to concrete object cfg names."""

    remaining_specs = normalize_required_object_specs(required_object_specs)
    available_cfgs = {
        str(cfg.get("name", "")).strip(): deepcopy(cfg)
        for cfg in object_cfgs
        if str(cfg.get("name", "")).strip()
    }
    matches: dict[str, str] = {}

    def _bind(symbol: str, cfg_name: str) -> None:
        matches[symbol] = cfg_name
        remaining_specs.pop(symbol, None)
        available_cfgs.pop(cfg_name, None)

    for symbol in sorted(tuple(remaining_specs)):
        if symbol not in available_cfgs:
            continue
        cfg = available_cfgs[symbol]
        if _object_cfg_matches_spec(cfg, remaining_specs[symbol]):
            _bind(symbol, symbol)

    for symbol in sorted(tuple(remaining_specs)):
        symbol_ordinal = _extract_trailing_ordinal(symbol)
        if symbol_ordinal is None:
            continue
        spec = remaining_specs[symbol]
        ordinal_candidates = [
            cfg_name
            for cfg_name, cfg in available_cfgs.items()
            if _object_cfg_matches_spec(cfg, spec)
            and _extract_trailing_ordinal(cfg_name) == symbol_ordinal
        ]
        if len(ordinal_candidates) == 1:
            _bind(symbol, ordinal_candidates[0])

    changed = True
    while changed:
        changed = False
        for symbol in sorted(tuple(remaining_specs)):
            spec = remaining_specs[symbol]
            candidates = [
                cfg_name
                for cfg_name, cfg in available_cfgs.items()
                if _object_cfg_matches_spec(cfg, spec)
            ]
            if not candidates:
                continue

            non_generated = [
                cfg_name
                for cfg_name in candidates
                if not is_generated_object_name(cfg_name)
            ]

            chosen = None
            if len(non_generated) == 1:
                chosen = non_generated[0]
            elif len(non_generated) == 0 and len(candidates) == 1:
                chosen = candidates[0]

            if chosen is None:
                continue

            _bind(symbol, chosen)
            changed = True
            break

    return matches


def should_keep_object_cfg_for_trajectory(
    object_cfg: dict[str, Any],
    *,
    required_object_names: Iterable[str],
    required_object_types: Iterable[str],
) -> bool:
    """Return whether one object config should survive trajectory pruning."""

    name_set = {str(name) for name in required_object_names if name}
    type_set = {str(obj_type) for obj_type in required_object_types if obj_type}

    name = str(object_cfg.get("name", "")).strip()
    if name in name_set:
        return True

    if not _object_cfg_matches_required_type(object_cfg, type_set):
        return False

    if not name:
        return True
    if name.endswith(_AUTO_GENERATED_SUFFIXES):
        return True
    if _GENERIC_OBJECT_NAME_RE.match(name):
        return True
    return False


def normalize_required_object_specs(
    required_object_specs: dict[str, dict[str, Any]] | None,
    *,
    required_object_names: Iterable[str] | None = None,
    required_object_types: Iterable[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Normalize symbolic trajectory object requirements into one dict."""

    normalized: dict[str, dict[str, Any]] = {}
    if required_object_specs:
        for symbol, spec in required_object_specs.items():
            if not symbol:
                continue
            normalized[str(symbol)] = {
                "object_type": str(spec.get("object_type", "")).strip() or None,
                "location": spec.get("location"),
            }
        return normalized

    names = [str(name) for name in (required_object_names or []) if name]
    types = [str(obj_type) for obj_type in (required_object_types or []) if obj_type]
    for idx, symbol in enumerate(names):
        object_type = types[idx] if idx < len(types) else None
        normalized[symbol] = {"object_type": object_type}
    return normalized


def is_generated_object_name(name: str) -> bool:
    """Return whether an object name is auto-generated rather than task-authored."""

    normalized = str(name).strip()
    if not normalized:
        return False
    return normalized.endswith(_AUTO_GENERATED_SUFFIXES) or bool(
        _GENERIC_OBJECT_NAME_RE.match(normalized)
    )


def _extract_trailing_ordinal(name: str) -> int | None:
    normalized = str(name).strip()
    if not normalized:
        return None
    tail = normalized.rsplit("_", 1)[-1]
    if tail.isdigit():
        return int(tail)
    return None


def _extract_required_object_requirements(
    *,
    initial_state: dict[str, Any],
    grounding_map: dict[str, Any],
) -> tuple[set[str], set[str]]:
    names: set[str] = set()
    object_types: set[str] = set()

    for name, obj_state in (initial_state.get("objects") or {}).items():
        names.add(str(name))
        object_type = obj_state.get("object_type")
        if object_type:
            object_types.add(str(object_type))

    for symbol, spec in (grounding_map.get("symbols") or {}).items():
        if spec.get("entity_type") != "object":
            continue
        names.add(str(symbol))
        object_type = spec.get("object_type")
        if object_type:
            object_types.add(str(object_type))

    return names, object_types


def _extract_required_object_specs(
    initial_state: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}
    for name, obj_state in (initial_state.get("objects") or {}).items():
        specs[str(name)] = {
            "object_type": obj_state.get("object_type"),
            "location": obj_state.get("location"),
        }
    return specs


def _extract_required_fixture_types(
    *,
    initial_state: dict[str, Any],
    grounding_map: dict[str, Any],
) -> set[str]:
    fixture_types: set[str] = set()

    for fixture_state in (initial_state.get("fixtures") or {}).values():
        fixture_type = fixture_state.get("fixture_type")
        if fixture_type:
            fixture_types.add(str(fixture_type))

    for spec in (grounding_map.get("symbols") or {}).values():
        fixture_type = spec.get("fixture_type")
        if fixture_type:
            fixture_types.add(str(fixture_type))
        for preferred_type in spec.get("preferred_fixture_types") or ():
            if preferred_type:
                fixture_types.add(str(preferred_type))

    return fixture_types


def _build_fixture_disable_updates(
    *,
    layout: int,
    required_fixture_types: set[str],
) -> dict[str, dict[str, bool]]:
    layout_config = _load_layout_config(layout)
    updates: dict[str, dict[str, bool]] = {}
    for fixture_cfg in _iter_fixture_configs(layout_config):
        fixture_name = fixture_cfg.get("name")
        fixture_type = fixture_cfg.get("type")
        if not fixture_name or fixture_type not in REMOVABLE_FIXTURE_TYPES:
            continue
        if fixture_type in required_fixture_types:
            continue
        updates[str(fixture_name)] = {"enable": False}
    return updates


def _load_layout_config(layout: int) -> dict[str, Any]:
    if layout <= 0:
        raise ValueError(f"Layout ids must be positive, got {layout!r}")
    split = "test" if layout <= 10 else "train"
    layout_path = _LAYOUT_ROOT / split / f"layout{layout:03d}.yaml"
    with open(layout_path, "r") as f:
        return yaml.safe_load(f)


def _iter_fixture_configs(value: Any):
    if isinstance(value, dict):
        if "name" in value and "type" in value:
            yield value
        for child in value.values():
            yield from _iter_fixture_configs(child)
        return
    if isinstance(value, list):
        for child in value:
            yield from _iter_fixture_configs(child)


def _object_cfg_matches_required_type(
    object_cfg: dict[str, Any],
    required_object_types: set[str],
) -> bool:
    if not required_object_types:
        return False

    candidate_types: set[str] = set()
    obj_groups = object_cfg.get("obj_groups")
    if isinstance(obj_groups, str):
        candidate_types.add(obj_groups)
    elif isinstance(obj_groups, (list, tuple, set)):
        candidate_types.update(str(group) for group in obj_groups if group)

    placement = object_cfg.get("placement") or {}
    container_type = placement.get("try_to_place_in")
    if isinstance(container_type, str) and container_type:
        candidate_types.add(container_type)

    return bool(candidate_types & required_object_types)


def _object_cfg_matches_spec(
    object_cfg: dict[str, Any],
    required_object_spec: dict[str, Any],
) -> bool:
    required_type = required_object_spec.get("object_type")
    if not required_type:
        return False

    # ``try_to_place_in`` describes an implicit generated container for this
    # object, not the object's own type. Binding a symbolic container such as
    # "pan" to the child steak cfg prevents Kitchen from later creating and
    # binding the native ``obj_container``.
    #
    # Include cfg ``name`` as a candidate semantic type. Several tasks use a
    # coarse symbolic type (e.g. "spice", "bottle"), while ``obj_groups``
    # enumerates concrete variants (e.g. turmeric/paprika). If we only match
    # against ``obj_groups``, trajectory pruning can drop required task objects
    # and fail env init.
    candidate_types: set[str] = set()
    cfg_name = object_cfg.get("name")
    if isinstance(cfg_name, str) and cfg_name.strip():
        candidate_types.add(cfg_name.strip())
    obj_groups = object_cfg.get("obj_groups")
    if isinstance(obj_groups, str):
        candidate_types.add(obj_groups)
    elif isinstance(obj_groups, (list, tuple, set)):
        candidate_types.update(str(group) for group in obj_groups if group)

    return str(required_type) in candidate_types
