"""Canonical symbolic workspace relationships shared across the pipeline.

Cabinets are storage entities, but an agent standing "at" a cabinet occupies
the same broad workspace as the counter supporting it.  Object containment is
deliberately not canonicalized: an object in a cabinet remains in the cabinet.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Iterable


CABINET_FIXTURE_TYPES = frozenset(
    {
        "cab",
        "cabinet",
        "cabinet_single_door",
        "cabinet_double_door",
        "cabinet_with_door",
    }
)


def fixture_type(initial_state: dict[str, Any], fixture_id: str | None) -> str:
    state = (initial_state.get("fixtures") or {}).get(fixture_id)
    return str((state or {}).get("fixture_type") or "").strip().lower()


def is_cabinet(initial_state: dict[str, Any], fixture_id: str | None) -> bool:
    return fixture_type(initial_state, fixture_id) in CABINET_FIXTURE_TYPES


def parent_fixture_id(
    initial_state: dict[str, Any], fixture_id: str | None
) -> str | None:
    state = (initial_state.get("fixtures") or {}).get(fixture_id)
    parent = (state or {}).get("parent_fixture")
    return parent if isinstance(parent, str) and parent else None


def canonical_agent_workspace(
    initial_state: dict[str, Any], fixture_id: str | None
) -> str | None:
    """Return the canonical workspace for an agent location or navigation.

    Only cabinets alias their parent. Exclusive child appliances retain their
    own identity even though they may access their parent surface.
    """

    if isinstance(fixture_id, str) and is_cabinet(initial_state, fixture_id):
        return parent_fixture_id(initial_state, fixture_id) or fixture_id
    return fixture_id


def cabinet_children_of(
    initial_state: dict[str, Any], parent_id: str
) -> tuple[str, ...]:
    return tuple(
        fixture_id
        for fixture_id in (initial_state.get("fixtures") or {})
        if is_cabinet(initial_state, fixture_id)
        and parent_fixture_id(initial_state, fixture_id) == parent_id
    )


def exclusive_children_of(
    initial_state: dict[str, Any],
    parent_id: str,
    *,
    exclusive_fixture_types: Iterable[str],
) -> tuple[str, ...]:
    exclusive = {str(value).lower() for value in exclusive_fixture_types}
    return tuple(
        fixture_id
        for fixture_id, state in (initial_state.get("fixtures") or {}).items()
        if parent_fixture_id(initial_state, fixture_id) == parent_id
        and str((state or {}).get("fixture_type") or "").lower() in exclusive
    )


def accepted_agent_workspaces(
    initial_state: dict[str, Any],
    expected_fixture_id: str,
    *,
    exclusive_fixture_types: Iterable[str] = (),
) -> tuple[str, ...]:
    """Return locations from which an agent may use ``expected_fixture_id``.

    Cabinet and parent aliases work in both directions. An agent at an
    exclusive child may use its parent while retaining the child location.
    """

    canonical = canonical_agent_workspace(initial_state, expected_fixture_id)
    accepted = [expected_fixture_id]
    if isinstance(canonical, str):
        accepted.append(canonical)
        accepted.extend(cabinet_children_of(initial_state, canonical))
        accepted.extend(
            exclusive_children_of(
                initial_state,
                canonical,
                exclusive_fixture_types=exclusive_fixture_types,
            )
        )
    return tuple(dict.fromkeys(accepted))


def canonicalize_agent_locations(
    initial_state: dict[str, Any],
) -> dict[str, Any]:
    """Copy a state and canonicalize agent locations only."""

    normalized = deepcopy(initial_state)
    for state in (normalized.get("agents") or {}).values():
        if not isinstance(state, dict):
            continue
        state["location"] = canonical_agent_workspace(
            normalized, state.get("location")
        )
    return normalized


def canonicalize_configuration(
    configuration: dict[str, Any], initial_state: dict[str, Any]
) -> dict[str, Any]:
    """Copy a compact configuration and canonicalize its agent locations."""

    normalized = deepcopy(configuration)
    locations = normalized.get("agent_locations") or {}
    normalized["agent_locations"] = {
        agent_id: canonical_agent_workspace(initial_state, location)
        for agent_id, location in locations.items()
    }
    return normalized
