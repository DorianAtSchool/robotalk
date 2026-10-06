"""Build symbolic per-run task instances and canonical raw trajectory payloads."""

from __future__ import annotations

from copy import deepcopy
from itertools import product
import random
from typing import Any, Callable, Sequence

from robotalk.tools.grounding_specs import build_grounding_map_for_task
from robotalk.utils import stable_json_sha256

from .constants import EXCLUSIVE_FIXTURE_TYPES
from .types import TaskInstance
from .workspace_semantics import canonical_agent_workspace


_ACCESS_PART_TO_TOOL = {
    "hinged_part": "open_hinged_part",
    "sliding_part": "open_sliding_part",
}


def _is_access_fixture_type(fixture_type: str) -> bool:
    """Recognize storage access fixtures without including appliances."""

    normalized = fixture_type.strip().lower()
    return (
        normalized == "cab"
        or normalized.startswith("cabinet")
        or normalized.startswith("fridge")
        or normalized.startswith("refrigerator")
        or normalized == "drawer"
        or normalized.startswith("drawer_")
        or normalized.endswith("_drawer")
    )


def _allowed_values(tool_spec: dict[str, Any], *names: str) -> set[str]:
    values: set[str] = set()
    for name in names:
        raw = tool_spec.get(name)
        if isinstance(raw, list):
            values.update(str(value) for value in raw if isinstance(value, str))
    return values


def eligible_access_state_parts(
    *,
    initial_state: dict[str, Any],
    allowed_tool_specs: dict[str, dict[str, Any]],
) -> tuple[tuple[str, str], ...]:
    """Return globally inferred access parts that can safely start open/closed."""

    eligible: list[tuple[str, str]] = []
    for fixture_id, fixture in (initial_state.get("fixtures") or {}).items():
        if not isinstance(fixture, dict):
            continue
        fixture_type = str(fixture.get("fixture_type") or "").lower()
        if not _is_access_fixture_type(fixture_type):
            continue
        for part_id, part in (fixture.get("parts") or {}).items():
            if not isinstance(part, dict) or part.get("state") not in {"open", "closed"}:
                continue
            tool_name = _ACCESS_PART_TO_TOOL.get(str(part.get("part_type") or "").lower())
            tool_spec = allowed_tool_specs.get(tool_name or "")
            if not isinstance(tool_spec, dict):
                continue
            targets = _allowed_values(
                tool_spec, "allowed_target_ids", "allowed_fixture_ids"
            )
            parts = _allowed_values(tool_spec, "allowed_part_ids")
            if fixture_id in targets and part_id in parts:
                eligible.append((str(fixture_id), str(part_id)))
    return tuple(sorted(eligible))


def _access_state_assignments(
    *,
    composite_task: str,
    initial_state: dict[str, Any],
    allowed_tool_specs: dict[str, dict[str, Any]],
) -> tuple[dict[tuple[str, str], str], ...]:
    """Enumerate eligible joint access states in reproducibly shuffled order."""

    parts = eligible_access_state_parts(
        initial_state=initial_state,
        allowed_tool_specs=allowed_tool_specs,
    )
    if not parts:
        return ({},)
    assignments = [
        dict(zip(parts, states, strict=True))
        for states in product(("closed", "open"), repeat=len(parts))
    ]
    rng = random.Random(
        stable_json_sha256(
            {"kind": "balanced_access_state_v2", "composite_task": composite_task}
        )
    )
    rng.shuffle(assignments)
    return tuple(assignments)


def _apply_access_state_assignment(
    initial_state: dict[str, Any],
    assignment: dict[tuple[str, str], str],
) -> None:
    for (fixture_id, part_id), state in assignment.items():
        initial_state["fixtures"][fixture_id]["parts"][part_id]["state"] = state


def _agent_location_assignments(
    *,
    composite_task: str,
    agent_ids: Sequence[str],
    fixture_ids: Sequence[str],
    fixture_states: dict[str, Any],
 ) -> tuple[dict[str, str], ...]:
    """Enumerate valid starts in reproducibly shuffled order."""

    seed_material = stable_json_sha256(
        {
            "composite_task": composite_task,
            "fixture_ids": tuple(fixture_ids),
            "agent_ids": tuple(agent_ids),
        }
    )
    rng = random.Random(seed_material)
    assignments: list[dict[str, str]] = []
    for locations in product(fixture_ids, repeat=len(agent_ids)):
        occupants: dict[str, int] = {}
        for location in locations:
            occupants[location] = occupants.get(location, 0) + 1
        if any(
            count > 1
            and str((fixture_states.get(location) or {}).get("fixture_type", "")).lower()
            in EXCLUSIVE_FIXTURE_TYPES
            for location, count in occupants.items()
        ):
            continue
        assignments.append(dict(zip(agent_ids, locations, strict=True)))
    if not assignments:
        raise ValueError(
            "Cannot sample executable agent starts: no valid fixture assignment exists."
        )
    rng.shuffle(assignments)
    return tuple(assignments)


def validate_initial_agent_locations(initial_state: dict[str, Any]) -> None:
    """Reject task instances that begin with two agents at one exclusive fixture."""

    fixtures = initial_state.get("fixtures") or {}
    agents = initial_state.get("agents") or {}
    occupants: dict[str, list[str]] = {}
    for agent_id, agent_state in agents.items():
        location = (agent_state or {}).get("location")
        if isinstance(location, str):
            canonical = canonical_agent_workspace(initial_state, location)
            occupants.setdefault(canonical or location, []).append(str(agent_id))
    conflicts = []
    for fixture_id, agent_ids in occupants.items():
        fixture = fixtures.get(fixture_id) or {}
        if len(agent_ids) > 1 and (
            isinstance(fixture, dict)
            and str(fixture.get("fixture_type", "")).lower()
            in EXCLUSIVE_FIXTURE_TYPES
        ):
            conflicts.append(f"{fixture_id}: {', '.join(sorted(agent_ids))}")
    if conflicts:
        raise ValueError(
            "Initial agent locations collide at exclusive fixtures: "
            + "; ".join(conflicts)
        )


def resolve_initial_position_fixture_ids(
    *,
    initial_state: dict[str, Any],
    allowed_tool_specs: dict[str, dict[str, Any]],
) -> tuple[str, ...]:
    """Resolves the fixture IDs agents may use as randomized starting positions."""

    navigate_spec = allowed_tool_specs.get("navigate_to_fixture", {})
    allowed_fixture_ids = navigate_spec.get("allowed_fixture_ids")
    if isinstance(allowed_fixture_ids, list) and allowed_fixture_ids:
        if not all(isinstance(fixture_id, str) for fixture_id in allowed_fixture_ids):
            raise ValueError(
                "navigate_to_fixture.allowed_fixture_ids must be a list of strings."
            )
        derived_reference_ids = set(
            navigate_spec.get("_derived_reference_fixture_ids") or ()
        )
        return tuple(dict.fromkeys(
            canonical_agent_workspace(initial_state, fixture_id) or fixture_id
            for fixture_id in allowed_fixture_ids
            if fixture_id not in derived_reference_ids
        ))

    fixture_state = initial_state.get("fixtures", {})
    if isinstance(fixture_state, dict) and fixture_state:
        fixture_ids = tuple(fixture_state)
        if not all(isinstance(fixture_id, str) for fixture_id in fixture_ids):
            raise ValueError("initial_state.fixtures keys must be strings.")
        return tuple(dict.fromkeys(
            canonical_agent_workspace(initial_state, fixture_id) or fixture_id
            for fixture_id in fixture_ids
        ))

    raise ValueError(
        "Tasks must define at least one fixture position for initial agent placement."
    )


def count_balanced_initial_configurations(
    *,
    composite_task: str,
    agent_ids: Sequence[str],
    initial_state: dict[str, Any],
    allowed_tool_specs: dict[str, dict[str, Any]],
    random_start_location: bool = True,
    random_access_state: bool = True,
) -> int:
    """Count joint start/access configurations used by the balanced sampler."""

    access_count = (
        len(
            _access_state_assignments(
                composite_task=composite_task,
                initial_state=initial_state,
                allowed_tool_specs=allowed_tool_specs,
            )
        )
        if random_access_state
        else 1
    )
    if not random_start_location:
        return access_count
    fixture_ids = resolve_initial_position_fixture_ids(
        initial_state=initial_state,
        allowed_tool_specs=allowed_tool_specs,
    )
    location_count = len(
        _agent_location_assignments(
            composite_task=composite_task,
            agent_ids=agent_ids,
            fixture_ids=fixture_ids,
            fixture_states=initial_state.get("fixtures") or {},
        )
    )
    return access_count * location_count


def build_randomized_fixture_task_instance(
    *,
    composite_task: str,
    agent_ids: Sequence[str],
    initial_state: dict[str, Any],
    allowed_tool_specs: dict[str, dict[str, Any]],
    run_index: int,
    runtime_config: Any | None = None,
) -> TaskInstance:
    """Builds one per-run task instance with configurable agent start fixtures."""

    sampled_initial_state = deepcopy(initial_state)
    random_start_location = True
    # Direct validator/spec construction has no generation policy and must
    # preserve the canonical task state. The raw-generation CLI opts in.
    random_access_state = False
    if runtime_config is not None:
        random_start_location = bool(
            getattr(runtime_config, "random_start_location", True)
        )
        random_access_state = bool(
            getattr(runtime_config, "random_access_state", True)
        )
    access_assignments = (
        _access_state_assignments(
            composite_task=composite_task,
            initial_state=sampled_initial_state,
            allowed_tool_specs=allowed_tool_specs,
        )
        if random_access_state
        else ({},)
    )
    if not random_start_location:
        access_assignment = access_assignments[run_index % len(access_assignments)]
        _apply_access_state_assignment(
            sampled_initial_state,
            access_assignment,
        )
        for agent_id in agent_ids:
            agent_state = sampled_initial_state.setdefault("agents", {}).setdefault(
                agent_id, {}
            )
            agent_state.setdefault("held_object", None)
        validate_initial_agent_locations(sampled_initial_state)
        return TaskInstance(
            initial_state=sampled_initial_state,
            physical_configuration={
                "agent_locations": {
                    agent_id: sampled_initial_state["agents"][agent_id]["location"]
                    for agent_id in agent_ids
                },
                "access_states": dict(access_assignment),
            },
        )

    fixture_ids = resolve_initial_position_fixture_ids(
        initial_state=initial_state,
        allowed_tool_specs=allowed_tool_specs,
    )
    location_assignments = _agent_location_assignments(
        composite_task=composite_task,
        agent_ids=agent_ids,
        fixture_ids=fixture_ids,
        fixture_states=sampled_initial_state.get("fixtures") or {},
    )
    # Cycle through the Cartesian product rather than drawing each dimension
    # independently. Across N runs, every valid joint initial configuration is
    # represented either floor(N / K) or ceil(N / K) times.
    access_index = run_index % len(access_assignments)
    location_index = (run_index // len(access_assignments)) % len(location_assignments)
    _apply_access_state_assignment(sampled_initial_state, access_assignments[access_index])
    sampled_agent_locations = location_assignments[location_index]

    for agent_id in agent_ids:
        agent_state = sampled_initial_state.setdefault("agents", {}).setdefault(
            agent_id, {}
        )
        agent_state["location"] = sampled_agent_locations[agent_id]
        agent_state.setdefault("held_object", None)

    validate_initial_agent_locations(sampled_initial_state)
    return TaskInstance(
        initial_state=sampled_initial_state,
        physical_configuration={
            "agent_locations": dict(sampled_agent_locations),
            "access_states": dict(access_assignments[access_index]),
        },
    )


def initial_state_from_configuration(
    initial_state: dict[str, Any], configuration: dict[str, Any]
) -> dict[str, Any]:
    """Apply one frozen compact configuration to a canonical task state."""

    resolved = deepcopy(initial_state)
    agents = resolved.setdefault("agents", {})
    for agent_id, location in (configuration.get("agent_locations") or {}).items():
        if agent_id not in agents:
            raise ValueError(f"configuration names unknown agent {agent_id!r}")
        agents[agent_id]["location"] = location
        agents[agent_id].setdefault("held_object", None)
    for path, state in (configuration.get("access_states") or {}).items():
        try:
            fixture_id, part_id = path.split(".", 1)
            part = resolved["fixtures"][fixture_id]["parts"][part_id]
        except (KeyError, ValueError) as exc:
            raise ValueError(f"configuration names unknown access part {path!r}") from exc
        if state not in {"open", "closed"}:
            raise ValueError(f"invalid access state {state!r} for {path}")
        part["state"] = state
    validate_initial_agent_locations(resolved)
    return resolved


def build_symbolic_trajectory_record(
    *,
    composite_task: str,
    agent_ids: Sequence[str],
    candidate: dict[str, Any],
    validation: dict[str, Any],
    trajectory_id: str,
    generation_usage: dict[str, Any],
    task_instance: TaskInstance,
) -> dict[str, Any]:
    """Builds the persisted raw trajectory payload for one symbolic task."""

    return {
        "trajectory_id": trajectory_id,
        "composite_task": composite_task,
        "agents": build_canonical_agents(agent_ids),
        "initial_state": task_instance.initial_state,
        "steps": candidate.get("steps"),
        "validation": validation,
        "generation_usage": generation_usage,
        "grounding_map": build_grounding_map_for_task(
            composite_task,
            task_instance.initial_state,
        ),
    }


def build_canonical_agents(agent_ids: Sequence[str]) -> list[dict[str, str]]:
    """Builds the shared persisted agent roster for trajectories."""

    if not agent_ids:
        raise ValueError("agent_ids must contain at least one agent.")
    return [{"agent": agent_id} for agent_id in agent_ids]


def make_symbolic_trajectory_record_builder(
    *,
    composite_task: str,
    agent_ids: Sequence[str],
) -> Callable[
    [dict[str, Any], dict[str, Any], str, dict[str, Any], TaskInstance], dict[str, Any]
]:
    """Builds a reusable symbolic raw-trajectory writer for one task."""

    def build_trajectory_record(
        candidate: dict[str, Any],
        validation: dict[str, Any],
        trajectory_id: str,
        generation_usage: dict[str, Any],
        task_instance: TaskInstance,
    ) -> dict[str, Any]:
        """Builds one symbolic raw trajectory record."""

        return build_symbolic_trajectory_record(
            composite_task=composite_task,
            agent_ids=agent_ids,
            candidate=candidate,
            validation=validation,
            trajectory_id=trajectory_id,
            generation_usage=generation_usage,
            task_instance=task_instance,
        )

    return build_trajectory_record
