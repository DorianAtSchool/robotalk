"""Load JSON-backed task specs that mirror task-level Python definitions."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any


def derive_effective_allowed_tool_specs(
    initial_state: dict[str, Any],
    allowed_tool_specs: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Add fixture-relative placement anchors as valid navigation targets.

    A fixture used by ``place_next_to`` may declare the nearby placement
    surface through ``machine_state[fixture].adjacent_location_id``.  Both the
    anchor and that surface are physically valid approach poses, so expose the
    anchor to navigation without duplicating the relationship in every spec.
    """

    effective = {
        str(tool_name): dict(tool_spec)
        for tool_name, tool_spec in allowed_tool_specs.items()
    }
    navigation = effective.get("navigate_to_fixture")
    next_to = effective.get("place_next_to")
    if not isinstance(navigation, dict) or not isinstance(next_to, dict):
        return effective

    explicit_targets = navigation.get("allowed_fixture_ids")
    reference_ids = next_to.get("allowed_reference_fixture_ids")
    if not isinstance(explicit_targets, list) or not isinstance(reference_ids, list):
        return effective

    fixtures = initial_state.get("fixtures") or {}
    machine_state = initial_state.get("machine_state") or {}
    targets = list(explicit_targets)
    derived_targets: list[str] = []
    for reference_id in reference_ids:
        if not isinstance(reference_id, str) or reference_id not in fixtures:
            continue
        reference_state = machine_state.get(reference_id)
        if not isinstance(reference_state, dict):
            continue
        adjacent_id = reference_state.get("adjacent_location_id")
        if not isinstance(adjacent_id, str) or adjacent_id not in fixtures:
            continue
        if reference_id not in targets:
            targets.append(reference_id)
            derived_targets.append(reference_id)

    navigation["allowed_fixture_ids"] = targets
    if derived_targets:
        # Initial-position sampling historically consumes allowed_fixture_ids.
        # Keep newly legal approach poses out of that independent distribution
        # unless a task explicitly listed them as starts before derivation.
        navigation["_derived_reference_fixture_ids"] = derived_targets
    return effective


@dataclass(frozen=True)
class PreflightTokenEstimateSpec:
    """Serializable preflight token estimate for one task spec."""

    prompt_tokens: int
    output_tokens: int
    reasoning_tokens: int = 0

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PreflightTokenEstimateSpec":
        """Build one token estimate from a JSON object."""

        return cls(
            prompt_tokens=int(payload["prompt_tokens"]),
            output_tokens=int(payload["output_tokens"]),
            reasoning_tokens=int(payload.get("reasoning_tokens", 0)),
        )


@dataclass(frozen=True)
class TaskSpec:
    """Structured representation of one JSON-backed task configuration."""

    spec_version: int
    composite_task: str
    source_python_module: str
    agent_ids: tuple[str, ...]
    max_reasoning_chars: int
    validator_checks: tuple[str, ...]
    preflight_token_estimate: PreflightTokenEstimateSpec
    initial_state: dict[str, Any]
    allowed_tool_specs: dict[str, dict[str, Any]]
    task_goal: str
    extra_execution_rules: tuple[str, ...]
    initial_public_state: dict[str, Any]
    task_preconditions: tuple[dict[str, Any], ...]
    goal_conditions: tuple[dict[str, Any], ...]
    task_effects: tuple[dict[str, Any], ...]
    grounding: dict[str, Any]
    example_trajectory: dict[str, Any]
    notes: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TaskSpec":
        """Build one validated task spec from a JSON object."""

        required_fields = (
            "spec_version",
            "composite_task",
            "source_python_module",
            "agent_ids",
            "max_reasoning_chars",
            "validator_checks",
            "preflight_token_estimate",
            "initial_state",
            "allowed_tool_specs",
            "task_goal",
            "extra_execution_rules",
            "initial_public_state",
            "task_preconditions",
            "goal_conditions",
            "task_effects",
            "grounding",
            "example_trajectory",
        )
        missing_fields = [field_name for field_name in required_fields if field_name not in payload]
        if missing_fields:
            missing = ", ".join(sorted(missing_fields))
            raise ValueError(f"Task spec is missing required fields: {missing}")

        initial_state = dict(payload["initial_state"])
        allowed_tool_specs = derive_effective_allowed_tool_specs(
            initial_state,
            {
                str(tool_name): dict(tool_spec)
                for tool_name, tool_spec in dict(payload["allowed_tool_specs"]).items()
            },
        )

        return cls(
            spec_version=int(payload["spec_version"]),
            composite_task=str(payload["composite_task"]),
            source_python_module=str(payload["source_python_module"]),
            agent_ids=tuple(str(agent_id) for agent_id in payload["agent_ids"]),
            max_reasoning_chars=int(payload["max_reasoning_chars"]),
            validator_checks=tuple(str(check_name) for check_name in payload["validator_checks"]),
            preflight_token_estimate=PreflightTokenEstimateSpec.from_dict(
                dict(payload["preflight_token_estimate"])
            ),
            initial_state=initial_state,
            allowed_tool_specs=allowed_tool_specs,
            task_goal=str(payload["task_goal"]),
            extra_execution_rules=tuple(
                str(rule_text) for rule_text in payload["extra_execution_rules"]
            ),
            initial_public_state=dict(payload["initial_public_state"]),
            task_preconditions=tuple(
                dict(condition_spec) for condition_spec in payload["task_preconditions"]
            ),
            goal_conditions=tuple(
                dict(condition_spec) for condition_spec in payload["goal_conditions"]
            ),
            task_effects=tuple(
                dict(effect_spec) for effect_spec in payload["task_effects"]
            ),
            grounding=dict(payload["grounding"]),
            example_trajectory=dict(payload["example_trajectory"]),
            notes=tuple(str(note_text) for note_text in payload.get("notes", ())),
        )


SPEC_DIRECTORY = Path(__file__).resolve().parent
VERIFIED_SPEC_DIRECTORY = SPEC_DIRECTORY / "verified"
TASK_SPEC_DIRECTORY_OVERRIDE_ENV_VAR = "ROBOCASA_TASK_SPEC_DIR"


def _default_spec_directories() -> tuple[Path, ...]:
    """Choose the checked-in spec inventories used when no override is configured."""

    directories: list[Path] = []
    if VERIFIED_SPEC_DIRECTORY.is_dir():
        directories.append(VERIFIED_SPEC_DIRECTORY)
    if any(SPEC_DIRECTORY.glob("*.json")):
        directories.append(SPEC_DIRECTORY)
    if not directories:
        directories.append(SPEC_DIRECTORY)
    return tuple(directories)


def _resolve_spec_directories() -> tuple[Path, ...]:
    override_directory = os.environ.get(TASK_SPEC_DIRECTORY_OVERRIDE_ENV_VAR)
    if override_directory:
        return (Path(override_directory).expanduser().resolve(),)
    return _default_spec_directories()


def _normalized_task_spec_filename(task_name: str) -> str:
    """Build the normalized JSON filename for one composite task name."""

    normalized_name = "".join(
        character.lower() if character.isalnum() else "_" for character in task_name
    )
    collapsed_name = "_".join(part for part in normalized_name.split("_") if part)
    return f"{collapsed_name}.json"


def _task_spec_paths(task_name: str) -> tuple[Path, ...]:
    """Resolve candidate JSON file paths for one composite task name."""

    filename = _normalized_task_spec_filename(task_name)
    return tuple(directory / filename for directory in _resolve_spec_directories())


def load_task_spec(task_name: str) -> TaskSpec:
    """Load one JSON-backed task spec by composite task name."""

    candidate_paths = _task_spec_paths(task_name)
    for spec_path in candidate_paths:
        if not spec_path.exists():
            continue
        with spec_path.open("r", encoding="utf-8") as handle:
            return TaskSpec.from_dict(json.load(handle))
    raise FileNotFoundError(
        f"Unknown task spec {task_name!r}: {candidate_paths[0]}"
    )


def load_all_task_specs() -> tuple[TaskSpec, ...]:
    """Load every JSON-backed task spec shipped in this package."""

    specs: list[TaskSpec] = []
    seen_filenames: set[str] = set()
    for directory in _resolve_spec_directories():
        for spec_path in sorted(directory.glob("*.json")):
            if spec_path.name in seen_filenames:
                continue
            seen_filenames.add(spec_path.name)
            with spec_path.open("r", encoding="utf-8") as handle:
                specs.append(TaskSpec.from_dict(json.load(handle)))
    return tuple(specs)


def load_verified_task_specs() -> tuple[TaskSpec, ...]:
    """Load the canonical flat verified JSON-backed task specs."""

    if not VERIFIED_SPEC_DIRECTORY.is_dir():
        return ()
    specs: list[TaskSpec] = []
    for spec_path in sorted(VERIFIED_SPEC_DIRECTORY.glob("*.json")):
        with spec_path.open("r", encoding="utf-8") as handle:
            specs.append(TaskSpec.from_dict(json.load(handle)))
    return tuple(specs)


def supported_verified_task_names() -> tuple[str, ...]:
    """Return the canonical verified composite task names in sorted order."""

    return tuple(sorted(task_spec.composite_task for task_spec in load_verified_task_specs()))
