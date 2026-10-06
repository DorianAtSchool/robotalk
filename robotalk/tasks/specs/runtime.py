"""Build runtime task definitions from JSON-backed task specs."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from typing import Any

from robotalk.tools.subatomic_tool_specs import (
    build_allowed_tool_specs,
)
from robotalk.tasks.shared.errors import (
    TaskPreconditionSemanticValidationError,
)
from robotalk.tasks.shared.fsm import FiniteStateTaskValidator
from robotalk.tasks.shared.instances import (
    build_randomized_fixture_task_instance,
    make_symbolic_trajectory_record_builder,
)
from robotalk.tasks.shared.prompting import make_task_prompt_builder
from robotalk.tasks.shared.schema import build_task_response_schema
from robotalk.generation.tick_format import TICK_FORMAT_RULES
from robotalk.generation.tick_format import build_tick_response_schema
from robotalk.tasks.shared.state import TaskRuntimeState
from robotalk.tasks.shared.types import (
    PreflightTokenEstimate,
    TaskDefinition,
    TaskInstance,
)

from . import TaskSpec, load_all_task_specs


def _hinged_parts_by_fixture(initial_state: dict[str, Any]) -> dict[str, list[str]]:
    hinged_parts: dict[str, list[str]] = {}
    for fixture_id, fixture_state in initial_state.get("fixtures", {}).items():
        if not isinstance(fixture_state, dict):
            continue
        fixture_parts = fixture_state.get("parts", {})
        if not isinstance(fixture_parts, dict):
            continue
        part_ids = [
            str(part_id)
            for part_id, part_state in fixture_parts.items()
            if isinstance(part_state, dict)
            and part_state.get("part_type") == "hinged_part"
        ]
        if part_ids:
            hinged_parts[str(fixture_id)] = sorted(part_ids)
    return hinged_parts


def _resolve_machine_path(
    machine_state: dict[str, Any],
    machine_path: list[str] | tuple[str, ...],
) -> Any:
    current_value: Any = machine_state
    for path_part in machine_path:
        if not isinstance(current_value, dict):
            return None
        current_value = current_value.get(path_part)
    return current_value


def _set_machine_path(
    machine_state: dict[str, Any],
    machine_path: list[str] | tuple[str, ...],
    value: Any,
) -> None:
    current_value = machine_state
    for path_part in machine_path[:-1]:
        next_value = current_value.get(path_part)
        if not isinstance(next_value, dict):
            next_value = {}
            current_value[path_part] = next_value
        current_value = next_value
    current_value[machine_path[-1]] = value


class SpecDrivenTaskValidator(FiniteStateTaskValidator):
    """Generic FSM validator that interprets TaskSpec preconditions and goals."""

    def __init__(
        self,
        task_spec: TaskSpec,
        task_instance: TaskInstance | None = None,
        *,
        allowed_tool_specs_override: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self._task_spec = task_spec
        self.coordinator_id = (
            task_instance.coordinator_id
            if task_instance is not None
            else None
        )
        effective_initial_state = (
            task_instance.initial_state
            if task_instance is not None
            else deepcopy(task_spec.initial_state)
        )
        super().__init__(
            composite_task=task_spec.composite_task,
            agent_ids=task_spec.agent_ids,
            initial_state=effective_initial_state,
            allowed_tool_specs=(
                deepcopy(allowed_tool_specs_override)
                if allowed_tool_specs_override is not None
                else task_spec.allowed_tool_specs
            ),
            checks=task_spec.validator_checks,
            max_reasoning_chars=task_spec.max_reasoning_chars,
            initial_public_state=task_spec.initial_public_state,
        )

    def _refresh_public_state(self, runtime_state: TaskRuntimeState) -> None:
        for public_key in tuple(runtime_state.public_state):
            if public_key.endswith("_location"):
                object_id = public_key[: -len("_location")]
                if object_id in runtime_state.objects:
                    runtime_state.public_state[public_key] = runtime_state.objects[object_id].get(
                        "location"
                    )
            elif public_key.endswith("_started"):
                machine_id = public_key[: -len("_started")]
                machine_entry = runtime_state.machine_state.get(machine_id, {})
                if isinstance(machine_entry, dict) and "started" in machine_entry:
                    runtime_state.public_state[public_key] = machine_entry.get("started")
            else:
                machine_path = public_key.split(".")
                if len(machine_path) > 1:
                    machine_value = _resolve_machine_path(
                        runtime_state.machine_state, machine_path
                    )
                    if machine_value is not None:
                        runtime_state.public_state[public_key] = machine_value

    def _object_location_matches(
        self,
        *,
        runtime_state: TaskRuntimeState,
        object_id: str,
        required_location: str,
    ) -> bool:
        actual_location = runtime_state.objects[object_id]["location"]
        if actual_location == required_location:
            return True

        resolved_actual = self._resolve_reference_location(
            reference_id=object_id,
            runtime_state=runtime_state,
        )
        if resolved_actual == required_location:
            return True

        required_fixture_id = self._resolve_fixture_for_location(
            location_id=required_location,
            runtime_state=runtime_state,
        )
        actual_fixture_id = self._resolve_fixture_for_location(
            location_id=resolved_actual if isinstance(resolved_actual, str) else actual_location,
            runtime_state=runtime_state,
        )
        return (
            isinstance(required_fixture_id, str)
            and isinstance(actual_fixture_id, str)
            and required_fixture_id == actual_fixture_id
        )

    def validate_task_preconditions(
        self,
        step: dict[str, Any],
        runtime_state: TaskRuntimeState,
    ) -> None:
        for condition in self._task_spec.task_preconditions:
            condition_kind = condition["kind"]
            if condition_kind == "object_must_remain_at_location":
                object_id = condition["object_id"]
                required_location = condition["location"]
                actual_location = runtime_state.objects[object_id]["location"]
                if not self._object_location_matches(
                    runtime_state=runtime_state,
                    object_id=object_id,
                    required_location=required_location,
                ):
                    raise TaskPreconditionSemanticValidationError(
                        condition["message"],
                        details={
                            "tool": step["tool"],
                            "object_id": object_id,
                            "required_location": required_location,
                            "actual_location": actual_location,
                        },
                    )
            elif condition_kind == "fixture_part_state_required_for_pickup":
                if step["tool"] != condition["tool"]:
                    continue
                if step["args"].get("source_id") != condition["source_id"]:
                    continue
                # Unguarded chained indexing turned a missing part into a bare
                # KeyError that surfaced as an uninterpretable rejection reason
                # (40 of them on prepare_cheese_station). A part absent from
                # runtime state simply has not been opened yet.
                actual_state = (
                    ((runtime_state.fixtures.get(condition["fixture_id"]) or {})
                     .get("parts") or {})
                    .get(condition["part_id"]) or {}
                ).get("state")
                if actual_state != condition["required_state"]:
                    raise TaskPreconditionSemanticValidationError(
                        condition.get("message")
                        or (
                            f"{condition['fixture_id']}.{condition['part_id']} must be "
                            f"{condition['required_state']} before {step['tool']} from "
                            f"{condition['source_id']}."
                        ),
                        details={
                            "tool": step["tool"],
                            "fixture_id": condition["fixture_id"],
                            "part_id": condition["part_id"],
                            "required_state": condition["required_state"],
                            "actual_state": actual_state,
                        },
                    )
            elif condition_kind == "fixture_part_state_required_for_action":
                if step["tool"] != condition["tool"]:
                    continue
                arg_name = condition.get("arg_name")
                arg_value = condition.get("arg_value")
                if isinstance(arg_name, str) and arg_value is not None:
                    if step["args"].get(arg_name) != arg_value:
                        continue
                # Unguarded chained indexing turned a missing part into a bare
                # KeyError that surfaced as an uninterpretable rejection reason
                # (40 of them on prepare_cheese_station). A part absent from
                # runtime state simply has not been opened yet.
                actual_state = (
                    ((runtime_state.fixtures.get(condition["fixture_id"]) or {})
                     .get("parts") or {})
                    .get(condition["part_id"]) or {}
                ).get("state")
                if actual_state != condition["required_state"]:
                    raise TaskPreconditionSemanticValidationError(
                        condition["message"],
                        details={
                            "tool": step["tool"],
                            "fixture_id": condition["fixture_id"],
                            "part_id": condition["part_id"],
                            "required_state": condition["required_state"],
                            "actual_state": actual_state,
                        },
                    )
            elif condition_kind == "object_location_required_for_action":
                if step["tool"] != condition["tool"]:
                    continue
                arg_name = condition.get("arg_name")
                arg_value = condition.get("arg_value")
                if isinstance(arg_name, str) and arg_value is not None:
                    if step["args"].get(arg_name) != arg_value:
                        continue
                else:
                    step_object_id = step["args"].get("object_id")
                    if (
                        isinstance(step_object_id, str)
                        and step_object_id != condition["object_id"]
                    ):
                        continue
                actual_location = runtime_state.objects[condition["object_id"]]["location"]
                if not self._object_location_matches(
                    runtime_state=runtime_state,
                    object_id=condition["object_id"],
                    required_location=condition["required_location"],
                ):
                    raise TaskPreconditionSemanticValidationError(
                        condition["message"],
                        details={
                            "tool": step["tool"],
                            "object_id": condition["object_id"],
                            "required_location": condition["required_location"],
                            "actual_location": actual_location,
                        },
                    )
            else:
                raise ValueError(
                    f"Unsupported TaskSpec precondition kind: {condition_kind}"
                )

    def apply_task_effects(
        self,
        step: dict[str, Any],
        runtime_state: TaskRuntimeState,
    ) -> None:
        for effect in self._task_spec.task_effects:
            effect_kind = effect["kind"]
            if effect_kind != "set_machine_flag_on_action":
                raise ValueError(f"Unsupported TaskSpec effect kind: {effect_kind}")
            if step["tool"] != effect["tool"]:
                continue
            if any(
                step["args"].get(arg_name) != arg_value
                for arg_name, arg_value in effect["args"].items()
                if not (step["tool"] == "communicate" and arg_name == "message")
            ):
                continue
            required_object_locations = effect.get("required_object_locations") or []
            if any(
                not self._object_location_matches(
                    runtime_state=runtime_state,
                    object_id=requirement["object_id"],
                    required_location=requirement["location"],
                )
                for requirement in required_object_locations
                if isinstance(requirement, dict)
                and isinstance(requirement.get("object_id"), str)
                and isinstance(requirement.get("location"), str)
            ):
                continue
            required_machine_values = effect.get("required_machine_values") or []
            if any(
                _resolve_machine_path(
                    runtime_state.machine_state,
                    tuple(requirement["machine_path"]),
                )
                != requirement.get("value")
                for requirement in required_machine_values
                if isinstance(requirement, dict)
                and isinstance(requirement.get("machine_path"), list)
            ):
                continue
            required_fixture_controls = effect.get("required_fixture_controls") or []
            if any(
                runtime_state.fixtures.get(requirement.get("fixture_id"), {})
                .get("controls", {})
                .get(requirement.get("control_id"), {})
                .get("state")
                != requirement.get("state")
                for requirement in required_fixture_controls
                if isinstance(requirement, dict)
            ):
                continue
            _set_machine_path(
                runtime_state.machine_state,
                tuple(effect["machine_path"]),
                effect["value"],
            )

        self._refresh_public_state(runtime_state)

    def is_goal_state_satisfied(self, runtime_state: TaskRuntimeState) -> bool:
        for condition in self._task_spec.goal_conditions:
            condition_kind = condition["kind"]
            if condition_kind == "object_at_location":
                if (
                    runtime_state.objects[condition["object_id"]]["location"]
                    != condition["location"]
                ):
                    return False
            elif condition_kind == "object_count_at_location":
                object_ids = [
                    object_id
                    for object_id in condition.get("object_ids", ())
                    if isinstance(object_id, str)
                ]
                actual_count = sum(
                    1
                    for object_id in object_ids
                    if runtime_state.objects.get(object_id, {}).get("location")
                    == condition.get("location")
                )
                if actual_count != condition.get("count"):
                    return False
            elif condition_kind == "object_count_at_locations":
                object_ids = [
                    object_id
                    for object_id in condition.get("object_ids", ())
                    if isinstance(object_id, str)
                ]
                locations = {
                    location
                    for location in condition.get("locations", ())
                    if isinstance(location, str)
                }
                actual_count = sum(
                    1
                    for object_id in object_ids
                    if runtime_state.objects.get(object_id, {}).get("location")
                    in locations
                )
                if actual_count != condition.get("count"):
                    return False
            elif condition_kind == "object_at_location_one_of":
                loc = runtime_state.objects[condition["object_id"]]["location"]
                if loc not in condition["locations"]:
                    return False
            elif condition_kind == "machine_flag_true":
                if not _resolve_machine_path(
                    runtime_state.machine_state,
                    tuple(condition["machine_path"]),
                ):
                    return False
            elif condition_kind == "machine_flag_equals":
                if _resolve_machine_path(
                    runtime_state.machine_state,
                    tuple(condition["machine_path"]),
                ) != condition.get("value"):
                    return False
            elif condition_kind == "fixture_part_state":
                fixture_state = runtime_state.fixtures.get(condition["fixture_id"], {})
                if not isinstance(fixture_state, dict):
                    return False
                fixture_parts = fixture_state.get("parts", {})
                if not isinstance(fixture_parts, dict):
                    return False
                actual_state = fixture_parts.get(condition["part_id"], {}).get("state")
                if actual_state != condition.get("state"):
                    return False
            elif condition_kind == "fixture_control_state":
                fixture_state = runtime_state.fixtures.get(condition["fixture_id"], {})
                if not isinstance(fixture_state, dict):
                    return False
                fixture_controls = fixture_state.get("controls", {})
                if not isinstance(fixture_controls, dict):
                    return False
                actual_state = fixture_controls.get(condition["control_id"], {}).get(
                    "state"
                )
                if actual_state != condition.get("state"):
                    return False
            else:
                raise ValueError(f"Unsupported TaskSpec goal kind: {condition_kind}")

        # Enforce mutual exclusion for object_at_location_one_of conditions
        # that opt in with "exclusive": true.
        exclusive_resolved: list[tuple[str, str, tuple[str, ...]]] = []
        for condition in self._task_spec.goal_conditions:
            if condition["kind"] != "object_at_location_one_of":
                continue
            if not condition.get("exclusive", False):
                continue
            obj_id = condition["object_id"]
            loc = runtime_state.objects[obj_id]["location"]
            exclusive_resolved.append((obj_id, loc, tuple(condition["locations"])))
        for i, (obj_a, loc_a, pool_a) in enumerate(exclusive_resolved):
            for obj_b, loc_b, pool_b in exclusive_resolved[i + 1 :]:
                if loc_a == loc_b and set(pool_a) & set(pool_b):
                    return False

        return True

    def goal_state_diagnostics(
        self, runtime_state: TaskRuntimeState
    ) -> dict[str, Any]:
        """Report unmet declarative goals and observed state, without a repair plan."""

        unmet: list[dict[str, Any]] = []
        for condition in self._task_spec.goal_conditions:
            kind = condition["kind"]
            actual: Any
            satisfied = False
            if kind == "object_at_location":
                actual = runtime_state.objects.get(condition["object_id"], {}).get(
                    "location"
                )
                satisfied = actual == condition["location"]
            elif kind == "object_count_at_location":
                actual = sum(
                    1
                    for object_id in condition.get("object_ids", ())
                    if runtime_state.objects.get(object_id, {}).get("location")
                    == condition.get("location")
                )
                satisfied = actual == condition.get("count")
            elif kind == "object_count_at_locations":
                locations = {
                    location
                    for location in condition.get("locations", ())
                    if isinstance(location, str)
                }
                actual = sum(
                    1
                    for object_id in condition.get("object_ids", ())
                    if runtime_state.objects.get(object_id, {}).get("location")
                    in locations
                )
                satisfied = actual == condition.get("count")
            elif kind == "object_at_location_one_of":
                actual = runtime_state.objects.get(condition["object_id"], {}).get(
                    "location"
                )
                satisfied = actual in condition["locations"]
            elif kind in {"machine_flag_true", "machine_flag_equals"}:
                actual = _resolve_machine_path(
                    runtime_state.machine_state, tuple(condition["machine_path"])
                )
                expected = True if kind == "machine_flag_true" else condition.get("value")
                satisfied = actual == expected
            elif kind == "fixture_part_state":
                actual = (
                    runtime_state.fixtures.get(condition["fixture_id"], {})
                    .get("parts", {})
                    .get(condition["part_id"], {})
                    .get("state")
                )
                satisfied = actual == condition.get("state")
            elif kind == "fixture_control_state":
                actual = (
                    runtime_state.fixtures.get(condition["fixture_id"], {})
                    .get("controls", {})
                    .get(condition["control_id"], {})
                    .get("state")
                )
                satisfied = actual == condition.get("state")
            else:
                actual = "unsupported goal kind"
            if not satisfied:
                unmet.append({"condition": deepcopy(condition), "actual": actual})

        details = super().goal_state_diagnostics(runtime_state)
        details["unmet_goal_conditions"] = unmet
        return details


# Tools the pipeline fills in after generation; the model never emits them.
DERIVED_TOOL_NAMES = frozenset({"wait_for_signal"})


def build_task_definition_from_spec(task_spec: TaskSpec) -> TaskDefinition:
    """Build one runtime TaskDefinition from a JSON-backed task spec."""

    hinged_parts_by_fixture = _hinged_parts_by_fixture(task_spec.initial_state)
    tool_names = list(task_spec.allowed_tool_specs)
    if hinged_parts_by_fixture and "open_hinged_part" not in tool_names:
        tool_names.append("open_hinged_part")
    overrides: dict[str, dict[str, Any]] = {}
    for tool_name, tool_spec in task_spec.allowed_tool_specs.items():
        tool_override = {
            key: deepcopy(value)
            for key, value in tool_spec.items()
            if key
            not in {
                "description",
                "tool_args",
                "optional_tool_args",
                "tool_arg_types",
                "tool_arg_any_of",
            }
        }
        if tool_override:
            overrides[tool_name] = tool_override

    if hinged_parts_by_fixture and "open_hinged_part" not in task_spec.allowed_tool_specs:
        overrides["open_hinged_part"] = {
            "allowed_target_ids": sorted(hinged_parts_by_fixture),
            "allowed_part_ids": sorted(
                {
                    part_id
                    for part_ids in hinged_parts_by_fixture.values()
                    for part_id in part_ids
                }
            ),
        }

    allowed_tool_specs = build_allowed_tool_specs(tuple(tool_names), overrides=overrides)
    from robotalk.tasks.shared.workspace_semantics import (
        canonical_agent_workspace,
    )

    for tool_name in ("navigate_to_fixture", "give_space"):
        tool_spec = allowed_tool_specs.get(tool_name)
        if not isinstance(tool_spec, dict):
            continue
        allowed_ids = tool_spec.get("allowed_fixture_ids")
        if not isinstance(allowed_ids, list):
            continue
        tool_spec["allowed_fixture_ids"] = list(dict.fromkeys(
            [*allowed_ids]
            + [
                canonical_agent_workspace(task_spec.initial_state, fixture_id)
                or fixture_id
                for fixture_id in allowed_ids
            ]
        ))
    # Demonstration generation is deliberately task-specific. The eventual
    # SFT/eval policy still receives the global tool interface, but its label
    # generator should not be invited to call tools that this task rejects.
    # build_allowed_tool_specs also prunes unusable legacy placement aliases.
    generation_tool_specs = deepcopy(allowed_tool_specs)
    # New demonstrations use the canonical parent workspace for cabinet
    # navigation/yielding. The validator retains the original cabinet IDs as
    # legacy aliases so old trajectories and model outputs remain executable.
    for tool_name in ("navigate_to_fixture", "give_space"):
        tool_spec = generation_tool_specs.get(tool_name)
        if not isinstance(tool_spec, dict):
            continue
        allowed_ids = tool_spec.get("allowed_fixture_ids")
        if not isinstance(allowed_ids, list):
            continue
        tool_spec["allowed_fixture_ids"] = list(dict.fromkeys(
            canonical_agent_workspace(task_spec.initial_state, fixture_id)
            or fixture_id
            for fixture_id in allowed_ids
        ))
    response_schema = build_task_response_schema(
        agent_ids=task_spec.agent_ids,
        allowed_tool_specs=generation_tool_specs,
    )
    # In tick form the model can see that an agent is blocked, so it is given
    # wait_for_signal back and asked to place its own coordination. That is the
    # whole point of the format: the defects the insertion pass exists to fix
    # all come from the model not being able to see execution order.
    tick_response_schema = build_tick_response_schema(
        agent_ids=task_spec.agent_ids,
        allowed_tool_specs=generation_tool_specs,
    )
    explicit_blocked_tick_response_schema = build_tick_response_schema(
        agent_ids=task_spec.agent_ids,
        allowed_tool_specs=generation_tool_specs,
        explicit_blocked_markers=True,
    )
    non_communicate_tool_names = tuple(
        tool_name for tool_name in generation_tool_specs if tool_name != "communicate"
    )
    build_prompt = make_task_prompt_builder(
        composite_task=task_spec.composite_task,
        task_goal=task_spec.task_goal,
        initial_state=task_spec.initial_state,
        allowed_tool_specs=generation_tool_specs,
        non_communicate_tool_names=non_communicate_tool_names,
        task_preconditions=task_spec.task_preconditions,
        task_effects=task_spec.task_effects,
        extra_execution_rules=task_spec.extra_execution_rules,
        agent_ids=task_spec.agent_ids,
    )
    build_trajectory_record = make_symbolic_trajectory_record_builder(
        composite_task=task_spec.composite_task,
        agent_ids=task_spec.agent_ids,
    )

    def _build_task_instance(run_index: int, runtime_config: Any | None = None) -> TaskInstance:
        # Coordinator is part of the model-visible initial configuration. Use
        # it as the fastest-cycling dimension, then advance the physical
        # start/access configuration. This prevents run-index parity from
        # correlating one coordinator with one open/closed access state.
        physical_configuration_index = run_index // len(task_spec.agent_ids)
        instance = build_randomized_fixture_task_instance(
            composite_task=task_spec.composite_task,
            agent_ids=task_spec.agent_ids,
            initial_state=task_spec.initial_state,
            allowed_tool_specs=allowed_tool_specs,
            run_index=physical_configuration_index,
            runtime_config=runtime_config,
        )
        rules = tuple(
            instance.extra_execution_rules or task_spec.extra_execution_rules
        ) + TICK_FORMAT_RULES
        return replace(
            instance,
            coordinator_id=task_spec.agent_ids[run_index % len(task_spec.agent_ids)],
            extra_execution_rules=tuple(rules),
            # Tick generation lets the model author waits, so its prompt must
            # display the same full tool set as the tick response schema.
            allowed_tool_specs=deepcopy(allowed_tool_specs),
        )

    def _validator_factory(task_instance: TaskInstance | None) -> SpecDrivenTaskValidator:
        return SpecDrivenTaskValidator(
            task_spec,
            task_instance,
            allowed_tool_specs_override=allowed_tool_specs,
        )

    return TaskDefinition(
        composite_task=task_spec.composite_task,
        response_schema=response_schema,
        tick_response_schema=tick_response_schema,
        explicit_blocked_tick_response_schema=explicit_blocked_tick_response_schema,
        preflight_token_estimate=PreflightTokenEstimate(
            prompt_tokens=task_spec.preflight_token_estimate.prompt_tokens,
            output_tokens=task_spec.preflight_token_estimate.output_tokens,
            reasoning_tokens=task_spec.preflight_token_estimate.reasoning_tokens,
        ),
        build_task_instance=_build_task_instance,
        build_prompt=build_prompt,
        build_trajectory_record=build_trajectory_record,
        validator_factory=_validator_factory,
    )


SPEC_TASK_REGISTRY: dict[str, TaskDefinition] = {
    task_spec.composite_task: build_task_definition_from_spec(task_spec)
    for task_spec in load_all_task_specs()
}
