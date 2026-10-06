"""Validation and normalization helpers for one-step structured VLM outputs."""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any

from robotalk.tasks.shared.schema import build_task_response_schema
from robotalk.tools.subatomic_tool_specs import KNOWN_OBSERVATION_VIEWS

GET_IMAGE_TOOL_NAME = "get_image"
GET_IMAGE_TOOL_SPEC: dict[str, Any] = {
    "tool_args": ["views"],
    "tool_arg_types": {"views": "STRING_ARRAY"},
    "allowed_views": list(KNOWN_OBSERVATION_VIEWS),
    "tool_arg_descriptions": {
        "views": (
            "A non-empty list containing only these exact camera-view names: "
            + ", ".join(KNOWN_OBSERVATION_VIEWS)
            + ". Copy the names exactly; do not invent directional, object, "
            "fixture, or scene-specific view names."
        )
    },
    "description": "Request one or more canonical camera views before choosing the next action.",
}


def augment_tool_specs_with_get_image(
    allowed_tool_specs: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Adds the get_image tool to a task's tool specs."""

    augmented = dict(allowed_tool_specs)
    augmented[GET_IMAGE_TOOL_NAME] = deepcopy(GET_IMAGE_TOOL_SPEC)
    return augmented


def build_single_step_response_schema(
    *,
    agent_ids: tuple[str, ...],
    allowed_tool_specs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Builds the task-level schema and narrows it to exactly one predicted step."""

    schema = build_task_response_schema(
        agent_ids=agent_ids,
        allowed_tool_specs=allowed_tool_specs,
        min_steps=1,
    )
    schema = deepcopy(schema)
    schema["properties"]["steps"]["maxItems"] = 1
    step_items = schema["properties"]["steps"]["items"]
    step_items.get("properties", {}).pop("reasoning", None)
    if isinstance(step_items.get("required"), list):
        step_items["required"] = [
            field_name
            for field_name in step_items["required"]
            if field_name != "reasoning"
        ]
    return schema


def compact_json_dumps(value: Any) -> str:
    """Serializes JSON in a stable compact format for targets and manifests."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"))


def parse_first_json_object(text: str) -> Any:
    """Extracts the first balanced JSON object from a model response."""

    start = text.find("{")
    if start < 0:
        raise ValueError("Response did not contain a JSON object.")

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue
        if char == "{":
            depth += 1
            continue
        if char == "}":
            depth -= 1
            if depth == 0:
                return json.loads(text[start : index + 1])

    raise ValueError("Response contained an unterminated JSON object.")


def canonicalize_for_comparison(value: Any) -> Any:
    """Normalizes JSON-like values for exact structured comparison."""

    if isinstance(value, dict):
        return {key: canonicalize_for_comparison(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [canonicalize_for_comparison(item) for item in value]
    if isinstance(value, str):
        return " ".join(value.strip().split())
    return value


def declared_tool_arg_names(
    tool_spec: dict[str, Any],
) -> tuple[list[str], set[str]]:
    """Returns (required_args, all_allowed_args) for one tool spec.

    Mirrors the generation-side schema vocabulary: `tool_args` are required,
    `optional_tool_args` may appear, and each `tool_arg_any_of` group requires
    at least one member (enforced by validate_single_step_payload).
    """

    required = list(tool_spec.get("tool_args", ()))
    allowed = set(required).union(tool_spec.get("optional_tool_args", ()))
    for group in tool_spec.get("tool_arg_any_of", ()):
        allowed.update(group)
    return required, allowed


def _allowed_ids_key_for_arg_name(arg_name: str) -> str | None:
    if not arg_name.endswith("_id"):
        return None
    return f"allowed_{arg_name[:-3]}_ids"


def _normalize_required_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field_name} must be a string.")
    normalized = " ".join(value.strip().split())
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty.")
    return normalized


def _normalize_integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer.")
    return value


def _normalize_string_array(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list of strings.")
    return [_normalize_required_string(item, f"{field_name}[]") for item in value]


def validate_single_step_payload(
    payload: Any,
    *,
    agent_ids: tuple[str, ...],
    allowed_tool_specs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Validates and normalizes one structured model output payload."""

    if not isinstance(payload, dict):
        raise ValueError("Payload must be a JSON object.")

    steps = payload.get("steps")
    if not isinstance(steps, list) or len(steps) != 1:
        raise ValueError("Payload.steps must contain exactly one item.")

    step = steps[0]
    if not isinstance(step, dict):
        raise ValueError("Payload.steps[0] must be an object.")

    normalized_step_index = _normalize_integer(step.get("step"), "steps[0].step")
    normalized_agent = _normalize_required_string(step.get("agent"), "steps[0].agent")
    if normalized_agent not in agent_ids:
        raise ValueError(f"steps[0].agent must be one of {agent_ids}.")

    normalized_tool = _normalize_required_string(step.get("tool"), "steps[0].tool")
    if normalized_tool not in allowed_tool_specs:
        raise ValueError(f"steps[0].tool must be one of {tuple(allowed_tool_specs)}.")

    raw_args = step.get("args")
    if not isinstance(raw_args, dict):
        raise ValueError("steps[0].args must be an object.")

    tool_spec = allowed_tool_specs[normalized_tool]
    expected_args, _ = declared_tool_arg_names(tool_spec)
    optional_args = list(tool_spec.get("optional_tool_args", ()))
    any_of_groups = [
        list(group) for group in tool_spec.get("tool_arg_any_of", ())
    ]
    provided_arg_names = set(raw_args)
    missing_arg_names = [
        arg_name for arg_name in expected_args if arg_name not in raw_args
    ]
    if missing_arg_names:
        missing = ", ".join(missing_arg_names)
        raise ValueError(f"steps[0].args is missing required fields: {missing}.")
    for group in any_of_groups:
        if not provided_arg_names.intersection(group):
            group_text = ", ".join(group)
            raise ValueError(
                f"steps[0].args must include at least one of: {group_text}."
            )

    allowed_arg_names = set(expected_args).union(optional_args)
    for group in any_of_groups:
        allowed_arg_names.update(group)
    unexpected_arg_names = provided_arg_names.difference(allowed_arg_names)
    if unexpected_arg_names:
        unexpected = ", ".join(sorted(unexpected_arg_names))
        raise ValueError(
            f"steps[0].args contains unsupported fields for {normalized_tool}: {unexpected}."
        )

    tool_arg_types = dict(tool_spec.get("tool_arg_types", {}))
    normalized_args: dict[str, Any] = {}
    # Normalize every provided allowed argument; required args are already
    # guaranteed present, optional/any-of args are validated when supplied.
    for arg_name in expected_args + [
        arg_name
        for arg_name in sorted(allowed_arg_names.difference(expected_args))
        if arg_name in raw_args
    ]:
        schema_type = tool_arg_types.get(arg_name, "STRING")
        raw_value = raw_args[arg_name]
        if schema_type == "STRING":
            normalized_value = _normalize_required_string(
                raw_value,
                f"steps[0].args.{arg_name}",
            )
        elif schema_type == "INTEGER":
            normalized_value = _normalize_integer(
                raw_value,
                f"steps[0].args.{arg_name}",
            )
        elif schema_type == "STRING_ARRAY":
            normalized_value = _normalize_string_array(
                raw_value,
                f"steps[0].args.{arg_name}",
            )
        else:
            raise ValueError(
                f"Unsupported schema type {schema_type!r} for argument {arg_name}."
            )

        if arg_name in {"agent", "agent_id", "to"}:
            allowed_values = set(agent_ids)
            if (
                isinstance(normalized_value, str)
                and normalized_value not in allowed_values
            ):
                allowed_text = ", ".join(sorted(allowed_values))
                raise ValueError(
                    f"steps[0].args.{arg_name} must be one of {allowed_text}."
                )

        allowed_values_key = f"allowed_{arg_name}"
        allowed_ids_key = _allowed_ids_key_for_arg_name(arg_name)
        constraint_key = (
            allowed_values_key
            if allowed_values_key in tool_spec
            else allowed_ids_key
            if allowed_ids_key and allowed_ids_key in tool_spec
            else None
        )
        declared_values = tool_spec.get("allowed_arg_values", {})
        explicit_values = (
            declared_values.get(arg_name)
            if isinstance(declared_values, dict)
            else None
        )
        if constraint_key is not None or explicit_values:
            allowed_values = set(
                tool_spec[constraint_key]
                if constraint_key is not None
                else explicit_values
            )
            if isinstance(normalized_value, str):
                if normalized_value not in allowed_values:
                    allowed_text = ", ".join(sorted(allowed_values))
                    raise ValueError(
                        f"steps[0].args.{arg_name} must be one of {allowed_text}."
                    )
            elif isinstance(normalized_value, list):
                invalid_values = [
                    value for value in normalized_value if value not in allowed_values
                ]
                if invalid_values:
                    invalid_text = ", ".join(sorted(invalid_values))
                    allowed_text = ", ".join(sorted(allowed_values))
                    raise ValueError(
                        f"steps[0].args.{arg_name} contains invalid values {invalid_text}; "
                        f"allowed values: {allowed_text}."
                    )

        normalized_args[arg_name] = normalized_value

    return {
        "steps": [
            {
                "step": normalized_step_index,
                "agent": normalized_agent,
                "tool": normalized_tool,
                "args": normalized_args,
            }
        ]
    }


def normalize_give_space_fixtures(
    allowed_tool_specs: dict[str, dict],
) -> dict[str, dict]:
    """Lets an agent yield at any fixture it is allowed to navigate to.

    The per-task give_space allowlist is generated from fixtures the expert
    happened to yield at, not from a simulator constraint -- SimToolExecutor
    .give_space only requires the fixture to exist and then steps >=1.5m away.
    When a reachable fixture is missing from that list, two agents contending
    there have no legal way to resolve it: observed on 7 tasks, including
    hot_dog_setup, whose unyieldable `counter` is a pick_up source both agents
    need and which scored 0/120.

    Widening only adds legal moves, so no existing expert trajectory is
    invalidated.
    """

    nav = allowed_tool_specs.get("navigate_to_fixture") or {}
    give = allowed_tool_specs.get("give_space")
    if not give:
        return allowed_tool_specs
    reachable = list(nav.get("allowed_fixture_ids") or [])
    if not reachable:
        return allowed_tool_specs
    current = list(give.get("allowed_fixture_ids") or [])
    merged = list(dict.fromkeys([*current, *reachable]))
    if merged == current:
        return allowed_tool_specs
    updated = dict(allowed_tool_specs)
    updated["give_space"] = {**give, "allowed_fixture_ids": merged}
    return updated


# Eval-only leniency. The benchmark names the same fixture `cab` in some tasks
# and `cabinet` in others, so a model that saw one spelling emits it where only
# the other is legal. Approved alias set is deliberately just this pair --
# counter/dining_counter/kitchen_counter are NOT aliased, since those may be
# genuinely distinct fixtures. Applied when validating a model's proposal in
# live-sim; training data and prompts are untouched.
_EVAL_FIXTURE_ALIASES = {"cab": "cabinet", "cabinet": "cab"}


def apply_eval_fixture_aliases(
    parsed_call: dict,
    allowed_tool_specs: dict[str, dict],
) -> dict:
    """Rewrites cab<->cabinet in a proposed call when only the alias is legal.

    Returns the call unchanged unless the emitted value is disallowed AND its
    alias is allowed, so this can never turn a legal call into an illegal one.
    """

    if not parsed_call:
        return parsed_call
    spec = (allowed_tool_specs or {}).get(parsed_call.get("name"))
    if not spec:
        return parsed_call
    args = dict(parsed_call.get("arguments") or {})
    changed = False
    for key, value in list(args.items()):
        alias = _EVAL_FIXTURE_ALIASES.get(value) if isinstance(value, str) else None
        if alias is None:
            continue
        for spec_key, allowed in spec.items():
            if not spec_key.startswith("allowed_") or not isinstance(allowed, list):
                continue
            if value not in allowed and alias in allowed:
                args[key] = alias
                changed = True
                break
    if not changed:
        return parsed_call
    return {**parsed_call, "arguments": args}
