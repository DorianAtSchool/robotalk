"""Helpers for JSON-schema tool definitions and Qwen tool-call parsing."""

from __future__ import annotations

import json
import re
from typing import Any

from robotalk.training.schema_utils import (
    declared_tool_arg_names,
    validate_single_step_payload,
)

_TOOL_CALL_PATTERN = re.compile(
    r"<tool_call>\s*<function=([^>\n]+)>\s*(.*?)</function>\s*</tool_call>",
    re.DOTALL,
)
_PARAMETER_PATTERN = re.compile(
    r"<parameter=([^>\n]+)>\s*(.*?)\s*</parameter>",
    re.DOTALL,
)
# Qwen's chat template renders assistant tool calls with a JSON body:
#   <tool_call>\n{"name": ..., "arguments": {...}}\n</tool_call>
# which is what a tool_call-format SFT model is trained on and emits. The
# <function=…>/<parameter=…> dialect above is a different rendering; support
# both so native generations parse.
_TOOL_CALL_JSON_PATTERN = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)
_THINK_BLOCK_PATTERN = re.compile(r"<think>.*?</think>", re.DOTALL)


def _allowed_ids_key_for_arg_name(arg_name: str) -> str | None:
    if not arg_name.endswith("_id"):
        return None
    return f"allowed_{arg_name[:-3]}_ids"


def _build_string_schema(
    *,
    arg_name: str,
    agent_ids: tuple[str, ...],
    tool_spec: dict[str, Any],
) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "string"}
    if arg_name in {"agent", "agent_id", "to"}:
        schema["enum"] = list(agent_ids)
        return schema

    declared_values = tool_spec.get("allowed_arg_values", {})
    if isinstance(declared_values, dict) and declared_values.get(arg_name):
        schema["enum"] = list(declared_values[arg_name])
        return schema

    allowed_values_key = f"allowed_{arg_name}"
    if allowed_values_key in tool_spec:
        schema["enum"] = list(tool_spec[allowed_values_key])
        return schema
    allowed_ids_key = _allowed_ids_key_for_arg_name(arg_name)
    if allowed_ids_key is not None and allowed_ids_key in tool_spec:
        schema["enum"] = list(tool_spec[allowed_ids_key])
    return schema


def _build_parameter_schema(
    *,
    arg_name: str,
    agent_ids: tuple[str, ...],
    tool_spec: dict[str, Any],
) -> dict[str, Any]:
    schema_type = dict(tool_spec.get("tool_arg_types", {})).get(arg_name, "STRING")
    if schema_type == "STRING":
        schema = _build_string_schema(
            arg_name=arg_name,
            agent_ids=agent_ids,
            tool_spec=tool_spec,
        )
    elif schema_type == "INTEGER":
        schema = {"type": "integer"}
    elif schema_type == "STRING_ARRAY":
        schema = {
            "type": "array",
            "items": _build_string_schema(
                arg_name=arg_name,
                agent_ids=agent_ids,
                tool_spec=tool_spec,
            ),
        }
    else:
        raise ValueError(f"Unsupported tool arg schema type: {schema_type!r}")
    description = dict(tool_spec.get("tool_arg_descriptions", {})).get(arg_name)
    if description:
        schema["description"] = str(description)
    return schema


# Argument name carrying the model-chosen acting agent when agent prediction
# is enabled. It is a schema-level parameter, never a task-spec tool arg.


def build_tool_schemas(
    *,
    agent_ids: tuple[str, ...],
    allowed_tool_specs: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Builds Hugging Face / OpenAI-style function schemas for one task.

    The acting agent is fixed by the caller, so no tool takes an "agent"
    argument.
    """

    tool_schemas: list[dict[str, Any]] = []
    for tool_name, tool_spec in allowed_tool_specs.items():
        required_args, allowed_args = declared_tool_arg_names(tool_spec)
        schema_args = required_args + sorted(allowed_args.difference(required_args))
        description = tool_spec.get("description", "").strip()
        any_of_groups = [list(group) for group in tool_spec.get("tool_arg_any_of", ())]
        for group in any_of_groups:
            group_text = " or ".join(group)
            description = f"{description} Requires at least one of: {group_text}."
        properties: dict[str, Any] = {}
        schema_required = list(required_args)
        for arg_name in schema_args:
            properties[arg_name] = _build_parameter_schema(
                arg_name=arg_name,
                agent_ids=agent_ids,
                tool_spec=tool_spec,
            )
        tool_schemas.append(
            {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": description.strip(),
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": schema_required,
                        "additionalProperties": False,
                    },
                },
            }
        )
    return tool_schemas


def build_assistant_tool_call_message(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    reasoning_text: str | None = None,
) -> dict[str, Any]:
    """Builds one assistant tool-call message for chat-template supervision.

    With reasoning_text (v1.5 probe), the assistant content becomes a
    `<think>...</think>` prefix supervising a rationale before the tool call.
    The eval scorer already strips think blocks before parsing
    (evaluation.py::split_reasoning), so this only adds supervision.
    """

    content = f"<think>{reasoning_text}</think>" if reasoning_text else ""
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": dict(arguments),
                },
            }
        ],
    }


def parse_first_qwen_tool_call(text: str) -> dict[str, Any]:
    """Extracts the first rendered Qwen tool call from decoded model text."""

    match = _TOOL_CALL_PATTERN.search(text)
    if match is not None:
        tool_name = match.group(1).strip()
        body = match.group(2)
        arguments: dict[str, str] = {}
        for parameter_match in _PARAMETER_PATTERN.finditer(body):
            arg_name = parameter_match.group(1).strip()
            arguments[arg_name] = parameter_match.group(2).strip()
        return {"name": tool_name, "arguments": arguments}

    json_match = _TOOL_CALL_JSON_PATTERN.search(text)
    if json_match is not None:
        try:
            payload = json.loads(json_match.group(1))
        except json.JSONDecodeError as exc:
            raise ValueError(f"<tool_call> body is not valid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError("<tool_call> body must be a JSON object.")
        tool_name = payload.get("name")
        if not isinstance(tool_name, str) or not tool_name.strip():
            raise ValueError('<tool_call> body must contain a non-empty "name".')
        raw_arguments = payload.get("arguments", {})
        if not isinstance(raw_arguments, dict):
            raise ValueError('<tool_call> body must contain object "arguments".')
        return {"name": tool_name.strip(), "arguments": dict(raw_arguments)}

    raise ValueError("Response did not contain a <tool_call> block.")


def extract_pre_tool_call_text(text: str) -> str:
    """Returns any natural-language prefix before the first tool call."""

    match = _TOOL_CALL_PATTERN.search(text)
    if match is None:
        return text.strip()
    prefix = text[: match.start()].strip()
    prefix = _THINK_BLOCK_PATTERN.sub("", prefix).strip()
    return prefix


def _parse_string_value(raw_value: str) -> str:
    candidate = raw_value.strip()
    if len(candidate) >= 2 and candidate[0] == '"' and candidate[-1] == '"':
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            return candidate
        if isinstance(parsed, str):
            return parsed
    return candidate


def _parse_integer_value(raw_value: str) -> int:
    candidate = raw_value.strip()
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        parsed = int(candidate)
    if isinstance(parsed, bool) or not isinstance(parsed, int):
        raise ValueError(f"Expected integer argument, got {raw_value!r}.")
    return parsed


def _parse_string_array_value(raw_value: str) -> list[str]:
    parsed = json.loads(raw_value.strip())
    if not isinstance(parsed, list) or not all(
        isinstance(item, str) for item in parsed
    ):
        raise ValueError(f"Expected JSON string array, got {raw_value!r}.")
    return parsed


def tool_call_to_single_step_payload(
    tool_call: dict[str, Any],
    *,
    step_index: int,
    agent_id: str,
    agent_ids: tuple[str, ...],
    allowed_tool_specs: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Converts one parsed tool call into the canonical single-step payload."""

    tool_name = str(tool_call.get("name", "")).strip()
    if tool_name not in allowed_tool_specs:
        raise ValueError(f"Unsupported tool call name: {tool_name!r}.")

    raw_arguments = tool_call.get("arguments")
    if not isinstance(raw_arguments, dict):
        raise ValueError("Tool call arguments must be an object.")

    tool_spec = allowed_tool_specs[tool_name]
    expected_args, allowed_args = declared_tool_arg_names(tool_spec)
    missing_args = [
        arg_name for arg_name in expected_args if arg_name not in raw_arguments
    ]
    if missing_args:
        missing_text = ", ".join(missing_args)
        raise ValueError(f"Tool call is missing required arguments: {missing_text}.")

    unexpected_args = sorted(set(raw_arguments).difference(allowed_args))
    if unexpected_args:
        unexpected_text = ", ".join(unexpected_args)
        raise ValueError(
            f"Tool call contains unexpected arguments for {tool_name}: {unexpected_text}."
        )

    tool_arg_types = dict(tool_spec.get("tool_arg_types", {}))
    normalized_args: dict[str, Any] = {}
    # Required args first, then any provided optional/any-of args; the
    # terminal validate_single_step_payload enforces any-of group presence.
    provided_extra_args = sorted(
        set(raw_arguments).intersection(allowed_args).difference(expected_args)
    )
    for arg_name in expected_args + provided_extra_args:
        raw_value = raw_arguments[arg_name]
        schema_type = tool_arg_types.get(arg_name, "STRING")
        if schema_type not in {"STRING", "INTEGER", "STRING_ARRAY"}:
            raise ValueError(
                f"Unsupported tool arg schema type {schema_type!r} for {arg_name}."
            )
        if not isinstance(raw_value, str):
            # A JSON-body <tool_call> yields already-typed values; pass them
            # through and let validate_single_step_payload type-check them.
            # (The <function=…>/<parameter=…> dialect yields raw strings, which
            # still need the per-type parsing below.)
            normalized_args[arg_name] = raw_value
            continue
        if schema_type == "STRING":
            normalized_args[arg_name] = _parse_string_value(raw_value)
        elif schema_type == "INTEGER":
            normalized_args[arg_name] = _parse_integer_value(raw_value)
        else:
            normalized_args[arg_name] = _parse_string_array_value(raw_value)

    return validate_single_step_payload(
        {
            "steps": [
                {
                    "step": step_index,
                    "agent": agent_id,
                    "tool": tool_name,
                    "args": normalized_args,
                }
            ]
        },
        agent_ids=agent_ids,
        allowed_tool_specs=allowed_tool_specs,
    )
