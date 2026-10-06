"""Gemini native function-calling helpers for the live-sim evaluator."""

from __future__ import annotations
from typing import Any
from robotalk.training.schema_utils import declared_tool_arg_names
from robotalk.training.task_registry import AGENT_IDS


def message_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return "\n".join(
        str(item.get("text", ""))
        for item in content
        if isinstance(item, dict) and item.get("type") == "text"
    )


def build_vertex_function_declarations(
    allowed_tool_specs: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Builds one Vertex FunctionDeclaration per allowed tool.

    Reuses the shared response-schema helpers so a tool's argument names and
    types are declared exactly as they are for the forced-JSON path — the only
    difference is the delivery mechanism (native function calling vs a
    response_schema), keeping the two comparable.
    """

    from robotalk.tasks.shared.schema import (
        _build_symbolic_field_schema,
        _iter_declared_tool_arg_names,
        _resolve_tool_arg_schema_type,
    )

    declarations: list[dict[str, Any]] = []
    for tool_name, tool_spec in allowed_tool_specs.items():
        required_args, _ = declared_tool_arg_names(tool_spec)
        properties: dict[str, Any] = {}
        for field_name in _iter_declared_tool_arg_names(tool_spec):
            properties[field_name] = _build_symbolic_field_schema(
                field_name,
                AGENT_IDS,
                _resolve_tool_arg_schema_type(field_name, tool_spec),
            )
            allowed_values_key = f"allowed_{field_name}"
            if allowed_values_key in tool_spec:
                target = properties[field_name]
                if target.get("type") == "ARRAY":
                    target = target["items"]
                target["enum"] = list(tool_spec[allowed_values_key])
            declared_values = tool_spec.get("allowed_arg_values", {})
            if isinstance(declared_values, dict) and declared_values.get(field_name):
                target = properties[field_name]
                if target.get("type") == "ARRAY":
                    target = target["items"]
                target["enum"] = list(declared_values[field_name])
        declarations.append(
            {
                "name": tool_name,
                "description": (tool_spec.get("description") or "").strip(),
                "parameters": {
                    "type": "OBJECT",
                    "properties": properties,
                    "required": list(required_args),
                },
            }
        )
    return declarations


GEMINI_MIME_BY_SUFFIX = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}
