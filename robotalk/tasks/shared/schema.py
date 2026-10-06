"""Build and validate shared task-level JSON response schemas."""

from __future__ import annotations

from typing import Any, Sequence

from .errors import TrajectoryStructureValidationError


def _normalize_text(value: Any, field_name: str) -> str:
    """Normalizes and validates a required string field."""

    if not isinstance(value, str):
        raise TrajectoryStructureValidationError(f"{field_name} must be a string.")
    normalized = " ".join(value.strip().split())
    if not normalized:
        raise TrajectoryStructureValidationError(f"{field_name} must be non-empty.")
    return normalized


def _normalize_mapping(value: Any, field_name: str) -> dict[str, Any]:
    """Normalizes and validates a required object field."""

    if not isinstance(value, dict):
        raise TrajectoryStructureValidationError(f"{field_name} must be an object.")
    return dict(value)


def _append_unique_field_names(
    destination: list[str],
    field_names: Sequence[str],
) -> None:
    """Appends schema field names while preserving their first-seen order."""

    for field_name in field_names:
        if field_name not in destination:
            destination.append(field_name)


def _iter_declared_tool_arg_names(tool_spec: dict[str, Any]) -> tuple[str, ...]:
    """Return every schema-visible arg name declared by a tool spec."""

    ordered_field_names: list[str] = []
    _append_unique_field_names(ordered_field_names, tool_spec.get("tool_args", ()))
    _append_unique_field_names(
        ordered_field_names,
        tool_spec.get("optional_tool_args", ()),
    )
    for arg_group in tool_spec.get("tool_arg_any_of", ()):
        if not isinstance(arg_group, (list, tuple)):
            raise ValueError("tool_arg_any_of entries must be lists or tuples.")
        _append_unique_field_names(
            ordered_field_names,
            tuple(
                field_name
                for field_name in arg_group
                if isinstance(field_name, str)
            ),
        )
    return tuple(ordered_field_names)


def _build_scalar_field_schema(
    field_name: str,
    agent_ids: Sequence[str],
    schema_type: str = "STRING",
) -> dict[str, Any]:
    """Builds a scalar schema property entry for one symbolic response field."""

    if field_name in {"agent", "agent_id", "to"}:
        return {
            "type": "STRING",
            "enum": list(agent_ids),
        }
    return {"type": schema_type}


def _build_symbolic_field_schema(
    field_name: str,
    agent_ids: Sequence[str],
    schema_type: str = "STRING",
) -> dict[str, Any]:
    """Builds a schema property entry for one symbolic response field."""

    if schema_type == "STRING_ARRAY":
        return {
            "type": "ARRAY",
            "items": _build_scalar_field_schema(field_name, agent_ids),
        }
    return _build_scalar_field_schema(field_name, agent_ids, schema_type)


def _allowed_ids_key_for_arg_name(arg_name: str) -> str | None:
    """Maps a symbolic tool argument like fixture_id to its allowed_* override key."""

    if not arg_name.endswith("_id"):
        return None
    return f"allowed_{arg_name[:-3]}_ids"


def _resolve_tool_arg_schema_type(
    field_name: str,
    tool_spec: dict[str, Any],
) -> str:
    """Resolves the shared response-schema type for one tool argument."""

    tool_arg_types = tool_spec.get("tool_arg_types", {})
    if not isinstance(tool_arg_types, dict):
        raise ValueError("tool_arg_types must be a mapping when provided.")
    schema_type = tool_arg_types.get(field_name, "STRING")
    if schema_type not in {"STRING", "INTEGER", "STRING_ARRAY"}:
        raise ValueError(f"Unsupported schema type {schema_type!r} for {field_name}.")
    return schema_type


def build_task_response_schema(
    *,
    agent_ids: Sequence[str],
    allowed_tool_specs: dict[str, dict[str, Any]],
    min_steps: int = 3,
) -> dict[str, Any]:
    """Builds the shared task-level response schema from agent and tool metadata."""

    if not agent_ids:
        raise ValueError("agent_ids must contain at least one agent.")
    if not allowed_tool_specs:
        raise ValueError("allowed_tool_specs must contain at least one tool.")
    if min_steps < 1:
        raise ValueError("min_steps must be at least 1.")

    def arg_property(field_name: str, tool_spec: dict[str, Any]) -> dict[str, Any]:
        property_schema = _build_symbolic_field_schema(
            field_name,
            agent_ids,
            _resolve_tool_arg_schema_type(field_name, tool_spec),
        )
        allowed_ids_key = _allowed_ids_key_for_arg_name(field_name)
        allowed_values = tool_spec.get(allowed_ids_key) if allowed_ids_key else None
        declared_values = tool_spec.get("allowed_arg_values", {})
        if not allowed_values and isinstance(declared_values, dict):
            allowed_values = declared_values.get(field_name)
        if isinstance(allowed_values, (list, tuple)) and allowed_values:
            if property_schema.get("type") == "ARRAY":
                property_schema["items"]["enum"] = list(allowed_values)
            else:
                property_schema["enum"] = list(allowed_values)
        return property_schema

    def args_schema(tool_spec: dict[str, Any]) -> dict[str, Any]:
        required = list(tool_spec.get("tool_args", ()))
        declared = list(_iter_declared_tool_arg_names(tool_spec))
        any_of_groups = tool_spec.get("tool_arg_any_of", ())
        if not any_of_groups:
            return {
                "type": "OBJECT",
                "required": required,
                "properties": {
                    name: arg_property(name, tool_spec) for name in declared
                },
            }

        # Each alternative gets its own closed args object. This prevents a
        # model from mixing aliases (or attaching another tool's arguments).
        alternatives: list[dict[str, Any]] = []
        for group in any_of_groups:
            for selected_name in group:
                branch_names = [
                    name
                    for name in declared
                    if name not in group or name == selected_name
                ]
                alternatives.append(
                    {
                        "type": "OBJECT",
                        "required": [*required, selected_name],
                        "properties": {
                            name: arg_property(name, tool_spec)
                            for name in branch_names
                        },
                    }
                )
        return {"anyOf": alternatives}

    step_variants = []
    for tool_name, tool_spec in allowed_tool_specs.items():
        step_variants.append(
            {
                "type": "OBJECT",
                "required": ["step", "agent", "tool", "args", "reasoning"],
                "properties": {
                    "step": {"type": "INTEGER"},
                    "agent": _build_symbolic_field_schema("agent", agent_ids),
                    "tool": {"type": "STRING", "enum": [tool_name]},
                    "args": args_schema(tool_spec),
                    "reasoning": {"type": "STRING"},
                },
            }
        )

    return {
        "type": "OBJECT",
        "required": ["steps"],
        "properties": {
            "steps": {
                "type": "ARRAY",
                "minItems": min_steps,
                "items": {"anyOf": step_variants},
            },
        },
    }
