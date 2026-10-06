"""Shared tool contracts and task-specific validator spec builders.

``TASK_LEVEL_ALLOWED_TOOL_SPECS`` is the canonical, task-agnostic interface.
Task specs may refine it with symbolic allowlists for validation, but those
refinements are deliberately not part of the model-facing interface.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

from robotalk.tools.subatomic_tool_calls import discover_subatomic_tools

BASIC_TASK_TOOL_NAMES = ("give_space",)

KNOWN_OBSERVATION_VIEWS = (
    "top_view",
    "room_view",
    "map",
    "wrist",
    "agentview_center",
    "agentview_left",
    "agentview_right",
)


# These descriptions are intentionally task agnostic.  They explain the
# meaning of each argument without revealing which IDs solve a particular
# task.  Both trajectory generation and policy train/eval schemas consume this
# same metadata.
TOOL_ARG_DESCRIPTIONS: dict[str, str] = {
    "about": "Exact symbolic object or fixture ID being awaited.",
    "control_id": (
        "Exact key of the control under "
        "initial_state.fixtures[target_id].controls; copy it verbatim."
    ),
    "coordination_phase": (
        "Optional opening-protocol phase, or portion_complete when reporting "
        "only this agent's assigned work complete. Omit for ordinary messages."
    ),
    "fixture_id": "Exact symbolic ID of the fixture or workspace.",
    "from": "Exact ID of the agent whose matching release will end the wait.",
    "goal": "Requested final symbolic state of the control.",
    "message": "Short natural-language coordination message for the other agent.",
    "object_id": "Exact symbolic ID of the movable object being manipulated.",
    "part_id": (
        "Exact key of the part under initial_state.fixtures[target_id].parts; "
        "copy it verbatim (for example door, hinged, lid, or sliding)."
    ),
    "receptacle_id": "Exact symbolic ID of the destination receptacle.",
    "reference_fixture_id": "Exact symbolic ID of the fixture used as the placement reference.",
    "reference_id": "Exact symbolic ID of the object or fixture used as the placement reference.",
    "reference_object_id": "Exact symbolic ID of the movable object used as the placement reference.",
    "relative_position": "Optional named relative placement; omit unless the goal requires one.",
    "releases": (
        "Optional exact symbolic object or fixture ID being handed over. Include "
        "only for a real handover; omit from all other messages."
    ),
    "source_id": "Exact symbolic ID of the object's current source region.",
    "source_site_id": "Optional exact sub-site within the source; omit when no named sub-site matters.",
    "support_id": "Exact symbolic ID of the fixture or surface supporting the placement.",
    "support_object_id": "Exact symbolic ID of the movable object supporting the placed object.",
    "target_id": "Exact symbolic ID of the fixture, receptacle, or destination targeted by the action.",
    "target_site_id": "Optional exact sub-site within the destination; omit when no named sub-site matters.",
    "to": "Exact ID of the agent receiving the message.",
    "views": (
        "A non-empty list containing only these exact camera-view names: "
        + ", ".join(KNOWN_OBSERVATION_VIEWS)
        + ". Copy the names exactly; do not invent directional, object, fixture, "
        "or scene-specific view names."
    ),
}


def _build_subatomic_allowed_tool_specs() -> dict[str, dict[str, Any]]:
    """Builds the shared allowed-tool metadata for every subatomic tool."""

    optional_arg_names_by_tool = {
        "pick_up_object": ["source_site_id"],
        "place_in_receptacle": [
            "target_site_id",
            "relative_position",
        ],
        "place_next_to": [
            "target_site_id",
            "relative_position",
        ],
        "place_on_object": ["relative_position"],
        "place_on_surface": [
            "target_site_id",
            "relative_position",
        ],
        "place_under": ["target_site_id"],
    }
    subatomic_allowed_tool_specs: dict[str, dict[str, Any]] = {}
    for tool_spec in discover_subatomic_tools():
        argument_names = [argument.name for argument in tool_spec.constructor_args]
        tool_arg_types = {
            argument.name: argument.schema_type
            for argument in tool_spec.constructor_args
            if argument.schema_type != "STRING"
        }
        subatomic_allowed_tool_spec = {
            "description": tool_spec.description,
            "tool_args": list(argument_names),
            "tool_arg_descriptions": {
                name: TOOL_ARG_DESCRIPTIONS[name]
                for name in argument_names
                if name in TOOL_ARG_DESCRIPTIONS
            },
        }
        if tool_arg_types:
            subatomic_allowed_tool_spec["tool_arg_types"] = tool_arg_types
        if tool_spec.name == "place_next_to":
            # `place_next_to` can anchor relative to either a movable object
            # or a fixture with an explicit adjacent symbolic surface.
            subatomic_allowed_tool_spec["description"] = (
                f"{tool_spec.description} When using reference_fixture_id, the "
                "agent may navigate either to that reference fixture or to its "
                "declared adjacent_location_id before placing."
            )
            subatomic_allowed_tool_spec["tool_args"] = ["object_id"]
            subatomic_allowed_tool_spec["tool_arg_any_of"] = [
                ["reference_object_id", "reference_fixture_id"]
            ]
        elif tool_spec.name == "give_space":
            subatomic_allowed_tool_spec["description"] = (
                "Leave a workspace when vacating it is genuinely needed so another "
                "agent can enter or work there. This is a handover action, not an "
                "idle, completion, or participation action. Roomy counters, islands, "
                "and dining tables normally need no give_space when agents use "
                "different objects."
            )
        elif tool_spec.name == "place_on_surface":
            subatomic_allowed_tool_spec["tool_args"] = ["object_id", "support_id"]
        elif tool_spec.name == "place_in_receptacle":
            subatomic_allowed_tool_spec["tool_args"] = ["object_id", "receptacle_id"]
        elif tool_spec.name == "place_on_object":
            subatomic_allowed_tool_spec["tool_args"] = ["object_id", "support_object_id"]
        elif tool_spec.name == "place_under":
            subatomic_allowed_tool_spec["tool_args"] = ["object_id", "reference_fixture_id"]
        optional_tool_args = optional_arg_names_by_tool.get(tool_spec.name)
        if optional_tool_args:
            subatomic_allowed_tool_spec["optional_tool_args"] = list(optional_tool_args)
        declared_names = set(subatomic_allowed_tool_spec.get("tool_args", ()))
        declared_names.update(subatomic_allowed_tool_spec.get("optional_tool_args", ()))
        for group in subatomic_allowed_tool_spec.get("tool_arg_any_of", ()):
            declared_names.update(group)
        subatomic_allowed_tool_spec["tool_arg_descriptions"] = {
            name: TOOL_ARG_DESCRIPTIONS[name]
            for name in sorted(declared_names)
            if name in TOOL_ARG_DESCRIPTIONS
        }
        subatomic_allowed_tool_specs[tool_spec.name] = subatomic_allowed_tool_spec
    return subatomic_allowed_tool_specs


# Keep the shared tool metadata in one place so individual task files only add
# task-specific symbolic constraints such as allowed fixture or object IDs.
SUBATOMIC_ALLOWED_TOOL_SPECS = _build_subatomic_allowed_tool_specs()

# Discovery knows that get_image takes a string array, but the permitted
# vocabulary belongs to the task-level observation contract rather than the
# low-level constructor. Put it on the shared model interface here so every
# backend receives the same item enum and description.
if "get_image" in SUBATOMIC_ALLOWED_TOOL_SPECS:
    SUBATOMIC_ALLOWED_TOOL_SPECS["get_image"].update(
        {
            "description": (
                "Request one or more canonical camera views before choosing "
                "the next action."
            ),
            "allowed_views": list(KNOWN_OBSERVATION_VIEWS),
        }
    )


TASK_LEVEL_ALLOWED_TOOL_SPECS = {
    "communicate": {
        "description": (
            "Send a short coordination message to the other agent in the scene. "
            "args.to must be that agent's exact ID. Set args.releases to the "
            "exact symbolic id of a thing you have finished with and are handing "
            "over -- that, and only that, wakes a partner who is waiting on it. "
            "Omit args.releases from every other message. "
            "args.releases must be an object id or a fixture id that appears in "
            "initial_state, copied exactly. It names a THING, never an event or "
            "a status: 'coffee_machine' and 'mug' are ids; 'mug_placed', "
            "'counter_free' and 'task_done' are not, and an id that does not "
            "exist releases nothing at all."
            " args.coordination_phase is required for protocol messages and must "
            "be one of propose, await_plan, await_confirmation, confirm, "
            "or portion_complete. Omit it from ordinary requests, corrections, "
            "handoffs, and informational messages."
        ),
        "tool_args": ["to", "message"],
        "optional_tool_args": ["releases", "coordination_phase"],
        "allowed_arg_values": {
            "coordination_phase": [
                "propose",
                "await_plan",
                "await_confirmation",
                "confirm",
                "portion_complete",
            ]
        },
        "tool_arg_descriptions": {
            name: TOOL_ARG_DESCRIPTIONS[name]
            for name in ("to", "message", "releases", "coordination_phase")
        },
    },
    "wait_for_signal": {
        "description": (
            "Pause until the other agent hands over what you are waiting for. "
            "args.about names that thing by its exact symbolic id -- an object "
            "id or a fixture id that appears in initial_state, copied exactly. "
            "It is a THING you are waiting FOR, never an event you are waiting "
            "ON: wait about='coffee_machine', never about='mug_placed' or "
            "'your_turn'. An id that is not in initial_state can never be "
            "released, so the wait blocks forever and the whole plan is thrown "
            "away. You stay "
            "paused until args.from sends a communicate whose args.releases "
            "names the same id -- their other messages do not wake you, so you "
            "do not need to wait again after an unrelated one."
        ),
        "tool_args": ["from", "about"],
        "tool_arg_descriptions": {
            name: TOOL_ARG_DESCRIPTIONS[name] for name in ("from", "about")
        },
    },
    **SUBATOMIC_ALLOWED_TOOL_SPECS,
}


_LEGACY_PLACEMENT_ANCHOR_BY_TOOL = {
    "place_in_receptacle": "receptacle_id",
    "place_on_object": "support_object_id",
    "place_on_surface": "support_id",
    "place_under": "reference_fixture_id",
}


def canonicalize_model_tool_args(
    tool_name: str,
    args: Mapping[str, Any],
) -> dict[str, Any]:
    """Translate unambiguous legacy placement aliases at the model boundary."""

    canonical = deepcopy(dict(args))
    anchor_name = _LEGACY_PLACEMENT_ANCHOR_BY_TOOL.get(tool_name)
    if anchor_name is not None:
        legacy_value = canonical.pop("target_id", None)
        if anchor_name not in canonical and legacy_value is not None:
            canonical[anchor_name] = legacy_value
    return canonical


def build_model_tool_specs(
    *,
    include_get_image: bool = False,
) -> dict[str, dict[str, Any]]:
    """Return the one global, task-agnostic interface shown to a model.

    Task-local allowlists and overrides never enter this view. ``get_image``
    is exposed only when active observation is enabled.
    """

    specs = deepcopy(TASK_LEVEL_ALLOWED_TOOL_SPECS)
    if not include_get_image:
        specs.pop("get_image", None)
    return specs


def build_allowed_tool_specs(
    tool_names: Sequence[str],
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Builds a task-local allowed-tool map from the shared task-level registry.

    Args:
        tool_names: Ordered tool names to expose for a specific task, such as
            ``("communicate", "navigate_to_fixture", "pick_up_object")``.
            Shared basic coordination tools such as ``give_space`` are appended
            automatically when omitted.
        overrides: Optional per-tool metadata patches applied on top of the
            shared registry entry. Use this to add task-specific symbolic
            constraints without redefining the base tool spec. For example,
            ``{"pick_up_object": {"allowed_object_ids": ["mug"]}}`` keeps the
            shared description and ``tool_args`` fields, then
            adds the task-specific ``allowed_object_ids`` constraint.

    Returns:
        A new dictionary containing only the requested tools, with any override
        fields merged into the copied base specification for each tool.

    Raises:
        KeyError: If ``tool_names`` includes an unknown tool or if
            ``overrides`` contains a tool name that was not requested.
    """

    requested_tool_names = list(tool_names)
    for basic_tool_name in BASIC_TASK_TOOL_NAMES:
        if basic_tool_name not in requested_tool_names:
            requested_tool_names.append(basic_tool_name)

    selected_tool_specs: dict[str, dict[str, Any]] = {}
    tool_overrides = dict(overrides or {})

    for tool_name in requested_tool_names:
        if tool_name not in TASK_LEVEL_ALLOWED_TOOL_SPECS:
            raise KeyError(f"Unknown task-level tool name: {tool_name}")
        selected_tool_specs[tool_name] = deepcopy(
            TASK_LEVEL_ALLOWED_TOOL_SPECS[tool_name]
        )
        if tool_name in tool_overrides:
            # Merge task-specific symbolic constraints into the shared base spec.
            selected_tool_specs[tool_name].update(
                deepcopy(dict(tool_overrides[tool_name]))
            )

    # Keep give_space aligned with navigation fixture constraints when tasks do
    # not need to repeat the same allowed fixture list twice.
    give_space_spec = selected_tool_specs.get("give_space")
    navigate_tool_spec = selected_tool_specs.get("navigate_to_fixture")
    if (
        give_space_spec is not None
        and "allowed_fixture_ids" not in give_space_spec
        and navigate_tool_spec is not None
        and "allowed_fixture_ids" in navigate_tool_spec
    ):
        give_space_spec["allowed_fixture_ids"] = deepcopy(
            navigate_tool_spec["allowed_fixture_ids"]
        )

    unknown_override_names = set(tool_overrides) - set(requested_tool_names)
    if unknown_override_names:
        unknown_names = ", ".join(sorted(unknown_override_names))
        raise KeyError(
            f"Overrides were provided for unavailable tool names: {unknown_names}"
        )

    # Specialize placement aliases to the task's actual symbolic vocabulary.
    # Shared constructors accept several destination spellings, but exposing an
    # unconstrained alias (for example target_site_id when the task defines no
    # sites) invites plausible-looking IDs that can never validate.
    for tool_spec in selected_tool_specs.values():
        optional_args = list(tool_spec.get("optional_tool_args", ()))
        any_of_groups = list(tool_spec.get("tool_arg_any_of", ()))
        declared_required = list(tool_spec.get("tool_args", ()))

        # A few imported verified specs use the legacy singular argument key
        # itself for its allowed-value list. Normalize that representation
        # before deciding which task-local aliases are meaningful.
        for arg_name in {
            *declared_required,
            *optional_args,
            *(name for group in any_of_groups for name in group),
        }:
            legacy_values = tool_spec.get(arg_name)
            if arg_name.endswith("_id") and isinstance(legacy_values, list):
                allowed_key = f"allowed_{arg_name[:-3]}_ids"
                tool_spec.setdefault(allowed_key, deepcopy(legacy_values))
                tool_spec.pop(arg_name, None)

        def has_task_values(arg_name: str) -> bool:
            if not arg_name.endswith("_id"):
                return True
            allowed_key = f"allowed_{arg_name[:-3]}_ids"
            values = tool_spec.get(allowed_key)
            return isinstance(values, (list, tuple)) and bool(values)

        pruned_groups: list[list[str]] = []
        for raw_group in any_of_groups:
            available = [name for name in raw_group if has_task_values(name)]
            if len(available) == 1:
                if available[0] not in declared_required:
                    declared_required.append(available[0])
            elif available:
                pruned_groups.append(available)
        grouped_names = {
            name for group in any_of_groups for name in group
        }
        optional_args = [
            name
            for name in optional_args
            if name not in grouped_names or has_task_values(name)
        ]
        optional_args = [
            name for name in optional_args if name not in declared_required
        ]
        # Site aliases are meaningful only when a task explicitly enumerates
        # sites; fixture/object destination aliases remain governed above.
        optional_args = [
            name
            for name in optional_args
            if not name.endswith("_site_id") or has_task_values(name)
        ]
        tool_spec["tool_args"] = declared_required
        if optional_args:
            tool_spec["optional_tool_args"] = optional_args
        else:
            tool_spec.pop("optional_tool_args", None)
        if pruned_groups:
            tool_spec["tool_arg_any_of"] = pruned_groups
        else:
            tool_spec.pop("tool_arg_any_of", None)

    return selected_tool_specs
