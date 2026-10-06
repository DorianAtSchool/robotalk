"""
Primitive simulator tool specifications for symbolic kitchen planning.

These specs are intentionally lightweight: they describe the simulator-facing
tool surface, not the higher-level semantic task decomposition that may sit on
top of it.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def _build_tool_spec(
    name: str,
    description: str,
    *arg_specs: str | tuple[str, str] | tuple[str, str, bool],
) -> dict[str, Any]:
    """Build a minimal tool spec with required parameters."""
    parameters = []
    for arg_spec in arg_specs:
        if isinstance(arg_spec, tuple):
            if len(arg_spec) == 3:
                arg_name, arg_type, required = arg_spec
            else:
                arg_name, arg_type = arg_spec
                required = True
        else:
            arg_name, arg_type, required = arg_spec, "string", True
        parameters.append(
            {
                "name": arg_name,
                "type": arg_type,
                "required": bool(required),
            }
        )
    return {
        "name": name,
        "description": description,
        "parameters": parameters,
    }


SIM_TOOL_SPECS: list[dict[str, Any]] = [
    _build_tool_spec(
        "get_image",
        "Capture and save one or more images from named views, including the placement map via map.",
        ("views", "array"),
        ("image_paths", "array"),
    ),
    _build_tool_spec(
        "close_hinged_part",
        "Close a hinged door, lid, or articulated head on a fixture.",
        "target_id",
        "part_id",
    ),
    _build_tool_spec(
        "close_sliding_part",
        "Push in a sliding drawer or rack on a fixture.",
        "target_id",
        "part_id",
    ),
    _build_tool_spec(
        "communicate",
        "Send a coordination message to another robot agent.",
        "to",
        "message",
    ),
    _build_tool_spec(
        "navigate_to_fixture",
        "Move the robot base to the working pose of a fixture.",
        "fixture_id",
    ),
    _build_tool_spec(
        "open_hinged_part",
        "Open a hinged door, lid, or articulated head on a fixture.",
        "target_id",
        "part_id",
    ),
    _build_tool_spec(
        "open_sliding_part",
        "Pull out a sliding drawer or rack on a fixture.",
        "target_id",
        "part_id",
    ),
    _build_tool_spec(
        "pick_up_object",
        "Grasp and lift an object from a symbolic source region.",
        "object_id",
        "source_id",
        ("source_site_id", "string", False),
    ),
    _build_tool_spec(
        "place_in_receptacle",
        "Place an object into an interior or container-like receptacle. Use target_site_id separately when the exact basin, bowl, rack, shelf, slot, or tray matters.",
        "object_id",
        "receptacle_id",
        ("target_id", "string", False),
        ("target_site_id", "string", False),
        ("relative_position", "string", False),
    ),
    _build_tool_spec(
        "place_next_to",
        "Place an object on the surrounding fixture surface, beside a reference object or fixture. This tool never places an object on or inside a movable object. To place something on a plate, tray, cutting board, or other movable support, use place_on_object. To place something inside a bowl or other container, use place_in_receptacle.",
        "object_id",
        "reference_object_id",
        ("reference_id", "string", False),
        ("reference_fixture_id", "string", False),
        ("target_site_id", "string", False),
        ("relative_position", "string", False),
    ),
    _build_tool_spec(
        "place_on_object",
        "Place an object on top of another movable support object.",
        "object_id",
        "support_object_id",
        "anchor_fixture_id",
        ("target_id", "string", False),
        ("relative_position", "string", False),
    ),
    _build_tool_spec(
        "place_on_surface",
        "Place an object onto an exposed support surface or attachment seat. Keep the fixture id in support_id or target_id and put the exact burner, rack, or other sub-location in target_site_id.",
        "object_id",
        "support_id",
        ("target_id", "string", False),
        ("target_site_id", "string", False),
        ("relative_position", "string", False),
    ),
    _build_tool_spec(
        "place_under",
        "Place an object directly beneath a reference fixture. For dispensers "
        "(e.g. coffee machine nozzle, sink faucet) the object is positioned at "
        "the dispenser output site; for other fixtures (e.g. wall cabinet) the "
        "object is placed on the nearest surface below.",
        "object_id",
        "reference_fixture_id",
        ("target_id", "string", False),
        ("target_site_id", "string", False),
    ),
    _build_tool_spec(
        "press_button",
        "Activate a discrete button-like control on a fixture.",
        "target_id",
        "control_id",
    ),
    _build_tool_spec(
        "press_lever",
        "Activate a discrete lever-like control on a fixture.",
        "target_id",
        "control_id",
    ),
    _build_tool_spec(
        "set_rotary_control",
        "Set a rotary or handle-like control to an explicit goal state.",
        "target_id",
        "control_id",
        "goal",
    ),
    _build_tool_spec(
        "give_space",
        "Move away from a fixture so another robot can access it.",
        "fixture_id",
    ),
    _build_tool_spec(
        "wait",
        "Pause and observe while another robot completes its step.",
    ),
    _build_tool_spec(
        "wait_for_signal",
        "Block until the named robot sends a message. Any message wakes you, so "
        "call this again if what arrives is not what you were waiting for.",
        "from",
        "about",
    ),
]

SIM_TOOL_SPEC_BY_NAME = {spec["name"]: spec for spec in SIM_TOOL_SPECS}


def get_sim_tool_specs() -> list[dict[str, Any]]:
    """Return a defensive copy of the simulator tool specs."""
    return deepcopy(SIM_TOOL_SPECS)


def get_sim_tool_spec(name: str) -> dict[str, Any]:
    """Return a defensive copy of a single simulator tool spec."""
    return deepcopy(SIM_TOOL_SPEC_BY_NAME[name])


__all__ = [
    "SIM_TOOL_SPECS",
    "SIM_TOOL_SPEC_BY_NAME",
    "get_sim_tool_specs",
    "get_sim_tool_spec",
]
