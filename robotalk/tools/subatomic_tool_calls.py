"""Define the shared subatomic tool-call catalog used by task-level prompts."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class SubatomicToolArg:
    """Represents one prompt-facing argument for a shared subatomic tool."""

    name: str
    schema_type: str = "STRING"


@dataclass(frozen=True)
class SubatomicToolSpec:
    """Stores prompt-facing metadata for a shared subatomic tool."""

    name: str
    description: str
    constructor_args: tuple[SubatomicToolArg, ...]

    def to_prompt_block(self) -> str:
        """Renders a compact prompt entry for a single tool."""

        if self.constructor_args:
            arg_text = ", ".join(arg.name for arg in self.constructor_args)
        else:
            arg_text = "no explicit constructor inputs"
        return f"- {self.name}: {self.description} Inputs: {arg_text}."


def _build_tool_spec(
    name: str,
    description: str,
    *arg_specs: str | tuple[str, str],
) -> SubatomicToolSpec:
    """Builds a static subatomic tool specification."""

    constructor_args: list[SubatomicToolArg] = []
    for arg_spec in arg_specs:
        if isinstance(arg_spec, tuple):
            arg_name, schema_type = arg_spec
            constructor_args.append(
                SubatomicToolArg(name=arg_name, schema_type=schema_type)
            )
            continue
        constructor_args.append(SubatomicToolArg(name=arg_spec))

    return SubatomicToolSpec(
        name=name,
        description=description,
        constructor_args=tuple(constructor_args),
    )


# Keep the shared subatomic catalog static so prompt construction is predictable.
SUBATOMIC_TOOL_SPECS: tuple[SubatomicToolSpec, ...] = (
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
        "get_image",
        "Capture one or more images from named views for later inspection.",
        ("views", "STRING_ARRAY"),
    ),
    _build_tool_spec(
        "give_space",
        "Yield working room at a fixture so the other agent can complete an action there.",
        "fixture_id",
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
    ),
    _build_tool_spec(
        "place_in_receptacle",
        "Place an object into an interior or container-like receptacle. Use target_site_id separately when the exact basin, bowl, rack, shelf, slot, or tray matters.",
        "object_id",
        "receptacle_id",
    ),
    _build_tool_spec(
        "place_next_to",
        "Place an object on the surrounding fixture surface, beside a reference object or fixture. This tool never places an object on or inside a movable object. To place something on a plate, tray, cutting board, or other movable support, use place_on_object. To place something inside a bowl or other container, use place_in_receptacle.",
        "object_id",
        "reference_object_id",
    ),
    _build_tool_spec(
        "place_on_object",
        "Place an object on top of another movable support object such as a plate or cutting_board.",
        "object_id",
        "support_object_id",
    ),
    _build_tool_spec(
        "place_on_surface",
        "Place an object onto an exposed support surface or attachment seat. Keep the fixture id in support_id or target_id and put the exact burner, rack, or other sub-location in target_site_id.",
        "object_id",
        "support_id",
    ),
    _build_tool_spec(
        "place_under",
        "Place an object directly beneath a reference fixture. For dispensers "
        "(e.g. coffee machine nozzle, sink faucet) the object is positioned at "
        "the dispenser output site; for other fixtures (e.g. wall cabinet) the "
        "object is placed on the nearest surface below.",
        "object_id",
        "reference_fixture_id",
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
)


@lru_cache(maxsize=1)
def discover_subatomic_tools() -> tuple[SubatomicToolSpec, ...]:
    """Returns the shared subatomic catalog used by task-level generation."""

    return tuple(sorted(SUBATOMIC_TOOL_SPECS, key=lambda tool: tool.name))


def render_subatomic_tool_catalog(
    tool_specs: tuple[SubatomicToolSpec, ...] | None = None,
) -> str:
    """Renders the subatomic tool catalog for inclusion in task prompts."""

    if tool_specs is None:
        tool_specs = discover_subatomic_tools()
    return "\n".join(tool.to_prompt_block() for tool in tool_specs)
