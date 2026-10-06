"""Define shared symbolic tool categories for task-level prompting and validation."""

from __future__ import annotations

# Keep the shared FSM vocabulary small so task validators stay intuitive.
NAVIGATION_TOOL_NAMES = frozenset({"navigate_to_fixture"})
ACQUIRE_TOOL_NAMES = frozenset({"pick_up_object"})
RELEASE_TOOL_NAMES = frozenset(
    {
        "place_in_receptacle",
        "place_next_to",
        "place_on_object",
        "place_on_surface",
        "place_under",
    }
)
OBSERVATION_TOOL_NAMES = frozenset(
    {
        "get_image",
    }
)
WAIT_TOOL_NAMES = frozenset({"wait_for_signal"})
# Steps that carry no symbolic effect: they neither move an agent nor change any
# object, so they are exempt from the observation bracketing every real action
# needs. `communicate` was handled by name before `wait_for_signal` joined it.
SOCIAL_TOOL_NAMES = frozenset({"communicate"}) | WAIT_TOOL_NAMES
GIVE_SPACE_TOOL_NAMES = frozenset({"give_space"})
OPEN_PART_TOOL_NAMES = frozenset({"open_hinged_part", "open_sliding_part"})
CLOSE_PART_TOOL_NAMES = frozenset({"close_hinged_part", "close_sliding_part"})
INTERACTION_TOOL_NAMES = frozenset(
    {"press_button", "press_lever", "set_rotary_control"}
)
PLACE_LOCATION_ARG_NAMES = (
    "target_id",
    "support_id",
    "receptacle_id",
    "support_object_id",
)

# Args that name a thing whose state one agent can change under another.
DEPENDENCY_ARG_NAMES = (
    "object_id",
    "source_id",
    "support_object_id",
    "reference_object_id",
    "receptacle_id",
)
# A release or announcement has to say something. Requiring only that the id
# appear verbatim is satisfied by a message that is literally just the id --
# observed in generated data, and it teaches the model that a bare token
# discharges an obligation.
MIN_SIGNAL_MESSAGE_WORDS = 4

# Fixtures only one agent can work at, whatever they are handling: either a
# single opening that must be approached head-on (mirrors _REQUIRE_FRONT_TYPES
# in the sim executor), or a footprint too small for two robots. Everything
# else -- counters, islands, dining tables -- is roomy enough to share, so two
# agents there only conflict if they touch the SAME object.
EXCLUSIVE_FIXTURE_TYPES = frozenset({
    "drawer", "top_drawer",
    "fridge", "microwave", "oven", "dishwasher", "stove", "sink",
    "toaster", "toaster_oven", "coffee_machine", "blender", "stand_mixer",
    "electric_kettle",
})
# Args naming a fixture the acting agent must be standing at.
FIXTURE_ARG_NAMES = ("fixture_id", "source_id", "support_id", "target_id")
