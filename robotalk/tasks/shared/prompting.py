"""Render shared task prompts from task metadata and task state."""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Sequence

from .constants import (
    ACQUIRE_TOOL_NAMES,
    CLOSE_PART_TOOL_NAMES,
    EXCLUSIVE_FIXTURE_TYPES,
    GIVE_SPACE_TOOL_NAMES,
    WAIT_TOOL_NAMES,
    INTERACTION_TOOL_NAMES,
    OPEN_PART_TOOL_NAMES,
    RELEASE_TOOL_NAMES,
)
from .types import TaskInstance, TaskPromptBuilder


def _canonical_arg_groups_for_prompt(
    tool_name: str,
    tool_spec: dict[str, Any],
) -> list[tuple[str, ...]]:
    """Render generation-facing arg alternatives without legacy aliases.

    We keep runtime alias support for backward compatibility, but generation
    prompts should teach one canonical form per semantic role.
    """

    canonical_overrides: dict[str, list[tuple[str, ...]]] = {
        "place_on_surface": [("support_id",)],
        "place_in_receptacle": [("receptacle_id",)],
        "place_on_object": [("support_object_id",)],
        "place_under": [("reference_fixture_id",)],
        "place_next_to": [("reference_object_id", "reference_fixture_id")],
    }
    if tool_name in canonical_overrides:
        declared = set(tool_spec.get("tool_args", ())) | set(
            tool_spec.get("optional_tool_args", ())
        )
        for group in tool_spec.get("tool_arg_any_of", ()):
            declared.update(group)
        filtered = [
            tuple(arg_name for arg_name in group if arg_name in declared)
            for group in canonical_overrides[tool_name]
        ]
        return [group for group in filtered if group]

    groups: list[tuple[str, ...]] = []
    for arg_group in tool_spec.get("tool_arg_any_of", ()):
        if not isinstance(arg_group, (list, tuple)):
            continue
        normalized_group = tuple(
            arg_name for arg_name in arg_group if isinstance(arg_name, str)
        )
        if normalized_group:
            groups.append(normalized_group)
    return groups


def _format_agent_id_list(agent_ids: Sequence[str]) -> str:
    """Formats agent IDs into a short prompt-facing list."""

    if not agent_ids:
        raise ValueError("agent_ids must contain at least one agent.")
    if len(agent_ids) == 1:
        return agent_ids[0]
    if len(agent_ids) == 2:
        return f"{agent_ids[0]} and {agent_ids[1]}"
    return f"{', '.join(agent_ids[:-1])}, and {agent_ids[-1]}"


def format_concurrency_facts(initial_state: dict[str, Any]) -> str:
    """Render fixture IDs using the same type policy as concurrent validation."""

    fixtures = initial_state.get("fixtures") or {}
    exclusive: list[str] = []
    shared: list[str] = []
    for fixture_id, fixture_state in sorted(fixtures.items()):
        fixture_type = str((fixture_state or {}).get("fixture_type", "")).lower()
        (exclusive if fixture_type in EXCLUSIVE_FIXTURE_TYPES else shared).append(
            fixture_id
        )
    exclusive_text = ", ".join(exclusive) if exclusive else "none"
    shared_text = ", ".join(shared) if shared else "none"
    return (
        "Concurrency facts (derived from the same fixture-type policy as the "
        "concurrent validator):\n"
        f"- Exclusive, one agent at a time: {exclusive_text}.\n"
        "- Roomy/shared for simultaneous work on different objects: "
        f"{shared_text}."
    )


def _format_initial_agent_positions(
    initial_state: dict[str, Any],
    agent_ids: Sequence[str],
) -> str:
    """Formats the prompt-facing list of starting fixture positions."""

    position_lines: list[str] = []
    for agent_id in agent_ids:
        agent_state = initial_state.get("agents", {}).get(agent_id, {})
        location = agent_state.get("location")
        location_label = (
            location if isinstance(location, str) and location else "unknown"
        )
        position_lines.append(f"- {agent_id}: {location_label}")
    return "\n".join(position_lines)


def _optional_arg_condition(arg_name: str) -> str:
    """Explain when a schema-optional tool argument belongs in a call."""

    conditions = {
        "coordination_phase": (
            "only for propose, await_plan, await_confirmation, confirm, or "
            "portion_complete protocol messages"
        ),
        "releases": (
            "one exact object or fixture ID; only when this message hands that "
            "resource to an agent already waiting for it"
        ),
        "source_site_id": (
            "only when the listed source has an exact sub-site that matters"
        ),
        "target_site_id": (
            "only when the listed destination has an exact sub-site that matters"
        ),
        "relative_position": (
            "only when the goal requires a specific relative placement"
        ),
    }
    return conditions.get(arg_name, "only when needed for this exact action")


def _format_tool_contracts(allowed_tool_specs: dict[str, dict[str, Any]]) -> str:
    """Render compact task-agnostic contracts from the shared interface."""

    blocks: list[str] = []
    for tool_name, spec in allowed_tool_specs.items():
        lines = [tool_name, f"  Purpose: {spec.get('description', '').strip()}"]
        descriptions = dict(spec.get("tool_arg_descriptions", {}))
        arg_types = dict(spec.get("tool_arg_types", {}))
        required = list(spec.get("tool_args", ()))
        lines.append("  Required:" if required else "  Required: none")
        for arg_name in required:
            lines.append(
                f"    {arg_name} ({arg_types.get(arg_name, 'STRING')}): "
                f"{descriptions.get(arg_name, 'Exact symbolic value required by this action.')}"
            )
        for group in spec.get("tool_arg_any_of", ()):
            lines.append("  At least one required alternative:")
            for arg_name in group:
                lines.append(
                    f"    {arg_name} ({arg_types.get(arg_name, 'STRING')}): "
                    f"{descriptions.get(arg_name, 'Exact symbolic value required by this action.')}"
                )
        optional = list(spec.get("optional_tool_args", ()))
        if optional:
            lines.append("  Optional (otherwise omit):")
            for arg_name in optional:
                lines.append(
                    f"    {arg_name} ({arg_types.get(arg_name, 'STRING')}): "
                    f"{descriptions.get(arg_name, _optional_arg_condition(arg_name))}"
                )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _format_placement_anchor_guide(
    allowed_tool_specs: dict[str, dict[str, Any]],
) -> str:
    """Render a compact, typed vocabulary for placement destinations."""

    guide = {
        "place_in_receptacle": ("receptacle_id", "inside a movable container or receptacle"),
        "place_under": ("reference_fixture_id", "under a fixture or appliance"),
        "place_on_object": ("support_object_id", "on a movable supporting object"),
        "place_on_surface": ("support_id", "on a fixture surface"),
    }
    lines = [
        "Placement-anchor vocabulary (use the exact field shown; never use target_id as an alias):"
    ]
    for tool_name, (arg_name, meaning) in guide.items():
        if tool_name in allowed_tool_specs:
            lines.append(f"- {tool_name}: {arg_name} = {meaning}.")
    next_to = allowed_tool_specs.get("place_next_to")
    if next_to is not None:
        declared = set(next_to.get("tool_args", ())) | set(
            next_to.get("optional_tool_args", ())
        )
        for group in next_to.get("tool_arg_any_of", ()):
            declared.update(group)
        options = []
        if "reference_object_id" in declared:
            options.append("reference_object_id = next to a movable object")
        if "reference_fixture_id" in declared:
            options.append("reference_fixture_id = next to a fixture")
        if options:
            lines.append("- place_next_to: " + " OR ".join(options) + ".")
    return "\n".join(lines)


def _build_fsm_prompt_rules(
    allowed_tool_specs: dict[str, dict[str, Any]],
    *,
    task_preconditions: Sequence[dict[str, Any]] | None = None,
    task_effects: Sequence[dict[str, Any]] | None = None,
    extra_rules: Sequence[str] | None = None,
) -> list[str]:
    """Builds concise prompt rules that mirror the FSM validator."""

    # Keep the prompt rules derived from the same tool set the validator uses so
    # the model sees the key legality constraints up front.
    allowed_tool_names = set(allowed_tool_specs)
    prompt_rules: list[str] = []

    if "communicate" in allowed_tool_names:
        prompt_rules.append(
            "Before the first task action, both agents must communicate at least once."
        )
    if "navigate_to_fixture" in allowed_tool_names:
        prompt_rules.append(
            "Before interacting with a fixture, surface, receptacle, or dispenser, first navigate to that place."
        )
    if allowed_tool_names & (OPEN_PART_TOOL_NAMES | CLOSE_PART_TOOL_NAMES):
        prompt_rules.append(
            "Only open or close a fixture part after navigating to that same fixture."
        )
        prompt_rules.append(
            "An agent must have empty hands to open or close a hinged part, drawer, "
            "door, or articulated appliance part. If it is already holding an object, "
            "it must place that object at a valid temporary location first. When "
            "practical, open the required fixture before picking up the object."
        )
    if allowed_tool_names & ACQUIRE_TOOL_NAMES:
        prompt_rules.append(
            "Only use pick_up_object when the object is still at the listed source_id, and never pick up a second object while already holding one."
        )
        prompt_rules.append(
            "After picking up an object, that agent should only navigate or place that same object until it is no longer holding anything."
        )
    if allowed_tool_names & RELEASE_TOOL_NAMES:
        prompt_rules.append(
            "Only use a placement tool for the exact object the acting agent is currently holding."
        )
        prompt_rules.append(
            "Do not use a movable object that any agent is currently holding as a source, support, receptacle, or reference target. If a bowl, plate, tray, or other movable support needs to receive an item, it must be resting on a fixture surface first."
        )
    if allowed_tool_names & GIVE_SPACE_TOOL_NAMES:
        prompt_rules.append(
            "Use give_space when an agent genuinely needs to vacate a workspace so "
            "the partner can enter or work there. It is a handover action, not a way "
            "to become idle, report completion, or manufacture participation. "
            "Counters, islands, and dining tables are roomy: both agents may stand "
            "there simultaneously when they use different objects, so that ordinary "
            "case needs no give_space."
        )
        prompt_rules.append(
            "A cabinet and its parent counter are one shared workspace. Navigate "
            "to the parent counter for cabinet work. Different agents may use "
            "different objects there without a handover. Open or close the cabinet "
            "before, not during, another agent's access to its contents."
        )
        prompt_rules.append(
            "Do not use give_space while holding an object unless there is no legal way to finish the current placement first."
        )
        prompt_rules.append(
            "After an agent executes give_space, that agent is no longer at the fixture. "
            "Before that agent can interact there again (pick up, place, open, close, or use a control), "
            "it must navigate_to_fixture first. Similarly, the arriving agent must give_space in turn "
            "before the original agent can navigate back."
        )
    if allowed_tool_names & WAIT_TOOL_NAMES:
        # One protocol stated once, as a recipe. This replaced five overlapping
        # prose rules that stated the trigger twice and buried the definition of
        # a conflict last.
        prompt_rules.append(
            "SHARING PROTOCOL. The two agents act at the same time, so whenever they "
            "need the same thing they must hand it over explicitly. A conflict is "
            "exactly one of:\n"
            "  (a) both agents use the SAME object, anywhere;\n"
            "  (b) both agents work at the same TIGHT fixture (drawer, "
            "fridge, oven, stove, sink, or a countertop appliance), even on "
            "different objects, because only one robot fits there.\n"
            "Two agents at a roomy counter, island, or dining table using DIFFERENT "
            "objects is NOT a conflict and needs nothing.\n"
            "For every conflict, write these four steps in this order, using the "
            "exact symbolic id X of the thing (never an invented event name):\n"
            "  1. the second agent names X as the release keyword, for example "
            "`When done, release \"X\"`;\n"
            "  2. the second agent calls wait_for_signal(from=<first agent>, about=X);\n"
            "  3. the first agent finishes with X -- its last use of the object, or "
            "give_space at the fixture;\n"
            "  4. only THEN the first agent sends a message naming X saying it is "
            "done, in a full sentence of at least four words.\n"
            "All four are required. Saying 'I will wait' without step 2 is not "
            "waiting, and sending step 4 before step 3 is a promise, not a release."
        )
        prompt_rules.append(
            "The other agent cannot see wait_for_signal. A waiting agent wakes "
            "only when the named holder sends communicate "
            "with a releases entry exactly matching wait_for_signal.about. Ordinary "
            "messages and releases for other objects or fixtures do not wake it."
        )
        prompt_rules.append(
            "The sole release exception is a terminal wait: if the partner's "
            "action on that same tick satisfies the global FSM goal, the episode "
            "ends and no release is needed. The waiter must still ask on the "
            "preceding tick and name the real object or fixture involved. Say "
            "only that your own assigned work is finished; never claim that the "
            "global task is complete before the goal action executes."
        )
    if allowed_tool_names & INTERACTION_TOOL_NAMES:
        prompt_rules.append(
            "Interaction tools (press_button, press_lever, set_rotary_control) require the agent to be at the target fixture. Navigate to the fixture first."
        )
    for tool_name, tool_spec in allowed_tool_specs.items():
        for arg_group in _canonical_arg_groups_for_prompt(tool_name, tool_spec):
            formatted_group = ", ".join(str(arg_name) for arg_name in arg_group)
            if len(arg_group) == 1:
                prompt_rules.append(
                    f"When using {tool_name}, use {formatted_group} as the canonical anchor arg. Do not include alias alternatives."
                )
            else:
                prompt_rules.append(
                    f"When using {tool_name}, provide exactly one of {formatted_group} and do not include multiple alternatives in the same step."
                )
    if "place_next_to" in allowed_tool_names:
        prompt_rules.append(
            "For place_next_to, prefer reference_object_id for object-relative placements and reference_fixture_id for fixture-relative placements."
        )
        prompt_rules.append(
            "For place_next_to, reference_object_id must name an object from initial_state.objects and reference_fixture_id must name a fixture from initial_state.fixtures. Never put a fixture id in reference_object_id."
        )
    if "place_on_surface" in allowed_tool_names:
        prompt_rules.append(
            "For place_on_surface, use support_id as the fixture anchor and use target_site_id for the exact sub-location such as a rack or burner."
        )
    if "place_in_receptacle" in allowed_tool_names:
        prompt_rules.append(
            "For place_in_receptacle, use receptacle_id as the receptacle anchor and use target_site_id only for the exact interior site."
        )
    if "place_under" in allowed_tool_names:
        prompt_rules.append(
            "For place_under, prefer reference_fixture_id for the fixture and use target_site_id when the exact dispenser or basin site matters."
        )
    # Keep later references aligned with prior FSM effects.
    prompt_rules.append(
        "Keep object locations consistent across steps. After an object moves, later source_id and destination references must match its new location."
    )
    prompt_rules.append(
        "Treat repeated symbolic objects as distinct physical instances. Do not reuse one concrete object to stand in for two different symbolic ids."
    )
    prompt_rules.append(
        "When a partitioned fixture exposes support_sites such as burners, shelves, basins, bowls, slots, trays, or racks, use the concrete source_site_id or target_site_id when the task depends on that exact sub-location."
    )
    prompt_rules.append(
        "If a step specifies a concrete target_site_id or source_site_id, honor that exact site. Do not silently switch to a different site on the same fixture."
    )
    prompt_rules.append(
        "Each id-valued tool arg must be one concrete symbolic id. Do not concatenate fixture ids, site ids, hardware labels, or free text into one field."
    )
    prompt_rules.append(
        "Stop as soon as the goal state is satisfied. Do not add extra task actions afterward."
    )

    for condition in task_preconditions or ():
        condition_kind = condition.get("kind")
        if condition_kind == "fixture_part_state_required_for_pickup":
            prompt_rules.append(
                "Open "
                f"{condition['fixture_id']}.{condition['part_id']} before using "
                f"{condition['tool']} from {condition['source_id']}."
            )
        elif condition_kind == "fixture_part_state_required_for_action":
            arg_name = condition.get("arg_name")
            arg_value = condition.get("arg_value")
            action_scope = ""
            if isinstance(arg_name, str) and isinstance(arg_value, str):
                action_scope = f" when {arg_name}={arg_value}"
            prompt_rules.append(
                "Open "
                f"{condition['fixture_id']}.{condition['part_id']} before using "
                f"{condition['tool']}{action_scope}."
            )
        elif condition_kind == "object_location_required_for_action":
            arg_name = condition.get("arg_name")
            arg_value = condition.get("arg_value")
            action_scope = ""
            if isinstance(arg_name, str) and arg_value is not None:
                action_scope = f" when {arg_name}={arg_value}"
            prompt_rules.append(
                f"Only use {condition['tool']}{action_scope} after "
                f"{condition['object_id']} is already at {condition['required_location']}."
            )

    for effect in task_effects or ():
        if effect.get("kind") != "set_machine_flag_on_action":
            continue
        tool_name = effect.get("tool")
        if not isinstance(tool_name, str):
            continue
        required_object_locations = effect.get("required_object_locations") or ()
        for requirement in required_object_locations:
            if not isinstance(requirement, dict):
                continue
            object_id = requirement.get("object_id")
            location = requirement.get("location")
            if isinstance(object_id, str) and isinstance(location, str):
                prompt_rules.append(
                    f"Only use {tool_name} after {object_id} is already at {location}."
                )
        required_machine_values = effect.get("required_machine_values") or ()
        for requirement in required_machine_values:
            if not isinstance(requirement, dict):
                continue
            machine_path = requirement.get("machine_path")
            value = requirement.get("value")
            if (
                isinstance(machine_path, list)
                and machine_path
                and all(isinstance(part, str) for part in machine_path)
            ):
                machine_path_text = ".".join(machine_path)
                prompt_rules.append(
                    f"Only use {tool_name} after {machine_path_text} is already {value}."
                )

    for rule in extra_rules or ():
        normalized_rule = " ".join(rule.strip().split())
        if normalized_rule:
            prompt_rules.append(normalized_rule)

    deduped_rules: list[str] = []
    seen_rules: set[str] = set()
    for rule in prompt_rules:
        normalized_rule = " ".join(rule.strip().split())
        if not normalized_rule or normalized_rule in seen_rules:
            continue
        seen_rules.add(normalized_rule)
        deduped_rules.append(normalized_rule)
    return deduped_rules


def make_task_prompt_builder(
    *,
    composite_task: str,
    task_goal: str,
    initial_state: dict[str, Any],
    allowed_tool_specs: dict[str, Any],
    non_communicate_tool_names: Sequence[str],
    task_preconditions: Sequence[dict[str, Any]] | None = None,
    task_effects: Sequence[dict[str, Any]] | None = None,
    extra_execution_rules: Sequence[str] | None = None,
    agent_ids: Sequence[str] = ("agent_0", "agent_1"),
) -> TaskPromptBuilder:
    """Builds a reusable task prompt function from the shared prompt template.

    Args:
        composite_task: Task name shown to the model and stored in generated data.
        task_goal: One-sentence goal description for the task.
        initial_state: Initial state presented to the model.
        allowed_tool_specs: Task-specific allowed tools and id constraints.
        non_communicate_tool_names: Non-communication task tools allowed in steps.
        extra_execution_rules: Optional task-specific sequencing rules enforced by validation.
        agent_ids: Ordered agent IDs the task expects the model to simulate.

    Returns:
        A callable that renders the task prompt for a specific variation key.
    """

    agent_count = len(agent_ids)
    _ = non_communicate_tool_names

    def build_prompt(
        variation_key: str,
        task_instance: TaskInstance | None = None,
        retry_feedback: str | None = None,
    ) -> str:
        """Renders the shared task-level prompt with task-specific content."""

        # Copy the initial state per run so prompt rendering can reflect sampled
        # task instances without mutating the task definition defaults.
        prompt_initial_state = (
            deepcopy(task_instance.initial_state)
            if task_instance is not None
            else deepcopy(initial_state)
        )
        prompt_allowed_tool_specs = deepcopy(allowed_tool_specs)
        prompt_task_goal = (
            task_instance.task_goal
            if task_instance is not None and isinstance(task_instance.task_goal, str)
            else task_goal
        )
        prompt_extra_execution_rules = (
            task_instance.extra_execution_rules
            if task_instance is not None and task_instance.extra_execution_rules
            else tuple(extra_execution_rules or ())
        )
        coordinator_id = (
            task_instance.coordinator_id
            if task_instance is not None and task_instance.coordinator_id
            else agent_ids[0]
        )
        initial_state_text = json.dumps(
            prompt_initial_state,
            indent=2,
            sort_keys=True,
        )
        initial_position_text = _format_initial_agent_positions(
            prompt_initial_state,
            agent_ids,
        )
        # The compact contract above replaces the legacy tick prose. Keep
        # only genuine task facts below it.
        from robotalk.generation.tick_format import TICK_FORMAT_RULES

        simplified_extra_rules = tuple(
            rule
            for rule in prompt_extra_execution_rules
            if rule not in TICK_FORMAT_RULES
            and not rule.startswith(
                "This division requires the agents to hand work"
            )
        )
        normalized_simplified_extra = {
            " ".join(rule.strip().split()) for rule in simplified_extra_rules
        }
        derived_task_facts = _build_fsm_prompt_rules(
            {},
            task_preconditions=task_preconditions,
            task_effects=task_effects,
            extra_rules=simplified_extra_rules,
        )
        task_specific_rules = [
            rule for rule in derived_task_facts
            if rule in normalized_simplified_extra
            or rule.startswith("Open ")
            or rule.startswith("Only use ")
        ]
        task_specific_text = "\n".join(
            f"- {rule}" for rule in task_specific_rules
        ) or "- None beyond the task state and tool contracts below."
        location_rule = (
            "Track each agent's current fixture. navigate_to_fixture(X) sets "
            "its location to X; give_space(X) removes it from X. Every pickup, "
            "placement, fixture-part, and control action requires the matching "
            "current fixture. Being near or able to see fixture X does not count as "
            "being at X. Before using an object, container, or appliance at X, call "
            "navigate_to_fixture(X) unless the agent's current symbolic location is "
            "already X. After give_space, navigate again before acting there. "
            "An agent can hold only one object at a time. There is no tool for "
            "passing an object between agents: the agent that picks up an object "
            "must place that same object itself before picking up another. "
            "A cabinet and its parent counter are one shared workspace: navigate "
            "to the parent counter and use cabinet tools from there. An agent at "
            "an exclusive child appliance may use its parent counter and objects "
            "on that counter without navigating again; it remains at the child. "
            "Opening or closing a fixture part requires empty hands: if holding an "
            "object, place it at a valid temporary location first. When practical, "
            "open the required fixture before picking up the object."
        )
        tail_rule = (
            "Communicate only a plan, request, release, correction, or one "
            "portion_complete message; never claim global completion. If an agent "
            "finishes its assigned physical calls while its partner still has calls "
            "remaining, it stays active: on its next call send exactly one "
            "communicate with coordination_phase=\"portion_complete\" saying only "
            "its own portion is done, using the finished-portion message from rule "
            "6; on the immediately following tick make the matching wait call. If "
            "the opening plan assigns an agent no physical actions, its first call "
            "after the handshake is that finished-portion message and its following "
            "call is the matching wait. If "
            "the partner's action on the same "
            "tick as portion_complete satisfies the FSM goal, end the episode on "
            "that tick: do not add a wait or any later tick. It then remains "
            "blocked until a "
            "later matching release makes it active on the following tick. If the "
            "partner reaches the FSM goal in the wait tick, end the episode without "
            "a release."
        )
        invocation_rule = (
            "Every tick must contain both agent fields. Give an unblocked "
            "agent exactly one real tool call. BLOCKED is not an action and "
            "cannot start a wait. Use {\"state\":\"blocked\"} only after that "
            "same agent has called wait_for_signal and before its matching "
            "release arrives. Write the blocked marker on "
            "each later tick while the wait is unresolved; never repeat the "
            "wait call. Keep the blocked marker on the release tick, then give "
            "the awakened agent a real call on the following tick. Set the "
            "top-level format to \"explicit_blocked_v1\"."
        )
        v3_tail_addendum = (
            " If the first required message after physical work is a resource "
            "release, combine it with portion_complete in that one communicate "
            "call by including both coordination_phase=\"portion_complete\" and "
            "releases=\"<resource id>\"."
        )
        blocked_example = (
            "\n\nExclusive-fixture handover example:\n"
            "- tick N: agent_0 communicates `When done, release \"fridge\"`; agent_1 finishes useful fridge work.\n"
            "- tick N+1: agent_0 calls wait_for_signal(from=agent_1, about=fridge); agent_1 calls give_space(fridge).\n"
            "- tick N+2: agent_0 is {\"state\":\"blocked\"}; agent_1 communicates to agent_0 with releases=fridge and coordination_phase=portion_complete if this was agent_1's final physical work.\n"
            "- tick N+3: agent_0 calls navigate_to_fixture(fridge); agent_1 calls wait_for_signal about one real object or fixture in agent_0's remaining work if agent_1 is finished.\n"
            "After give_space(X), that agent is no longer at X and must navigate before acting there again."
        )
        assignment_rule = (
            "Choose an efficient division of work from the initial state. "
            "Prefer giving both agents a coherent physical responsibility when the "
            "task has multiple objects or separable actions. Keep each object's "
            "pickup and placement with one owner. A single agent may do all physical "
            "work only when the task is one inherently sequential chain or splitting "
            "it would create work that does not help complete the goal. State the "
            "chosen exact IDs in the opening proposal."
        )
        prompt = f"""
You are simulating {agent_count} cooperative robot agents in a physical kitchen.
Generate one valid trajectory for {composite_task} as a list of simultaneous ticks.

Eight rules:
1. {invocation_rule}
2. The sampled coordinator is {coordinator_id}. Tick 0: {coordinator_id} proposes one concrete division of work with coordination_phase="propose" while the partner uses "await_plan". Tick 1: {coordinator_id} uses "await_confirmation" while the partner uses "confirm". Start physical work on tick 2.
3. {assignment_rule}
4. {location_rule}
5. A tick is atomic. The agents may work together at a roomy counter or cabinet workspace on different objects, but may not use the same object or the same exclusive drawer, appliance, sink, or stove workspace in one tick. A cabinet door may not open or close during another agent's access to that cabinet's contents. give_space(X) frees X only after its tick finishes: the partner must not enter or use X during the give_space tick. An unblocked partner may enter or use X on the following tick without waiting for a release.
6. wait_for_signal has only two uses. To hand over an exclusive fixture X, the incoming agent communicates `When done, release \"X\"` and on its very next call waits with about=X. The current user finishes at X, calls give_space(X), and on a later tick communicates with releases=X. Use the fixture ID X, not the ID of an object at X. Waiting does not move an agent, and releasing an object at X does not free X. A finished agent follows the same visible-request rule: communicate `My portion is done. Release \"X\" if you need me again` with coordination_phase=\"portion_complete\", then on its very next call wait with about=X. The other agent cannot see the wait call. In every case, the communicate call immediately before wait must name the exact same X and say to release it; never omit the preceding message or put another action between it and wait. A waiter stays blocked through the matching release tick and resumes on the following tick. Never wait and release in the same tick; an ordinary message does not wake it.
7. {tail_rule}{v3_tail_addendum}
8. Use only the tools and exact symbolic IDs below. Required args must appear. Optional args appear only under the condition stated by that tool; otherwise omit them. Every real tool call needs one short first-person reasoning sentence. Output JSON only.{blocked_example}

Task-specific facts and work-division guidance:
{task_specific_text}

Composite task:
- {composite_task}
- Goal: {prompt_task_goal}

Initial agent positions:
{initial_position_text}

Initial task state:
{initial_state_text}

{format_concurrency_facts(prompt_initial_state)}

{_format_placement_anchor_guide(prompt_allowed_tool_specs)}

Tool contracts:
{_format_tool_contracts(prompt_allowed_tool_specs)}

Variation key: {variation_key}
""".strip()
        if retry_feedback is None:
            return prompt
        return f"{prompt}\n\n{retry_feedback.strip()}"

    return build_prompt
