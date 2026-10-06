#!/usr/bin/env python
"""Tick-based trajectory format: the plan written the way it executes.

The flat step list is a *serialization* of two independent per-agent streams.
Reading it top to bottom gives one interleaving, but that is an artifact of how
the JSON was written, not a constraint on anything -- the agents advance at
their own rates, so position in the list says nothing about when a step runs.
Every coordination defect measured on this dataset traces to that one gap:

    2 of 2   lock-step collisions   (plan alternates correctly, ticks collide)
  332 of 357 inert waits            (release is later in the file, earlier in time)

In tick form there is no gap to fall into. Each row is one instant; both agents
act on it simultaneously; a blocked agent is visibly blocked. The model writes
in the frame the simulator executes in, so it can see that it is about to be
held up -- which is what it needs in order to say "tell me when the machine is
free" rather than "heading over now" and then wait.

    rows  -> steps : `to_steps`, for the FSM and everything downstream
    steps -> rows  : `to_rows`, for prompting and for inspection
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Sequence

BLOCKED = "(blocked)"
IDLE = "(idle)"


def to_steps(rows: Sequence[dict[str, Any]], agent_ids: Sequence[str]) -> list[dict[str, Any]]:
    """Flatten tick rows into the numbered step list the FSM validates.

    Within one tick the agents are simultaneous, so their order in the flat
    list is arbitrary; `agent_ids` order is used for determinism. Anything the
    model leaves out, or marks blocked or idle, contributes no step.
    """

    steps: list[dict[str, Any]] = []
    for row in rows:
        for agent_id in agent_ids:
            action = row.get(agent_id)
            if not isinstance(action, dict) or not action.get("tool"):
                continue
            step = deepcopy(action)
            step["agent"] = agent_id
            step["step"] = len(steps)
            steps.append(step)
    return steps


def to_rows(
    steps: Sequence[dict[str, Any]],
    agent_ids: Sequence[str],
    ticks: Sequence[int | None] | None = None,
) -> list[dict[str, Any]]:
    """Group steps into tick rows, scheduling them if ticks are not supplied."""

    if ticks is None:
        from robotalk.generation.insert_waits import schedule

        ticks, _ = schedule(list(steps))

    rows: dict[int, dict[str, Any]] = {}
    for step, tick in zip(steps, ticks):
        if tick is None:
            continue
        row = rows.setdefault(tick, {"tick": tick})
        entry = {k: v for k, v in step.items() if k not in {"agent", "step"}}
        row[step["agent"]] = entry
    return [rows[tick] for tick in sorted(rows)]


def render(rows: Sequence[dict[str, Any]], agent_ids: Sequence[str], width: int = 46) -> str:
    """A readable two-column view, used in the prompt and for inspection."""

    lines = [f"{'tick':>4}  " + "  ".join(a.ljust(width) for a in agent_ids)]
    lines.append("-" * (6 + (width + 2) * len(agent_ids)))
    for row in rows:
        cells = []
        for agent_id in agent_ids:
            action = row.get(agent_id)
            if not isinstance(action, dict):
                cells.append("".ljust(width))
                continue
            args = action.get("args") or {}
            detail = ", ".join(f"{k}={v}" for k, v in args.items() if not isinstance(v, (dict, list)))
            cells.append(f"{action.get('tool')}({detail})"[:width].ljust(width))
        lines.append(f"{row.get('tick', 0):>4}  " + "  ".join(cells))
    return "\n".join(lines)


def build_tick_response_schema(
    *,
    agent_ids: Sequence[str],
    allowed_tool_specs: dict[str, Any],
    explicit_blocked_markers: bool = False,
) -> dict[str, Any]:
    """A response schema whose top level is ticks rather than steps."""

    from robotalk.tasks.shared.schema import (
        build_task_response_schema,
    )

    # Reuse the per-step schema the flat format already validates, then re-key
    # it by agent under each tick so one row is one instant.
    flat = build_task_response_schema(
        agent_ids=agent_ids,
        allowed_tool_specs=allowed_tool_specs,
    )
    step_schema = _find_step_schema(flat)

    def without_flat_fields(branch: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                key: deepcopy(value)
                for key, value in (branch.get("properties") or {}).items()
                if key not in {"agent", "step"}
            },
            "required": [
                name
                for name in (branch.get("required") or [])
                if name not in {"agent", "step"}
            ],
        }

    step_variants = step_schema.get("anyOf")
    action_schema = (
        {"anyOf": [without_flat_fields(branch) for branch in step_variants]}
        if isinstance(step_variants, list)
        else without_flat_fields(step_schema)
    )
    if explicit_blocked_markers:
        blocked_schema = {
            "type": "object",
            "properties": {
                "state": {"type": "string", "enum": ["blocked"]},
            },
            "required": ["state"],
        }
        if isinstance(action_schema.get("anyOf"), list):
            action_schema = {
                "anyOf": [*action_schema["anyOf"], blocked_schema]
            }
        else:
            action_schema = {"anyOf": [action_schema, blocked_schema]}

    def schema_for_agent(agent_id: str) -> dict[str, Any]:
        """Constrain recipient-like arguments that depend on the row owner."""

        specialized = deepcopy(action_schema)
        other_agents = [value for value in agent_ids if value != agent_id]

        def specialize_branch(branch: dict[str, Any]) -> None:
            properties = branch.get("properties") or {}
            tool_values = (properties.get("tool") or {}).get("enum") or []
            if not tool_values:
                return
            tool = tool_values[0]
            arg_roots = [properties.get("args") or {}]
            while arg_roots:
                root = arg_roots.pop()
                if isinstance(root.get("anyOf"), list):
                    arg_roots.extend(root["anyOf"])
                    continue
                arg_properties = root.get("properties") or {}
                if tool == "communicate" and "to" in arg_properties:
                    arg_properties["to"]["enum"] = other_agents
                if tool == "wait_for_signal" and "from" in arg_properties:
                    arg_properties["from"]["enum"] = other_agents

        for branch in specialized.get("anyOf", [specialized]):
            specialize_branch(branch)
        return specialized

    row_properties: dict[str, Any] = {
        "tick": {"type": "integer", "description": "0-based instant; both agents act on it."},
    }
    for agent_id in agent_ids:
        row_properties[agent_id] = {
            **schema_for_agent(agent_id),
            "description": (
                f"What {agent_id} does at this tick. This field is required "
                f"whenever {agent_id} is not blocked by an earlier "
                f"wait_for_signal; omit it only while that wait remains blocked."
            ),
        }

    schema = {
        "type": "object",
        "properties": {
            "ticks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": row_properties,
                    "required": (
                        ["tick", *agent_ids]
                        if explicit_blocked_markers
                        else ["tick"]
                    ),
                },
            }
        },
        "required": ["ticks"],
    }
    if explicit_blocked_markers:
        schema["properties"]["format"] = {
            "type": "string",
            "enum": ["explicit_blocked_v1"],
        }
        schema["required"].append("format")
    return schema


def _find_step_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Locate the per-step object schema inside the flat response schema."""

    properties = schema.get("properties") or {}
    steps = properties.get("steps") or {}
    items = steps.get("items")
    if isinstance(items, dict):
        return items
    raise ValueError("flat response schema has no steps.items to re-key by tick")


TICK_FORMAT_RULES: tuple[str, ...] = (
    "Write the trajectory as a list of ticks, not as a list of steps. One tick "
    "is one instant of time.",
    "Both agents act SIMULTANEOUSLY on each tick. Whatever you put under "
    "agent_0 and under agent_1 in the same tick happens at the same moment.",
    "Because of that, the two agents must never act on the same object, or at "
    "the same exclusive fixture, on the same tick. Two robots cannot reach "
    "into one cabinet at once.",
    "Every unblocked agent must appear exactly once in every tick, including "
    "after its assigned physical work is finished. The live scheduler invokes "
    "every unblocked agent and has no implicit idle or finished action. Omit an "
    "agent only while it is blocked by an earlier wait_for_signal; otherwise "
    "give it a real, useful, non-conflicting tool call.",
    "wait_for_signal(from, about) blocks the calling agent until the other "
    "agent sends a communicate whose releases names the same id. Nothing else "
    "wakes it, so put the wait where the agent genuinely cannot proceed.",
    "`about` and `releases` name a THING, never an event. Both must be an "
    "object id or a fixture id that appears in initial_state, copied exactly: "
    "about=\"coffee_machine\", releases=[\"coffee_machine\"]. Names like "
    "\"mug_placed\", \"counter_free\", \"your_turn\" or \"task_done\" are not "
    "ids -- they describe a moment rather than a thing, nothing can ever "
    "release them, and a single one of them throws the whole trajectory away. "
    "Ask yourself: is this the name of something in the kitchen? If not, it is "
    "wrong. You are waiting FOR an object or a fixture, not ON an event.",
    "A release is still a MESSAGE. communicate always needs a real, non-empty "
    "`message` saying what is happening, and `releases` is an EXTRA field "
    "alongside it -- never a replacement for it. A communicate with releases "
    "and no message is rejected.",
    "A WAIT IS HALF A PROTOCOL, AND YOU MUST WRITE BOTH HALVES unless the "
    "partner satisfies the global FSM goal on the very tick the wait begins. Every other "
    "wait_for_signal(from=X, about=R) you write obliges agent X to send "
    "communicate(to=<waiter>, message=\"...\", releases=[\"R\"]) on a LATER "
    "tick, naming R "
    "exactly. A plain message does not count -- only the releases field wakes "
    "the waiter. If you write the wait and forget X's release, that agent "
    "never moves again and every remaining tick of the plan is dead. Before "
    "you finish, go through your waits one by one and find the matching "
    "release for each.",
    "TERMINAL WAIT EXCEPTION. If one agent has finished its assigned work and "
    "the partner's action on the same tick as wait_for_signal satisfies the "
    "global FSM goal, the episode ends atomically and no release is needed. "
    "The waiter must still ask on the preceding tick and wait on the real "
    "object or fixture used by the partner. Say only that YOUR assigned work "
    "is finished; never announce that the GLOBAL task is complete before the "
    "FSM goal action executes.",
    "The other agent cannot see wait_for_signal. Immediately before waiting, "
    "communicate the exact release keyword, for example: `When done, release "
    "\"<X>\"`. On your very next call, use wait_for_signal(..., "
    "about=\"<X>\")`. Do nothing between that message and the wait.",
    "The waiter must begin waiting no later than the tick when the holder lets "
    "go, and the matching release must come on a later tick. A release only "
    "wakes an agent that is already waiting. If the release comes before the "
    "wait, it is lost; if it comes in the wait tick, it takes effect too early.",
    "Release only what you have genuinely let go of. Before X sends "
    "releases=[\"R\"]: if R is a fixture, X must already have called "
    "give_space on it or navigated somewhere else; if R is an object, X must "
    "have put it down. The release may be reported later, but X must not "
    "reoccupy or reuse R before reporting it. Saying \"you can take it\" while "
    "still standing at the fixture is not a handover at all.",
    "Order what an agent says around what it is about to do. If you are about "
    "to be held up, ask BEFORE announcing that you are on your way: say "
    "\"tell me when the machine is free\", wait, and only then say \"heading "
    "over\". Announcing a move you cannot make yet is wrong.",
    "A handover has FOUR parts. agent_1 wants the cabinet that agent_0 is "
    "using:\n"
    "    tick 3   agent_1: communicate(to=agent_0, \"When done, release "
    "\\\"cab\\\".\")     <- 1. ASK\n"
    "    tick 4   agent_1: wait_for_signal(from=agent_0, about=\"cab\")"
    "                    <- 2. BLOCK, the very next tick\n"
    "    ticks 5-8  agent_0 carries on with its own work; agent_1 does not "
    "appear in any\n"
    "               of these rows, because it is blocked. This can be as long "
    "as it needs\n"
    "               to be -- what matters is the order, not the gap.\n"
    "    tick 9   agent_0: give_space(fixture_id=\"cab\")"
    "                               <- 3a. LEAVE, only now\n"
    "    tick 10  agent_0: communicate(to=agent_1, message=\"the cabinet is "
    "yours now\", releases=[\"cab\"])   <- 3b. REPORT on a later tick\n"
    "    tick 11  agent_1: navigate_to_fixture(fixture_id=\"cab\")"
    "                     <- 4. MOVE IN, the tick AFTER the release,\n"
    "                                                                        "
    "                          never on the same tick as it\n"
    "ASK then BLOCK are adjacent. LEAVE must happen before REPORT, but the "
    "report may be later if the holder does not reuse the resource. The waiter "
    "must already be blocked before the REPORT arrives.",
    "A released agent starts moving on the tick AFTER the release, not on the "
    "same one. The message has to arrive before it can be acted on, so put "
    "the waiter's next call one tick later than the communicate that freed "
    "it. The tick the release is sent belongs to the holder alone.",
    "Being idle is a cost. If an agent has nothing to do for many ticks, the "
    "work is badly divided -- give it something, or shorten the wait.",
)
