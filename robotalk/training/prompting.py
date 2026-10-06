"""Prompt construction helpers for task-level VLM fine-tuning."""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Iterable

from robotalk.tasks.shared.prompting import format_concurrency_facts
from robotalk.training.communication_profiles import get_communication_profile
from robotalk.training.schema_utils import compact_json_dumps


PROMPT_CONTRACT_VERSION = (
    "state_grounded_global_tools_no_index_v9_unambiguous_garnish_cake_goal"
)

PHYSICAL_HANDOFF_RULES = (
    "Physical workspace handoffs:\n"
    "- give_space(X) frees X only after that tool call finishes. The partner "
    "must not enter or use X in the same concurrent tick.\n"
    "- An unblocked partner may enter or use X on the following tick without "
    "calling wait_for_signal and without receiving a release.\n"
    "- Use wait_for_signal only when the agent must actually block before X "
    "becomes free. Once waiting, it remains blocked throughout a later matching "
    "release tick and resumes only on the following tick. Never wait and receive "
    "the matching release in the same tick.\n"
)

WORKSPACE_RULES = (
    "Workspace relationships:\n"
    "- A cabinet and its parent counter are one shared workspace. Navigate to "
    "the parent counter for cabinet work. Cabinet tools may be used from that "
    "counter, and different agents may use different objects there without a "
    "handover. Do not open or close the cabinet during another agent's access "
    "to its contents.\n"
    "- If an agent is at an exclusive child appliance on a parent counter, it "
    "may use that counter and objects on it without navigating again. This does "
    "not move the agent: its location remains the exclusive child appliance.\n"
)

PHYSICAL_GIVE_SPACE_RULE = (
    "Physical workspace handoffs:\n"
    "- give_space(X) frees X only after that tool call finishes. The other "
    "agent may enter or use X starting on the following tick, not during the "
    "same tick.\n"
)

MINIMAL_COMMUNICATION_RULES = (
    "Communication guidance:\n"
    "- Communicate with the other agent to complete the task together.\n"
    "- When you need to wait for X, first communicate a request naming the "
    "exact symbolic ID X. On your next call, use wait_for_signal with about=X. "
    "The wait call is private. A later message with releases=X wakes the "
    "waiting agent.\n"
)

INTERMEDIATE_COMMUNICATION_RULES = (
    "Communication guidance:\n"
    "- Communicate with the other agent to divide work, share relevant state, "
    "and coordinate access to workspaces.\n"
    "- When you need to wait for X, first communicate a request naming the "
    "exact symbolic ID X. On your next call, use wait_for_signal with about=X. "
    "The wait call is private. A later message with releases=X wakes the "
    "waiting agent.\n"
    "- Never claim that the global task is complete; the FSM alone ends the "
    "episode. If your own portion finishes first, communicate only that your "
    "part is done and name one exact release keyword X, then call "
    "wait_for_signal with about=X on your next turn. Stay blocked until the "
    "partner sends a later matching release.\n"
)

ACTIVE_OBSERVATION_RULES = (
    "Camera observations:\n"
    "- get_image.views must be a non-empty list containing only these exact "
    "names: top_view, room_view, map, wrist, agentview_center, agentview_left, "
    "agentview_right.\n"
    "- Copy these names exactly. Do not invent view names from a direction, "
    "task, scene, object, or fixture. For example, use "
    '`get_image(views=["agentview_center", "wrist"])`. Names such as front, '
    "back, overhead, left, right, counter, and fridge_interior are invalid.\n"
)

# The agent is told who it is, so it must not choose an acting agent; it only
# decides whether to look first and what tool to call.
SYSTEM_PROMPT = (
    "You are a robot task planner. Predict exactly one next tool call for the "
    "current acting agent. Request the camera views you need before an action "
    "by calling get_image. Use images only as scene context. Do not output "
    "image paths or describe the images."
)


def format_history_steps(history_steps: Iterable[dict[str, Any]]) -> str:
    """Renders prior symbolic action history compactly.

    A step may carry an "error" key (live-sim rejected attempts); it renders as
    a trailing `FAILED: <reason>` so the model can see why nothing changed.
    Training data never sets it, so v1 prompts are unaffected.
    """

    rendered_steps = []
    for step in history_steps:
        # No step index is rendered: a joint demonstration index would leak how
        # much hidden activity the other agent performed between this agent's
        # turns.
        line = (
            f"- agent={step['agent']} tool={step['tool']} "
            f"args={compact_json_dumps(step['args'])}"
        )
        error_text = step.get("error")
        if error_text:
            line = f"{line} FAILED: {error_text}"
        rendered_steps.append(line)
    if not rendered_steps:
        return "- none"
    return "\n".join(rendered_steps)


def strip_history_step_indices(
    history_steps: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return a copy of private history with every step index removed.

    Both offline SFT construction and closed-loop evaluation must call this
    helper, so the context seen at evaluation matches training exactly.
    """

    normalized = [deepcopy(step) for step in history_steps]
    for step in normalized:
        step["step"] = None
    return normalized


def build_partial_user_prompt(
    *,
    composite_task: str,
    task_instruction: str,
    agent_id: str,
    history_steps: list[dict[str, Any]],
    observation_views: list[str],
    allowed_tool_specs: dict[str, dict[str, Any]],
    coordinator_id: str | None = None,
    initial_state: dict[str, Any] | None = None,
    communication_mode: str = "full",
) -> str:
    """Build the canonical private-history prompt used by SFT and live eval.

    The acting agent is fixed, history carries no step indices, and the tool
    schemas are supplied separately as function definitions.
    """

    history_steps = strip_history_step_indices(history_steps)

    # "none" (not "unknown") when no images are attached: under partial-v3
    # consume-once semantics that is the normal state after an observation has
    # been spent, and "unknown" would imply views exist but are unidentified.
    observations_text = ", ".join(observation_views) if observation_views else "none"
    agent_rule = (
        "- The acting agent is fixed by the prompt; do not choose actions for the other agent.\n"
    )
    communication_profile = get_communication_profile(communication_mode)
    initial_communication_rule = (
        "- Before ANY non-communicate action, BOTH agents must each have sent "
        "at least one communicate message; task actions are rejected until then.\n"
        if communication_profile.require_initial_communication
        else ""
    )
    output_rules = (
        "The tool schemas are provided separately as function definitions.\n\n"
        "Rules:\n"
        "- Predict exactly one next tool call.\n"
        "- Use symbolic IDs only, never concrete simulator IDs.\n"
        "- Symbolic IDs are dictionary keys in the initial state; copy the "
        "relevant object, fixture, part, or control key verbatim. Use a site "
        "ID only when the state explicitly provides a named site.\n"
        f"{agent_rule}"
        f"{initial_communication_rule}"
        "- The tool must be one of the provided function schemas.\n"
        "- Supply exactly the arguments required by the selected tool.\n"
        "- Prefer the most immediate executable next action.\n"
        "- Do not emit markdown or narrative after the tool call."
    )

    acting_agent_line = f"Current acting agent: {agent_id}\n"
    participation_line = {
        "full": "",
        "intermediate": (
            "You are one of two agents working together to complete the task.\n"
        ),
        "minimal": (
            "You are one of two agents working together to complete the task.\n"
        ),
        "unguided": (
            "You are one of two agents jointly completing the task. "
            "Communication with the other agent is possible.\n"
        ),
        "none": (
            "You are one of two agents jointly completing the task. The other "
            "agent acts independently, but you cannot exchange messages. Choose "
            "actions that contribute to the shared task goal.\n"
        ),
    }[communication_mode]
    coordinator_line = (
        f"Coordinator for this episode: {coordinator_id}\n"
        if coordinator_id and communication_profile.require_opening_protocol
        else ""
    )
    coordinator_rules = (
        "\nCoordinator handshake:\n"
        "- Opening coordination uses two communication ticks; optional ticks "
        "containing only get_image do not count.\n"
        "- Communication tick 1: the coordinator sends one concrete division "
        "using exact symbolic IDs and coordination_phase=propose; the partner "
        "sends coordination_phase=await_plan.\n"
        "- Communication tick 2: the coordinator sends "
        "coordination_phase=await_confirmation; the partner sends "
        "coordination_phase=confirm.\n"
        "- Physical work may begin only on the following tick.\n"
        "- Never claim that the global task is complete; the FSM alone ends the "
        "episode. If your own portion finishes first, send exactly one "
        "coordination_phase=portion_complete message saying only that your part "
        "is done and naming one exact release keyword X, for example `My portion "
        "is done. Release \"X\" if you need me again`. On your very next call, "
        "call wait_for_signal with about=X. The private wait call is not visible "
        "to the partner, so the message must name the same X and say to release "
        "it. Stay blocked until the "
        "partner sends a later matching release.\n"
        if coordinator_id and communication_profile.require_opening_protocol
        else ""
    )
    communication_rules = {
        "full": PHYSICAL_HANDOFF_RULES,
        "intermediate": INTERMEDIATE_COMMUNICATION_RULES + PHYSICAL_HANDOFF_RULES,
        "minimal": MINIMAL_COMMUNICATION_RULES + PHYSICAL_GIVE_SPACE_RULE,
        "unguided": PHYSICAL_GIVE_SPACE_RULE,
        "none": PHYSICAL_GIVE_SPACE_RULE,
    }[communication_mode]
    initial_state_block = (
        "Symbolic initial state:\n"
        f"{json.dumps(initial_state, indent=2, sort_keys=True)}\n\n"
        f"{format_concurrency_facts(initial_state)}\n\n"
        if initial_state is not None
        else ""
    )
    active_observation_rules = (
        ACTIVE_OBSERVATION_RULES if "get_image" in allowed_tool_specs else ""
    )
    return (
        f"Task family: {composite_task}\n"
        f"Task instruction: {task_instruction}\n"
        f"{acting_agent_line}"
        f"{participation_line}"
        f"{coordinator_line}"
        f"Observation views attached in order: {observations_text}\n\n"
        f"{initial_state_block}"
        "Previous executed symbolic action history:\n"
        f"{format_history_steps(history_steps)}\n\n"
        f"{coordinator_rules}"
        f"{active_observation_rules}"
        f"\n{communication_rules}"
        f"{WORKSPACE_RULES}"
        f"{output_rules}"
    )


def build_system_message() -> dict[str, Any]:
    """Builds the fixed system instruction for one chat conversation."""

    return {
        "role": "system",
        "content": [{"type": "text", "text": SYSTEM_PROMPT}],
    }


def build_user_message(*, user_prompt: str, num_images: int) -> dict[str, Any]:
    """Builds one multimodal user turn with placeholder image slots."""

    user_content = [{"type": "image"} for _ in range(num_images)]
    user_content.append({"type": "text", "text": user_prompt})
    return {
        "role": "user",
        "content": user_content,
    }


def build_assistant_text_message(
    *, target_text: str, reasoning_text: str | None = None
) -> dict[str, Any]:
    """Builds one assistant text message (an empty one marks a generation turn)."""

    content = f"<think>{reasoning_text}</think>{target_text}" if reasoning_text else target_text
    return {
        "role": "assistant",
        "content": content,
    }


def build_messages(
    *,
    user_prompt: str,
    num_images: int,
    target_tool_call: dict[str, Any] | None = None,
    target_text: str | None = None,
    reasoning_text: str | None = None,
) -> list[dict[str, Any]]:
    """Builds one chat conversation for training or generation.

    With reasoning_text, the assistant turn is supervised with a
    `<think>{reasoning_text}</think>` prefix before the target.
    """

    if target_tool_call is not None and target_text is not None:
        raise ValueError("Provide either target_tool_call or target_text, not both.")

    messages: list[dict[str, Any]] = [
        build_system_message(),
        build_user_message(user_prompt=user_prompt, num_images=num_images),
    ]
    if target_text is not None:
        messages.append(
            build_assistant_text_message(
                target_text=target_text, reasoning_text=reasoning_text
            )
        )
    elif target_tool_call is not None:
        from robotalk.training.tool_calling import build_assistant_tool_call_message

        messages.append(
            build_assistant_tool_call_message(
                tool_name=target_tool_call["name"],
                arguments=target_tool_call["arguments"],
                reasoning_text=reasoning_text,
            )
        )
    return messages
