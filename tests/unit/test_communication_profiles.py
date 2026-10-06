from copy import deepcopy
import json

import pytest

from robotalk.tools.subatomic_tool_specs import build_model_tool_specs
from robotalk.training.communication_profiles import apply_communication_profile
from robotalk.training.prompting import build_partial_user_prompt


def _prompt(mode: str) -> str:
    return build_partial_user_prompt(
        composite_task="PrepareCoffee",
        task_instruction="Prepare coffee.",
        agent_id="agent_0",
        history_steps=[],
        observation_views=[],
        allowed_tool_specs=apply_communication_profile(
            build_model_tool_specs(include_get_image=True), mode
        ),
        coordinator_id="agent_0",
        initial_state={
            "agents": {
                "agent_0": {"location": "counter"},
                "agent_1": {"location": "counter"},
            },
            "objects": {"mug": {"location": "counter"}},
            "fixtures": {"counter": {"shared": True}},
        },
        communication_mode=mode,
    )


def test_intermediate_retains_guidance_without_protocol():
    prompt = _prompt("intermediate")
    assert "divide work, share relevant state" in prompt
    assert "Never claim that the global task is complete" in prompt
    assert "give_space(X) frees X only after" in prompt
    assert "Use wait_for_signal only when the agent must actually block" in prompt
    assert "Coordinator for this episode" not in prompt
    assert "Coordinator handshake" not in prompt
    assert "coordination_phase" not in prompt
    assert "Before ANY non-communicate action" not in prompt
    assert "Communication guidance:" in prompt


def test_full_adds_protocol_to_intermediate_guidance():
    prompt = _prompt("full")
    assert "Never claim that the global task is complete" in prompt
    assert "Coordinator for this episode: agent_0" in prompt
    assert "Coordinator handshake" in prompt
    assert "coordination_phase=propose" in prompt
    assert "Before ANY non-communicate action" in prompt


def test_intermediate_keeps_detailed_tool_semantics_but_drops_phase():
    specs = apply_communication_profile(
        build_model_tool_specs(include_get_image=True), "intermediate"
    )
    communicate = specs["communicate"]
    assert "that, and only that, wakes a partner" in communicate["description"]
    assert "coordination_phase" not in communicate["description"]
    assert "coordination_phase" not in communicate.get("optional_tool_args", [])
    assert "coordination_phase" not in communicate.get("allowed_arg_values", {})


@pytest.mark.parametrize("mode", ("unguided", "minimal", "intermediate"))
def test_communicative_modes_describe_releasable_waits(mode):
    canonical = build_model_tool_specs(include_get_image=True)
    before = deepcopy(canonical)
    specs = apply_communication_profile(canonical, mode)
    wait = specs["wait_for_signal"]
    assert "communicate" in specs
    assert "releases" in wait["description"]
    assert wait["tool_arg_descriptions"] == canonical["wait_for_signal"]["tool_arg_descriptions"]
    description = json.dumps(wait)
    assert "Communication is unavailable" not in description
    assert "No message can arrive" not in description
    assert "Stop this agent from acting for the rest of the episode" not in description
    assert canonical == before
    assert "coordination_phase" not in json.dumps(specs)
    assert "Coordinator handshake" not in _prompt(mode)


def test_none_is_the_only_mode_with_unreleasable_wait_description():
    specs = apply_communication_profile(build_model_tool_specs(include_get_image=True), "none")
    assert "communicate" not in specs
    assert "Communication is unavailable" in specs["wait_for_signal"]["description"]


def test_full_is_unmodified_and_intermediate_preserves_detailed_wait():
    canonical = build_model_tool_specs(include_get_image=True)
    assert apply_communication_profile(canonical, "full") == canonical
    intermediate = apply_communication_profile(canonical, "intermediate")
    assert intermediate["wait_for_signal"] == canonical["wait_for_signal"]


def test_unguided_and_minimal_have_same_tools_but_different_guidance():
    canonical = build_model_tool_specs(include_get_image=True)
    assert apply_communication_profile(canonical, "unguided") == apply_communication_profile(canonical, "minimal")
    assert "Communicate with the other agent to complete the task together" not in _prompt("unguided")
    assert "Communicate with the other agent to complete the task together" in _prompt("minimal")
    assert "On your next call, use wait_for_signal" not in _prompt("unguided")
    assert "On your next call, use wait_for_signal" in _prompt("minimal")
