import pytest

from robotalk.generation.raw.config import RuntimeConfig
from robotalk.tasks import get_task_definition
from robotalk.tasks.shared.instances import (
    validate_initial_agent_locations,
)


def _config() -> RuntimeConfig:
    return RuntimeConfig(
        composite_task="AddSugarCubes",
        num_runs=1,
        model="gemini-3-flash-preview",
        sdk="google-genai",
        project=None,
        location="global",
        temperature=0.6,
    )


def test_simplified_prompt_has_eight_rules():
    definition = get_task_definition("AddSugarCubes")
    instance = definition.build_task_instance(0, _config())
    prompt = definition.build_prompt(
        "test", task_instance=instance
    )

    assert sum(f"\n{number}. " in prompt for number in range(1, 9)) == 8
    assert "Choose an efficient division of work from the initial state." in prompt


def test_tick_prompt_displays_wait_and_optional_communication_contract():
    definition = get_task_definition("AddSugarCubes")
    instance = definition.build_task_instance(0, _config())
    prompt = definition.build_prompt(
        "test", task_instance=instance
    )

    assert "wait_for_signal\n" in prompt
    assert "coordination_phase (STRING): Optional opening-protocol phase" in prompt
    assert "releases (STRING): Optional exact symbolic object or fixture ID" in prompt
    assert "matching release tick" in prompt
    assert 'releases=["' not in prompt


def test_simplified_v2_spells_out_location_and_finished_agent_transitions():
    definition = get_task_definition("AddSugarCubes")
    instance = definition.build_task_instance(0, _config())
    prompt = definition.build_prompt(
        "test", task_instance=instance
    )

    assert "give_space(X) removes it from X" in prompt
    assert "Being near or able to see fixture X does not count as being at X" in prompt
    assert "unless the agent's current symbolic location is already X" in prompt
    assert "on its next call send exactly one communicate" in prompt
    assert "opening plan assigns an agent no physical actions" in prompt
    assert "on the immediately following tick make the matching wait call" in prompt
    assert "It then remains blocked until" in prompt
    assert "using the finished-portion message from rule 6" in prompt
    assert "same tick as portion_complete satisfies the FSM goal" in prompt
    assert "do not add a wait or any later tick" in prompt
    assert "give_space(X) frees X only after its tick finishes" in prompt
    assert "must not enter or use X during the give_space tick" in prompt
    assert "following tick without waiting for a release" in prompt
    assert "wait_for_signal has only two uses" in prompt


def test_prepare_cheese_station_requires_the_goal_reference_object():
    definition = get_task_definition("PrepareCheeseStation")
    instance = definition.build_task_instance(0, _config())
    prompt = definition.build_prompt(
        "test", task_instance=instance
    )

    assert "place the cheese and grater next to salad_bowl" in prompt
    refs = instance.allowed_tool_specs["place_next_to"][  # type: ignore[index]
        "allowed_reference_object_ids"
    ]
    assert refs == ["salad_bowl"]


def test_simplified_v3_uses_explicit_blocked_schema_and_prompt():
    definition = get_task_definition("AddSugarCubes")
    config = _config()
    instance = definition.build_task_instance(0, config)
    prompt = definition.build_prompt(
        "test", task_instance=instance
    )
    schema = definition.explicit_blocked_tick_response_schema

    assert '{"state":"blocked"}' in prompt
    assert "never repeat the wait call" in prompt
    assert "explicit_blocked_v1" in prompt
    assert "Exclusive-fixture handover example" in prompt
    assert "The other agent cannot see the wait call" in prompt
    assert 'incoming agent communicates `When done, release "X"`' in prompt
    assert "Use the fixture ID X, not the ID of an object at X" in prompt
    assert "Waiting does not move an agent" in prompt
    assert "releasing an object at X does not free X" in prompt
    assert 'My portion is done. Release "X" if you need me again' in prompt
    assert "never omit the preceding message or put another action between" in prompt
    assert "Every real tool call needs one short first-person reasoning sentence" in prompt
    assert "An agent can hold only one object at a time" in prompt
    assert "There is no tool for passing an object between agents" in prompt
    assert schema["properties"]["format"]["enum"] == ["explicit_blocked_v1"]
    row = schema["properties"]["ticks"]["items"]
    assert set(row["required"]) == {"tick", "agent_0", "agent_1"}


def test_simplified_v3_no_partition_prefers_coherent_two_agent_work():
    definition = get_task_definition("AddSugarCubes")
    config = _config()
    instance = definition.build_task_instance(0, config)
    prompt = definition.build_prompt(
        "test", task_instance=instance
    )

    assert "Prefer giving both agents a coherent physical responsibility" in prompt
    assert "single agent may do all physical work only" in prompt


def test_random_starts_canonicalize_cabinet_to_shared_parent_workspace():
    definition = get_task_definition("PrepareSoupServing")
    instance = definition.build_task_instance(0, _config())
    agents = instance.initial_state["agents"]
    assert all(state["location"] != "cab" for state in agents.values())
    validate_initial_agent_locations(instance.initial_state)


def test_task_instance_validation_accepts_duplicate_cabinet_start_as_shared():
    shared = {
        "agents": {
            "agent_0": {"location": "cab"},
            "agent_1": {"location": "cab"},
        },
        "fixtures": {"cab": {"fixture_type": "cabinet"}},
    }
    validate_initial_agent_locations(shared)


def test_task_instance_validation_still_rejects_duplicate_appliance_start():
    invalid = {
        "agents": {
            "agent_0": {"location": "toaster"},
            "agent_1": {"location": "toaster"},
        },
        "fixtures": {"toaster": {"fixture_type": "toaster_oven"}},
    }
    with pytest.raises(ValueError, match="toaster: agent_0, agent_1"):
        validate_initial_agent_locations(invalid)
