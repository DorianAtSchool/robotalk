from __future__ import annotations

from types import SimpleNamespace

from robotalk.tasks.shared.instances import (
    build_randomized_fixture_task_instance,
    count_balanced_initial_configurations,
    eligible_access_state_parts,
)
from robotalk.tasks.specs import load_all_task_specs


def _spec(composite_task: str):
    return next(
        spec for spec in load_all_task_specs() if spec.composite_task == composite_task
    )


def test_prepare_coffee_cabinet_is_globally_eligible() -> None:
    spec = _spec("PrepareCoffee")
    assert eligible_access_state_parts(
        initial_state=spec.initial_state,
        allowed_tool_specs=spec.allowed_tool_specs,
    ) == (("cab", "hinged"),)


def test_double_door_cabinet_is_globally_eligible() -> None:
    spec = _spec("SpicyMarinade")
    assert eligible_access_state_parts(
        initial_state=spec.initial_state,
        allowed_tool_specs=spec.allowed_tool_specs,
    ) == (("cabinet", "hinged"),)


def test_access_sampling_is_deterministic_and_reaches_both_states() -> None:
    spec = _spec("PrepareCoffee")
    states = []
    for run_index in range(20):
        kwargs = dict(
            composite_task=spec.composite_task,
            agent_ids=spec.agent_ids,
            initial_state=spec.initial_state,
            allowed_tool_specs=spec.allowed_tool_specs,
            run_index=run_index,
            runtime_config=SimpleNamespace(
                random_start_location=False, random_access_state=True
            ),
        )
        first = build_randomized_fixture_task_instance(**kwargs)
        second = build_randomized_fixture_task_instance(**kwargs)
        state = first.initial_state["fixtures"]["cab"]["parts"]["hinged"]["state"]
        assert second.initial_state["fixtures"]["cab"]["parts"]["hinged"]["state"] == state
        states.append(state)
    assert set(states) == {"closed", "open"}
    assert states.count("closed") == states.count("open") == 10


def test_joint_start_and_access_configurations_are_evenly_covered() -> None:
    spec = _spec("PrepareCoffee")
    config_count = count_balanced_initial_configurations(
        composite_task=spec.composite_task,
        agent_ids=spec.agent_ids,
        initial_state=spec.initial_state,
        allowed_tool_specs=spec.allowed_tool_specs,
    )
    observed: dict[tuple[object, ...], int] = {}
    for run_index in range(config_count * 3 + 1):
        instance = build_randomized_fixture_task_instance(
            composite_task=spec.composite_task,
            agent_ids=spec.agent_ids,
            initial_state=spec.initial_state,
            allowed_tool_specs=spec.allowed_tool_specs,
            run_index=run_index,
            runtime_config=SimpleNamespace(
                random_start_location=True, random_access_state=True
            ),
        )
        state = instance.initial_state
        signature = (
            state["agents"]["agent_0"]["location"],
            state["agents"]["agent_1"]["location"],
            state["fixtures"]["cab"]["parts"]["hinged"]["state"],
        )
        observed[signature] = observed.get(signature, 0) + 1
    assert len(observed) == config_count
    assert max(observed.values()) - min(observed.values()) == 1


def test_access_sampling_can_be_disabled_without_changing_spec() -> None:
    spec = _spec("PrepareCoffee")
    instance = build_randomized_fixture_task_instance(
        composite_task=spec.composite_task,
        agent_ids=spec.agent_ids,
        initial_state=spec.initial_state,
        allowed_tool_specs=spec.allowed_tool_specs,
        run_index=3,
        runtime_config=SimpleNamespace(
            random_start_location=False, random_access_state=False
        ),
    )
    assert instance.initial_state["fixtures"]["cab"]["parts"]["hinged"]["state"] == "open"
    assert spec.initial_state["fixtures"]["cab"]["parts"]["hinged"]["state"] == "open"


def test_appliance_state_is_not_randomized() -> None:
    spec = _spec("PrepareCoffee")
    for run_index in range(10):
        instance = build_randomized_fixture_task_instance(
            composite_task=spec.composite_task,
            agent_ids=spec.agent_ids,
            initial_state=spec.initial_state,
            allowed_tool_specs=spec.allowed_tool_specs,
            run_index=run_index,
            runtime_config=SimpleNamespace(
                random_start_location=False, random_access_state=True
            ),
        )
        assert instance.initial_state["machine_state"]["coffee_machine"]["turned_on"] is False
        assert instance.initial_state["objects"]["mug"]["location"] == "cab"
