"""Scheduler-contract tests for partial-observability live sim.

These exercise the pieces that encode the distributed contract — resource
claims, consume-once observations, private history/delivery, and rejection
handling — without a model, a GPU, or a simulator.
"""

from __future__ import annotations

import pytest

from robotalk.tasks.shared.concurrent_fsm import (
    atomic_handover_conflicts,
    contention_resources,
    simultaneous_contentions,
)
from robotalk.tasks.shared.state import (
    AgentRuntimeState,
    TaskRuntimeState,
)

from robotalk.evaluation.live_sim_eval import (
    REJECTION_MODE_REPORT_FAILED,
    AgentRuntime,
    _count_rejected_partial_records,
    _handle_rejection,
    _partial_budget_termination,
    _proposal_key,
    _tool_duration,
)


INITIAL_STATE = {
    "objects": {
        "cheese": {"object_type": "cheese", "location": "counter"},
        "tomato": {"object_type": "tomato", "location": "counter"},
        "tray": {"object_type": "tray", "location": "counter"},
    },
    "fixtures": {
        "counter": {"fixture_type": "counter"},
        "fridge": {"fixture_type": "fridge"},
        "toaster": {
            "fixture_type": "toaster_oven",
            "parent_fixture": "counter",
        },
    },
}


def _nav(fixture: str) -> dict:
    return {"tool": "navigate_to_fixture", "args": {"fixture_id": fixture}}


def _pick(obj: str, source: str) -> dict:
    return {"tool": "pick_up_object", "args": {"object_id": obj, "source_id": source}}


def _comm(to: str, message: str = "hi") -> dict:
    return {"tool": "communicate", "args": {"to": to, "message": message}}


def _conflicts(a: dict, b: dict) -> bool:
    return bool(simultaneous_contentions(a, b, INITIAL_STATE))


# --- resource claims -------------------------------------------------------


def test_disjoint_fixtures_do_not_conflict():
    assert not _conflicts(_nav("counter"), _nav("fridge"))


def test_same_roomy_counter_does_not_conflict():
    assert not _conflicts(_nav("counter"), _nav("counter"))


def test_different_objects_at_same_roomy_counter_do_not_conflict():
    assert not _conflicts(
        _pick("cheese", "counter"),
        _pick("tomato", "counter"),
    )


def test_same_exclusive_fixture_conflicts():
    assert _conflicts(_nav("fridge"), _nav("fridge"))


def test_distinct_objects_from_same_stationary_movable_source_do_not_conflict():
    assert not _conflicts(
        _pick("cheese", "tray"),
        _pick("tomato", "tray"),
    )


def test_same_object_pickup_conflicts():
    assert _conflicts(
        _pick("cheese", "tray"),
        _pick("cheese", "tray"),
    )


def test_moving_pickup_source_while_partner_reads_it_conflicts():
    assert _conflicts(
        _pick("tray", "counter"),
        _pick("tomato", "tray"),
    )


def test_shared_stationary_placement_target_does_not_conflict():
    left = {
        "tool": "place_in_receptacle",
        "args": {"object_id": "cheese", "receptacle_id": "tray"},
    }
    right = {
        "tool": "place_in_receptacle",
        "args": {"object_id": "tomato", "receptacle_id": "tray"},
    }
    assert not _conflicts(left, right)


def test_moving_a_support_while_partner_uses_it_conflicts():
    move_tray = _pick("tray", "counter")
    use_tray = {
        "tool": "place_in_receptacle",
        "args": {"object_id": "tomato", "receptacle_id": "tray"},
    }

    assert _conflicts(move_tray, use_tray)


def test_read_only_tools_never_conflict():
    gi = {"tool": "get_image", "args": {"views": ["wrist"]}}
    assert contention_resources(gi, INITIAL_STATE) == frozenset()
    assert contention_resources(_comm("agent_1"), INITIAL_STATE) == frozenset()
    assert not _conflicts(_comm("agent_1"), _comm("agent_0"))


def _runtime_state(agent_0_location: str, agent_1_location: str) -> TaskRuntimeState:
    return TaskRuntimeState(
        agents={
            "agent_0": AgentRuntimeState(
                location=agent_0_location, held_object=None
            ),
            "agent_1": AgentRuntimeState(
                location=agent_1_location, held_object=None
            ),
        },
        objects={}, fixtures={}, machine_state={},
    )


def test_same_tick_give_space_and_exclusive_entry_conflict():
    state = _runtime_state("counter", "fridge")
    calls = [
        {"agent": "agent_0", **_nav("fridge")},
        {
            "agent": "agent_1",
            "tool": "give_space",
            "args": {"fixture_id": "fridge"},
        },
    ]
    assert atomic_handover_conflicts(calls, state, INITIAL_STATE) == {
        "agent_0": frozenset({"fridge"})
    }
    assert atomic_handover_conflicts(
        list(reversed(calls)), state, INITIAL_STATE
    ) == {"agent_0": frozenset({"fridge"})}


def test_entry_after_prior_give_space_is_accepted():
    state = _runtime_state("counter", "counter")
    assert not atomic_handover_conflicts(
        [{"agent": "agent_0", **_nav("fridge")}], state, INITIAL_STATE
    )


def test_exclusive_entry_is_rejected_while_partner_remains_there():
    state = _runtime_state("counter", "fridge")
    assert atomic_handover_conflicts(
        [{"agent": "agent_0", **_nav("fridge")}], state, INITIAL_STATE
    ) == {"agent_0": frozenset({"fridge"})}


def test_same_tick_give_space_and_exclusive_manipulation_conflict():
    state = _runtime_state("counter", "fridge")
    calls = [
        {
            "agent": "agent_0",
            "tool": "open_hinged_part",
            "args": {"target_id": "fridge", "part_id": "door"},
        },
        {
            "agent": "agent_1",
            "tool": "give_space",
            "args": {"fixture_id": "fridge"},
        },
    ]
    assert atomic_handover_conflicts(calls, state, INITIAL_STATE) == {
        "agent_0": frozenset({"fridge"})
    }


def test_parented_exclusive_appliance_uses_same_atomic_handover_rule():
    state = _runtime_state("counter", "toaster")
    calls = [
        {"agent": "agent_0", **_nav("toaster")},
        {
            "agent": "agent_1",
            "tool": "give_space",
            "args": {"fixture_id": "toaster"},
        },
    ]
    assert atomic_handover_conflicts(calls, state, INITIAL_STATE) == {
        "agent_0": frozenset({"toaster"})
    }


def test_roomy_counter_give_space_does_not_create_atomic_handover_lock():
    state = _runtime_state("fridge", "counter")
    calls = [
        {"agent": "agent_0", **_nav("counter")},
        {
            "agent": "agent_1",
            "tool": "give_space",
            "args": {"fixture_id": "counter"},
        },
    ]
    assert not atomic_handover_conflicts(calls, state, INITIAL_STATE)


# --- consume-once observations ---------------------------------------------


def test_observation_is_consumed_exactly_once():
    agent = AgentRuntime("agent_0")
    agent.pending_obs = (["/img/a.png"], ["wrist"])

    paths, views = agent.take_observation()
    assert paths == ["/img/a.png"] and views == ["wrist"]

    # second decision gets nothing: the image was spent
    assert agent.take_observation() == ([], [])


# --- private history and delivery ------------------------------------------


def test_private_history_is_agent_scoped():
    a0, a1 = AgentRuntime("agent_0"), AgentRuntime("agent_1")
    a0.deliver({"agent": "agent_0", "tool": "navigate_to_fixture",
                "args": {"fixture_id": "counter"}})

    assert len(a0.private_history) == 1
    assert a1.private_history == []  # agent_1 never sees agent_0's physical act


def test_communicate_reaches_recipient_but_images_do_not():
    a0, a1 = AgentRuntime("agent_0"), AgentRuntime("agent_1")
    a0.pending_obs = (["/img/secret.png"], ["wrist"])
    msg = {"agent": "agent_0", "tool": "communicate",
           "args": {"to": "agent_1", "message": "counter is clear"}}
    a0.deliver(msg)
    a1.deliver(msg)  # delivery at execution time

    assert a1.private_history[-1]["args"]["message"] == "counter is clear"
    assert a1.pending_obs is None  # images are never transferred


# --- rejection handling ----------------------------------------------------


def _reject(agent, agents, *, mode, clock=0.0):
    record: dict = {}
    _handle_rejection(
        agent=agent, agents=agents, step=_nav("counter"), reason="resource conflict",
        clock=clock, mode=mode, multiplier=1.0, record=record,
    )
    return record


def test_report_failed_mode_records_failure_immediately():
    a0, a1 = AgentRuntime("agent_0"), AgentRuntime("agent_1")
    record = _reject(a0, {"agent_0": a0, "agent_1": a1},
                     mode=REJECTION_MODE_REPORT_FAILED)

    assert record["failure_reported"] is True
    assert "error" in a0.private_history[-1]


def test_rejection_never_touches_the_other_agent():
    a0, a1 = AgentRuntime("agent_0"), AgentRuntime("agent_1")
    agents = {"agent_0": a0, "agent_1": a1}
    _reject(a0, agents, mode=REJECTION_MODE_REPORT_FAILED)
    assert a1.private_history == []


def test_partial_rejection_records_are_counted_across_failure_types():
    records = [
        {"legal": False, "executed": False},
        {"legal": True, "sim_success": False},
        {"legal": True, "sim_success": True},
    ]
    assert _count_rejected_partial_records(records) == 2


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"rejected_total": 8}, "rejection_budget_exhausted"),
        ({"consecutive_rejections": 3}, "max_consecutive_rejections"),
        ({"proposal_index": 16}, "proposal_budget_exhausted"),
    ],
)
def test_partial_scheduler_budget_terminations(overrides, expected):
    values = {
        "rejected_total": 0,
        "max_rejected": 8,
        "consecutive_rejections": 0,
        "max_consecutive_rejections": 3,
        "proposal_index": 0,
        "max_proposals": 16,
        "turn_index": 0,
        "step_budget": 8,
    }
    values.update(overrides)
    assert _partial_budget_termination(**values) == expected


# --- durations -------------------------------------------------------------


def test_durations_separate_tool_classes_even_when_measurement_is_flat():
    """Teleporting execution can make measured sim-steps near-uniform; the
    per-class floors are what stop both agents staying in lockstep."""

    comm = _tool_duration("communicate", sim_steps=None, multiplier=1.0)
    nav = _tool_duration("navigate_to_fixture", sim_steps=None, multiplier=1.0)
    pick = _tool_duration("pick_up_object", sim_steps=None, multiplier=1.0)
    assert comm < pick < nav


def test_measured_steps_override_the_floor_when_larger():
    assert _tool_duration("communicate", sim_steps=100, multiplier=1.0) == 100.0


# --- repeat detection ------------------------------------------------------


def test_proposal_key_detects_identical_retries():
    assert _proposal_key(_nav("counter")) == _proposal_key(_nav("counter"))
    assert _proposal_key(_nav("counter")) != _proposal_key(_nav("fridge"))
