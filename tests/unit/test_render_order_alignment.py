"""The dataset builder must line raw steps up with the order the sim ran them.

``original_trajectory.json`` is stored in file order; with ``--step-order
concurrent`` the sweep runs the plan in executor order, so ``plan.json`` and
``metadata.json`` come back permuted. Zipping the three positionally would pair
each raw step with another step's execution result and its images.
"""

import pytest

from robotalk.training import dataset


def _raw(step: int, agent: str, tool: str) -> dict:
    return {"step": step, "agent": agent, "tool": tool, "args": {}}


def _plan(source: int, tool: str, agent: str) -> dict:
    return {
        "tool": tool,
        "args": {},
        "metadata": {"step_index": source, "source_agent": agent},
    }


RAW = [
    _raw(0, "agent_0", "communicate"),
    _raw(1, "agent_1", "communicate"),
    _raw(2, "agent_1", "navigate_to_fixture"),
    _raw(3, "agent_0", "get_image"),
    _raw(4, "agent_0", "pick_up_object"),
]
# agent_0's get_image ran before agent_1's navigate: they share a tick, and the
# executor acts in agent-id order within a tick.
ORDER = [0, 1, 3, 2, 4]


def _plans(order):
    return [_plan(i, RAW[i]["tool"], RAW[i]["agent"]) for i in order]


def _executed(order):
    return [
        {"step_index": position, "source_step_index": source, "success": True}
        for position, source in enumerate(order)
    ]


def _align(order):
    plan_steps = _plans(order)
    ordered = dataset._order_raw_steps_as_executed(
        task_name="t",
        trajectory_id="traj_0",
        raw_steps=list(RAW),
        plan_steps=plan_steps,
    )
    dataset._validate_alignment(
        task_name="t",
        trajectory_id="traj_0",
        raw_steps=ordered,
        plan_steps=plan_steps,
        executed_steps=_executed(order),
    )
    return ordered


def test_a_file_order_render_is_left_exactly_as_it_was():
    assert _align([0, 1, 2, 3, 4]) == RAW


def test_a_concurrent_order_render_lines_up_with_the_run():
    ordered = _align(ORDER)
    assert [s["step"] for s in ordered] == ORDER
    assert [s["tool"] for s in ordered] == [RAW[i]["tool"] for i in ORDER]


def test_every_step_survives_the_permutation():
    ordered = _align(ORDER)
    assert sorted(s["step"] for s in ordered) == [s["step"] for s in RAW]


def test_a_plan_naming_an_unknown_source_step_is_rejected():
    plan_steps = _plans(ORDER)
    plan_steps[2]["metadata"]["step_index"] = 99
    with pytest.raises(ValueError, match="not in the trajectory"):
        dataset._order_raw_steps_as_executed(
            task_name="t",
            trajectory_id="traj_0",
            raw_steps=list(RAW),
            plan_steps=plan_steps,
        )


def test_a_plan_missing_its_source_index_is_rejected():
    """An older render without the metadata must fail loudly, not misalign."""

    plan_steps = _plans(ORDER)
    plan_steps[1]["metadata"] = {}
    with pytest.raises(ValueError, match="not in the trajectory"):
        dataset._order_raw_steps_as_executed(
            task_name="t",
            trajectory_id="traj_0",
            raw_steps=list(RAW),
            plan_steps=plan_steps,
        )


def test_a_plan_that_repeats_a_source_step_is_rejected():
    plan_steps = _plans([0, 1, 3, 3, 4])
    ordered = dataset._order_raw_steps_as_executed(
        task_name="t",
        trajectory_id="traj_0",
        raw_steps=list(RAW),
        plan_steps=plan_steps,
    )
    with pytest.raises(ValueError, match="not a permutation"):
        dataset._validate_alignment(
            task_name="t",
            trajectory_id="traj_0",
            raw_steps=ordered,
            plan_steps=plan_steps,
            executed_steps=_executed([0, 1, 3, 3, 4]),
        )


def test_metadata_disagreeing_with_the_plan_is_rejected():
    plan_steps = _plans(ORDER)
    ordered = dataset._order_raw_steps_as_executed(
        task_name="t",
        trajectory_id="traj_0",
        raw_steps=list(RAW),
        plan_steps=plan_steps,
    )
    executed = _executed(ORDER)
    executed[3]["source_step_index"] = 4
    with pytest.raises(ValueError, match="source mismatch"):
        dataset._validate_alignment(
            task_name="t",
            trajectory_id="traj_0",
            raw_steps=ordered,
            plan_steps=plan_steps,
            executed_steps=executed,
        )


def test_an_older_render_without_source_step_index_still_validates():
    """Renders predating the reorder carry no source_step_index; accept them."""

    order = [0, 1, 2, 3, 4]
    plan_steps = _plans(order)
    executed = [{"step_index": i, "success": True} for i in order]
    ordered = dataset._order_raw_steps_as_executed(
        task_name="t",
        trajectory_id="traj_0",
        raw_steps=list(RAW),
        plan_steps=plan_steps,
    )
    dataset._validate_alignment(
        task_name="t",
        trajectory_id="traj_0",
        raw_steps=ordered,
        plan_steps=plan_steps,
        executed_steps=executed,
    )
