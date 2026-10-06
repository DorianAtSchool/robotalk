"""Rendering must walk the steps the way the concurrent executor will."""

import json
import os
from pathlib import Path

import pytest

from robotalk.tasks.shared.render_order import (
    concurrent_step_order,
    reorder_for_concurrent_render,
)

# The training layout built by robotalk.release.materialize_training_layout.
CORPUS = Path(os.environ.get("ROBOTALK_DATA_ROOT", "data/robotalk_rendered"))


def _sample(count: int = 12) -> list[Path]:
    # An empty sample makes pytest skip the parametrized tests.
    paths = sorted(CORPUS.glob("*/traj_*/original_trajectory.json"))
    if not paths:
        return []
    stride = max(len(paths) // count, 1)
    return paths[::stride][:count]


def test_a_trajectory_with_one_step_is_left_alone():
    trajectory = {"composite_task": "nope", "steps": [{"step": 0}]}
    assert concurrent_step_order(trajectory) == [0]


def test_an_unloadable_task_is_an_error_not_a_file_order_render():
    trajectory = {
        "composite_task": "definitely_not_a_task",
        "agents": ["agent_0"],
        "steps": [{"step": 0}, {"step": 1}],
    }
    with pytest.raises(ValueError, match="cannot load task"):
        concurrent_step_order(trajectory)


@pytest.mark.parametrize("path", _sample(), ids=lambda p: f"{p.parent.parent.name}/{p.parent.name}")
def test_the_order_is_a_permutation_that_loses_no_step(path):
    trajectory = json.loads(path.read_text())
    order = concurrent_step_order(trajectory)
    assert sorted(order) == list(range(len(trajectory["steps"])))


@pytest.mark.parametrize("path", _sample(), ids=lambda p: f"{p.parent.parent.name}/{p.parent.name}")
def test_reordering_preserves_every_step_object_and_its_step_field(path):
    trajectory = json.loads(path.read_text())
    reordered, order = reorder_for_concurrent_render(trajectory)
    original = trajectory["steps"]
    assert len(reordered["steps"]) == len(original)
    # Same objects, same `step` labels -- only the sequence differs, so image
    # paths and any downstream join on step identity survive intact.
    assert {s["step"] for s in reordered["steps"]} == {s["step"] for s in original}
    for position, index in enumerate(order):
        assert reordered["steps"][position] is original[index]


@pytest.mark.parametrize("path", _sample(), ids=lambda p: f"{p.parent.parent.name}/{p.parent.name}")
def test_each_agents_own_steps_keep_their_relative_order(path):
    """Reordering interleaves the agents; it never reshuffles one agent's plan."""

    trajectory = json.loads(path.read_text())
    reordered, _ = reorder_for_concurrent_render(trajectory)
    for agent in trajectory.get("agents", []):
        agent_id = agent if isinstance(agent, str) else agent.get("agent_id")
        before = [s["step"] for s in trajectory["steps"] if s.get("agent") == agent_id]
        after = [s["step"] for s in reordered["steps"] if s.get("agent") == agent_id]
        assert before == after


def test_published_trajectories_are_already_in_concurrent_order():
    """Tick-format trajectories are written tick by tick, so the concurrent
    executor walks them in file order; reordering only matters for
    trajectories written agent by agent."""

    paths = _sample(24)
    if not paths:
        pytest.skip(f"corpus not present at {CORPUS}")
    for path in paths:
        trajectory = json.loads(path.read_text())
        order = concurrent_step_order(trajectory)
        assert order == list(range(len(trajectory["steps"])))
