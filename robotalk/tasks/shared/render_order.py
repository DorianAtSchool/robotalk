"""The order concurrent sim runs a trajectory's steps in.

A trajectory is stored as one flat list of steps. Replaying that list top to
bottom is NOT what the concurrent executor does: it runs ticks on a clock and,
within a tick, applies each ready agent's call in agent-id order
(``live_sim_eval`` sorts its ``ready`` list by ``agent_id``;
``ConcurrentTaskValidator.replay`` sorts ``acting`` the same way). Across ticks
the two agree, because the flat list is a serialization of the tick rows.
Within a tick they need not, and measurably do not -- 20.5% of the corpus's
observations have a different set of actions applied before them under the two
orders.

That matters for rendering. An image taken in flat order can show a world the
model would never see at that point in concurrent sim, and the model is trained
to condition on it. This returns the executor's order so the renderer can walk
the steps the way the executor will.

The stored trajectory is never modified -- only the order it is executed in.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any


def concurrent_step_order(trajectory: dict[str, Any]) -> list[int]:
    """Positions into ``trajectory["steps"]``, in concurrent-executor order.

    Raises ``ValueError`` when the concurrent replay cannot place every step
    (an unknown task, or a plan that blocks), rather than rendering in another
    order.
    """

    steps = trajectory.get("steps") or []
    if len(steps) < 2:
        return list(range(len(steps)))

    from robotalk.tasks.shared import concurrent_fsm as cf
    from robotalk.tasks.specs import load_task_spec
    from robotalk.tasks.specs.runtime import SpecDrivenTaskValidator

    try:
        inner = SpecDrivenTaskValidator(load_task_spec(trajectory["composite_task"]))
    except Exception as exc:
        raise ValueError(f"cannot load task {trajectory.get('composite_task')!r}: {exc}") from exc
    inner.initial_state = deepcopy(trajectory.get("initial_state") or {})
    replay = cf.ConcurrentTaskValidator(inner, models=(cf.LOCK_STEP,)).replay(
        {"agents": trajectory.get("agents"), "steps": steps},
        model=cf.LOCK_STEP,
        # A step error must not truncate the render: the sim is the
        # authority on what physically happens, and it runs the whole plan.
        stop_on_step_error=False,
    )

    order: list[int] = []
    seen: set[int] = set()
    for event in replay.events:
        if 0 <= event.index < len(steps) and event.index not in seen:
            seen.add(event.index)
            order.append(event.index)
    if len(order) != len(steps):
        raise ValueError(
            f"concurrent replay placed {len(order)}/{len(steps)} steps "
            f"(blocked: {', '.join(replay.blocked) or 'none'})"
        )
    return order


def reorder_for_concurrent_render(
    trajectory: dict[str, Any],
) -> tuple[dict[str, Any], list[int]]:
    """A copy of ``trajectory`` whose steps are in concurrent-executor order.

    Each step keeps its original ``step`` field, so image paths, reasoning, and
    any downstream join on step identity survive the permutation intact.
    """

    order = concurrent_step_order(trajectory)
    steps = trajectory.get("steps") or []
    if order == list(range(len(steps))):
        return trajectory, order

    reordered = dict(trajectory)
    reordered["steps"] = [steps[index] for index in order]
    return reordered, order


__all__ = ["concurrent_step_order", "reorder_for_concurrent_render"]
