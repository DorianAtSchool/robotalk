#!/usr/bin/env python
"""The tick observation injector must not move anything relative to anything else.

Coordination in tick output is carried entirely by which agents share a row.
Padding one agent's stream without padding the other's -- which is what the flat
injector does, since it pads per action and never pads `communicate` -- slides
releases past the waits they are meant to discharge. These tests pin the
property that makes the tick injector safe: every source tick maps to a
contiguous block of output ticks that all agents enter and leave together.
"""

from __future__ import annotations

import unittest


from robotalk.generation.image.processor import (  # noqa: E402
    GET_IMAGE_TOOL_NAME,
    TICK_KEY,
    post_process_trajectory,
    rebuild_tick_rows_with_image_observations,
)
from robotalk.tasks.shared.scheduling import (  # noqa: E402
    observation_transparent_tick_rows,
)

AGENTS = ("agent_0", "agent_1")


def _rows(*specs: dict[str, tuple[str, dict] | None]) -> list[dict]:
    """Builds tick rows from {agent: (tool, args)} shorthand."""

    rows = []
    for tick, spec in enumerate(specs):
        row = {TICK_KEY: tick}
        for agent_id, action in spec.items():
            if action is None:
                continue
            tool, args = action
            row[agent_id] = {"tool": tool, "args": args, "reasoning": "because"}
        rows.append(row)
    return rows


def _tick_of(rows: list[dict], agent_id: str, tool: str) -> list[int]:
    return [
        row[TICK_KEY]
        for row in rows
        if isinstance(row.get(agent_id), dict) and row[agent_id]["tool"] == tool
    ]


class DilationTests(unittest.TestCase):
    def test_observation_projection_recovers_source_schedule(self) -> None:
        rows = _rows(
            {"agent_0": ("pick_up_object", {"object_id": "mug"}),
             "agent_1": ("navigate_to_fixture", {"fixture_id": "sink"})},
            {"agent_0": ("communicate", {"to": "agent_1", "message": "done"})},
        )
        injected = rebuild_tick_rows_with_image_observations(rows, agent_ids=AGENTS)
        self.assertEqual(observation_transparent_tick_rows(injected), rows)

    def test_simultaneous_actions_stay_simultaneous(self) -> None:
        rows = _rows(
            {"agent_0": ("pick_up_object", {"object_id": "mug"}),
             "agent_1": ("navigate_to_fixture", {"fixture_id": "sink"})},
        )
        out = rebuild_tick_rows_with_image_observations(rows, agent_ids=AGENTS)
        self.assertEqual(
            _tick_of(out, "agent_0", "pick_up_object"),
            _tick_of(out, "agent_1", "navigate_to_fixture"),
        )

    def test_handover_ordering_survives(self) -> None:
        """ASK, block, work, let go, report -- the order the FSM checks."""

        rows = _rows(
            {"agent_0": ("communicate", {"to": "agent_1", "message": "mine?"}),
             "agent_1": ("pick_up_object", {"object_id": "mug"})},
            {"agent_0": ("wait_for_signal", {"from": "agent_1", "about": "sink"}),
             "agent_1": ("navigate_to_fixture", {"fixture_id": "sink"})},
            {"agent_1": ("give_space", {"fixture_id": "sink"})},
            {"agent_1": ("communicate",
                         {"to": "agent_0", "message": "yours", "releases": ["sink"]})},
            {"agent_0": ("navigate_to_fixture", {"fixture_id": "sink"})},
        )
        out = rebuild_tick_rows_with_image_observations(rows, agent_ids=AGENTS)
        ask = _tick_of(out, "agent_0", "communicate")[0]
        wait = _tick_of(out, "agent_0", "wait_for_signal")[0]
        space = _tick_of(out, "agent_1", "give_space")[0]
        release = _tick_of(out, "agent_1", "communicate")[-1]
        resume = _tick_of(out, "agent_0", "navigate_to_fixture")[0]
        self.assertLess(ask, wait)
        self.assertLess(wait, space)
        self.assertLess(space, release)
        self.assertLess(release, resume)

    def test_waits_and_messages_are_not_bracketed(self) -> None:
        rows = _rows(
            {"agent_0": ("communicate", {"to": "agent_1", "message": "hi"})},
            {"agent_0": ("wait_for_signal", {"from": "agent_1", "about": "sink"})},
        )
        out = rebuild_tick_rows_with_image_observations(rows, agent_ids=AGENTS)
        observed = [
            row for row in out
            if any(
                isinstance(row.get(a), dict)
                and row[a]["tool"] == GET_IMAGE_TOOL_NAME
                for a in AGENTS
            )
        ]
        # Only the opening scene inspection, nothing around the two calls.
        self.assertEqual(len(observed), 1)
        self.assertEqual(observed[0][TICK_KEY], 0)

    def test_every_action_is_bracketed_once_each_side(self) -> None:
        rows = _rows({"agent_0": ("press_button", {"fixture_id": "coffee_machine"})})
        out = rebuild_tick_rows_with_image_observations(rows, agent_ids=AGENTS)
        tools = [
            row["agent_0"]["tool"]
            for row in out
            if isinstance(row.get("agent_0"), dict)
        ]
        self.assertEqual(
            tools,
            [GET_IMAGE_TOOL_NAME, GET_IMAGE_TOOL_NAME, "press_button",
             GET_IMAGE_TOOL_NAME],
        )

    def test_opening_observation_is_one_shared_tick(self) -> None:
        rows = _rows({"agent_0": ("press_button", {"fixture_id": "coffee_machine"})})
        out = rebuild_tick_rows_with_image_observations(rows, agent_ids=AGENTS)
        self.assertEqual(out[0][TICK_KEY], 0)
        for agent_id in AGENTS:
            self.assertEqual(out[0][agent_id]["tool"], GET_IMAGE_TOOL_NAME)

    def test_rerunning_is_idempotent(self) -> None:
        rows = _rows(
            {"agent_0": ("pick_up_object", {"object_id": "mug"}),
             "agent_1": ("communicate", {"to": "agent_0", "message": "go"})},
            {"agent_1": ("give_space", {"fixture_id": "sink"})},
        )
        once = rebuild_tick_rows_with_image_observations(rows, agent_ids=AGENTS)
        twice = rebuild_tick_rows_with_image_observations(once, agent_ids=AGENTS)
        self.assertEqual(once, twice)


class RecordTests(unittest.TestCase):
    def _record(self) -> dict:
        return {
            "trajectory_id": "traj_000000",
            "agents": [{"agent": a} for a in AGENTS],
            "steps": [],
            "tick_rows": _rows(
                {"agent_0": ("pick_up_object", {"object_id": "mug"}),
                 "agent_1": ("communicate", {"to": "agent_0", "message": "go"})},
            ),
        }

    def test_steps_are_rederived_from_the_rebuilt_rows(self) -> None:
        out = post_process_trajectory(self._record())
        from robotalk.generation.tick_format import to_steps

        # image_paths is attached to the flat steps afterwards; everything else
        # about the two views of the plan has to agree.
        stripped = [
            {k: v for k, v in step.items() if k != "image_paths"}
            for step in out["steps"]
        ]
        self.assertEqual(stripped, to_steps(out["tick_rows"], AGENTS))

    def test_observation_steps_carry_image_paths(self) -> None:
        out = post_process_trajectory(self._record())
        for step in out["steps"]:
            if step["tool"] == GET_IMAGE_TOOL_NAME:
                self.assertTrue(step.get("image_paths"))
            else:
                self.assertNotIn("image_paths", step)


if __name__ == "__main__":
    unittest.main(verbosity=1)
