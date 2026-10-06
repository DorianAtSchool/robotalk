"""The FSM must prove every wait_for_signal is announced, valid, and released.

The runtime harness wakes a waiter on ANY message on purpose -- judging relevance
is the model's job at inference. Validation has the opposite duty: a demo is only
usable if the dependency is stated and really discharged, so both the
announcement and the release must name `about` verbatim, and `about` must name
something that exists. Without these a trajectory that deadlocks in live-sim
still validates, because a wait has no symbolic effect to contradict.
"""

from __future__ import annotations

import unittest

from robotalk.tasks.shared.errors import (
    WaitSignalSemanticValidationError,
)
from robotalk.tasks.shared.fsm import FiniteStateTaskValidator


class _NeverSatisfied(FiniteStateTaskValidator):
    """Goal never fires, so validation runs the whole step list."""

    def is_goal_state_satisfied(self, runtime_state) -> bool:  # noqa: D102
        return False


def _validator() -> FiniteStateTaskValidator:
    return _NeverSatisfied(
        composite_task="WaitProbe",
        agent_ids=("agent_0", "agent_1"),
        initial_state={
            "agents": {
                "agent_0": {"location": "counter", "held_object": None},
                "agent_1": {"location": "counter", "held_object": None},
            },
            "objects": {
                "bowl": {"location": "counter"},
                "salad_bowl": {"location": "counter"},
                "sugar_cube_1": {"location": "counter"},
                "sugar_cube_2": {"location": "counter"},
            },
            "fixtures": {"counter": {"fixture_type": "counter"}},
        },
        allowed_tool_specs={
            "communicate": {"tool_args": ["to", "message"]},
            "wait_for_signal": {"tool_args": ["from", "about"]},
        },
    )


def _msg(step, agent, to, message):
    return {
        "step": step,
        "agent": agent,
        "tool": "communicate",
        "args": {"to": to, "message": message},
        "reasoning": "I am speaking.",
    }


def _announce(step, agent="agent_0", to="agent_1", about="bowl"):
    return _msg(step, agent, to, f"I am waiting on {about} before I continue.")


def _wait(step, agent="agent_0", frm="agent_1", about="bowl"):
    return {
        "step": step,
        "agent": agent,
        "tool": "wait_for_signal",
        "args": {"from": frm, "about": about},
        "reasoning": "I am waiting.",
    }


def _act(step, agent="agent_1", about="bowl"):
    """The releasing action: the partner's last use of the contested thing."""

    return {
        "step": step,
        "agent": agent,
        "tool": "place_on_surface",
        "args": {"object_id": about, "support_id": "counter"},
        "reasoning": "I am done with it.",
    }


def _release(step, agent="agent_1", to="agent_0", about="bowl"):
    return _msg(step, agent, to, f"I am done with {about}, it is yours.")


class WaitRuleTest(unittest.TestCase):
    def _run(self, steps):
        return _validator().validate(
            {
                "agents": [{"agent": "agent_0"}, {"agent": "agent_1"}],
                "steps": steps,
            }
        )

    def _error(self, steps) -> str:
        with self.assertRaises(WaitSignalSemanticValidationError) as caught:
            self._run(steps)
        return str(caught.exception)

    # --- the shape that should pass -------------------------------------

    def test_announced_and_released_is_accepted(self):
        steps = [_announce(0), _wait(1), _act(2), _release(3)]
        try:
            self._run(steps)
        except WaitSignalSemanticValidationError:  # pragma: no cover
            self.fail("an announced and released wait must satisfy the rule")
        except Exception:
            pass  # later goal checks are not this test's concern

    # --- `about` must name something real -------------------------------

    def test_invented_milestone_name_is_rejected(self):
        # the real SpicyMarinade failure: an event name nothing can release
        message = self._error(
            [
                _announce(0, about="cabinet_items_moved"),
                _wait(1, about="cabinet_items_moved"),
                _release(2, about="cabinet_items_moved"),
            ]
        )
        self.assertIn("not a symbolic id", message)

    def test_fixture_id_is_a_valid_about(self):
        steps = [_announce(0, about="counter"), _wait(1, about="counter"),
                 {"step": 2, "agent": "agent_1", "tool": "give_space",
                  "args": {"fixture_id": "counter"}, "reasoning": "Yielding."},
                 _release(3, about="counter")]
        try:
            self._run(steps)
        except WaitSignalSemanticValidationError:  # pragma: no cover
            self.fail("waiting on a declared fixture must be allowed")
        except Exception:
            pass

    # --- the announcement ------------------------------------------------


    # --- the release ------------------------------------------------------

    def test_wait_with_no_release_is_rejected(self):
        message = self._error(
            [_announce(0), _wait(1), _msg(2, "agent_1", "agent_0", "on my way.")]
        )
        self.assertIn("never released", message)

    def test_release_must_come_from_the_named_partner(self):
        message = self._error(
            [_announce(0), _wait(1), _msg(2, "agent_0", "agent_1", "bowl is free.")]
        )
        self.assertIn("never released", message)

    def test_paraphrase_does_not_release(self):
        message = self._error(
            [
                _announce(0, about="salad_bowl"),
                _wait(1, about="salad_bowl"),
                _msg(2, "agent_1", "agent_0", "I am finished with the bowl."),
            ]
        )
        self.assertIn("never released", message)

    def test_sibling_object_does_not_release(self):
        # the collision a token matcher would let through
        message = self._error(
            [
                _announce(0, about="sugar_cube_2"),
                _wait(1, about="sugar_cube_2"),
                _msg(2, "agent_1", "agent_0", "I am done with sugar_cube_1."),
            ]
        )
        self.assertIn("never released", message)

    def test_earlier_mention_does_not_count_as_release(self):
        message = self._error(
            [_announce(0), _msg(1, "agent_1", "agent_0", "I will use bowl first."),
             _wait(2)]
        )
        self.assertIn("never released", message)

    # --- arg shape --------------------------------------------------------

    def test_wait_rejects_self_as_partner(self):
        self.assertIn("other agent", self._error([_announce(0), _wait(1, frm="agent_0")]))

    def test_wait_rejects_empty_about(self):
        self.assertIn("non-empty about", self._error([_announce(0), _wait(1, about="  ")]))

    def test_wait_rejects_extra_args(self):
        step = _wait(1)
        step["args"]["timeout"] = 5
        self.assertIn("only contain from and about", self._error([_announce(0), step]))


if __name__ == "__main__":
    unittest.main()
