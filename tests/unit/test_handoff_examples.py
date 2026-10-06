"""Hand-checked examples of correct and incorrect handoffs.

These are the ground truth for the sharing protocol. Every rule in the FSM
should be derivable from them; when a rule and an example disagree, the example
wins and the rule is wrong. Three rules were written, unit-tested, and shipped
before real trajectories showed they rejected legitimate plans -- guessing the
rule and patching it against aggregate pass rates does not converge.

The scenarios, in plain terms:

  1. OBJECT HANDOFF      agent_0 fills the bowl, agent_1 carries it away.
                         agent_1 must not touch it until agent_0 says it is done.
  2. TIGHT FIXTURE       both need the cabinet, which fits one robot.
                         agent_1 waits until agent_0 has actually left.
  3. REVISITED FIXTURE   agent_0 uses the cabinet, hands it over, and comes back
                         later once agent_1 has finished. Legitimate.
  4. NO CONFLICT         two agents, two different bowls, one roomy counter.
                         Nothing is shared, so nothing is needed.
  5. PROMISE ONLY        agent_0 says it will yield, then keeps using the
                         cabinet anyway. Not a release.
  6. NO WAIT AT ALL      the handoff is simply unguarded.

1-4 must be ACCEPTED. 5-6 must be REJECTED.
"""

from __future__ import annotations

import unittest

from robotalk.tasks.shared.errors import (
    WaitSignalSemanticValidationError,
)
from robotalk.tasks.shared.fsm import FiniteStateTaskValidator


class _Probe(FiniteStateTaskValidator):
    """Goal never fires, so the whole step list is examined."""

    def is_goal_state_satisfied(self, runtime_state) -> bool:  # noqa: D102
        return False


def _validator() -> FiniteStateTaskValidator:
    return _Probe(
        composite_task="HandoffProbe",
        agent_ids=("agent_0", "agent_1"),
        initial_state={
            "agents": {
                "agent_0": {"location": "counter", "held_object": None},
                "agent_1": {"location": "counter", "held_object": None},
            },
            "objects": {
                "bowl": {"location": "counter"},
                "bread": {"location": "counter"},
                "plate": {"location": "counter"},
                "cup": {"location": "counter"},
            },
            "fixtures": {
                "counter": {"fixture_type": "counter"},
                "cabinet": {"fixture_type": "cabinet"},
            },
        },
        allowed_tool_specs={
            "communicate": {"tool_args": ["to", "message"]},
            "wait_for_signal": {"tool_args": ["from", "about"]},
            "navigate_to_fixture": {"tool_args": ["fixture_id"]},
            "pick_up_object": {"tool_args": ["object_id", "source_id"]},
            "place_on_surface": {"tool_args": ["object_id", "support_id"]},
            "give_space": {"tool_args": ["fixture_id"]},
        },
    )


def say(agent, to, text):
    return {"agent": agent, "tool": "communicate",
            "args": {"to": to, "message": text}, "reasoning": "Speaking."}


def wait(agent, frm, about):
    return {"agent": agent, "tool": "wait_for_signal",
            "args": {"from": frm, "about": about}, "reasoning": "Waiting."}


def do(agent, tool, **args):
    return {"agent": agent, "tool": tool, "args": args, "reasoning": "Acting."}


def _numbered(steps):
    for index, step in enumerate(steps):
        step["step"] = index
    return steps


class HandoffExampleTests(unittest.TestCase):
    def _validate(self, steps):
        return _validator().validate({
            "agents": [{"agent": "agent_0"}, {"agent": "agent_1"}],
            "steps": _numbered(steps),
        })

    def _accepts(self, steps, why):
        try:
            self._validate(steps)
        except WaitSignalSemanticValidationError as exc:
            self.fail(f"{why}\n  FSM wrongly rejected it: {exc}")
        except Exception:
            pass  # unrelated goal/precondition checks are not what we assert

    def _rejects(self, steps, why):
        with self.assertRaises(WaitSignalSemanticValidationError, msg=why):
            self._validate(steps)

    # ---------------------------------------------------------------- 1

    def test_object_handoff(self):
        """agent_0 puts bread in the bowl; agent_1 then carries the bowl off."""

        self._accepts([
            say("agent_0", "agent_1", "I am putting the bread into the bowl now."),
            say("agent_1", "agent_0", "Understood, I will take the bowl after."),
            do("agent_0", "pick_up_object", object_id="bread", source_id="counter"),
            do("agent_0", "place_on_surface", object_id="bread", support_id="bowl"),
            say("agent_1", "agent_0", "I need the bowl next; tell me when it is free."),
            wait("agent_1", "agent_0", "bowl"),
            say("agent_0", "agent_1", "I am finished with the bowl, it is yours now."),
            do("agent_1", "pick_up_object", object_id="bowl", source_id="counter"),
        ], "a plain object handoff with announce, wait, release")

    # ---------------------------------------------------------------- 2

    def test_tight_fixture_handoff(self):
        """Both need the cabinet. agent_1 waits until agent_0 has left."""

        self._accepts([
            say("agent_0", "agent_1", "I am going to the cabinet for the plate."),
            say("agent_1", "agent_0", "Fine, I need the cabinet after you."),
            do("agent_0", "navigate_to_fixture", fixture_id="cabinet"),
            do("agent_0", "pick_up_object", object_id="plate", source_id="cabinet"),
            say("agent_1", "agent_0", "Tell me when the cabinet is free."),
            wait("agent_1", "agent_0", "cabinet"),
            do("agent_0", "give_space", fixture_id="cabinet"),
            say("agent_0", "agent_1", "I have left the cabinet, it is free for you."),
            do("agent_1", "navigate_to_fixture", fixture_id="cabinet"),
        ], "a tight-fixture handoff where the holder actually leaves")

    # ---------------------------------------------------------------- 3

    def test_fixture_revisited_after_the_handoff(self):
        """agent_0 hands the cabinet over, then returns once agent_1 is done.

        This is legitimate and the current rule rejects it: 'the partner never
        touches it again' is true for objects but wrong for a fixture that both
        agents use in turn.
        """

        self._accepts([
            say("agent_0", "agent_1", "Taking the plate from the cabinet first."),
            say("agent_1", "agent_0", "I need the cabinet after you are done."),
            do("agent_0", "navigate_to_fixture", fixture_id="cabinet"),
            do("agent_0", "pick_up_object", object_id="plate", source_id="cabinet"),
            say("agent_1", "agent_0", "Let me know when the cabinet is clear."),
            wait("agent_1", "agent_0", "cabinet"),
            do("agent_0", "give_space", fixture_id="cabinet"),
            say("agent_0", "agent_1", "The cabinet is free now, go ahead."),
            do("agent_1", "navigate_to_fixture", fixture_id="cabinet"),
            do("agent_1", "pick_up_object", object_id="cup", source_id="cabinet"),
            say("agent_0", "agent_1", "Tell me when you are finished there."),
            wait("agent_0", "agent_1", "cabinet"),
            do("agent_1", "give_space", fixture_id="cabinet"),
            say("agent_1", "agent_0", "I am done at the cabinet, you can return."),
            do("agent_0", "navigate_to_fixture", fixture_id="cabinet"),
        ], "a fixture legitimately revisited after the other agent finished")

    # ---------------------------------------------------------------- 4

    def test_no_conflict_needs_nothing(self):
        """Two agents, two different objects, one roomy counter."""

        self._accepts([
            say("agent_0", "agent_1", "I will handle the bowl on the counter."),
            say("agent_1", "agent_0", "I will handle the plate at the same time."),
            do("agent_0", "pick_up_object", object_id="bowl", source_id="counter"),
            do("agent_1", "pick_up_object", object_id="plate", source_id="counter"),
        ], "no shared object and a roomy fixture: no protocol required")

    # ---------------------------------------------------------------- 5

    def test_promise_is_not_a_release(self):
        """agent_0 says it will yield, then keeps using the cabinet."""

        self._rejects([
            say("agent_0", "agent_1", "I am at the cabinet getting the plate."),
            say("agent_1", "agent_0", "I need the cabinet when you are done."),
            do("agent_0", "navigate_to_fixture", fixture_id="cabinet"),
            say("agent_1", "agent_0", "Tell me when the cabinet is free."),
            wait("agent_1", "agent_0", "cabinet"),
            say("agent_0", "agent_1", "I will give you space at the cabinet shortly."),
            do("agent_0", "pick_up_object", object_id="plate", source_id="cabinet"),
            do("agent_0", "pick_up_object", object_id="cup", source_id="cabinet"),
            do("agent_1", "navigate_to_fixture", fixture_id="cabinet"),
        ], "a promise followed by continued use is not a release")

    # ---------------------------------------------------------------- 6

    def test_unguarded_handoff_is_rejected(self):
        """agent_1 takes the bowl with no wait at all."""

        self._rejects([
            say("agent_0", "agent_1", "I am filling the bowl."),
            say("agent_1", "agent_0", "Understood."),
            do("agent_0", "pick_up_object", object_id="bowl", source_id="counter"),
            do("agent_0", "place_on_surface", object_id="bowl", support_id="counter"),
            do("agent_1", "pick_up_object", object_id="bowl", source_id="counter"),
        ], "an unguarded cross-agent handoff must be rejected")


class InsertionExampleTests(unittest.TestCase):
    """Phase-2 insertion must produce trajectories the FSM accepts.

    The case that broke it: a wait whose holder is still using the thing
    afterwards. Dropping the release straight after the wait produced exactly
    the promise scenario 5 rejects -- the release said "done" while the holder
    went on to use it twice more. The release has to land after the holder's
    LAST use and before the waiter needs it. This was 36 points of validity on
    real data (60% -> 96%), from one misplaced line.
    """


    def test_model_written_waits_are_discarded(self):
        """A wait on something its holder never holds must not survive.

        arrangetea/000000: agent_0 waited on the tray it was itself using.
        Nobody releases a resource they do not hold, so it blocks until the
        deadlock breaker fires. Placement is derivable, so phase 2 owns it and
        generated waits are dropped rather than repaired.
        """

        import random

        from robotalk.generation.insert_waits import insert

        steps = _numbered([
            say("agent_0", "agent_1", "I will put the bread in the bowl."),
            say("agent_1", "agent_0", "Understood."),
            do("agent_0", "pick_up_object", object_id="bread", source_id="counter"),
            # nonsense: agent_0 waits on the bowl it is using itself
            wait("agent_0", "agent_1", "bowl"),
            do("agent_0", "place_on_surface", object_id="bread", support_id="bowl"),
        ])
        out, _ = insert(steps, {"counter": {"fixture_type": "counter"}},
                        {"bowl", "bread"}, random.Random(0))
        spurious = [
            s for s in out
            if s["tool"] == "wait_for_signal"
            and s["agent"] == "agent_0"
            and (s.get("args") or {}).get("about") == "bowl"
        ]
        self.assertEqual(spurious, [], "a wait whose holder never holds the "
                                       "thing must be discarded, not kept")

    def test_nothing_is_inserted_after_the_last_action(self):
        """The FSM forbids work once the goal is reached.

        servemealjuice/000008 failed because a release landed past the final
        productive step, so a valid plan became post-goal work.
        """

        import random

        from robotalk.generation.insert_waits import insert

        steps = _numbered([
            say("agent_0", "agent_1", "Starting on the bowl."),
            say("agent_1", "agent_0", "Understood."),
            do("agent_0", "pick_up_object", object_id="bowl", source_id="counter"),
            do("agent_0", "place_on_surface", object_id="bowl", support_id="counter"),
            do("agent_1", "pick_up_object", object_id="bowl", source_id="counter"),
        ])
        out, _ = insert(steps, {"counter": {"fixture_type": "counter"}},
                        {"bowl"}, random.Random(0))
        last_action = max(
            i for i, s in enumerate(out)
            if s["tool"] not in ("communicate", "wait_for_signal")
        )
        self.assertEqual(
            last_action, len(out) - 1,
            "no step may be inserted after the final productive action",
        )


if __name__ == "__main__":
    unittest.main()
