"""Tests for structural wait discharge via communicate.releases."""

from __future__ import annotations

import os
import unittest
from pathlib import Path




class CommunicateArgumentTests(unittest.TestCase):
    """The FSM's handling of the releases argument."""

    def _validator(self):
        from robotalk.tasks.specs import load_task_spec
        from robotalk.tasks.specs.runtime import (
            build_task_definition_from_spec,
        )

        spec = load_task_spec("PrepareCoffee")
        return build_task_definition_from_spec(spec).validator_factory(None)

    def _step(self, **args):
        return {"step": 0, "agent": "agent_0", "tool": "communicate",
                "reasoning": "r", "args": {"to": "agent_1", "message": "done", **args}}

    def test_known_id_is_accepted_and_normalized_to_a_list(self):
        step = self._step(releases="mug")
        self._validator()._validate_communicate_step(step)
        self.assertEqual(step["args"]["releases"], ["mug"])

    def test_unknown_id_is_rejected(self):
        from robotalk.tasks.shared.errors import (
            CommunicationStepSemanticValidationError,
        )

        with self.assertRaises(CommunicationStepSemanticValidationError):
            self._validator()._validate_communicate_step(self._step(releases="teapot"))

    def test_message_without_releases_is_unchanged(self):
        step = self._step()
        self._validator()._validate_communicate_step(step)
        self.assertNotIn("releases", step["args"])

    def test_unsupported_argument_is_still_rejected(self):
        from robotalk.tasks.shared.errors import (
            CommunicationStepSemanticValidationError,
        )

        with self.assertRaises(CommunicationStepSemanticValidationError):
            self._validator()._validate_communicate_step(self._step(urgency="high"))


class InsertWaitsTests(unittest.TestCase):
    def test_every_derived_wait_gets_a_matching_structural_release(self):
        import glob
        import json
        import random
        from copy import deepcopy

        import robotalk.generation.insert_waits as insert_waits

        root = os.environ.get("ROBOTALK_DATA_ROOT", "data/robotalk_rendered")
        paths = sorted(glob.glob(root + "/*/traj_000000/original_trajectory.json"))
        if not paths:
            self.skipTest("trajectory subset not present")

        waits_seen = 0
        for path in paths[:20]:
            record = json.loads(Path(path).read_text())
            initial = record["initial_state"]
            steps, _ = insert_waits.insert(
                deepcopy(record["steps"]),
                initial.get("fixtures") or {},
                set(initial.get("objects") or {}),
                random.Random(0),
                {a: (s or {}).get("location")
                 for a, s in (initial.get("agents") or {}).items()},
            )
            released = set()
            for step in steps:
                value = (step.get("args") or {}).get("releases")
                if isinstance(value, str):
                    released.add(value)
                elif value:
                    released.update(value)
            for step in steps:
                if step["tool"] != "wait_for_signal":
                    continue
                waits_seen += 1
                self.assertIn(step["args"]["about"], released)
        self.assertGreater(waits_seen, 0)


if __name__ == "__main__":
    unittest.main()
