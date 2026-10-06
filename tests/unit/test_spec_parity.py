"""The generator and the evaluator must resolve the SAME tool set.

They used to diverge: build_task_definition_from_spec injected open_hinged_part
for fixtures with hinged parts, while task_registry read the raw declared spec.
That single divergence caused "Tool open_hinged_part is not allowed here",
KeyError: 'hinged', and the over-narrow give_space allowlist. The specs are now
materialised, so this test is what keeps them from drifting apart again.
"""
import unittest

from robotalk.tasks.specs import load_all_task_specs
from robotalk.tasks.specs.runtime import (
    build_task_definition_from_spec,
)
from robotalk.training.task_registry import (
    _camel_to_snake_case,
    get_task_metadata,
)


class SpecParity(unittest.TestCase):
    def test_generator_and_evaluator_agree(self):
        mismatches = []
        for spec in load_all_task_specs():
            validator = build_task_definition_from_spec(spec).validator_factory(None)
            generator = dict(getattr(validator, "allowed_tool_specs", {}) or {})
            evaluator = get_task_metadata(
                _camel_to_snake_case(spec.composite_task)
            ).allowed_tool_specs
            if set(generator) != set(evaluator):
                mismatches.append(
                    (spec.composite_task,
                     sorted(set(generator) - set(evaluator)),
                     sorted(set(evaluator) - set(generator)))
                )
        self.assertEqual(mismatches, [], f"tool sets drifted: {mismatches}")


if __name__ == "__main__":
    unittest.main()
