from __future__ import annotations

import os
from pathlib import Path
import shutil
import tempfile
import unittest
import unittest.mock
from types import SimpleNamespace

from robotalk.training.dataset import (
    build_same_task_trajectory_split,
    estimate_example_length,
)


class SameTaskTrajectorySplitTests(unittest.TestCase):
    @unittest.mock.patch(
        "robotalk.training.dataset.list_task_trajectory_ids",
        return_value=[f"traj_{index:06d}" for index in range(100)],
    )
    def test_seeded_split_is_reproducible_and_not_a_suffix(self, _mock_list):
        kwargs = dict(
            dataset_root=Path("/unused"),
            train_task_names=["task_a"],
            validation_task_names=["task_a"],
            validation_fraction=0.1,
            min_validation_trajectories_per_task=1,
        )
        first = build_same_task_trajectory_split(**kwargs, seed=42)
        repeated = build_same_task_trajectory_split(**kwargs, seed=42)
        changed = build_same_task_trajectory_split(**kwargs, seed=43)
        held_out = first.validation_trajectory_ids_by_task["task_a"]
        self.assertEqual(first, repeated)
        self.assertNotEqual(first, changed)
        self.assertEqual(len(held_out), 10)
        self.assertNotEqual(held_out, [f"traj_{index:06d}" for index in range(90, 100)])


# A real rendered trajectory, vendored into the repo. These tests used to copy
# from a generated dataset directory (data/image/<timestamp>/), which was deleted
# when that vintage was superseded -- every test here then failed on a missing
# file rather than on anything about the code. The fixture is checked in so the
# suite does not depend on which corpus happens to be on disk.
#
# It is a --step-order concurrent render, so plan.json is a genuine permutation
# of the trajectory's steps and these tests exercise the reordered path.
FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "data" / "render_fixture"
FIXTURE_TASK = "add_lemon_to_fish"
FIXTURE_TRAJECTORY_ID = "traj_000000"
SOURCE_TRAJECTORY_DIR = FIXTURE_ROOT / FIXTURE_TASK / FIXTURE_TRAJECTORY_ID


class AgentTurnExampleTests(unittest.TestCase):
    def setUp(self) -> None:
        # The fixture's image paths point at the render that produced it, which
        # is not part of the repo. These tests are about example construction,
        # not about pixels being on disk.
        patcher = unittest.mock.patch.dict(
            os.environ, {"ROBOCASA_VALIDATE_IMAGE_PATHS": "false"}
        )
        patcher.start()
        self.addCleanup(patcher.stop)

    def _build_single_trajectory_dataset_root(self) -> Path:
        temp_root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        trajectory_dir = temp_root / FIXTURE_TASK / FIXTURE_TRAJECTORY_ID
        trajectory_dir.mkdir(parents=True, exist_ok=True)
        for file_name in ("original_trajectory.json", "plan.json", "metadata.json"):
            shutil.copy2(SOURCE_TRAJECTORY_DIR / file_name, trajectory_dir / file_name)
        return temp_root


    def test_length_estimate_accounts_for_images_and_respects_image_cap(self):
        example = SimpleNamespace(
            messages=[
                {"role": "user", "content": [{"type": "text", "text": "move"}]}
            ],
            tool_schemas=[{"type": "function", "function": {"name": "move"}}],
            image_paths=["one.png", "two.png", "three.png"],
        )

        uncapped = estimate_example_length(example)
        capped = estimate_example_length(
            example,
            max_images_per_sample=1,
        )

        self.assertGreater(uncapped, capped)
        self.assertEqual(
            uncapped - capped,
            256 * (len(example.image_paths) - 1),
        )


if __name__ == "__main__":
    unittest.main()
