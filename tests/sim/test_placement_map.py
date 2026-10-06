from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from robocasa.utils import placement_map as pm


class _FakeAxis:
    def __init__(self):
        self.patches = []
        self.plots = []
        self.texts = []
        self.images = []
        self.title = None

    def add_patch(self, patch):
        self.patches.append(patch)

    def imshow(self, image, **kwargs):
        self.images.append((np.asarray(image).copy(), kwargs))

    def plot(self, *args, **kwargs):
        self.plots.append((args, kwargs))

    def text(self, x, y, label, **kwargs):
        self.texts.append((x, y, label, kwargs))
        return SimpleNamespace()

    def set_xlim(self, *_args):
        pass

    def set_ylim(self, *_args):
        pass

    def set_aspect(self, *_args):
        pass

    def set_xlabel(self, *_args):
        pass

    def set_ylabel(self, *_args):
        pass

    def set_title(self, title):
        self.title = title


class PlacementMapTests(unittest.TestCase):
    def test_draw_robots_uses_agent_labels(self):
        ax = _FakeAxis()
        runner = SimpleNamespace(
            _num_robots=2,
            _get_robot_position=lambda ridx: np.array([float(ridx), float(ridx), 0.0]),
        )

        pm._draw_robots(ax, runner)

        self.assertEqual(
            [label for _, _, label, _ in ax.texts],
            ["agent 0", "agent 1"],
        )
    def test_raster_grid_preserves_cell_classes_in_one_artist(self):
        grid = SimpleNamespace(
            _rows=2,
            _cols=2,
            _origin=np.array([0.0, 0.0]),
            cell_size=1.0,
            _grid=np.array([[False, True], [False, False]], dtype=bool),
            _grid_to_world=lambda r, c: np.array([c + 0.5, r + 0.5]),
            is_standable=lambda xy: not np.allclose(xy, [0.5, 1.5]),
        )
        runner = SimpleNamespace(
            _occupancy_grid=grid,
            _fixtures={},
            _num_robots=0,
            env=SimpleNamespace(objects={}),
        )

        raster_ax = _FakeAxis()
        pm.draw_grid_map(raster_ax, runner, grid_renderer="raster")
        self.assertEqual(len(raster_ax.images), 1)
        self.assertEqual(len(raster_ax.patches), 0)
        np.testing.assert_array_equal(
            raster_ax.images[0][0],
            np.array([[0, 2], [1, 0]], dtype=np.uint8),
        )
        self.assertIn("occ=1, enclosed=1, free=2", raster_ax.title)

        legacy_ax = _FakeAxis()
        pm.draw_grid_map(legacy_ax, runner)
        self.assertEqual(len(legacy_ax.images), 0)
        self.assertEqual(len(legacy_ax.patches), 4)

        with self.assertRaisesRegex(ValueError, "grid_renderer"):
            pm.draw_grid_map(_FakeAxis(), runner, grid_renderer="unknown")

    def test_draw_fixtures_keeps_full_names_and_offsets_overlapping_labels(self):
        ax = _FakeAxis()
        fixture_aabbs = {
            "counter_main_group": (
                np.array([0.0, 0.0, 0.0]),
                np.array([1.0, 1.0, 0.0]),
            ),
            "counter_main_group_2": (
                np.array([0.15, 0.05, 0.0]),
                np.array([1.15, 1.05, 0.0]),
            ),
        }
        fixtures = {name: object() for name in fixture_aabbs}

        with (
            patch.object(pm, "get_fixture_aabb", side_effect=lambda fixture: fixture_aabbs[next(
                name for name, candidate in fixtures.items() if candidate is fixture
            )]),
            patch.object(pm, "is_ground_obstacle", return_value=False),
        ):
            pm._draw_fixtures(ax, fixtures, label_fontsize=5)

        self.assertEqual(len(ax.texts), 2)
        labels = [label for _, _, label, _ in ax.texts]
        self.assertIn("counter main group", labels)
        self.assertIn("counter main group 2", labels)

        first_pos = np.array(ax.texts[0][:2], dtype=float)
        second_pos = np.array(ax.texts[1][:2], dtype=float)
        self.assertFalse(np.allclose(first_pos, second_pos))


if __name__ == "__main__":
    unittest.main()
