"""Tests for OccupancyGrid — unit tests (mock fixtures) and functional tests (real sim)."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock

import numpy as np

from robocasa.utils.occupancy_grid import (
    OccupancyGrid,
    _is_countertop_appliance,
    _is_small_reference_fixture,
)
from robocasa.utils.placement import (
    get_front_alignment_metrics,
    is_in_front_workspace_corridor,
)


# ---------------------------------------------------------------------------
# Helpers — lightweight mock fixtures
# ---------------------------------------------------------------------------

def _make_mock_fixture(
    pos: tuple[float, float, float],
    size: tuple[float, float, float] = (0.5, 0.5, 0.9),
    rot: float = 0.0,
    name: str = "fixture",
) -> MagicMock:
    """Create a mock Fixture with pos, rot, and get_ext_sites returning an AABB."""
    fxtr = MagicMock()
    fxtr.pos = np.array(pos, dtype=float)
    fxtr.rot = rot
    fxtr.name = name

    # Build 8 AABB corners (world frame, no rotation for simplicity)
    hx, hy, hz = size[0] / 2, size[1] / 2, size[2] / 2
    cx, cy, cz = pos
    corners = [
        np.array([cx - hx, cy - hy, cz - hz]),  # p0
        np.array([cx + hx, cy - hy, cz - hz]),  # px
        np.array([cx - hx, cy + hy, cz - hz]),  # py
        np.array([cx - hx, cy - hy, cz + hz]),  # pz
        np.array([cx - hx, cy + hy, cz + hz]),
        np.array([cx + hx, cy + hy, cz + hz]),
        np.array([cx + hx, cy + hy, cz - hz]),
        np.array([cx + hx, cy - hy, cz + hz]),
    ]
    fxtr.get_ext_sites = MagicMock(return_value=corners)
    return fxtr


# ---------------------------------------------------------------------------
# Unit tests (no sim needed)
# ---------------------------------------------------------------------------

class TestOccupancyGridUnit(unittest.TestCase):
    """Unit tests using mock fixtures."""

    def test_stovetop_uses_countertop_appliance_approach_semantics(self):
        stovetop = _make_mock_fixture((0.0, 0.0, 0.9), name="stovetop_left_group")
        self.assertTrue(_is_countertop_appliance(stovetop))

    def test_only_seat_references_use_small_fixture_lateral_extension(self):
        stool = _make_mock_fixture((0.0, 0.0, 0.0), name="stool_2_room")
        chair = _make_mock_fixture((0.0, 0.0, 0.0), name="dining_chair")
        sink = _make_mock_fixture((0.0, 0.0, 0.0), name="sink_island_group")

        self.assertTrue(_is_small_reference_fixture(stool))
        self.assertTrue(_is_small_reference_fixture(chair))
        self.assertFalse(_is_small_reference_fixture(sink))

    def test_grid_construction_bounds(self):
        """Grid should have correct dimensions for a simple layout."""
        f1 = _make_mock_fixture((0.0, 0.0, 0.0), size=(1.0, 1.0, 0.9))
        grid = OccupancyGrid({"f1": f1}, cell_size=0.50)

        # Fixture spans [-0.5, 0.5] in both X and Y
        # With 1.0m margin: [-1.5, 1.5] → 3.0m extent → 6 cells per axis
        self.assertGreaterEqual(grid._rows, 4)
        self.assertGreaterEqual(grid._cols, 4)

    def test_rasterization_marks_fixture_cells(self):
        """Cells overlapping a fixture footprint should be occupied."""
        f1 = _make_mock_fixture((0.0, 0.0, 0.0), size=(1.0, 1.0, 0.9))
        grid = OccupancyGrid({"f1": f1}, cell_size=0.50)

        # Center of fixture should be occupied
        self.assertFalse(grid.is_free(np.array([0.0, 0.0])))
        # Far away should be free
        self.assertTrue(grid.is_free(np.array([1.2, 1.2])))

    def test_find_placement_returns_adjacent_free_cell(self):
        """Placement should return a position adjacent to the fixture, not on it."""
        f1 = _make_mock_fixture((0.0, 0.0, 0.0), size=(1.0, 1.0, 0.9))
        grid = OccupancyGrid({"f1": f1}, cell_size=0.50)

        result = grid.find_placement(f1)
        self.assertIsNotNone(result)
        pos_xy, yaw = result

        # Position should be free
        self.assertTrue(grid.is_free(pos_xy))
        # Position should be near fixture (within ~1.5 cell sizes)
        dist = np.linalg.norm(pos_xy - np.array([0.0, 0.0]))
        self.assertLess(dist, 2.0)

    def test_parent_placement_avoids_exclusive_child_working_region(self):
        """A broad parent target must not borrow a child's front work pose."""
        parent = _make_mock_fixture(
            (0.0, 0.0, 0.0), size=(2.0, 0.5, 0.9), name="counter"
        )
        child = _make_mock_fixture(
            (0.0, 0.0, 0.9), size=(0.4, 0.3, 0.3), name="cabinet"
        )
        grid = OccupancyGrid({"counter": parent, "cabinet": child}, cell_size=0.05)

        result = grid.find_placement(
            parent,
            prohibited_working_fixtures=[child],
        )

        self.assertIsNotNone(result)
        pos_xy, _yaw = result
        metrics = get_front_alignment_metrics(child, pos_xy)
        self.assertFalse(
            metrics
            and metrics["on_front_face"]
            and metrics["within_span"]
            and metrics["front_gap"] <= 0.75
        )

    def test_explicit_child_target_can_use_its_working_region(self):
        """Reserved child poses remain legal when that child is the target."""
        child = _make_mock_fixture(
            (0.0, 0.0, 0.0), size=(0.4, 0.3, 0.9), name="cabinet"
        )
        grid = OccupancyGrid({"cabinet": child}, cell_size=0.05)

        result = grid.find_placement(child, require_front=True)

        self.assertIsNotNone(result)
        pos_xy, _yaw = result
        metrics = get_front_alignment_metrics(child, pos_xy)
        self.assertTrue(metrics and metrics["on_front_face"] and metrics["within_span"])

    def test_countertop_appliance_samples_across_reserved_corridor_width(self):
        """Explicit appliance navigation may use the free side of its corridor."""
        appliance = _make_mock_fixture(
            (0.0, 0.0, 0.0), size=(0.1, 0.1, 0.3), name="coffee_machine"
        )
        grid = OccupancyGrid({"coffee_machine": appliance}, cell_size=0.05)

        result = grid.find_placement(
            appliance,
            robot_positions=[np.array([-0.25, -0.45])],
        )

        self.assertIsNotNone(result)
        pos_xy, _yaw = result
        self.assertGreaterEqual(
            np.linalg.norm(pos_xy - np.array([-0.25, -0.45])),
            grid._MIN_ROBOT_SEPARATION,
        )
        self.assertGreater(pos_xy[0], 0.05)

    def test_front_corridor_does_not_reserve_side_or_back_poses(self):
        """The new directional rule must not recreate an all-sides radius."""
        parent = _make_mock_fixture(
            (0.0, 0.0, 0.0), size=(0.4, 0.4, 0.9), name="counter"
        )
        child = _make_mock_fixture(
            (0.0, 0.0, 0.9), size=(2.0, 2.0, 0.3), name="toaster_oven"
        )
        grid = OccupancyGrid({"counter": parent, "cabinet": child}, cell_size=0.05)

        result = grid.find_placement(
            parent,
            prohibited_working_fixtures=[child],
        )

        self.assertIsNotNone(result)
        pos_xy, _yaw = result
        self.assertFalse(is_in_front_workspace_corridor(child, pos_xy))

    def test_corner_counter_front_prefers_kitchen_side_face(self):
        """Corner counters should not use the modeled dining-side front first."""
        corner = _make_mock_fixture(
            (0.325, -0.325, 0.0),
            size=(0.65, 0.60, 0.9),
            name="counter_corner_main_group",
        )
        grid = OccupancyGrid({"counter_corner": corner}, cell_size=0.05)

        result = grid.find_placement(corner, require_front=True)

        self.assertIsNotNone(result)
        pos_xy, _yaw = result
        self.assertLess(pos_xy[1], -0.60)

    def test_corner_counter_surface_approach_uses_kitchen_side_face(self):
        """Surface placement around corner counters should not stand through the wall side."""
        corner = _make_mock_fixture(
            (0.325, -0.325, 0.0),
            size=(0.65, 0.60, 0.9),
            name="counter_corner_main_group",
        )
        grid = OccupancyGrid({"counter_corner": corner}, cell_size=0.05)

        result = grid.find_placement(corner, require_front=False)

        self.assertIsNotNone(result)
        pos_xy, _yaw = result
        self.assertLess(pos_xy[1], -0.60)

    def test_multi_robot_exclusion(self):
        """Second robot should get a different cell than the first."""
        f1 = _make_mock_fixture((0.0, 0.0, 0.0), size=(1.0, 1.0, 0.9))
        grid = OccupancyGrid({"f1": f1}, cell_size=0.50)

        result1 = grid.find_placement(f1, robot_cells=[])
        self.assertIsNotNone(result1)
        pos1, _ = result1

        # Mark first robot's cell
        cell1 = grid._world_to_grid(pos1)
        result2 = grid.find_placement(f1, robot_cells=[cell1])
        self.assertIsNotNone(result2)
        pos2, _ = result2

        # Positions should differ
        self.assertFalse(np.allclose(pos1, pos2, atol=0.01),
                         f"Both robots got same position: {pos1}")

    def test_corner_fixture_only_open_side(self):
        """A fixture surrounded by walls on 3 sides should only have candidates on the open side."""
        # Place a fixture at origin, with walls on 3 sides (left, right, back)
        fixtures = {
            "target": _make_mock_fixture((0.0, 0.0, 0.0), size=(0.4, 0.4, 0.9)),
            "block_left": _make_mock_fixture((-0.5, 0.0, 0.0), size=(0.4, 0.4, 0.9)),
            "block_right": _make_mock_fixture((0.5, 0.0, 0.0), size=(0.4, 0.4, 0.9)),
            "block_back": _make_mock_fixture((0.0, -0.5, 0.0), size=(0.4, 0.4, 0.9)),
        }
        grid = OccupancyGrid(fixtures, cell_size=0.50)

        result = grid.find_placement(fixtures["target"])
        self.assertIsNotNone(result)
        pos_xy, _ = result

        # Robot should be placed in the open direction (positive Y)
        self.assertGreater(pos_xy[1], -0.1,
                           f"Robot placed behind blocked fixture: y={pos_xy[1]}")

    def test_large_fixture_candidates_surround_footprint(self):
        """A large fixture spanning multiple cells should have candidates around its full extent."""
        # Long counter: 2.0m x 0.5m
        f1 = _make_mock_fixture((0.0, 0.0, 0.0), size=(2.0, 0.5, 0.9))
        grid = OccupancyGrid({"f1": f1}, cell_size=0.50)

        fixture_cells = grid._get_fixture_cells(f1)
        # Should span multiple cells in X
        cols = {c for _, c in fixture_cells}
        self.assertGreater(len(cols), 1, "Large fixture should span multiple columns")

        # Placement should work
        result = grid.find_placement(f1)
        self.assertIsNotNone(result)

    def test_enclosed_corner_requires_two_walls_and_two_counters(self):
        """A true room-corner trap with two walls and two counters is rejected."""
        fixtures = {
            "wall_pos_x": _make_mock_fixture((0.55, 0.0, 0.0), size=(0.10, 1.20, 0.9)),
            "wall_pos_y": _make_mock_fixture((0.0, 0.55, 0.0), size=(1.20, 0.10, 0.9)),
            "counter_neg_x": _make_mock_fixture((-0.55, 0.0, 0.0), size=(0.10, 1.20, 0.9)),
            "counter_neg_y": _make_mock_fixture((0.0, -0.55, 0.0), size=(1.20, 0.10, 0.9)),
        }
        grid = OccupancyGrid(fixtures, cell_size=0.05)

        self.assertTrue(grid._is_enclosed(np.array([0.0, 0.0], dtype=float)))

    def test_enclosed_corner_rejects_entire_bounded_region(self):
        """The full wall-and-structure corner region should be non-standable."""
        fixtures = {
            "wall_pos_x": _make_mock_fixture((0.60, 0.0, 0.0), size=(0.10, 1.60, 0.9)),
            "wall_pos_y": _make_mock_fixture((0.0, 0.60, 0.0), size=(1.60, 0.10, 0.9)),
            "counter_neg_x": _make_mock_fixture((-0.40, 0.0, 0.0), size=(0.10, 1.60, 0.9)),
            "counter_neg_y": _make_mock_fixture((0.0, -0.40, 0.0), size=(1.60, 0.10, 0.9)),
        }
        grid = OccupancyGrid(fixtures, cell_size=0.05)

        self.assertTrue(grid._is_enclosed(np.array([0.45, 0.45], dtype=float)))
        self.assertTrue(grid._is_enclosed(np.array([0.45, -0.25], dtype=float)))
        self.assertTrue(grid._is_enclosed(np.array([-0.25, 0.45], dtype=float)))
        self.assertTrue(grid._is_enclosed(np.array([-0.25, -0.25], dtype=float)))

    def test_enclosed_corner_bridges_single_cell_gap(self):
        """A one-cell overlap gap between blocking structures should still fill the pocket."""
        fixtures = {
            "wall_pos_x": _make_mock_fixture((0.60, 0.0, 0.0), size=(0.10, 1.60, 0.9)),
            "wall_pos_y": _make_mock_fixture((0.0, 0.60, 0.0), size=(1.60, 0.10, 0.9)),
            "counter_neg_x": _make_mock_fixture((-0.40, -0.30, 0.0), size=(0.10, 0.60, 0.9)),
            "counter_neg_y": _make_mock_fixture((0.0, -0.40, 0.0), size=(1.60, 0.10, 0.9)),
        }
        grid = OccupancyGrid(fixtures, cell_size=0.05)

        self.assertTrue(grid._is_enclosed(np.array([0.20, -0.35], dtype=float)))

    def test_l_corner_without_opposite_counter_pair_is_not_enclosed(self):
        """A simple wall-plus-counter L-shape should remain standable."""
        fixtures = {
            "wall_pos_x": _make_mock_fixture((0.55, 0.0, 0.0), size=(0.10, 1.20, 0.9)),
            "wall_pos_y": _make_mock_fixture((0.0, 0.55, 0.0), size=(1.20, 0.10, 0.9)),
            "counter_neg_x": _make_mock_fixture((-0.55, 0.0, 0.0), size=(0.10, 1.20, 0.9)),
        }
        grid = OccupancyGrid(fixtures, cell_size=0.05)

        self.assertFalse(grid._is_enclosed(np.array([0.0, 0.0], dtype=float)))

    def test_occupy_and_release(self):
        """occupy() and release() should toggle cell state."""
        f1 = _make_mock_fixture((0.0, 0.0, 0.0), size=(0.4, 0.4, 0.9))
        grid = OccupancyGrid({"f1": f1}, cell_size=0.50)

        free_pos = np.array([1.0, 1.0])
        self.assertTrue(grid.is_free(free_pos))

        grid.occupy(free_pos)
        self.assertFalse(grid.is_free(free_pos))

        grid.release(free_pos)
        self.assertTrue(grid.is_free(free_pos))

    def test_mark_world_aabb_occupied_blocks_then_restores_free_space(self):
        """Transient AABBs should participate in occupancy until cleared."""
        f1 = _make_mock_fixture((0.0, 0.0, 0.0), size=(0.4, 0.4, 0.9))
        grid = OccupancyGrid({"f1": f1}, cell_size=0.10)
        free_pos = np.array([0.55, 0.0], dtype=float)

        self.assertTrue(grid.is_free(free_pos))

        obstacle_id = grid.mark_world_aabb_occupied(
            (np.array([0.45, -0.05]), np.array([0.65, 0.05])),
            obstacle_id="drawer_sweep",
        )

        self.assertEqual(obstacle_id, "drawer_sweep")
        self.assertFalse(grid.is_free(free_pos))

        grid.clear_world_aabb_occupied("drawer_sweep")
        self.assertTrue(grid.is_free(free_pos))

    def test_corner_cabinet_can_be_included_as_obstacle(self):
        """Corner cabinets should be optional ground obstacles for robot routing."""
        corner_cab = _make_mock_fixture((0.0, 0.0, 0.0), size=(0.6, 0.6, 0.9), name="corner_cab")
        corner_cab.is_corner_cab = True

        excluded_grid = OccupancyGrid(
            {"corner_cab": corner_cab},
            cell_size=0.10,
            include_corner_cabinets=False,
        )
        included_grid = OccupancyGrid(
            {"corner_cab": corner_cab},
            cell_size=0.10,
            include_corner_cabinets=True,
        )

        self.assertTrue(excluded_grid.is_free(np.array([0.0, 0.0], dtype=float)))
        self.assertFalse(included_grid.is_free(np.array([0.0, 0.0], dtype=float)))

    def test_ref_object_pos_biases_placement(self):
        """When ref_object_pos is given, placement should prefer the closer side."""
        # Long counter
        f1 = _make_mock_fixture((0.0, 0.0, 0.0), size=(2.0, 0.5, 0.9))
        grid = OccupancyGrid({"f1": f1}, cell_size=0.50)

        # Object on the left side
        result_left = grid.find_placement(f1, ref_object_pos=np.array([-0.8, 0.5]))
        self.assertIsNotNone(result_left)
        pos_left, _ = result_left

        # Object on the right side
        result_right = grid.find_placement(f1, ref_object_pos=np.array([0.8, 0.5]))
        self.assertIsNotNone(result_right)
        pos_right, _ = result_right

        # Left-biased placement should be further left than right-biased
        self.assertLess(pos_left[0], pos_right[0],
                        f"Left ref should produce leftward placement: {pos_left[0]} vs {pos_right[0]}")

    def test_front_required_ignores_ref_object_pos_bias(self):
        """Front-required placement should honor an explicit front working target."""
        target = _make_mock_fixture(
            (0.0, 0.0, 0.0),
            size=(2.0, 0.5, 0.9),
            rot=-np.pi / 2,
            name="fridge",
        )
        grid = OccupancyGrid({"target": target}, cell_size=0.10)

        result_left = grid.find_placement(
            target,
            ref_object_pos=np.array([-0.8, 0.6]),
            require_front=True,
        )
        result_right = grid.find_placement(
            target,
            ref_object_pos=np.array([0.8, 0.6]),
            require_front=True,
        )

        self.assertIsNotNone(result_left)
        self.assertIsNotNone(result_right)
        pos_left, _ = result_left
        pos_right, _ = result_right

        self.assertLess(
            pos_left[0],
            pos_right[0],
            f"Front target should bias front placement laterally: {pos_left} vs {pos_right}",
        )
        self.assertAlmostEqual(pos_left[0], -0.8, delta=0.12)
        self.assertAlmostEqual(pos_right[0], 0.8, delta=0.12)
        self.assertGreater(pos_left[1], 0.25)
        self.assertGreater(pos_right[1], 0.25)

    def test_front_required_fallback_stays_on_front_face(self):
        """Blocked front-center should still keep the robot on the front face."""
        target = _make_mock_fixture(
            (0.0, 0.0, 0.0),
            size=(2.0, 0.5, 0.9),
            rot=-np.pi / 2,
            name="fridge",
        )
        blocker = _make_mock_fixture(
            (0.0, 0.65, 0.0),
            size=(0.08, 0.4, 0.9),
            name="blocker",
        )
        grid = OccupancyGrid({"target": target, "blocker": blocker}, cell_size=0.10)

        result = grid.find_placement(target, require_front=True)

        self.assertIsNotNone(result)
        pos_xy, _ = result
        self.assertGreater(pos_xy[1], 0.25, f"Expected front-face placement, got {pos_xy}")
        self.assertGreaterEqual(pos_xy[0], -1.05, f"Expected lateral front offset, got {pos_xy}")
        self.assertLessEqual(pos_xy[0], 1.05, f"Expected lateral front offset, got {pos_xy}")

    def test_front_required_does_not_fall_back_to_side_face(self):
        """If the entire front face is blocked, front-required placement should fail."""
        target = _make_mock_fixture(
            (0.0, 0.0, 0.0),
            size=(2.0, 0.5, 0.9),
            rot=-np.pi / 2,
            name="fridge",
        )
        blocker = _make_mock_fixture(
            (0.0, 0.50, 0.0),
            size=(2.6, 0.40, 0.9),
            name="front_blocker",
        )
        grid = OccupancyGrid({"target": target, "blocker": blocker}, cell_size=0.10)

        result = grid.find_placement(target, require_front=True)

        self.assertIsNone(result, "Front-required placement should not fall back to a side face")

    def test_front_required_clamps_away_from_side_edge(self):
        """Front-required grid placement should reject edge-hugging corner approaches."""
        target = _make_mock_fixture(
            (0.0, 0.0, 0.0),
            size=(2.0, 0.5, 0.9),
            rot=-np.pi / 2,
            name="fridge",
        )
        grid = OccupancyGrid({"target": target}, cell_size=0.10)

        result = grid.find_placement(
            target,
            ref_object_pos=np.array([1.2, 0.6]),
            require_front=True,
        )

        self.assertIsNotNone(result)
        pos_xy, _ = result
        self.assertLessEqual(pos_xy[0], 0.90, f"Expected side filtering to reject edge pose, got {pos_xy}")
        self.assertGreater(pos_xy[1], 0.25, f"Expected front-face placement, got {pos_xy}")

    def test_front_required_uses_target_side_over_fixture_rot(self):
        """A front target should override an incorrect rot-derived front face."""
        target = _make_mock_fixture(
            (0.0, 0.0, 0.0),
            size=(2.0, 0.5, 0.9),
            rot=0.0,
            name="fridge",
        )
        grid = OccupancyGrid({"target": target}, cell_size=0.10)

        result = grid.find_placement(
            target,
            ref_object_pos=np.array([0.0, 0.7]),
            require_front=True,
        )

        self.assertIsNotNone(result)
        pos_xy, _ = result
        self.assertGreater(pos_xy[1], 0.25, f"Expected target-inferred front face, got {pos_xy}")
        self.assertGreater(abs(pos_xy[1]), abs(pos_xy[0]), f"Expected front along +Y, got {pos_xy}")

    def test_three_sided_wall_pocket_is_not_a_counter_wall_corner_trap(self):
        """Wall-only pockets should not trigger the counter-wall corner heuristic."""
        fixtures = {
            "wall_left": _make_mock_fixture((-0.25, 0.0, 0.0), size=(0.10, 0.80, 0.9)),
            "wall_right": _make_mock_fixture((0.25, 0.0, 0.0), size=(0.10, 0.80, 0.9)),
            "wall_front": _make_mock_fixture((0.0, 0.25, 0.0), size=(0.80, 0.10, 0.9)),
        }
        grid = OccupancyGrid(fixtures, cell_size=0.05)
        pocket = np.array([0.0, 0.0], dtype=float)

        self.assertFalse(grid._is_enclosed(pocket))

    def test_open_gap_is_not_treated_as_corner_pocket(self):
        """Two nearby obstacles should not block an otherwise reachable stance cell."""
        fixtures = {
            "wall_left": _make_mock_fixture((-0.25, 0.0, 0.0), size=(0.10, 0.80, 0.9)),
            "wall_right": _make_mock_fixture((0.25, 0.0, 0.0), size=(0.10, 0.80, 0.9)),
        }
        grid = OccupancyGrid(fixtures, cell_size=0.05)
        gap = np.array([0.0, 0.0], dtype=float)

        self.assertFalse(grid._is_enclosed(gap))

    def test_yaw_faces_fixture(self):
        """Robot yaw should point toward the fixture center."""
        f1 = _make_mock_fixture((0.0, 0.0, 0.0), size=(0.4, 0.4, 0.9))
        grid = OccupancyGrid({"f1": f1}, cell_size=0.50)

        result = grid.find_placement(f1)
        self.assertIsNotNone(result)
        pos_xy, yaw = result

        # Vector from robot to fixture center
        to_fixture = np.array([0.0, 0.0]) - pos_xy
        expected_yaw = np.arctan2(to_fixture[1], to_fixture[0])

        # Yaw should be close to expected (within ~45 degrees for diagonal snap)
        angle_diff = abs((yaw - expected_yaw + np.pi) % (2 * np.pi) - np.pi)
        self.assertLess(angle_diff, np.pi / 4 + 0.1,
                        f"Yaw {yaw:.2f} too far from expected {expected_yaw:.2f}")

    def test_no_placement_outside_bounds(self):
        """Grid boundary cells should be clamped; no placement outside world bounds."""
        f1 = _make_mock_fixture((0.0, 0.0, 0.0), size=(0.4, 0.4, 0.9))
        grid = OccupancyGrid({"f1": f1}, cell_size=0.50)

        result = grid.find_placement(f1)
        self.assertIsNotNone(result)
        pos_xy, _ = result

        # Should be within grid bounds (origin to origin + extent)
        extent_x = grid._cols * grid.cell_size
        extent_y = grid._rows * grid.cell_size
        self.assertGreaterEqual(pos_xy[0], grid._origin[0])
        self.assertLessEqual(pos_xy[0], grid._origin[0] + extent_x)
        self.assertGreaterEqual(pos_xy[1], grid._origin[1])
        self.assertLessEqual(pos_xy[1], grid._origin[1] + extent_y)

    def test_2ring_fallback(self):
        """When all 1-ring cells are occupied, 2-ring should be used."""
        f1 = _make_mock_fixture((0.0, 0.0, 0.0), size=(0.4, 0.4, 0.9))
        # Surround with fixtures to block 1-ring
        fixtures = {"target": f1}
        for i, (dx, dy) in enumerate([
            (-0.5, -0.5), (0.0, -0.5), (0.5, -0.5),
            (-0.5, 0.0),               (0.5, 0.0),
            (-0.5, 0.5),  (0.0, 0.5),  (0.5, 0.5),
        ]):
            fixtures[f"block_{i}"] = _make_mock_fixture((dx, dy, 0.0), size=(0.4, 0.4, 0.9))

        grid = OccupancyGrid(fixtures, cell_size=0.50)
        result = grid.find_placement(f1)
        # Should still find something in 2-ring
        self.assertIsNotNone(result)


# ---------------------------------------------------------------------------
# Functional tests (require sim) — run with full environment
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    unittest.main()
