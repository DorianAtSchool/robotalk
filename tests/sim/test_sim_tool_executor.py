import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import robosuite.utils.transform_utils as T

import robocasa.utils.object_utils as OU
import robocasa.utils.trajectory_runner as trajectory_runner_module
from robocasa.utils.trajectory_runner import TrajectoryRunner  # noqa: E402
from robocasa.utils.sim_tool_executor import SimToolExecutor  # noqa: E402


def _objects_intersect(executor: SimToolExecutor, object_a: str, object_b: str) -> bool:
    obj_a = executor._require_object(object_a)
    obj_b = executor._require_object(object_b)
    pos_a, quat_a_wxyz = executor._get_object_pose(object_a)
    pos_b, quat_b_wxyz = executor._get_object_pose(object_b)
    return OU.objs_intersect(
        obj_a,
        pos_a,
        T.convert_quat(quat_a_wxyz, to="xyzw"),
        obj_b,
        pos_b,
        T.convert_quat(quat_b_wxyz, to="xyzw"),
    )


class TestSimToolExecutor(unittest.TestCase):
    def test_communicate_accepts_protocol_metadata_without_splitting_release(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        result = executor.communicate(
            to="agent_1",
            message="The fridge is free.",
            releases="fridge",
            coordination_phase="portion_complete",
        )
        self.assertEqual(result.details["releases"], "fridge")
        self.assertEqual(result.details["coordination_phase"], "portion_complete")

    def test_non_mutating_dispatch_preserves_visual_caches(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._camera_frame_cache = {"top_view": np.ones((1, 1, 3))}
        executor._placement_map_cache = {(True, "jpg", 300): b"map"}
        executor.communicate = MagicMock(
            return_value=SimpleNamespace(tool_name="communicate", success=True)
        )
        executor.wait = MagicMock(
            return_value=SimpleNamespace(tool_name="wait", success=True)
        )

        executor.execute("communicate", to="agent_1", message="hello")
        executor.execute("wait", robot_idx=0)

        self.assertIn("top_view", executor._camera_frame_cache)
        self.assertIn((True, "jpg", 300), executor._placement_map_cache)

    def test_mutating_dispatch_invalidates_visual_caches(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._camera_frame_cache = {"top_view": np.ones((1, 1, 3))}
        executor._placement_map_cache = {(True, "jpg", 300): b"map"}
        executor.navigate_to_fixture = MagicMock(
            return_value=SimpleNamespace(
                tool_name="navigate_to_fixture", success=True
            )
        )

        executor.execute(
            "navigate_to_fixture", robot_idx=0, fixture_id="counter"
        )

        self.assertEqual(executor._camera_frame_cache, {})
        self.assertEqual(executor._placement_map_cache, {})

    def test_map_cache_key_includes_configured_dpi_and_renderer(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._camera_frame_cache = {}
        executor._placement_map_cache = {}
        executor._map_dpi = 300
        executor._map_renderer = "legacy"
        executor._render_map_image_bytes = MagicMock(
            side_effect=[b"dpi-300", b"dpi-60", b"raster-60"]
        )

        with tempfile.TemporaryDirectory() as output_dir:
            first = executor._save_map_image(Path(output_dir) / "first.jpg")
            second = executor._save_map_image(Path(output_dir) / "second.jpg")
            self.assertEqual(first.read_bytes(), second.read_bytes())
            self.assertEqual(executor._render_map_image_bytes.call_count, 1)

            executor._map_dpi = 60
            low_dpi = executor._save_map_image(Path(output_dir) / "low.jpg")
            self.assertEqual(low_dpi.read_bytes(), b"dpi-60")

            executor._map_renderer = "raster"
            raster = executor._save_map_image(Path(output_dir) / "raster.jpg")
            self.assertEqual(raster.read_bytes(), b"raster-60")

        self.assertEqual(executor._render_map_image_bytes.call_count, 3)
        self.assertEqual(
            executor._render_map_image_bytes.call_args.kwargs,
            {
                "clean_labels": True,
                "image_format": "jpg",
                "map_dpi": 60,
                "map_renderer": "raster",
            },
        )

    def test_configure_mujoco_gl_backend_updates_cached_binding_choice(self):
        original_gl = os.environ.get("MUJOCO_GL")
        original_egl_device = os.environ.get("MUJOCO_EGL_DEVICE_ID")
        fake_binding_utils = SimpleNamespace(_MUJOCO_GL="egl")
        observed_gl = None
        observed_egl_device = None

        try:
            os.environ["MUJOCO_GL"] = "egl"
            os.environ["MUJOCO_EGL_DEVICE_ID"] = "7"
            with patch.dict(
                sys.modules,
                {"robosuite.utils.binding_utils": fake_binding_utils},
            ):
                configured_backend = (
                    trajectory_runner_module._configure_mujoco_gl_backend("osmesa")
                )
                observed_gl = os.environ.get("MUJOCO_GL")
                observed_egl_device = os.environ.get("MUJOCO_EGL_DEVICE_ID")
        finally:
            if original_gl is None:
                os.environ.pop("MUJOCO_GL", None)
            else:
                os.environ["MUJOCO_GL"] = original_gl
            if original_egl_device is None:
                os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)
            else:
                os.environ["MUJOCO_EGL_DEVICE_ID"] = original_egl_device

        self.assertEqual(configured_backend, "osmesa")
        self.assertEqual(observed_gl, "osmesa")
        self.assertIsNone(observed_egl_device)
        self.assertEqual(fake_binding_utils._MUJOCO_GL, "osmesa")

    def test_restore_baseline_state_resets_runtime_and_model_state(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        fixture = SimpleNamespace(_turned_on=True, _num_steps_on=3)
        model = SimpleNamespace(
            site_rgba=np.ones((2, 4)),
            site_size=np.ones((2, 3)),
            geom_rgba=np.ones((3, 4)),
        )
        sim = MagicMock()
        sim.model = model
        env = SimpleNamespace(
            sim=sim,
            update_sites=MagicMock(),
            update_state=MagicMock(),
        )
        runner = SimpleNamespace(
            _fixtures={"coffee_machine": fixture},
            _object_locations={"mug": "counter_mutated"},
            _scene={"objects": {"mug": {"location": "counter_mutated"}}},
        )

        executor.env = env
        executor.runner = runner
        executor._held_objects = {0: "mug"}
        executor._baseline_sim_state = np.array([1.0, 2.0, 3.0])
        executor._baseline_model_site_rgba = np.zeros((2, 4))
        executor._baseline_model_site_size = np.full((2, 3), 0.5)
        executor._baseline_model_geom_rgba = np.full((3, 4), 0.25)
        executor._baseline_fixture_runtime_state = {
            "coffee_machine": {"_turned_on": False, "_num_steps_on": 0}
        }
        executor._baseline_object_locations = {"mug": "counter_clean"}
        executor._baseline_scene = {"objects": {"mug": {"location": "counter_clean"}}}

        executor.restore_baseline_state()

        sim.set_state_from_flattened.assert_called_once()
        np.testing.assert_array_equal(
            sim.set_state_from_flattened.call_args.args[0],
            np.array([1.0, 2.0, 3.0]),
        )
        self.assertEqual(sim.forward.call_count, 2)
        env.update_sites.assert_called_once()
        env.update_state.assert_called_once()
        self.assertEqual(executor._held_objects, {})
        self.assertFalse(fixture._turned_on)
        self.assertEqual(fixture._num_steps_on, 0)
        self.assertEqual(runner._object_locations["mug"], "counter_clean")
        self.assertEqual(
            runner._scene["objects"]["mug"]["location"],
            "counter_clean",
        )
        np.testing.assert_array_equal(model.site_rgba, np.zeros((2, 4)))
        np.testing.assert_array_equal(model.site_size, np.full((2, 3), 0.5))
        np.testing.assert_array_equal(model.geom_rgba, np.full((3, 4), 0.25))


class TestFrontRetryUnit(unittest.TestCase):
    @patch("robocasa.utils.sim_tool_executor._is_approach_center", return_value=True)
    def test_pickup_from_countertop_appliance_matches_non_front_navigation(
        self, _mock_is_approach_center
    ):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor.env = SimpleNamespace(objects={"pan": object()})
        executor.runner = MagicMock()
        executor.runner._fixtures = {"stove": object()}
        executor._require_object = MagicMock()
        executor._get_scene_object_location = MagicMock(return_value="stove")
        executor._resolve_pick_source_target = MagicMock(
            return_value=("stove", None, None)
        )
        executor._held_by_robot = MagicMock(return_value=None)
        executor._fixture_is_drawer = MagicMock(return_value=False)
        executor._fixture_requires_front_approach = MagicMock(return_value=False)
        executor._surface_fixture_prefers_front_approach = MagicMock(
            return_value=False
        )
        executor._robot_near_fixture = MagicMock(return_value=False)
        executor._move_robot_near_fixture_with_retries = MagicMock(
            return_value=True
        )
        executor._sync_held_object = MagicMock()
        executor._get_object_pose = MagicMock(
            return_value=(
                np.zeros(3, dtype=float),
                np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            )
        )
        executor._held_pose_offset = MagicMock(return_value=np.zeros(3))
        executor._set_support_parent = MagicMock()
        executor._held_objects = {}
        executor._held_object_offsets = {}

        result = executor.pick_up_object("pan", "stove", robot_idx=1)

        self.assertTrue(result.success)
        executor._move_robot_near_fixture_with_retries.assert_called_once_with(
            1, "stove", require_front=False
        )

    def test_safe_compute_object_target_pos_forwards_support_site(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor.runner = MagicMock()
        executor.runner._fixtures = {"island": object()}
        executor._find_contained_objects = MagicMock(return_value=[])
        executor.runner._compute_object_target_pos = MagicMock(
            return_value=np.array([1.0, 2.0, 3.0])
        )

        target = executor._safe_compute_object_target_pos(
            "island",
            "condiment",
            preferred_xy=np.array([9.0, 8.0]),
            support_site_id="rear_left",
        )

        self.assertTrue(np.allclose(target, np.array([1.0, 2.0, 3.0])))
        executor.runner._compute_object_target_pos.assert_called_once()
        _, kwargs = executor.runner._compute_object_target_pos.call_args
        self.assertEqual(kwargs["site_id"], "rear_left")
        self.assertTrue(np.allclose(kwargs["preferred_xy"], np.array([9.0, 8.0])))

    def test_safe_compute_object_target_pos_retries_same_site_without_switching(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor.runner = MagicMock()
        executor.runner._fixtures = {"blender": object()}
        executor._find_contained_objects = MagicMock(return_value=[])
        executor.runner._compute_object_target_pos = MagicMock(
            side_effect=[
                RuntimeError("no candidate"),
                np.array([0.1, 0.2, 0.3]),
            ]
        )

        target = executor._safe_compute_object_target_pos(
            "blender",
            "tomato",
            preferred_xy=np.array([1.0, 2.0]),
            support_site_id="bowl",
            ignored_fixture_ids={"counter_main"},
        )

        self.assertTrue(np.allclose(target, np.array([0.1, 0.2, 0.3])))
        self.assertEqual(2, executor.runner._compute_object_target_pos.call_count)
        first_call = executor.runner._compute_object_target_pos.call_args_list[0].kwargs
        second_call = executor.runner._compute_object_target_pos.call_args_list[1].kwargs
        self.assertEqual(first_call["site_id"], "bowl")
        self.assertEqual(second_call["site_id"], "bowl")
        self.assertTrue(np.allclose(first_call["preferred_xy"], np.array([1.0, 2.0])))
        self.assertIsNone(second_call["preferred_xy"])
        self.assertEqual(first_call["ignored_fixture_ids"], {"counter_main"})
        self.assertEqual(second_call["ignored_fixture_ids"], {"counter_main"})

    def test_safe_compute_object_target_pos_ignores_contained_children(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor.runner = MagicMock()
        executor.runner._fixtures = {"counter": object()}
        executor._find_contained_objects = MagicMock(return_value=["slice_a", "slice_b"])
        executor.runner._compute_object_target_pos = MagicMock(
            return_value=np.array([0.0, 0.0, 0.0])
        )

        executor._safe_compute_object_target_pos(
            "counter",
            "ingredient_bowl",
            ignored_object_ids={"plate"},
        )

        _, kwargs = executor.runner._compute_object_target_pos.call_args
        self.assertEqual(
            kwargs["ignored_object_ids"],
            {"plate", "slice_a", "slice_b"},
        )

    def test_explicit_site_candidate_generation_keeps_compact_region(self):
        runner = TrajectoryRunner.__new__(TrajectoryRunner)
        runner.env = SimpleNamespace(
            objects={
                "mug": SimpleNamespace(bottom_offset=np.array([0.0, 0.0, -0.02]))
            }
        )
        runner._get_object_placement_metadata = MagicMock(
            return_value={
                "size": np.array([0.20, 0.20, 0.10], dtype=float),
                "xy_radius": 0.09,
            }
        )
        runner._fixture_local_to_world = MagicMock(
            side_effect=lambda _fixture, local: np.asarray(local, dtype=float).copy()
        )
        target_fixture = SimpleNamespace(
            get_reset_regions=MagicMock(
                return_value={
                    "rack": {
                        "offset": (0.0, 0.0, 0.0),
                        "size": (0.06, 0.06),
                    }
                }
            )
        )

        regions = runner._get_fixture_reset_regions(
            target_fixture,
            min_size=np.array([0.20, 0.20, 0.10], dtype=float),
            site_id="rack",
        )
        candidates = runner._iter_object_target_candidates(
            target_fixture,
            "mug",
            site_id="rack",
        )

        self.assertEqual(1, len(regions))
        self.assertTrue(candidates)

    def test_explicit_site_candidate_generation_uses_fixture_aligned_extent(self):
        runner = TrajectoryRunner.__new__(TrajectoryRunner)
        runner.env = SimpleNamespace(
            objects={
                "baguette": SimpleNamespace(bottom_offset=np.array([0.0, 0.0, -0.02]))
            }
        )
        runner._get_object_placement_metadata = MagicMock(
            return_value={
                "size": np.array([0.22, 0.22, 0.06], dtype=float),
                "xy_radius": 0.11,
            }
        )
        runner._get_object_fixture_xy_half_extent = MagicMock(
            return_value=np.array([0.04, 0.01], dtype=float)
        )
        runner._fixture_local_to_world = MagicMock(
            side_effect=lambda _fixture, local: np.asarray(local, dtype=float).copy()
        )
        target_fixture = SimpleNamespace(
            rot=None,
            get_reset_regions=MagicMock(
                return_value={
                    "rack": {
                        "offset": (0.0, 0.0, 0.0),
                        "size": (0.12, 0.06),
                    }
                }
            )
        )

        candidates = runner._iter_object_target_candidates(
            target_fixture,
            "baguette",
            site_id="rack",
        )

        self.assertGreater(len(candidates), 1)

    def test_explicit_site_candidate_generation_samples_full_site_span(self):
        runner = TrajectoryRunner.__new__(TrajectoryRunner)
        runner.env = SimpleNamespace(
            objects={
                "bread": SimpleNamespace(bottom_offset=np.array([0.0, 0.0, -0.02]))
            }
        )
        runner._get_object_placement_metadata = MagicMock(
            return_value={
                "size": np.array([0.24, 0.08, 0.06], dtype=float),
                "xy_radius": 0.12,
            }
        )
        runner._get_object_fixture_xy_half_extent = MagicMock(
            return_value=np.array([0.12, 0.03], dtype=float)
        )
        runner._fixture_local_to_world = MagicMock(
            side_effect=lambda _fixture, local: np.asarray(local, dtype=float).copy()
        )
        target_fixture = SimpleNamespace(
            rot=None,
            get_reset_regions=MagicMock(
                return_value={
                    "rack": {
                        "offset": (0.0, 0.0, 0.0),
                        "size": (0.10, 0.04),
                    }
                }
            )
        )

        candidates = runner._iter_object_target_candidates(
            target_fixture,
            "bread",
            site_id="rack",
        )

        self.assertGreater(len(candidates), 1)
        xs = sorted({round(float(candidate[0]), 4) for candidate in candidates})
        self.assertGreater(len(xs), 1)

    def test_resolve_joint_name_normalizes_timer_knob(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        fixture = SimpleNamespace(
            name="toaster_oven_main_group",
            _joint_names={"time": "toaster_oven_knob_time_joint"},
        )

        resolved = executor._resolve_joint_name(fixture, "timer_knob")

        self.assertEqual(resolved, "toaster_oven_knob_time_joint")

    def test_support_object_prefers_upright_for_deep_cup_like_receptacles(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor.get_scene_description = MagicMock(
            return_value={
                "objects": {
                    "cup_main": {"object_type": "glass_cup"},
                    "straw_main": {"object_type": "straw"},
                }
            }
        )
        executor._get_object_placement_metadata = MagicMock(
            return_value={"size": np.array([0.18, 0.01, 0.01], dtype=float)}
        )

        self.assertTrue(executor._support_object_prefers_upright("cup_main", "straw_main"))

    def test_support_object_prefers_upright_from_support_object_id_tokens(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor.get_scene_description = MagicMock(
            return_value={
                "objects": {
                    "pot_main": {"object_type": ""},
                    "straw_main": {"object_type": "straw"},
                }
            }
        )
        executor._get_object_placement_metadata = MagicMock(
            return_value={"size": np.array([0.18, 0.01, 0.01], dtype=float)}
        )

        self.assertTrue(executor._support_object_prefers_upright("pot_main", "straw_main"))

    def test_support_object_floor_clearance_is_tighter_for_bowl_like_supports(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor.get_scene_description = MagicMock(
            return_value={"objects": {"bowl_main": {"object_type": "mixing_bowl"}}}
        )

        clearance = executor._support_object_floor_clearance(
            "bowl_main",
            {"extent_z": 0.10},
        )

        self.assertLessEqual(clearance, 0.01)
        self.assertGreaterEqual(clearance, 0.002)

    def test_quat_align_vectors_rotates_source_onto_target(self):
        quat_xyzw = SimToolExecutor._quat_align_vectors(
            np.array([1.0, 0.0, 0.0], dtype=float),
            np.array([0.0, 0.0, 1.0], dtype=float),
        )

        rotated = T.quat2mat(quat_xyzw) @ np.array([1.0, 0.0, 0.0], dtype=float)

        np.testing.assert_allclose(rotated, np.array([0.0, 0.0, 1.0]), atol=1e-6)

    def test_object_is_upright_checks_dominant_axis_alignment(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._dominant_object_local_axis = MagicMock(
            return_value=np.array([1.0, 0.0, 0.0], dtype=float)
        )

        upright_quat_xyzw = SimToolExecutor._quat_align_vectors(
            np.array([1.0, 0.0, 0.0], dtype=float),
            np.array([0.0, 0.0, 1.0], dtype=float),
        )
        upright_quat_wxyz = T.convert_quat(upright_quat_xyzw, to="wxyz")

        self.assertTrue(executor._object_is_upright("straw_main", upright_quat_wxyz))
        self.assertFalse(
            executor._object_is_upright(
                "straw_main",
                np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            )
        )

    def test_fixture_site_slot_positions_front_bias_interior_row(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor.runner = MagicMock()
        executor.runner._get_fixture_reset_regions = MagicMock(
            return_value=[
                {
                    "name": "shelf_0",
                    "offset": (0.0, 0.0, 0.0),
                    "size": (0.40, 0.20, 0.10),
                }
            ]
        )
        executor.runner._fixture_local_to_world = MagicMock(
            side_effect=lambda _fixture, local: np.asarray(local, dtype=float).copy()
        )
        executor._require_fixture = MagicMock(return_value=object())
        executor._get_fixture_type_name = MagicMock(return_value="cabinet")

        positions = executor._fixture_site_slot_positions("cab_main", "shelf_0", count=2)

        self.assertEqual(len(positions), 2)
        self.assertLess(float(positions[0][1]), 0.0)
        self.assertLess(float(positions[1][1]), 0.0)

    def test_support_object_slot_positions_use_wider_spread_for_plate_like_supports(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._support_object_geometry = MagicMock(
            return_value={
                "extent_x": 0.30,
                "extent_y": 0.24,
                "center_xy": np.array([0.0, 0.0], dtype=float),
                "is_concave": False,
                "rot_xy": np.eye(2, dtype=float),
            }
        )
        executor._support_object_placement_family = MagicMock(
            return_value="shallow_receptacle"
        )

        positions = executor._support_object_slot_positions("plate_main", count=2)

        self.assertEqual(len(positions), 2)
        self.assertGreater(
            float(np.linalg.norm(np.asarray(positions[0]) - np.asarray(positions[1]))),
            0.13,
        )
        self.assertLess(float(positions[0][0]), 0.0)
        self.assertGreater(float(positions[1][0]), 0.0)

    @patch("robocasa.utils.sim_tool_executor.get_front_alignment_metrics")
    @patch("robocasa.utils.sim_tool_executor._is_approach_center")
    def test_robot_near_fixture_requires_bounded_front_gap(
        self,
        mock_is_approach_center,
        mock_get_front_alignment_metrics,
    ):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        fixture = SimpleNamespace(pos=np.array([0.0, 0.0, 0.0]))
        executor.runner = MagicMock()
        executor.runner._fixtures = {"fridge": fixture}
        executor.runner._get_robot_position.return_value = np.array([0.0, 0.0, 0.0])
        executor.runner._get_fixture_front_target_xy.return_value = np.array([0.0, 0.0])

        mock_is_approach_center.return_value = True
        mock_get_front_alignment_metrics.return_value = {
            "on_front_face": True,
            "within_span": True,
            "lateral_offset": 0.0,
            "front_gap": 1.2,
        }

        self.assertFalse(executor._robot_near_fixture(0, "fridge"))

    def test_front_retry_retries_after_clearing_blockers(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor.runner = MagicMock()
        executor.runner._move_robot_near_fixture = MagicMock(side_effect=[False, True])
        executor._clear_fixture_blockers = MagicMock(return_value=True)

        placed = executor._move_robot_near_fixture_with_retries(
            1,
            "fridge",
            require_front=True,
        )

        self.assertTrue(placed)
        executor._clear_fixture_blockers.assert_called_once_with(1, "fridge")
        self.assertEqual(executor.runner._move_robot_near_fixture.call_count, 2)

    def test_non_front_retry_does_not_clear_blockers(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor.runner = MagicMock()
        executor.runner._move_robot_near_fixture = MagicMock(return_value=False)
        executor._clear_fixture_blockers = MagicMock(return_value=True)

        placed = executor._move_robot_near_fixture_with_retries(
            0,
            "counter",
            require_front=False,
        )

        self.assertFalse(placed)
        executor._clear_fixture_blockers.assert_not_called()
        executor.runner._move_robot_near_fixture.assert_called_once()

    def test_place_in_receptacle_uses_front_retry_for_front_required_fixture(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._require_object = MagicMock()
        executor._held_by_robot = MagicMock(return_value=0)
        executor._resolve_support_target = MagicMock(
            side_effect=[("fridge", "shelf_0"), ("fridge", "shelf_0")]
        )
        executor._require_explicit_site_if_needed = MagicMock()
        executor._fixture_placement_semantics = MagicMock(return_value="receptacle")
        executor._preferred_xy_for_fixture_target = MagicMock(
            return_value=np.array([1.0, 2.0], dtype=float)
        )
        executor._incoming_fixture_site_preference = MagicMock(return_value=None)
        executor._ignored_fixture_ids_for_placement = MagicMock(return_value=set())
        executor._compute_object_target_pose = MagicMock(
            return_value=(
                np.array([1.0, 2.0, 0.9], dtype=float),
                np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            )
        )
        executor._fixture_requires_front_approach = MagicMock(return_value=True)
        executor._surface_fixture_prefers_front_approach = MagicMock(return_value=False)
        executor._move_robot_near_fixture_with_retries = MagicMock(return_value=True)
        executor._sync_held_object = MagicMock()
        executor._set_object_pose = MagicMock()
        executor._set_support_parent = MagicMock()
        executor._settle_scene = MagicMock()
        executor._held_objects = {0: "bowl"}
        executor._held_object_offsets = {0: np.zeros(3, dtype=float)}
        executor.runner = MagicMock()
        executor.runner._fixtures = {"fridge": object()}
        executor.runner._set_object_location = MagicMock()
        executor.env = SimpleNamespace(objects={"bowl": object()})

        result = executor.place_in_receptacle(
            "bowl",
            receptacle_id="fridge",
            target_site_id="shelf_0",
            robot_idx=0,
        )

        self.assertTrue(result.success)
        executor._move_robot_near_fixture_with_retries.assert_called_once()
        self.assertTrue(
            executor._move_robot_near_fixture_with_retries.call_args.kwargs[
                "require_front"
            ]
        )

    def test_partitioned_fixture_requires_explicit_site(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor.runner = MagicMock()
        executor.runner._fixtures = {"cab_main": object()}
        executor.get_scene_description = MagicMock(
            return_value={
                "fixtures": {
                    "cab_main": {"fixture_type": "cabinet_double_door"},
                }
            }
        )
        executor.get_support_sites = MagicMock(
            return_value=["shelf_1", "shelf_2"]
        )
        executor._resolve_fixture_site_id = MagicMock(return_value="shelf_1")

        with self.assertRaises(ValueError):
            executor._require_explicit_site_if_needed(
                "cab_main",
                None,
                action_name="place_in_receptacle",
            )

    def test_normalize_target_site_drops_generic_surface_geom(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor.get_scene_description = MagicMock(
            return_value={
                "fixtures": {
                    "counter_main": {"fixture_type": "counter_non_dining"},
                    "stove_main": {"fixture_type": "stove"},
                }
            }
        )
        executor._resolve_fixture_site_id = MagicMock(
            side_effect=lambda fixture_id, site_id: site_id
        )

        self.assertIsNone(
            executor._normalize_target_site_id_for_placement("counter_main", "geom_0")
        )
        self.assertEqual(
            executor._normalize_target_site_id_for_placement("stove_main", "front_right_burner"),
            "burner",
        )

    def test_default_fixture_surface_preference_avoids_child_fixture_blockers(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._objects_on_fixture_site = MagicMock(return_value=[])
        executor._fixture_surface_blocker_positions = MagicMock(
            return_value=[np.array([0.0, 0.0], dtype=float)]
        )
        executor._surface_fixture_slot_positions = MagicMock(
            return_value=[
                np.array([0.0, 0.0], dtype=float),
                np.array([0.35, 0.0], dtype=float),
            ]
        )
        executor._preferred_xy_for_fixture_target = MagicMock(
            return_value=np.array([0.0, 0.0], dtype=float)
        )

        preferred_xy = executor._default_fixture_surface_preference(
            "counter_main",
            "bowl_main",
        )

        np.testing.assert_allclose(preferred_xy, np.array([0.35, 0.0], dtype=float))

    def test_default_fixture_surface_preference_keeps_duplicate_cubes_compact(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._objects_on_fixture_site = MagicMock(return_value=["sugar_cube_1"])
        executor._get_object_pose = MagicMock(
            return_value=(
                np.array([1.0, 2.0, 0.9], dtype=float),
                np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            )
        )
        executor._scene_object_tokens = MagicMock(return_value={"sugar", "cube"})
        executor._require_fixture = MagicMock(
            return_value=SimpleNamespace(rot=0.0)
        )
        executor._project_xy_onto_fixture = MagicMock(
            side_effect=lambda fixture_id, world_xy: np.asarray(world_xy, dtype=float)
        )
        executor._fixture_surface_blocker_positions = MagicMock()
        executor._surface_fixture_slot_positions = MagicMock()

        preferred_xy = executor._default_fixture_surface_preference(
            "dining_counter",
            "sugar_cube_2",
        )

        np.testing.assert_allclose(preferred_xy, np.array([1.08, 2.0], dtype=float))
        executor._surface_fixture_slot_positions.assert_not_called()

    def test_default_fixture_surface_preference_anchors_first_cube_near_plate(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._objects_on_fixture_site = MagicMock(return_value=["plate"])

        def object_pose(object_id):
            return (
                np.array([1.0, 2.0, 0.9], dtype=float),
                np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            )

        def object_tokens(object_id):
            if object_id == "sugar_cube_1":
                return {"sugar", "cube"}
            return {object_id}

        executor._get_object_pose = MagicMock(side_effect=object_pose)
        executor._scene_object_tokens = MagicMock(side_effect=object_tokens)
        executor._require_fixture = MagicMock(return_value=SimpleNamespace(rot=0.0))
        executor._project_xy_onto_fixture = MagicMock(
            side_effect=lambda fixture_id, world_xy: np.asarray(world_xy, dtype=float)
        )
        executor._fixture_surface_blocker_positions = MagicMock()
        executor._surface_fixture_slot_positions = MagicMock()

        preferred_xy = executor._default_fixture_surface_preference(
            "dining_counter",
            "sugar_cube_1",
        )

        np.testing.assert_allclose(preferred_xy, np.array([1.08, 2.0], dtype=float))
        executor._surface_fixture_slot_positions.assert_not_called()

    def test_place_on_surface_uses_surface_slot_preference_and_target_pose(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._require_object = MagicMock()
        executor._held_by_robot = MagicMock(return_value=0)
        executor._resolve_support_target = MagicMock(return_value=("counter_main", None))
        executor._require_explicit_site_if_needed = MagicMock()
        executor._fixture_placement_semantics = MagicMock(return_value="surface")
        executor._default_fixture_surface_preference = MagicMock(
            return_value=np.array([0.25, 0.1], dtype=float)
        )
        executor._preferred_xy_for_fixture_target = MagicMock()
        executor._ignored_fixture_ids_for_placement = MagicMock(return_value={"wall_back"})
        target_pos = np.array([0.4, 0.15, 0.9], dtype=float)
        target_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        executor._compute_object_target_pose = MagicMock(
            return_value=(target_pos, target_quat)
        )
        executor.runner = MagicMock()
        executor.runner._move_robot_near_fixture = MagicMock()
        executor.runner._set_object_location = MagicMock()
        executor._sync_held_object = MagicMock()
        executor._set_object_pose = MagicMock()
        executor._set_support_parent = MagicMock()
        executor._held_objects = {0: "condiment"}
        executor._settle_scene = MagicMock()

        result = executor.place_on_surface("condiment", "counter_main", robot_idx=0)

        self.assertTrue(result.success)
        executor._default_fixture_surface_preference.assert_called_once_with(
            "counter_main",
            "condiment",
        )
        np.testing.assert_allclose(
            executor._compute_object_target_pose.call_args.kwargs["preferred_xy"],
            np.array([0.25, 0.1], dtype=float),
        )
        executor._set_object_pose.assert_called_once_with(
            "condiment",
            target_pos,
            target_quat,
        )

    def test_ignored_fixture_ids_for_placement_ignores_structural_runner_fixtures(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor.runner = MagicMock()
        executor.runner._fixtures = {
            "counter_main": object(),
            "box_left_group": object(),
            "fridge_housing_right_group": object(),
        }
        executor.get_scene_description = MagicMock(
            return_value={
                "fixtures": {
                    "counter_main": {
                        "fixture_type": "counter_non_dining",
                        "parent_fixture": None,
                    }
                }
            }
        )

        ignored = executor._ignored_fixture_ids_for_placement("counter_main")

        self.assertIn("box_left_group", ignored)
        self.assertIn("fridge_housing_right_group", ignored)

    def test_ignored_fixture_ids_for_placement_ignores_box_structures_without_box_token(self):
        class Box:
            pass

        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor.runner = MagicMock()
        executor.runner._fixtures = {
            "counter_main": object(),
            "counter_stack_main_group_base": Box(),
        }
        executor.get_scene_description = MagicMock(
            return_value={
                "fixtures": {
                    "counter_main": {
                        "fixture_type": "counter_non_dining",
                        "parent_fixture": None,
                    }
                }
            }
        )

        ignored = executor._ignored_fixture_ids_for_placement("counter_main")

        self.assertIn("counter_stack_main_group_base", ignored)

    def test_safe_compute_object_target_pos_ignores_supported_descendants(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor.runner = MagicMock()
        executor.runner._fixtures = {"counter_main": object()}
        executor.runner._compute_object_target_pos = MagicMock(
            return_value=np.array([0.2, -0.1, 0.9], dtype=float)
        )
        executor._iter_supported_descendants = MagicMock(
            return_value=["child_a", "child_b"]
        )
        executor._find_contained_objects = MagicMock(return_value=["child_c"])

        target = executor._safe_compute_object_target_pos(
            "counter_main",
            "bowl_main",
            ignored_object_ids={"existing"},
        )

        np.testing.assert_allclose(target, np.array([0.2, -0.1, 0.9], dtype=float))
        self.assertEqual(
            executor.runner._compute_object_target_pos.call_args.kwargs[
                "ignored_object_ids"
            ],
            {"existing", "child_a", "child_b", "child_c"},
        )

    def test_resolve_fixture_site_id_maps_unique_family_match(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor.env = MagicMock()
        fixture = MagicMock()
        fixture.get_reset_regions.return_value = {
            "shelf_0": {
                "offset": (0.0, 0.0, 0.0),
                "size": (0.2, 0.2, 0.1),
            }
        }
        executor._require_fixture = MagicMock(return_value=fixture)

        self.assertEqual(
            executor._resolve_fixture_site_id("cab_main", "shelf_1"),
            "shelf_0",
        )

    def test_place_on_surface_rejects_receptacle_fixture(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._held_objects = {}
        executor._require_object = MagicMock()
        executor._held_by_robot = MagicMock(return_value=0)
        executor._resolve_support_target = MagicMock(return_value=("cab_main", "shelf_1"))
        executor._fixture_placement_semantics = MagicMock(return_value="receptacle")

        with self.assertRaises(ValueError):
            executor.place_on_surface(
                "mug",
                target_id="cab_main",
                target_site_id="shelf_1",
                robot_idx=0,
            )

    def test_place_in_receptacle_rejects_surface_fixture(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._held_objects = {}
        executor._require_object = MagicMock()
        executor._held_by_robot = MagicMock(return_value=0)
        executor.env = SimpleNamespace(objects={})
        executor.runner = MagicMock()
        executor.runner._fixtures = {"counter_main": object()}
        executor._resolve_support_target = MagicMock(
            return_value=("counter_main", None)
        )
        executor._fixture_placement_semantics = MagicMock(return_value="surface")

        with self.assertRaises(ValueError):
            executor.place_in_receptacle(
                "mug",
                target_id="counter_main",
                robot_idx=0,
            )

    def test_place_in_receptacle_accepts_support_site_identifier(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._held_objects = {}
        executor._require_object = MagicMock()
        executor._held_by_robot = MagicMock(return_value=0)
        executor.env = SimpleNamespace(objects={})
        executor.runner = MagicMock()
        executor.runner._fixtures = {"blender_main": object()}
        executor._resolve_support_target = MagicMock(
            return_value=("blender_main", "bowl")
        )
        executor._fixture_placement_semantics = MagicMock(return_value="receptacle")
        executor._preferred_xy_for_fixture_target = MagicMock(return_value=None)
        executor._compute_object_target_pose = MagicMock(
            return_value=(
                np.array([0.0, 0.0, 0.0]),
                np.array([1.0, 0.0, 0.0, 0.0]),
            )
        )
        executor._sync_held_object = MagicMock()
        executor._set_object_pose = MagicMock()
        executor._set_support_parent = MagicMock()
        executor._settle_scene = MagicMock()
        executor.runner._move_robot_near_fixture = MagicMock()
        executor.runner._set_object_location = MagicMock()

        result = executor.place_in_receptacle(
            "mug",
            receptacle_id="blender_bowl",
            robot_idx=0,
        )

        self.assertTrue(result.success)
        executor._resolve_support_target.assert_any_call("blender_bowl", None)
        executor.runner._set_object_location.assert_called_once_with("mug", "blender_main")

    def test_place_in_receptacle_prefers_object_receptacle(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._held_objects = {}
        executor._require_object = MagicMock()
        executor._held_by_robot = MagicMock(return_value=0)
        executor.env = SimpleNamespace(objects={"chicken": object(), "serving_tray": object()})
        executor.runner = MagicMock()
        executor.runner._fixtures = {}
        executor._resolve_support_target = MagicMock()
        executor._get_scene_object_location = MagicMock(return_value="table_main")
        executor._get_object_pose = MagicMock(
            return_value=(np.array([0.1, 0.2, 0.3]), np.array([1.0, 0.0, 0.0, 0.0]))
        )
        executor._sync_held_object = MagicMock()
        executor._place_on_object_center = MagicMock()
        executor._support_object_prefers_upright = MagicMock(return_value=False)
        executor._support_object_geometry = MagicMock(
            return_value={"is_concave": False, "top_z": 0.0, "bottom_z": 0.0, "extent_z": 0.1}
        )
        executor._supported_object_vertical_gap = MagicMock(return_value=0.0)
        executor._set_support_parent = MagicMock()
        executor._settle_scene = MagicMock()
        executor.runner._move_robot_near_fixture = MagicMock()
        executor.runner._set_object_location = MagicMock()

        result = executor.place_in_receptacle(
            "chicken",
            receptacle_id="serving_tray",
            robot_idx=0,
        )

        self.assertTrue(result.success)
        executor._resolve_support_target.assert_not_called()
        self.assertEqual(
            executor._place_on_object_center.call_args_list,
            [
                unittest.mock.call(
                    "chicken",
                    "serving_tray",
                    relative_position=None,
                ),
                unittest.mock.call(
                    "chicken",
                    "serving_tray",
                    relative_position=None,
                ),
            ],
        )
        self.assertGreaterEqual(executor.runner._set_object_location.call_count, 2)

    def test_place_in_receptacle_keeps_explicit_bowl_on_tray(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._held_objects = {}
        executor._require_object = MagicMock()
        executor._held_by_robot = MagicMock(return_value=0)
        executor.env = SimpleNamespace(objects={"bowl": object(), "tray": object()})
        executor.runner = MagicMock()
        executor.runner._fixtures = {}
        executor._resolve_support_target = MagicMock()
        executor._scene_object_tokens = MagicMock(return_value={"bowl"})
        executor._support_object_tokens = MagicMock(return_value={"tray"})
        executor._get_scene_object_location = MagicMock(return_value="counter_main")
        executor._get_object_pose = MagicMock(
            return_value=(np.array([0.1, 0.2, 0.3]), np.array([1.0, 0.0, 0.0, 0.0]))
        )
        executor._sync_held_object = MagicMock()
        executor._place_on_object_center = MagicMock()
        executor._settle_and_reseat_supported_object = MagicMock()
        executor._support_object_prefers_upright = MagicMock(return_value=False)
        executor._support_object_accepts_child_without_settle = MagicMock(return_value=False)
        executor._settle_scene = MagicMock()
        executor.place_next_to = MagicMock()

        result = executor.place_in_receptacle(
            "bowl",
            receptacle_id="tray",
            robot_idx=0,
        )

        self.assertTrue(result.success)
        executor.place_next_to.assert_not_called()
        executor._place_on_object_center.assert_called_once_with(
            "bowl",
            "tray",
            relative_position=None,
        )

    def test_place_in_receptacle_skips_final_generic_settle_for_upright_insertions(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._held_objects = {}
        executor._require_object = MagicMock()
        executor._held_by_robot = MagicMock(return_value=0)
        executor.env = SimpleNamespace(objects={"straw": object(), "glass_cup": object()})
        executor.runner = MagicMock()
        executor.runner._fixtures = {}
        executor._resolve_support_target = MagicMock()
        executor._get_scene_object_location = MagicMock(return_value="dining_main")
        executor._get_object_pose = MagicMock(
            return_value=(np.array([0.1, 0.2, 0.3]), np.array([1.0, 0.0, 0.0, 0.0]))
        )
        executor._sync_held_object = MagicMock()
        executor._place_on_object_center = MagicMock()
        executor._settle_and_reseat_supported_object = MagicMock()
        executor._support_object_prefers_upright = MagicMock(return_value=True)
        executor._settle_scene = MagicMock()
        executor.runner._move_robot_near_fixture = MagicMock()

        result = executor.place_in_receptacle(
            "straw",
            receptacle_id="glass_cup",
            robot_idx=0,
        )

        self.assertTrue(result.success)
        executor._settle_and_reseat_supported_object.assert_called_once_with(
            "straw",
            "glass_cup",
            relative_position=None,
        )
        executor._settle_scene.assert_not_called()

    def test_yogurt_prefers_upright_on_surface(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._scene_object_tokens = MagicMock(return_value={"yogurt"})
        executor._get_object_placement_metadata = MagicMock()

        self.assertTrue(executor._object_prefers_upright_on_surface("yogurt_1"))
        executor._get_object_placement_metadata.assert_not_called()

    def test_load_initial_state_skips_explicit_site_check_when_object_already_on_fixture(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor.env = MagicMock()
        executor.env.sim = MagicMock()
        executor.get_scene_description = MagicMock(
            return_value={
                "objects": {"milk": {"location": "fridge_main"}},
                "object_placements": {"milk": "fridge_main"},
                "fixtures": {"fridge_main": {"fixture_type": "fridge"}},
            }
        )
        executor._parse_agent_idx = MagicMock(return_value=0)
        executor._set_fixture_machine_state = MagicMock()
        executor._initialize_support_graph_from_scene = MagicMock()
        executor._infer_object_support_site = MagicMock(return_value="shelf_1")
        executor._current_support_object = MagicMock(return_value=None)
        executor._set_support_parent = MagicMock()
        executor._place_object_on_fixture = MagicMock()
        executor._settle_scene = MagicMock()
        executor._held_objects = {}
        executor._require_fixture = MagicMock()
        executor._require_object = MagicMock()
        executor.runner = MagicMock()
        executor.runner._fixtures = {"fridge_main": object()}
        executor._resolve_support_target = MagicMock(return_value=("fridge_main", None))
        executor._require_explicit_site_if_needed = MagicMock(
            side_effect=AssertionError("should not be called")
        )

        summary = executor.load_initial_state(
            {
                "agents": {"agent_0": {"location": "fridge_main", "held_object": None}},
                "objects": {"milk": {"location": "fridge_main", "object_type": "milk"}},
                "fixtures": {"fridge_main": {"fixture_type": "fridge"}},
                "machine_state": {},
            }
        )

        self.assertTrue(summary["loaded"])
        executor._place_object_on_fixture.assert_not_called()

    def test_load_initial_state_uses_target_site_id_with_fixture_location(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor.env = MagicMock()
        executor.env.sim = MagicMock()
        executor.get_scene_description = MagicMock(
            return_value={
                "objects": {"mug": {"location": "cab_main"}},
                "object_placements": {},
            }
        )
        executor._parse_agent_idx = MagicMock(return_value=0)
        executor._set_fixture_machine_state = MagicMock()
        executor._initialize_support_graph_from_scene = MagicMock()
        executor._infer_object_support_site = MagicMock(return_value=None)
        executor._surface_pose_needs_reset = MagicMock(return_value=False)
        executor._set_support_parent = MagicMock()
        executor._place_object_on_fixture = MagicMock()
        executor._incoming_fixture_site_preference = MagicMock(
            return_value=np.array([0.12, -0.04], dtype=float)
        )
        executor._preferred_xy_for_fixture_target = MagicMock()
        executor._settle_scene = MagicMock()
        executor._held_objects = {}
        executor._support_parents = {}
        executor._require_fixture = MagicMock()
        executor._require_object = MagicMock()
        executor.runner = MagicMock()
        executor.runner._fixtures = {"cab_main": object()}
        executor._resolve_support_target = MagicMock(return_value=("cab_main", "shelf_0"))
        executor._require_explicit_site_if_needed = MagicMock()

        summary = executor.load_initial_state(
            {
                "agents": {"agent_0": {"location": "cab_main", "held_object": None}},
                "objects": {
                    "mug": {
                        "location": "cab_main",
                        "target_site_id": "shelf_0",
                        "object_type": "mug",
                    }
                },
                "fixtures": {"cab_main": {"fixture_type": "cabinet"}},
                "machine_state": {},
            }
        )

        self.assertTrue(summary["loaded"])
        executor._resolve_support_target.assert_any_call("cab_main", "shelf_0")
        args, kwargs = executor._place_object_on_fixture.call_args
        self.assertEqual(args, ("mug", "cab_main"))
        np.testing.assert_allclose(
            kwargs["preferred_xy"],
            np.array([0.12, -0.04], dtype=float),
        )
        self.assertEqual(kwargs["target_site_id"], "shelf_0")
        self.assertFalse(kwargs["settle"])

    def test_load_initial_state_uses_default_site_when_partitioned_fixture_is_unspecified(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor.env = MagicMock()
        executor.env.sim = MagicMock()
        executor.env.objects = {"baguette_main": object()}
        executor.get_scene_description = MagicMock(
            return_value={
                "objects": {"baguette_main": {"location": "fridge_main"}},
                "object_placements": {},
            }
        )
        executor._parse_agent_idx = MagicMock(return_value=0)
        executor._set_fixture_machine_state = MagicMock()
        executor._initialize_support_graph_from_scene = MagicMock()
        executor._infer_object_support_site = MagicMock(return_value=None)
        executor._current_support_object = MagicMock(return_value=None)
        executor._surface_pose_needs_reset = MagicMock(return_value=False)
        executor._set_support_parent = MagicMock()
        executor._place_object_on_fixture = MagicMock()
        executor._incoming_fixture_site_preference = MagicMock(
            return_value=np.array([0.08, -0.03], dtype=float)
        )
        executor._preferred_xy_for_fixture_target = MagicMock()
        executor._settle_scene = MagicMock()
        executor._held_objects = {}
        executor._support_parents = {}
        executor._robot_spawn = "sim"
        executor._require_fixture = MagicMock()
        executor._require_object = MagicMock()
        executor._get_object_pose = MagicMock(
            return_value=(np.zeros(3, dtype=float), np.array([1.0, 0.0, 0.0, 0.0]))
        )
        executor.runner = MagicMock()
        executor.runner._fixtures = {"fridge_main": object()}
        executor.runner._set_object_location = MagicMock()
        executor._resolve_support_target = MagicMock(return_value=("fridge_main", None))
        executor._require_explicit_site_if_needed = MagicMock()
        executor._fixture_requires_explicit_site = MagicMock(return_value=True)
        executor._default_support_site_for_unspecified_fixture = MagicMock(
            return_value="shelf_1"
        )
        executor._normalize_target_site_id_for_placement = MagicMock(side_effect=lambda fixture_id, site_id: site_id)
        executor._repair_supported_children_vertical_gaps = MagicMock()

        summary = executor.load_initial_state(
            {
                "agents": {"agent_0": {"location": "fridge_main", "held_object": None}},
                "objects": {
                    "baguette_main": {
                        "location": "fridge_main",
                        "object_type": "baguette",
                    }
                },
                "fixtures": {"fridge_main": {"fixture_type": "fridge"}},
                "machine_state": {},
            }
        )

        self.assertTrue(summary["loaded"])
        executor._default_support_site_for_unspecified_fixture.assert_called_once_with(
            "fridge_main",
            incoming_object_id="baguette_main",
        )
        args, kwargs = executor._place_object_on_fixture.call_args
        self.assertEqual(args, ("baguette_main", "fridge_main"))
        self.assertEqual(kwargs["target_site_id"], "shelf_1")

    def test_load_initial_state_preserves_support_object_with_contained_children(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor.env = MagicMock()
        executor.env.sim = MagicMock()
        executor.env.objects = {"bowl_main": object(), "berry_1": object()}
        executor.get_scene_description = MagicMock(
            return_value={
                "objects": {
                    "bowl_main": {"location": "counter_main"},
                    "berry_1": {"location": "counter_main"},
                },
                "object_placements": {"bowl_main": "counter_main"},
            }
        )
        executor._parse_agent_idx = MagicMock(return_value=0)
        executor._set_fixture_machine_state = MagicMock()
        executor._initialize_support_graph_from_scene = MagicMock()
        executor._infer_object_support_site = MagicMock(return_value=None)
        executor._current_support_object = MagicMock(return_value=None)
        executor._surface_pose_needs_reset = MagicMock(return_value=False)
        executor._set_support_parent = MagicMock()
        executor._place_object_on_fixture = MagicMock()
        executor._place_on_object_center = MagicMock()
        executor._settle_and_reseat_supported_object = MagicMock()
        executor._repack_supported_children = MagicMock()
        executor._repair_supported_children_vertical_gaps = MagicMock()
        executor._settle_scene = MagicMock()
        executor._held_objects = {}
        executor._support_parents = {}
        executor._robot_spawn = "sim"
        executor._require_fixture = MagicMock()
        executor._require_object = MagicMock()
        executor.runner = MagicMock()
        executor.runner._fixtures = {"counter_main": object()}
        executor.runner._set_object_location = MagicMock()
        executor._resolve_support_target = MagicMock(return_value=("counter_main", None))
        executor._require_explicit_site_if_needed = MagicMock()

        summary = executor.load_initial_state(
            {
                "agents": {"agent_0": {"location": "counter_main", "held_object": None}},
                "objects": {
                    "bowl_main": {
                        "location": "counter_main",
                        "object_type": "bowl",
                    },
                    "berry_1": {
                        "location": "bowl_main",
                        "object_type": "berry",
                    },
                },
                "fixtures": {"counter_main": {"fixture_type": "counter_non_dining"}},
                "machine_state": {},
            }
        )

        self.assertTrue(summary["loaded"])
        executor._place_object_on_fixture.assert_not_called()
        self.assertEqual(
            summary["placement_events"][0]["reason"],
            "already_on_target_fixture",
        )

    def test_load_initial_state_reseats_surface_object_when_pose_needs_reset(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor.env = MagicMock()
        executor.env.sim = MagicMock()
        executor.env.objects = {"fork_main": object()}
        executor.get_scene_description = MagicMock(
            return_value={
                "objects": {"fork_main": {"location": "counter_main"}},
                "object_placements": {"fork_main": "counter_main"},
            }
        )
        executor._parse_agent_idx = MagicMock(return_value=0)
        executor._set_fixture_machine_state = MagicMock()
        executor._initialize_support_graph_from_scene = MagicMock()
        executor._infer_object_support_site = MagicMock(return_value="geom_0")
        executor._current_support_object = MagicMock(return_value=None)
        executor._surface_pose_needs_reset = MagicMock(return_value=True)
        executor._set_support_parent = MagicMock()
        executor._place_object_on_fixture = MagicMock()
        executor._place_on_object_center = MagicMock()
        executor._settle_and_reseat_supported_object = MagicMock()
        executor._repack_supported_children = MagicMock()
        executor._repair_supported_children_vertical_gaps = MagicMock()
        executor._settle_scene = MagicMock()
        executor._held_objects = {}
        executor._support_parents = {}
        executor._robot_spawn = "sim"
        executor._require_fixture = MagicMock()
        executor._require_object = MagicMock()
        executor._get_object_pose = MagicMock(
            return_value=(np.zeros(3, dtype=float), np.array([1.0, 0.0, 0.0, 0.0]))
        )
        executor.runner = MagicMock()
        executor.runner._fixtures = {"counter_main": object()}
        executor.runner._set_object_location = MagicMock()
        executor._resolve_support_target = MagicMock(return_value=("counter_main", None))
        executor._require_explicit_site_if_needed = MagicMock()
        executor._normalize_target_site_id_for_placement = MagicMock(return_value=None)
        executor._fixture_requires_explicit_site = MagicMock(return_value=False)
        executor._default_fixture_surface_preference = MagicMock(
            return_value=np.array([0.25, 0.0], dtype=float)
        )

        summary = executor.load_initial_state(
            {
                "agents": {"agent_0": {"location": "counter_main", "held_object": None}},
                "objects": {
                    "fork_main": {
                        "location": "counter_main",
                        "object_type": "fork",
                    }
                },
                "fixtures": {"counter_main": {"fixture_type": "counter_non_dining"}},
                "machine_state": {},
            }
        )

        self.assertTrue(summary["loaded"])
        args, kwargs = executor._place_object_on_fixture.call_args
        self.assertEqual(args, ("fork_main", "counter_main"))
        np.testing.assert_allclose(
            kwargs["preferred_xy"],
            np.array([0.25, 0.0], dtype=float),
        )
        self.assertIsNone(kwargs["target_site_id"])
        self.assertFalse(kwargs["settle"])

    def test_current_support_object_detects_supported_item(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._support_parents = {}
        executor.env = SimpleNamespace(objects={"plate": object(), "sugar_cube_1": object()})
        executor._find_objects_on_support = MagicMock(
            side_effect=lambda support_object_id, exclude=None: ["sugar_cube_1"]
            if support_object_id == "plate"
            else []
        )

        self.assertEqual(executor._current_support_object("sugar_cube_1"), "plate")

    def test_load_initial_state_does_not_skip_object_resting_on_other_object(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor.env = MagicMock()
        executor.env.sim = MagicMock()
        executor.get_scene_description = MagicMock(
            return_value={
                "objects": {"sugar_cube_1": {"location": "dining_main"}},
                "object_placements": {"sugar_cube_1": "dining_main"},
            }
        )
        executor._parse_agent_idx = MagicMock(return_value=0)
        executor._set_fixture_machine_state = MagicMock()
        executor._initialize_support_graph_from_scene = MagicMock()
        executor._infer_object_support_site = MagicMock(return_value=None)
        executor._current_support_object = MagicMock(return_value="plate_main")
        executor._set_support_parent = MagicMock()
        executor._place_object_on_fixture = MagicMock()
        executor._settle_scene = MagicMock()
        executor._held_objects = {}
        executor._support_parents = {}
        executor._require_fixture = MagicMock()
        executor._require_object = MagicMock()
        executor.runner = MagicMock()
        executor.runner._fixtures = {"dining_main": object()}
        executor.runner._set_object_location = MagicMock()
        executor._resolve_support_target = MagicMock(return_value=("dining_main", None))
        executor._require_explicit_site_if_needed = MagicMock()

        summary = executor.load_initial_state(
            {
                "agents": {"agent_0": {"location": "dining_main", "held_object": None}},
                "objects": {
                    "sugar_cube_1": {
                        "location": "dining_main",
                        "object_type": "sugar_cube",
                    }
                },
                "fixtures": {"dining_main": {"fixture_type": "dining_counter"}},
                "machine_state": {},
            }
        )

        self.assertTrue(summary["loaded"])
        executor._place_object_on_fixture.assert_called_once_with(
            "sugar_cube_1",
            "dining_main",
            preferred_xy=None,
            target_site_id=None,
            settle=False,
        )

    def test_load_initial_state_reseats_supported_objects_after_settle(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor.env = MagicMock()
        executor.env.sim = MagicMock()
        executor.get_scene_description = MagicMock(
            return_value={
                "objects": {
                    "bowl_main": {"location": "counter_main"},
                    "berry_1": {"location": "counter_main"},
                },
                "object_placements": {},
            }
        )
        executor._parse_agent_idx = MagicMock(return_value=0)
        executor._set_fixture_machine_state = MagicMock()
        executor._initialize_support_graph_from_scene = MagicMock()
        executor._infer_object_support_site = MagicMock(return_value=None)
        executor._current_support_object = MagicMock(return_value=None)
        executor._surface_pose_needs_reset = MagicMock(return_value=False)
        executor._set_support_parent = MagicMock()
        executor._set_object_pose = MagicMock()
        executor._place_object_on_fixture = MagicMock()
        executor._place_on_object_center = MagicMock()
        executor._settle_scene = MagicMock()
        executor._held_objects = {}
        executor._support_parents = {}
        executor._require_fixture = MagicMock()
        executor._require_object = MagicMock()
        executor._get_object_pose = MagicMock(
            side_effect=[
                (np.array([0.0, 0.0, 0.20]), np.array([1.0, 0.0, 0.0, 0.0])),
                (np.array([0.0, 0.0, 0.24]), np.array([1.0, 0.0, 0.0, 0.0])),
            ]
        )
        executor._support_object_prefers_upright = MagicMock(return_value=False)
        executor._support_object_geometry = MagicMock(
            return_value={"is_concave": True, "top_z": 0.0, "bottom_z": 0.0, "extent_z": 0.1}
        )
        executor._supported_object_vertical_gap = MagicMock(return_value=0.0)
        executor.runner = MagicMock()
        executor.runner._fixtures = {"counter_main": object()}
        executor.runner._set_object_location = MagicMock()
        executor._resolve_support_target = MagicMock(return_value=("counter_main", None))
        executor._require_explicit_site_if_needed = MagicMock()

        summary = executor.load_initial_state(
            {
                "agents": {"agent_0": {"location": "counter_main", "held_object": None}},
                "objects": {
                    "bowl_main": {
                        "location": "counter_main",
                        "object_type": "bowl",
                    },
                    "berry_1": {
                        "location": "bowl_main",
                        "object_type": "berry",
                    },
                },
                "fixtures": {"counter_main": {"fixture_type": "counter_non_dining"}},
                "machine_state": {},
            }
        )

        self.assertTrue(summary["loaded"])
        self.assertEqual(
            executor._place_on_object_center.call_args_list,
            [
                unittest.mock.call("berry_1", "bowl_main"),
                unittest.mock.call(
                    "berry_1",
                    "bowl_main",
                    relative_position=None,
                ),
            ],
        )
        self.assertEqual(executor._settle_scene.call_count, 1)
        executor._set_object_pose.assert_not_called()

    def test_load_initial_state_places_nested_supports_parent_first(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor.env = MagicMock()
        executor.env.sim = MagicMock()
        executor.env.objects = {
            "tray_main": object(),
            "bowl_main": object(),
            "berry_1": object(),
        }
        executor.get_scene_description = MagicMock(
            return_value={
                "objects": {
                    "tray_main": {"location": "counter_main"},
                    "bowl_main": {"location": "counter_main"},
                    "berry_1": {"location": "counter_main"},
                },
                "object_placements": {},
            }
        )
        executor._parse_agent_idx = MagicMock(return_value=0)
        executor._set_fixture_machine_state = MagicMock()
        executor._initialize_support_graph_from_scene = MagicMock()
        executor._infer_object_support_site = MagicMock(return_value=None)
        executor._current_support_object = MagicMock(return_value=None)
        executor._set_support_parent = MagicMock()
        executor._place_object_on_fixture = MagicMock()
        executor._place_on_object_center = MagicMock()
        executor._settle_and_reseat_supported_object = MagicMock()
        executor._repack_supported_children = MagicMock()
        executor._repair_supported_children_vertical_gaps = MagicMock()
        executor._settle_scene = MagicMock()
        executor._held_objects = {}
        executor._support_parents = {}
        executor._robot_spawn = "sim"
        executor._require_fixture = MagicMock()
        executor._require_object = MagicMock()
        executor.runner = MagicMock()
        executor.runner._fixtures = {"counter_main": object()}
        executor.runner._set_object_location = MagicMock()
        executor._resolve_support_target = MagicMock(return_value=("counter_main", None))
        executor._require_explicit_site_if_needed = MagicMock()

        summary = executor.load_initial_state(
            {
                "agents": {"agent_0": {"location": "counter_main", "held_object": None}},
                "objects": {
                    "tray_main": {
                        "location": "counter_main",
                        "object_type": "tray",
                    },
                    "berry_1": {
                        "location": "bowl_main",
                        "object_type": "berry",
                    },
                    "bowl_main": {
                        "location": "tray_main",
                        "object_type": "bowl",
                    },
                },
                "fixtures": {"counter_main": {"fixture_type": "counter_non_dining"}},
                "machine_state": {},
            }
        )

        self.assertTrue(summary["loaded"])
        self.assertEqual(
            executor._place_on_object_center.call_args_list,
            [
                unittest.mock.call("bowl_main", "tray_main"),
                unittest.mock.call("berry_1", "bowl_main"),
            ],
        )

    def test_fixture_pose_needs_reset_for_sideways_upright_object_in_receptacle(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._surface_pose_needs_reset = MagicMock(return_value=False)
        executor._fixture_placement_semantics = MagicMock(return_value="receptacle")
        executor._object_prefers_upright_on_surface = MagicMock(return_value=True)
        executor._object_is_upright = MagicMock(return_value=False)

        needs_reset = executor._fixture_pose_needs_reset(
            "condiment",
            "cab_main",
            site_id="shelf_0",
            quat_wxyz=np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
        )

        self.assertTrue(needs_reset)

    def test_settle_and_reseat_supported_object_does_not_raise_settled_z(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._set_support_parent = MagicMock()
        executor.runner = MagicMock()
        executor.runner._set_object_location = MagicMock()
        executor._settle_scene = MagicMock()
        executor._place_on_object_center = MagicMock()
        executor._set_object_pose = MagicMock()
        executor._support_object_prefers_upright = MagicMock(return_value=False)
        executor._support_object_geometry = MagicMock(
            return_value={"is_concave": False, "top_z": 0.0, "bottom_z": 0.0, "extent_z": 0.1}
        )
        executor._supported_object_vertical_gap = MagicMock(return_value=0.0)
        executor._get_object_pose = MagicMock(
            side_effect=[
                (np.array([0.0, 0.0, 0.18]), np.array([1.0, 0.0, 0.0, 0.0])),
                (np.array([0.0, 0.0, 0.26]), np.array([1.0, 0.0, 0.0, 0.0])),
            ]
        )

        executor._settle_and_reseat_supported_object("straw_main", "cup_main")

        pose_args = executor._set_object_pose.call_args.args
        self.assertEqual(pose_args[0], "straw_main")
        np.testing.assert_allclose(pose_args[1], np.array([0.0, 0.0, 0.18]))
        np.testing.assert_allclose(pose_args[2], np.array([1.0, 0.0, 0.0, 0.0]))
        self.assertEqual(executor._settle_scene.call_count, 2)

    def test_support_object_slot_positions_spread_two_items_across_tray(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._support_object_geometry = MagicMock(
            return_value={
                "center_xy": np.array([0.0, 0.0], dtype=float),
                "rot_xy": np.eye(2, dtype=float),
                "extent_x": 1.0,
                "extent_y": 0.4,
                "support_tokens": {"tray"},
                "is_concave": False,
            }
        )
        executor._support_object_placement_family = MagicMock(
            return_value="shallow_receptacle"
        )

        slot_positions = executor._support_object_slot_positions(
            "tray_main",
            count=2,
        )

        self.assertEqual(len(slot_positions), 2)
        self.assertLess(float(slot_positions[0][0]), 0.0)
        self.assertGreater(float(slot_positions[1][0]), 0.0)
        self.assertAlmostEqual(abs(float(slot_positions[0][0])), 0.22)
        self.assertAlmostEqual(abs(float(slot_positions[1][0])), 0.22)
        self.assertAlmostEqual(float(slot_positions[0][1]), 0.0)
        self.assertAlmostEqual(float(slot_positions[1][1]), 0.0)

    def test_supported_children_should_balance_bulky_pair_on_tray(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor.env = SimpleNamespace(objects={"tray_main": object(), "mug": object(), "kettle": object()})
        executor._held_by_robot = MagicMock(return_value=None)
        executor._support_object_tokens = MagicMock(return_value={"tray"})
        executor._scene_object_tokens = MagicMock(
            side_effect=lambda object_id: {object_id}
        )

        self.assertTrue(
            executor._supported_children_should_balance_slots(
                "tray_main",
                ["mug", "kettle"],
            )
        )

    def test_first_bulky_tray_item_uses_reserved_pair_slot(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._iter_direct_supported_children = MagicMock(return_value=[])
        executor._find_objects_on_support = MagicMock(return_value=[])
        executor._support_object_tokens = MagicMock(return_value={"tray"})
        executor._scene_object_tokens = MagicMock(return_value={"kettle"})
        executor._support_object_geometry = MagicMock(
            return_value={
                "center_xy": np.array([0.0, 0.0], dtype=float),
                "rot_xy": np.eye(2, dtype=float),
                "extent_x": 1.0,
                "extent_y": 0.4,
                "support_tokens": {"tray"},
                "is_concave": False,
            }
        )
        executor._support_object_placement_family = MagicMock(
            return_value="shallow_receptacle"
        )

        preferred = executor._incoming_support_slot_preference("tray_main", "kettle")

        np.testing.assert_allclose(preferred, np.array([-0.22, 0.0], dtype=float))

    def test_supported_children_does_not_balance_flat_meats_on_tray(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor.env = SimpleNamespace(objects={"tray_main": object(), "meat1": object(), "meat2": object()})
        executor._held_by_robot = MagicMock(return_value=None)
        executor._support_object_tokens = MagicMock(return_value={"tray"})
        executor._scene_object_tokens = MagicMock(return_value={"meat"})

        self.assertFalse(
            executor._supported_children_should_balance_slots(
                "tray_main",
                ["meat1", "meat2"],
            )
        )

    def test_supported_children_balances_hotdog_pair_on_plate_for_portion_task(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._task_name = "PortionHotDogs"
        executor.env = SimpleNamespace(
            objects={"plate1": object(), "hotdog_bun1": object(), "sausage1": object()}
        )
        executor._held_by_robot = MagicMock(return_value=None)
        executor._support_object_tokens = MagicMock(return_value={"plate"})
        executor._scene_object_tokens = MagicMock(
            side_effect=lambda object_id: (
                {"bun", "hotdog"} if "bun" in object_id else {"sausage"}
            )
        )

        self.assertTrue(
            executor._supported_children_should_balance_slots(
                "plate1",
                ["hotdog_bun1", "sausage1"],
            )
        )

    def test_portion_hotdogs_second_plate_item_uses_inboard_opposite_slot(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._task_name = "PortionHotDogs"
        executor._iter_direct_supported_children = MagicMock(return_value=["hotdog_bun1"])
        executor._find_objects_on_support = MagicMock(return_value=["hotdog_bun1"])
        executor._support_object_tokens = MagicMock(return_value={"plate"})
        executor._scene_object_tokens = MagicMock(
            side_effect=lambda object_id: (
                {"bun", "hotdog"} if "bun" in object_id else {"sausage"}
            )
        )
        executor._support_object_geometry = MagicMock(
            return_value={
                "center_xy": np.array([0.0, 0.0], dtype=float),
                "rot_xy": np.eye(2, dtype=float),
                "extent_x": 1.0,
                "extent_y": 1.0,
                "support_tokens": {"plate"},
                "is_concave": False,
            }
        )
        executor._support_object_anchor_axes_xy = MagicMock(
            return_value=(
                np.array([1.0, 0.0], dtype=float),
                np.array([0.0, 1.0], dtype=float),
            )
        )
        executor._get_object_pose = MagicMock(
            return_value=(np.array([-0.16, -0.05, 0.0], dtype=float), np.array([1.0, 0.0, 0.0, 0.0]))
        )

        preferred = executor._incoming_support_slot_preference("plate1", "sausage1")

        self.assertIsNotNone(preferred)
        self.assertGreater(float(preferred[0]), 0.0)
        self.assertLess(float(preferred[0]), 0.2)
        self.assertGreater(float(preferred[1]), 0.0)

    def test_settle_and_reseat_supported_object_reapplies_upright_pose_after_tip(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._set_support_parent = MagicMock()
        executor.runner = MagicMock()
        executor.runner._set_object_location = MagicMock()
        executor._settle_scene = MagicMock()
        executor._place_on_object_center = MagicMock()
        executor._set_object_pose = MagicMock()
        executor._support_object_prefers_upright = MagicMock(return_value=True)
        executor._object_is_upright = MagicMock(return_value=False)
        executor._get_object_pose = MagicMock(
            side_effect=[
                (np.array([0.0, 0.0, 0.18]), np.array([1.0, 0.0, 0.0, 0.0])),
                (np.array([0.0, 0.0, 0.18]), np.array([1.0, 0.0, 0.0, 0.0])),
                (np.array([0.0, 0.0, 0.18]), np.array([1.0, 0.0, 0.0, 0.0])),
            ]
        )

        executor._settle_and_reseat_supported_object("straw_main", "cup_main")

        self.assertEqual(executor._place_on_object_center.call_count, 2)

    def test_settle_and_reseat_supported_object_lowers_flat_support_gap_after_settle(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._set_support_parent = MagicMock()
        executor.runner = MagicMock()
        executor.runner._set_object_location = MagicMock()
        executor._settle_scene = MagicMock()
        executor._place_on_object_center = MagicMock()
        executor._set_object_pose = MagicMock()
        executor._support_object_prefers_upright = MagicMock(return_value=False)
        executor._support_object_geometry = MagicMock(
            return_value={"is_concave": False, "top_z": 0.0, "bottom_z": 0.0, "extent_z": 0.1}
        )
        executor._supported_object_vertical_gap = MagicMock(side_effect=[0.02, 0.0])
        executor._get_object_pose = MagicMock(
            side_effect=[
                (np.array([0.0, 0.0, 0.18]), np.array([1.0, 0.0, 0.0, 0.0])),
                (np.array([0.0, 0.0, 0.18]), np.array([1.0, 0.0, 0.0, 0.0])),
                (np.array([0.0, 0.0, 0.18]), np.array([1.0, 0.0, 0.0, 0.0])),
            ]
        )

        executor._settle_and_reseat_supported_object("bun_main", "plate_main")

        self.assertEqual(executor._settle_scene.call_count, 3)
        pose_args = executor._set_object_pose.call_args.args
        self.assertEqual(pose_args[0], "bun_main")
        np.testing.assert_allclose(pose_args[1], np.array([0.0, 0.0, 0.16]))

    def test_settle_and_reseat_supported_object_does_not_raise_sunk_deep_support_child(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._set_support_parent = MagicMock()
        executor.runner = MagicMock()
        executor.runner._set_object_location = MagicMock()
        executor._settle_scene = MagicMock()
        executor._place_on_object_center = MagicMock()
        executor._set_object_pose = MagicMock()
        executor._support_object_prefers_upright = MagicMock(return_value=False)
        executor._support_object_geometry = MagicMock(
            return_value={"is_concave": True, "top_z": 0.0, "bottom_z": 0.0, "extent_z": 0.1}
        )
        executor._support_object_uses_interior_floor = MagicMock(return_value=True)
        executor._supported_object_vertical_gap = MagicMock(return_value=-0.03)
        executor._iter_direct_supported_children = MagicMock(return_value=[])
        executor._supported_children_need_repack = MagicMock(return_value=False)
        executor._repair_supported_children_vertical_gaps = MagicMock()
        executor._get_object_pose = MagicMock(
            side_effect=[
                (np.array([0.0, 0.0, 0.18]), np.array([1.0, 0.0, 0.0, 0.0])),
                (np.array([0.0, 0.0, 0.18]), np.array([1.0, 0.0, 0.0, 0.0])),
                (np.array([0.0, 0.0, 0.15]), np.array([1.0, 0.0, 0.0, 0.0])),
            ]
        )

        executor._settle_and_reseat_supported_object("sausage_main", "bowl_main")

        executor._set_object_pose.assert_not_called()
        executor._repair_supported_children_vertical_gaps.assert_not_called()
        executor._settle_scene.assert_not_called()

    def test_repair_supported_children_vertical_gaps_lowers_floating_deep_child(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._support_object_uses_interior_floor = MagicMock(return_value=True)
        executor._supported_object_vertical_gap = MagicMock(return_value=0.03)
        executor._get_object_pose = MagicMock(
            return_value=(np.array([0.0, 0.0, 0.18]), np.array([1.0, 0.0, 0.0, 0.0]))
        )
        executor._set_object_pose = MagicMock()
        executor._set_support_parent = MagicMock()
        executor.runner = MagicMock()
        executor.runner._set_object_location = MagicMock()

        executor._repair_supported_children_vertical_gaps(
            "bowl_main",
            ["bun_main"],
            support_geometry={"bottom_z": 0.0, "top_z": 0.2, "extent_z": 0.1},
        )

        pose_args = executor._set_object_pose.call_args.args
        self.assertEqual(pose_args[0], "bun_main")
        np.testing.assert_allclose(pose_args[1], np.array([0.0, 0.0, 0.15]))

    def test_repair_supported_children_vertical_gaps_does_not_raise_sunken_deep_child(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._support_object_uses_interior_floor = MagicMock(return_value=True)
        executor._supported_object_vertical_gap = MagicMock(return_value=-0.03)
        executor._get_object_pose = MagicMock(
            return_value=(np.array([0.0, 0.0, 0.18]), np.array([1.0, 0.0, 0.0, 0.0]))
        )
        executor._set_object_pose = MagicMock()
        executor._set_support_parent = MagicMock()
        executor.runner = MagicMock()
        executor.runner._set_object_location = MagicMock()

        executor._repair_supported_children_vertical_gaps(
            "bowl_main",
            ["bun_main"],
            support_geometry={"bottom_z": 0.0, "top_z": 0.2, "extent_z": 0.1},
        )

        executor._set_object_pose.assert_not_called()

    def test_find_objects_on_support_accepts_children_below_bowl_center(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor.env = SimpleNamespace(
            obj_body_id={"bowl_main": 0},
            objects={"bowl_main": object(), "bun_main": object()},
            sim=SimpleNamespace(
                data=SimpleNamespace(
                    body_xpos=np.array([[0.0, 0.0, 0.30]], dtype=float)
                )
            ),
        )
        executor._support_object_geometry = MagicMock(
            return_value={"top_z": 0.38, "bottom_z": 0.22, "extent_z": 0.16}
        )
        executor._support_object_uses_interior_floor = MagicMock(return_value=True)
        executor._get_object_pose = MagicMock(return_value=(np.array([0.01, 0.0, 0.24]), np.array([1.0, 0.0, 0.0, 0.0])))

        bowl = SimpleNamespace(horizontal_radius=0.10)
        bun = SimpleNamespace(horizontal_radius=0.04)
        executor.env.objects = {"bowl_main": bowl, "bun_main": bun}

        contained = executor._find_objects_on_support("bowl_main")

        self.assertEqual(contained, ["bun_main"])

    def test_repack_supported_children_uses_size_aware_object_center_placement(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor.env = SimpleNamespace(objects={"big_child": object(), "small_child": object()})
        executor._held_by_robot = MagicMock(return_value=None)
        executor._require_object = MagicMock(
            side_effect=lambda object_id: SimpleNamespace(
                horizontal_radius=0.08 if object_id == "big_child" else 0.04
            )
        )
        executor._place_on_object_center = MagicMock()
        executor._set_support_parent = MagicMock()
        executor.runner = MagicMock()
        executor.runner._set_object_location = MagicMock()

        executor._repack_supported_children(
            "bowl_main",
            ["small_child", "big_child"],
        )

        self.assertEqual(
            executor._place_on_object_center.call_args_list,
            [
                unittest.mock.call("big_child", "bowl_main"),
                unittest.mock.call("small_child", "bowl_main"),
            ],
        )

    def test_support_object_placement_family_uses_shallow_receptacle_tokens(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._support_object_tokens = MagicMock(return_value={"plate"})

        family = executor._support_object_placement_family(
            "plate_main",
            {"is_concave": False},
        )

        self.assertEqual(family, "shallow_receptacle")

    def test_support_object_placement_family_uses_deep_receptacle_tokens(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._support_object_tokens = MagicMock(return_value={"bowl"})

        family = executor._support_object_placement_family(
            "bowl_main",
            {"is_concave": False},
        )

        self.assertEqual(family, "deep_receptacle")

    def test_object_prefers_upright_on_surface_for_bottle_tokens(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._scene_object_tokens = MagicMock(return_value={"bottle"})
        executor._get_object_placement_metadata = MagicMock(
            return_value={"size": [0.05, 0.05, 0.20]}
        )

        self.assertTrue(executor._object_prefers_upright_on_surface("bottle_main"))

    def test_object_prefers_upright_on_surface_rejects_flat_food_shapes(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._scene_object_tokens = MagicMock(return_value={"sausage"})
        executor._get_object_placement_metadata = MagicMock(
            return_value={"size": [0.13, 0.19, 0.19]}
        )

        self.assertFalse(executor._object_prefers_upright_on_surface("sausage_main"))

    def test_object_prefers_upright_on_surface_rejects_baguette_and_cutlery_profiles(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._scene_object_tokens = MagicMock(return_value={"baguette"})
        executor._get_object_placement_metadata = MagicMock(
            return_value={"size": [0.19198154, 0.06302114, 0.04465542]}
        )
        self.assertFalse(executor._object_prefers_upright_on_surface("baguette_main"))

        executor._scene_object_tokens = MagicMock(return_value={"fork"})
        executor._get_object_placement_metadata = MagicMock(
            return_value={"size": [0.04250569, 0.25490584, 0.07001984]}
        )
        self.assertFalse(executor._object_prefers_upright_on_surface("fork_main"))

    def test_object_prefers_upright_on_surface_rejects_bun_tokens(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._scene_object_tokens = MagicMock(return_value={"hotdog", "bun"})
        executor._get_object_placement_metadata = MagicMock(
            return_value={"size": [0.12495396, 0.18144512, 0.14319583]}
        )

        self.assertFalse(executor._object_prefers_upright_on_surface("hotdog_bun1"))

    def test_object_has_directional_long_axis_for_bun_tokens(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._scene_object_tokens = MagicMock(return_value={"hotdog", "bun"})
        executor._get_object_placement_metadata = MagicMock(
            return_value={"size": [0.12495396, 0.18144512, 0.14319583]}
        )

        self.assertTrue(executor._object_has_directional_long_axis("hotdog_bun1"))

    def test_object_is_flat_top_up_on_surface_rejects_inverted_mesh_normal(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._flat_alignment_up_axis = MagicMock(
            return_value=np.array([0.0, 0.0, 1.0], dtype=float)
        )

        self.assertTrue(
            executor._object_is_flat_top_up_on_surface(
                "hotdog_bun1",
                np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            )
        )
        self.assertFalse(
            executor._object_is_flat_top_up_on_surface(
                "hotdog_bun1",
                np.array([0.0, 1.0, 0.0, 0.0], dtype=float),
            )
        )

    def test_object_requires_top_up_flat_on_support_for_hotdog_items_on_plate(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._scene_object_tokens = MagicMock(return_value={"hotdog", "bun"})
        executor._support_object_tokens = MagicMock(return_value={"plate"})
        self.assertTrue(
            executor._object_requires_top_up_flat_on_support(
                "hotdog_bun1",
                "plate1",
            )
        )

        executor._support_object_tokens = MagicMock(return_value={"tray"})
        self.assertFalse(
            executor._object_requires_top_up_flat_on_support(
                "hotdog_bun1",
                "tray1",
            )
        )

    def test_support_object_candidate_quaternions_uses_single_sign_for_top_up_items(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._get_object_pose = MagicMock(
            return_value=(
                np.array([0.0, 0.0, 0.0], dtype=float),
                np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            )
        )
        executor._support_object_prefers_upright = MagicMock(return_value=False)
        executor._object_requires_top_up_flat_on_support = MagicMock(return_value=True)
        executor._object_prefers_upright_on_surface = MagicMock(return_value=False)
        executor._support_object_tokens = MagicMock(return_value=set())
        executor._object_has_directional_long_axis = MagicMock(return_value=False)
        executor._thinnest_object_local_axis = MagicMock(
            return_value=np.array([0.0, 0.0, 1.0], dtype=float)
        )
        executor._quat_align_vectors = MagicMock(
            return_value=np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        )
        executor._object_is_flat_top_up_on_surface = MagicMock(return_value=True)
        executor._object_bbox_metadata_for_quat = MagicMock(
            return_value={"spans": np.array([0.2, 0.1, 0.05], dtype=float)}
        )

        quats = executor._support_object_candidate_quaternions(
            "hotdog_bun1",
            "plate1",
        )

        self.assertTrue(quats)
        self.assertEqual(executor._quat_align_vectors.call_count, 1)

    def test_flat_alignment_up_axis_flips_utensil_mesh_normal(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._thinnest_object_local_axis = MagicMock(
            return_value=np.array([0.0, 0.0, 1.0], dtype=float)
        )
        executor._scene_object_tokens = MagicMock(return_value={"fork"})

        np.testing.assert_allclose(
            executor._flat_alignment_up_axis("fork_main"),
            np.array([0.0, 0.0, -1.0], dtype=float),
        )

        executor._scene_object_tokens = MagicMock(return_value={"baguette"})
        np.testing.assert_allclose(
            executor._flat_alignment_up_axis("baguette_main"),
            np.array([0.0, 0.0, 1.0], dtype=float),
        )

    @patch("robocasa.utils.sim_tool_executor.get_fixture_aabb")
    def test_open_sliding_part_marks_swept_volume_and_moves_children(
        self,
        mock_get_fixture_aabb,
    ):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        fixture = MagicMock()
        executor._require_fixture = MagicMock(return_value=fixture)
        executor._robot_near_fixture = MagicMock(return_value=True)
        executor._move_robot_near_fixture_with_retries = MagicMock()
        executor._sync_held_object = MagicMock()
        executor._resolve_joint_name = MagicMock(return_value="top_drawer_slidejoint")
        executor._sliding_joint_open_fraction = MagicMock(return_value=0.55)
        executor._set_named_joint = MagicMock()
        executor._iter_direct_supported_children = MagicMock(return_value=["cup_main"])
        executor._held_by_robot = MagicMock(return_value=None)
        executor._get_object_pose = MagicMock(
            return_value=(
                np.array([0.10, 0.0, 0.20], dtype=float),
                np.array([1.0, 0.0, 0.0, 0.0], dtype=float),
            )
        )
        executor._set_object_pose = MagicMock()
        executor._set_support_parent = MagicMock()
        executor._settle_scene = MagicMock()
        executor.env = SimpleNamespace(objects={"cup_main": object()})
        executor.runner = MagicMock()
        executor.runner._set_object_location = MagicMock()
        executor.runner._occupancy_grid = MagicMock()
        executor.runner._occupancy_grid.is_free = MagicMock(return_value=False)
        executor.runner._get_robot_position = MagicMock(
            return_value=np.array([0.25, 0.0, 0.0], dtype=float)
        )
        executor.runner._move_robot_near_fixture = MagicMock()

        mock_get_fixture_aabb.side_effect = [
            (np.array([0.0, -0.1]), np.array([0.2, 0.1])),
            (np.array([0.0, -0.1]), np.array([0.35, 0.1])),
        ]

        result = executor.open_sliding_part("drawer_main", "drawer", robot_idx=0)

        self.assertTrue(result.success)
        pose_args = executor._set_object_pose.call_args.args
        self.assertEqual(pose_args[0], "cup_main")
        np.testing.assert_allclose(
            pose_args[1],
            np.array([0.25, 0.0, 0.20], dtype=float),
        )
        executor.runner._occupancy_grid.mark_world_aabb_occupied.assert_called_once()
        executor.runner._move_robot_near_fixture.assert_called_once_with(
            0,
            "drawer_main",
            require_front=True,
        )

    def test_sliding_joint_open_fraction_keeps_drawers_partial(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)

        self.assertAlmostEqual(
            executor._sliding_joint_open_fraction("top_drawer_slidejoint"),
            0.55,
        )
        self.assertAlmostEqual(
            executor._sliding_joint_open_fraction(
                "toaster_oven_rack_joint",
                part_id="sliding",
            ),
            1.0,
        )

    def test_support_object_prefers_upright_for_straw_tokens(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._support_object_tokens = MagicMock(return_value={"cup"})
        executor._scene_object_tokens = MagicMock(return_value={"straw"})
        executor._get_object_placement_metadata = MagicMock(
            return_value={"size": [0.01, 0.01, 0.20]}
        )

        self.assertTrue(executor._support_object_prefers_upright("cup_main", "straw_main"))

    def test_supported_object_vertical_gap_uses_interior_floor_for_bowl_family(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._support_object_uses_interior_floor = MagicMock(return_value=True)
        executor._support_object_floor_clearance = MagicMock(return_value=0.01)
        executor._get_object_pose = MagicMock(
            return_value=(np.array([0.0, 0.0, 0.95]), np.array([1.0, 0.0, 0.0, 0.0]))
        )
        fake_obj = SimpleNamespace(
            get_bbox_points=lambda trans, rot: np.array(
                [
                    [-0.02, -0.02, trans[2] - 0.05],
                    [0.02, 0.02, trans[2] + 0.05],
                ],
                dtype=float,
            )
        )
        executor._require_object = MagicMock(return_value=fake_obj)

        gap = executor._supported_object_vertical_gap(
            "bun_main",
            "bowl_main",
            support_geometry={
                "top_z": 1.0,
                "bottom_z": 0.9,
                "extent_z": 0.1,
                "placement_family": "deep_receptacle",
                "is_concave": False,
            },
        )

        self.assertAlmostEqual(gap, -0.024)

    def test_bowl_support_children_skip_settle_and_repack(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._support_object_tokens = MagicMock(return_value={"bowl"})
        executor._support_object_prefers_upright = MagicMock(return_value=False)
        executor._set_support_parent = MagicMock()
        executor.runner = SimpleNamespace(_set_object_location=MagicMock())
        executor._place_on_object_center = MagicMock()
        executor._settle_scene = MagicMock()

        executor._settle_and_reseat_supported_object("bun_main", "bowl_main")

        executor._place_on_object_center.assert_called_once_with(
            "bun_main",
            "bowl_main",
            relative_position=None,
        )
        executor._settle_scene.assert_not_called()

    def test_cutting_board_support_children_skip_settle_and_repack(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._support_object_tokens = MagicMock(return_value={"cutting", "board"})
        executor._support_object_prefers_upright = MagicMock(return_value=False)
        executor._set_support_parent = MagicMock()
        executor.runner = SimpleNamespace(_set_object_location=MagicMock())
        executor._place_on_object_center = MagicMock()
        executor._settle_scene = MagicMock()

        executor._settle_and_reseat_supported_object(
            "sausage_main",
            "cutting_board_main",
        )

        executor._place_on_object_center.assert_called_once_with(
            "sausage_main",
            "cutting_board_main",
            relative_position=None,
        )
        executor._settle_scene.assert_not_called()

    def test_resolve_fixture_site_id_accepts_dispenser_alias_for_bottom_site(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._get_fixture_reset_regions = MagicMock(return_value={"bottom": {}})
        executor.get_support_sites = MagicMock(return_value=["bottom"])

        site_id = executor._resolve_fixture_site_id(
            "coffee_machine_left_group",
            "coffee_machine_dispenser",
        )

        self.assertEqual(site_id, "bottom")

    def test_resolve_support_target_prefers_fixture_tokens_for_ambiguous_sites(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor.runner = SimpleNamespace(
            _fixtures={
                "dishwasher_left_group": SimpleNamespace(name="dishwasher_left_group"),
                "toaster_oven_main_group": SimpleNamespace(name="toaster_oven_main_group"),
            }
        )
        executor._get_fixture_type_name = MagicMock(
            side_effect=lambda fixture_id: {
                "dishwasher_left_group": "dishwasher",
                "toaster_oven_main_group": "toaster_oven",
            }[fixture_id]
        )
        executor._resolve_fixture_site_id = MagicMock(return_value="rack")

        fixture_id, site_id = executor._resolve_support_target("toaster_oven_rack")

        self.assertEqual(fixture_id, "toaster_oven_main_group")
        self.assertEqual(site_id, "rack")

    def test_find_placeable_surface_near_stove_prefers_adjacent_counter(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor.get_scene_description = MagicMock(
            return_value={
                "fixtures": {
                    "stove_1_main_group": {
                        "fixture_type": "stove",
                        "can_place_objects": True,
                        "position": [2.0, 0.0, 0.0],
                    },
                    "counter_1_main_group": {
                        "fixture_type": "counter",
                        "can_place_objects": True,
                        "position": [1.8, 0.0, 0.0],
                    },
                }
            }
        )

        support_id = executor._find_placeable_surface_near_fixture(
            "stove_1_main_group"
        )

        self.assertEqual(support_id, "counter_1_main_group")


class TestTrajectoryRunnerPlacementSites(unittest.TestCase):
    def test_explicit_site_disables_fallback_region(self):
        runner = TrajectoryRunner.__new__(TrajectoryRunner)
        runner.env = object()
        fixture = SimpleNamespace(
            get_reset_regions=lambda env: {
                "front_left": {"offset": (0.0, 0.0, 0.0), "size": (0.2, 0.2, 0.1)}
            },
            sample_reset_region=lambda env, min_size=None: {
                "offset": (1.0, 1.0, 0.0),
                "size": (0.1, 0.1, 0.1),
            },
        )

        regions = runner._get_fixture_reset_regions(
            fixture,
            site_id="rear_right",
        )

        self.assertEqual(regions, [])


if __name__ == "__main__":
    unittest.main()
