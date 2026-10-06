"""Rendering and camera helpers for ``TrajectoryRunner``."""

from __future__ import annotations

from typing import Sequence

import numpy as np

# Per-robot cameras to keep, in priority order.
# Not all robots have agentview cameras (robot1 often only has robotview + eye_in_hand).
_AGENT_CAMERA_SUFFIXES = [
    "agentview_center",
    "agentview_left",
    "agentview_right",
    "eye_in_hand",
]

# Room-view framing parameters. These are intentionally separate from top-view
# distance tuning so we can keep the oblique room camera tighter around the
# active task workspace while preserving enough margin to avoid accidental
# cropping.
ROOM_VIEW_FIXTURE_RADIUS = 1.35
ROOM_VIEW_XY_MARGIN = 1.36
ROOM_VIEW_Z_LOOKAT_FRACTION = 0.45
ROOM_VIEW_BASE_DISTANCE_SCALE = 1.00
ROOM_VIEW_MIN_DISTANCE = 4.0
TOP_VIEW_XY_MARGIN = 1.12
TOP_VIEW_MIN_DISTANCE = 6.0
DEFAULT_MULTI_ROBOT_COLORS: tuple[tuple[float, float, float, float], ...] = (
    (1.0, 1.0, 1.0, 1.0),  # white
    (1.0, 0.55, 0.10, 1.0),  # orange
)


class TrajectoryRunnerRenderingMixin:
    def _normalize_robot_colors(
        self,
        robot_colors: Sequence[Sequence[float]] | None,
    ) -> tuple[tuple[float, float, float, float], ...]:
        """Normalize robot colors to RGBA tuples aligned to the active robots."""
        if robot_colors is None:
            return tuple()

        normalized: list[tuple[float, float, float, float]] = []
        for color in robot_colors[: len(self.env.robots)]:
            rgba = tuple(float(channel) for channel in color)
            if len(rgba) == 3:
                rgba = rgba + (1.0,)
            if len(rgba) != 4:
                raise ValueError(
                    "Each robot color must be an RGB or RGBA sequence."
                )
            normalized.append(rgba)
        return tuple(normalized)

    def _apply_robot_colors(self) -> None:
        """Tint each robot's visual geoms and materials while preserving alpha."""
        if not self._robot_colors:
            return

        for robot_idx, robot in enumerate(self.env.robots):
            if robot_idx >= len(self._robot_colors):
                break
            rgb = np.asarray(self._robot_colors[robot_idx][:3], dtype=float)
            seen_material_ids: set[int] = set()
            for geom_name in robot.robot_model.visual_geoms:
                geom_id = self.env.sim.model.geom_name2id(geom_name)
                geom_rgba = self.env.sim.model.geom_rgba[geom_id]
                geom_rgba[:3] = rgb
                material_id = int(self.env.sim.model.geom_matid[geom_id])
                if material_id >= 0 and material_id not in seen_material_ids:
                    material_rgba = self.env.sim.model.mat_rgba[material_id]
                    material_rgba[:3] = rgb
                    seen_material_ids.add(material_id)

    def _build_camera_list(self) -> list[str]:
        """Build the trimmed camera list: per-robot cameras + shared room_view."""
        all_cams = set(
            self.env.sim.model.camera_id2name(i)
            for i in range(self.env.sim.model.ncam)
        )
        cameras = []
        for robot_idx in range(self._num_robots):
            for suffix in _AGENT_CAMERA_SUFFIXES:
                cam = f"robot{robot_idx}_{suffix}"
                if cam in all_cams:
                    cameras.append(cam)
        # room_view is rendered via free camera (shared across agents)
        cameras.append("room_view")
        cameras.append("top_view")
        return cameras

    def _collect_scene_points(
        self,
        include_objects: bool = True,
        include_robots: bool = True,
        fixture_ids: set[str] | None = None,
    ) -> np.ndarray:
        points = []

        for fixture_id, fixture in self.env.fixtures.items():
            if fixture_ids is not None and fixture_id not in fixture_ids:
                continue
            if hasattr(fixture, "get_bbox_points"):
                try:
                    bbox_points = np.asarray(fixture.get_bbox_points(), dtype=float)
                except Exception:
                    bbox_points = None
                if bbox_points is not None and bbox_points.ndim == 2 and bbox_points.shape[1] >= 3:
                    points.append(bbox_points[:, :3])
                    continue

            if hasattr(fixture, "pos") and fixture.pos is not None:
                points.append(np.asarray(fixture.pos, dtype=float).reshape(1, 3))

        if include_objects and hasattr(self.env, "obj_body_id"):
            for body_id in self.env.obj_body_id.values():
                try:
                    obj_pos = self.env.sim.data.body_xpos[body_id].copy()
                except Exception:
                    continue
                points.append(np.asarray(obj_pos, dtype=float).reshape(1, 3))

        if include_robots:
            for robot_idx in range(self._num_robots):
                points.append(self._get_robot_position(robot_idx).reshape(1, 3))

        if not points:
            return np.zeros((0, 3), dtype=float)
        return np.concatenate(points, axis=0)

    def _collect_focus_points(self, fixture_radius: float) -> np.ndarray:
        """Collect points around fixtures near the current task workspace."""
        focus_fixture_ids: set[str] = set()
        focus_centers_xy = []

        if hasattr(self.env, "obj_body_id"):
            for body_id in self.env.obj_body_id.values():
                try:
                    obj_pos = self.env.sim.data.body_xpos[body_id].copy()
                except Exception:
                    continue
                focus_centers_xy.append(obj_pos[:2])

        for robot_idx in range(self._num_robots):
            focus_centers_xy.append(self._get_robot_position(robot_idx)[:2])

        if not focus_centers_xy:
            return self._collect_scene_points(include_objects=True, include_robots=True)

        for fixture_id, fixture in self.env.fixtures.items():
            if not hasattr(fixture, "pos") or fixture.pos is None:
                continue
            fixture_xy = np.asarray(fixture.pos[:2], dtype=float)
            if any(
                float(np.linalg.norm(fixture_xy - center_xy)) <= fixture_radius
                for center_xy in focus_centers_xy
            ):
                focus_fixture_ids.add(fixture_id)

        if not focus_fixture_ids:
            return self._collect_scene_points(include_objects=True, include_robots=True)

        return self._collect_scene_points(
            include_objects=True,
            include_robots=True,
            fixture_ids=focus_fixture_ids,
        )

    def _collect_room_view_points(self) -> np.ndarray:
        """Collect points that determine room camera framing.

        In full_scene_view mode, frames all fixtures so the entire kitchen is
        visible (useful for trajectory execution where robots navigate widely).
        Otherwise, frames only fixtures near the task workspace.
        """
        if self._full_scene_view:
            return self._collect_scene_points(
                include_objects=True,
                include_robots=True,
            )
        return self._collect_focus_points(ROOM_VIEW_FIXTURE_RADIUS)

    def _collect_top_view_points(self) -> np.ndarray:
        """Collect points that determine top camera framing.

        Uses the same point set as room_view for consistency.
        """
        return self._collect_room_view_points()

    def _compute_room_cam_config(self, base_cam_config: dict) -> dict:
        """Derive an oblique room camera from the current scene footprint."""
        scene_points = self._collect_room_view_points()
        if scene_points.size == 0:
            return dict(base_cam_config)

        min_xyz = np.min(scene_points, axis=0)
        max_xyz = np.max(scene_points, axis=0)
        center_xyz = 0.5 * (min_xyz + max_xyz)
        z_extent = max_xyz[2] - min_xyz[2]

        fovy_deg = float(getattr(self.env.sim.model.vis.global_, "fovy", 45.0))
        lookat = np.asarray(base_cam_config["lookat"], dtype=float).copy()
        lookat[0] = center_xyz[0]
        lookat[1] = center_xyz[1]
        lookat[2] = max(
            lookat[2],
            float(min_xyz[2] + ROOM_VIEW_Z_LOOKAT_FRACTION * max(z_extent, 1.0)),
        )

        azimuth_rad = np.deg2rad(float(base_cam_config["azimuth"]))
        elevation_rad = np.deg2rad(float(base_cam_config["elevation"]))
        forward = np.array(
            [
                -np.cos(elevation_rad) * np.cos(azimuth_rad),
                -np.cos(elevation_rad) * np.sin(azimuth_rad),
                -np.sin(elevation_rad),
            ],
            dtype=float,
        )
        forward /= np.linalg.norm(forward)
        world_up = np.array([0.0, 0.0, 1.0], dtype=float)
        right = np.cross(forward, world_up)
        if np.linalg.norm(right) < 1e-6:
            right = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        up /= np.linalg.norm(up)

        centered_points = scene_points - lookat
        right_extent = float(np.max(np.abs(centered_points @ right)))
        up_extent = float(np.max(np.abs(centered_points @ up)))

        half_fovy_rad = np.deg2rad(np.clip(fovy_deg, 1.0, 89.0) / 2.0)
        aspect = max(float(self.render_width) / float(self.render_height), 1e-6)
        half_fovx_rad = np.arctan(np.tan(half_fovy_rad) * aspect)
        required_distance_x = right_extent / max(np.tan(half_fovx_rad), 1e-6)
        required_distance_y = up_extent / max(np.tan(half_fovy_rad), 1e-6)
        required_distance = max(required_distance_x, required_distance_y) * ROOM_VIEW_XY_MARGIN

        return dict(
            lookat=lookat.tolist(),
            distance=float(
                max(
                    ROOM_VIEW_MIN_DISTANCE,
                    base_cam_config["distance"] * ROOM_VIEW_BASE_DISTANCE_SCALE,
                    required_distance,
                )
            ),
            azimuth=float(base_cam_config["azimuth"]),
            elevation=float(base_cam_config["elevation"]),
        )

    def _compute_top_cam_config(self, room_cam_config: dict) -> dict:
        """Derive an overhead camera that keeps the active kitchen room in frame."""
        scene_points = self._collect_top_view_points()
        if scene_points.size == 0:
            scene_points = self._collect_scene_points(include_objects=True, include_robots=True)
        if scene_points.size == 0:
            return dict(
                lookat=list(room_cam_config["lookat"]),
                distance=max(room_cam_config["distance"] * 1.8, 8.0),
                azimuth=room_cam_config["azimuth"],
                elevation=-89.0,
            )

        xy_points = scene_points[:, :2]
        min_xy = np.min(xy_points, axis=0)
        max_xy = np.max(xy_points, axis=0)
        center_xy = 0.5 * (min_xy + max_xy)
        max_radius = np.max(np.linalg.norm(xy_points - center_xy, axis=1))

        fovy_deg = float(getattr(self.env.sim.model.vis.global_, "fovy", 45.0))
        half_fovy_rad = np.deg2rad(np.clip(fovy_deg, 1.0, 89.0) / 2.0)
        required_distance = (max_radius / np.tan(half_fovy_rad)) * TOP_VIEW_XY_MARGIN

        lookat = np.asarray(room_cam_config["lookat"], dtype=float).copy()
        lookat[:2] = center_xy
        return dict(
            lookat=lookat.tolist(),
            distance=float(max(TOP_VIEW_MIN_DISTANCE, required_distance)),
            azimuth=float(room_cam_config["azimuth"]),
            elevation=-89.0,
        )

    def _render_free_camera(self, cam_config: dict) -> np.ndarray:
        """Render a free camera with an explicit camera config."""
        render_ctx = self._ensure_offscreen_render_context()
        if render_ctx is None:
            return np.zeros((self.render_height, self.render_width, 3), dtype=np.uint8)

        render_ctx.cam.lookat[:] = cam_config["lookat"]
        render_ctx.cam.distance = cam_config["distance"]
        render_ctx.cam.azimuth = cam_config["azimuth"]
        render_ctx.cam.elevation = cam_config["elevation"]
        try:
            render_ctx.render(
                width=self.render_width,
                height=self.render_height,
                camera_id=-1,
            )
            return render_ctx.read_pixels(self.render_width, self.render_height)[::-1]
        except AttributeError:
            # Some MuJoCo offscreen contexts can become partially torn down
            # after repeated sweeps. Rendering should not fail trajectory
            # execution; regular camera frames still cover the action.
            return np.zeros((self.render_height, self.render_width, 3), dtype=np.uint8)

    def _ensure_offscreen_render_context(self):
        """Ensure MuJoCo offscreen context exists; return it when available."""
        render_ctx = getattr(self.env.sim, "_render_context_offscreen", None)
        if render_ctx is not None:
            return render_ctx
        try:
            from robosuite.utils.binding_utils import MjRenderContextOffscreen

            device_id = int(getattr(self.env, "render_gpu_device_id", -1))
            render_ctx = MjRenderContextOffscreen(self.env.sim, device_id=device_id)
            self.env.sim.add_render_context(render_ctx)
            return getattr(self.env.sim, "_render_context_offscreen", None)
        except Exception:
            return None

    def _render_room_view(self) -> np.ndarray:
        """Render the layout-wide room camera."""
        return self._render_free_camera(self._room_cam_config)

    def _render_top_view(self) -> np.ndarray:
        """Render an overhead free camera focused on the active kitchen room."""
        return self._render_free_camera(self._top_cam_config)
