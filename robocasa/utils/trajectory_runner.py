"""
Trajectory runner for multi-agent task visualization.

Builds a Kitchen environment, extracts the scene description (fixtures, objects),
and executes trajectories defined as sequences of three primitives:

  - move_object(object, from, to)
  - interact(fixture, action, ...)
  - communicate(to, message)

Each step produces before/after visual observations from multiple cameras.
Robots are teleported near the fixture they interact with so camera views match.

Usage:

    runner = TrajectoryRunner(task_name="Kitchen", robots=2, layout=11, style=34)
    scene = runner.get_scene_description()
    # ... pass scene to LLM, get trajectory back ...
    result = runner.run(trajectory)
    # result.steps[i].before / .after  -> dict of camera_name -> np.ndarray
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import robosuite.utils.transform_utils as T
from robosuite.controllers import load_composite_controller_config
from robosuite.environments.base import make

import robocasa  # noqa: F401  — registers envs
import robocasa.utils.camera_utils as CamUtils
import robocasa.utils.env_utils as EnvUtils
import robocasa.utils.object_utils as OU
from robocasa.models.fixtures import FixtureType
from robocasa.wrappers.enclosing_wall_render_wrapper import EnclosingWallRenderWrapper
from robocasa.models.fixtures.fixture import Fixture
from robocasa.models.fixtures.fixture_utils import fixture_is_type
from robocasa.utils.occupancy_grid import OccupancyGrid
from robotalk.tasks.shared.constants import EXCLUSIVE_FIXTURE_TYPES
from robocasa.utils.trajectory_runner_rendering import (
    DEFAULT_MULTI_ROBOT_COLORS,
    TrajectoryRunnerRenderingMixin,
)

# Minimum 2-D distance between robots; used as a safety check.
# Two robots side-by-side need ~0.40 m (0.18 radius × 2 + margin).
MIN_ROBOT_SEPARATION = 0.40
_OBJECT_PLACEMENT_MARGIN = 0.01
_OBJECT_PLACEMENT_STEP = 0.08
_OBJECT_PLACEMENT_MAX_AXIS_SAMPLES = 7
_SWEEP_VERBOSE_ENV_VAR = "ROBOCASA_SWEEP_VERBOSE"
_COUNTERTOP_APPLIANCE_TYPES = frozenset({
    "coffee_machine", "toaster", "toaster_oven", "blender", "stand_mixer",
    "electric_kettle",
})
_HORIZONTAL_SUPPORT_TYPES = frozenset({
    "counter", "counter_non_corner", "counter_non_dining", "dining_counter",
    "island",
})


# ---------------------------------------------------------------------------
# Fixture type classification
# ---------------------------------------------------------------------------

# Fixture types where objects can be placed on/in.
_PLACEABLE_FIXTURE_TYPES: set[int] = {
    FixtureType.COUNTER,
    FixtureType.COUNTER_NON_CORNER,
    FixtureType.COUNTER_NON_DINING,
    FixtureType.DINING_COUNTER,
    FixtureType.ISLAND,
    FixtureType.CABINET,
    FixtureType.CABINET_WITH_DOOR,
    FixtureType.CABINET_SINGLE_DOOR,
    FixtureType.CABINET_DOUBLE_DOOR,
    FixtureType.DRAWER,
    FixtureType.TOP_DRAWER,
    FixtureType.SINK,
    FixtureType.STOVE,
    FixtureType.MICROWAVE,
    FixtureType.OVEN,
    FixtureType.FRIDGE,
    FixtureType.COFFEE_MACHINE,
    FixtureType.DISH_RACK,
}

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


def _configure_mujoco_gl_backend(gl_backend: str) -> str:
    """Keep the requested MuJoCo backend aligned with robosuite runtime state."""

    normalized_backend = str(gl_backend).strip().lower()
    os.environ["MUJOCO_GL"] = normalized_backend
    if normalized_backend != "egl":
        # Remove stale EGL routing when the caller switches a reused process
        # back to CPU rendering.
        os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)

    binding_utils = sys.modules.get("robosuite.utils.binding_utils")
    if binding_utils is not None:
        # Robosuite caches the chosen backend at import time, so update the
        # cached value before any new render context is created.
        binding_utils._MUJOCO_GL = normalized_backend
    return normalized_backend


def _classify_fixture(fixture: Fixture) -> str | None:
    """Return the most specific FixtureType name for a fixture, or None."""
    # Check specific types before general ones to get the most useful label.
    priority_order = [
        FixtureType.COFFEE_MACHINE,
        FixtureType.MICROWAVE,
        FixtureType.STOVE,
        FixtureType.OVEN,
        FixtureType.SINK,
        FixtureType.FRIDGE,
        FixtureType.DISHWASHER,
        FixtureType.TOASTER,
        FixtureType.TOASTER_OVEN,
        FixtureType.BLENDER,
        FixtureType.STAND_MIXER,
        FixtureType.ELECTRIC_KETTLE,
        FixtureType.DISH_RACK,
        FixtureType.TOP_DRAWER,
        FixtureType.DRAWER,
        FixtureType.CABINET_SINGLE_DOOR,
        FixtureType.CABINET_DOUBLE_DOOR,
        FixtureType.CABINET_WITH_DOOR,
        FixtureType.CABINET,
        FixtureType.ISLAND,
        FixtureType.DINING_COUNTER,
        FixtureType.COUNTER_NON_DINING,
        FixtureType.COUNTER_NON_CORNER,
        FixtureType.COUNTER,
        FixtureType.STOOL,
        FixtureType.WINDOW,
    ]
    for ftype in priority_order:
        try:
            if fixture_is_type(fixture, ftype):
                return ftype.name.lower()
        except (ValueError, Exception):
            continue
    return None


def _get_interactions(fixture: Fixture) -> list[str]:
    """Return list of interaction verbs this fixture supports."""
    interactions = []
    if hasattr(fixture, "door_joint_names") and fixture.door_joint_names:
        interactions.extend(["open", "close"])
    # Knob / button joints — exclude door, drawer, and stack joints
    if hasattr(fixture, "_joint_infos"):
        skip_patterns = ("door", "drawer", "stack")
        non_door = [
            j
            for j in fixture._joint_infos
            if not any(p in j.lower() for p in skip_patterns)
        ]
        if non_door:
            interactions.append("turn_on")
    return interactions


def _fixture_extents_3d(fixture: Fixture) -> tuple[np.ndarray, np.ndarray] | None:
    """Return world-space 3-D bounds from a fixture's external sites."""
    try:
        points = np.asarray(
            [np.asarray(point, dtype=float) for point in fixture.get_ext_sites(
                all_points=True, relative=False
            )],
            dtype=float,
        )
    except Exception:
        return None
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] < 3:
        return None
    return points[:, :3].min(axis=0), points[:, :3].max(axis=0)


def _countertop_support_parent(
    appliance_name: str,
    appliance: Fixture,
    fixtures: dict[str, Fixture],
    fixture_types: dict[str, str],
) -> str:
    """Resolve the horizontal surface physically underneath an appliance.

    This deliberately does not consider cabinets or drawers. Their existing
    parent relationship describes a paired workspace and must remain intact.
    """
    appliance_bounds = _fixture_extents_3d(appliance)
    if appliance_bounds is None:
        raise RuntimeError(
            f"Cannot resolve supporting surface for {appliance_name!r}: "
            "appliance bounds are unavailable."
        )
    appliance_min, appliance_max = appliance_bounds
    candidates: list[tuple[tuple[float, float, float], str]] = []
    for surface_name, surface in fixtures.items():
        if surface_name == appliance_name:
            continue
        if fixture_types.get(surface_name) not in _HORIZONTAL_SUPPORT_TYPES:
            continue
        surface_bounds = _fixture_extents_3d(surface)
        if surface_bounds is None:
            continue
        surface_min, surface_max = surface_bounds
        overlap_x = min(appliance_max[0], surface_max[0]) - max(
            appliance_min[0], surface_min[0]
        )
        overlap_y = min(appliance_max[1], surface_max[1]) - max(
            appliance_min[1], surface_min[1]
        )
        if overlap_x <= 1e-6 or overlap_y <= 1e-6:
            continue
        vertical_gap = float(appliance_min[2] - surface_max[2])
        # Allow small mesh interpenetration and asset-dependent feet, but not
        # an overhead or distant nearby fixture.
        if not -0.08 <= vertical_gap <= 0.30:
            continue
        overlap_area = float(overlap_x * overlap_y)
        center_distance = float(np.linalg.norm(
            (appliance_min[:2] + appliance_max[:2]) / 2.0
            - (surface_min[:2] + surface_max[:2]) / 2.0
        ))
        candidates.append(((abs(vertical_gap), -overlap_area, center_distance), surface_name))

    if not candidates:
        raise RuntimeError(
            f"Cannot resolve supporting surface for countertop appliance "
            f"{appliance_name!r}."
        )
    candidates.sort()
    if len(candidates) > 1 and all(
        abs(a - b) <= 1e-6 for a, b in zip(candidates[0][0], candidates[1][0])
    ):
        raise RuntimeError(
            f"Ambiguous supporting surfaces for countertop appliance "
            f"{appliance_name!r}: {candidates[0][1]!r} and {candidates[1][1]!r}."
        )
    return candidates[0][1]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class FixtureInfo:
    """Scene description of a single fixture for the LLM."""

    fixture_id: str
    fixture_type: str
    position: list[float]
    interactions: list[str]
    can_place_objects: bool
    nearby_fixtures: list[str] = field(default_factory=list)
    parent_fixture: str | None = None  # counter/surface this fixture sits on

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class ObjectInfo:
    """Scene description of a single object."""

    object_id: str
    object_type: str
    # Either a fixture_id or another object_id when object-on-object support is
    # observable in the simulator state (e.g., steak on plate).
    location: str

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class StepResult:
    """Visual observations for one trajectory step."""

    step_index: int
    agent_id: str
    action: str
    args: dict
    before: dict[str, np.ndarray]  # camera_name -> image
    after: dict[str, np.ndarray]

    def save_images(self, output_dir: str | Path, format: str = "jpg"):
        """Save before/after images to disk."""
        import imageio

        output_dir = Path(output_dir)
        kwargs = {"quality": 85} if format in ("jpg", "jpeg") else {}
        for phase in ("before", "after"):
            images = getattr(self, phase)
            for cam_name, img in images.items():
                path = (
                    output_dir
                    / f"step_{self.step_index:03d}_{phase}_{cam_name}.{format}"
                )
                imageio.imwrite(str(path), img, **kwargs)


@dataclass
class AgentStepView:
    """One step from a single agent's perspective."""

    step_index: int
    action: str | None  # the action taken, or None if this agent didn't act
    args: dict | None
    message_received: str | None  # incoming message from other agent
    before: dict[str, np.ndarray]
    after: dict[str, np.ndarray]


@dataclass
class AgentTrajectory:
    """Per-agent view of a trajectory, for VLM training."""

    agent_id: str
    camera_names: list[str]
    initial_obs: dict[str, np.ndarray]
    steps: list[AgentStepView]
    scene: dict

    def save(self, output_dir: str | Path, format: str = "jpg"):
        """Save per-agent training data."""
        import imageio

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        kwargs = {"quality": 85} if format in ("jpg", "jpeg") else {}

        for cam_name, img in self.initial_obs.items():
            imageio.imwrite(
                str(output_dir / f"initial_{cam_name}.{format}"), img, **kwargs
            )
        for step in self.steps:
            for phase in ("before", "after"):
                images = getattr(step, phase)
                for cam_name, img in images.items():
                    path = (
                        output_dir
                        / f"step_{step.step_index:03d}_{phase}_{cam_name}.{format}"
                    )
                    imageio.imwrite(str(path), img, **kwargs)

        meta = {
            "agent_id": self.agent_id,
            "cameras": self.camera_names,
            "steps": [
                {
                    "step_index": s.step_index,
                    "action": s.action,
                    "args": s.args,
                    "message_received": s.message_received,
                }
                for s in self.steps
            ],
        }
        with open(output_dir / "agent_trajectory.json", "w") as f:
            json.dump(meta, f, indent=2)


@dataclass
class TrajectoryResult:
    """Full result from running a trajectory."""

    initial_obs: dict[str, np.ndarray]
    steps: list[StepResult]
    scene: dict

    def save(self, output_dir: str | Path, format: str = "jpg"):
        """Save all images and the scene description."""
        import imageio

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        kwargs = {"quality": 85} if format in ("jpg", "jpeg") else {}
        for cam_name, img in self.initial_obs.items():
            imageio.imwrite(
                str(output_dir / f"initial_{cam_name}.{format}"), img, **kwargs
            )
        for step in self.steps:
            step.save_images(output_dir, format=format)
        with open(output_dir / "scene.json", "w") as f:
            json.dump(self.scene, f, indent=2)

    def split_by_agent(self) -> dict[str, AgentTrajectory]:
        """
        Split the combined trajectory into per-agent views for VLM training.

        Each agent sees every step but with different info:
          - Steps where it acts: action/args populated
          - Steps where the other agent acts: action=None, just before/after images
          - Incoming communications: message_received is populated
          - Camera images filtered to this agent's cameras + shared room_view
        """
        agent_ids = sorted({s.agent_id for s in self.steps})
        all_cameras = list(self.initial_obs.keys())

        def agent_cameras(agent_id: str) -> list[str]:
            idx = agent_id.replace("agent_", "")
            prefix = f"robot{idx}_"
            agent_cams = [c for c in all_cameras if c.startswith(prefix)]
            shared = [c for c in all_cameras if c == "room_view"]
            return agent_cams + shared if agent_cams else all_cameras

        def filter_cameras(
            images: dict[str, np.ndarray], cameras: list[str]
        ) -> dict[str, np.ndarray]:
            return {k: v for k, v in images.items() if k in cameras}

        result = {}
        for agent_id in agent_ids:
            cams = agent_cameras(agent_id)
            agent_steps = []

            for step in self.steps:
                is_actor = step.agent_id == agent_id
                message_received = None

                if not is_actor and step.action == "communicate":
                    to = (step.args or {}).get("to")
                    if to == agent_id:
                        message_received = (step.args or {}).get("message")

                agent_steps.append(
                    AgentStepView(
                        step_index=step.step_index,
                        action=step.action if is_actor else None,
                        args=step.args if is_actor else None,
                        message_received=message_received,
                        before=filter_cameras(step.before, cams),
                        after=filter_cameras(step.after, cams),
                    )
                )

            result[agent_id] = AgentTrajectory(
                agent_id=agent_id,
                camera_names=cams,
                initial_obs=filter_cameras(self.initial_obs, cams),
                steps=agent_steps,
                scene=self.scene,
            )

        return result


# ---------------------------------------------------------------------------
# TrajectoryRunner
# ---------------------------------------------------------------------------

class TrajectoryRunner(TrajectoryRunnerRenderingMixin):
    """
    Builds a Kitchen env, extracts scene info, and executes trajectories
    by teleporting objects and toggling fixture states.
    Robots are moved near the fixture they interact with each step.
    """

    def __init__(
        self,
        task_name: str = "Kitchen",
        robots: int = 2,
        layout: int | None = None,
        style: int | None = None,
        seed: int | None = None,
        split: str | None = None,
        camera_names: list[str] | None = None,
        render_width: int = 512,
        render_height: int = 512,
        gl_backend: str = "osmesa",
        placement: str = "grid",
        cell_size: float = 0.05,
        align_to_wall: bool = True,
        standoff: float = 0.40,
        sample_spacing: float = 0.08,
        robot_radius: float = 0.18,
        full_scene_view: bool = False,
        robot_colors: Sequence[Sequence[float]] | None = DEFAULT_MULTI_ROBOT_COLORS,
        update_fxtr_cfg_dict: dict | None = None,
        trajectory_object_names: list[str] | tuple[str, ...] | None = None,
        trajectory_object_types: list[str] | tuple[str, ...] | None = None,
        trajectory_object_specs: dict | list | None = None,
    ):
        gl_backend = _configure_mujoco_gl_backend(gl_backend)

        self._full_scene_view = full_scene_view
        self._num_robots = robots
        self._grid_kwargs = dict(
            cell_size=cell_size,
            align_to_wall=align_to_wall,
            standoff=standoff,
            sample_spacing=sample_spacing,
        )
        robot_list = ["PandaOmron"] * robots

        env_kwargs = dict(
            env_name=task_name,
            robots=robot_list,
            controller_configs=load_composite_controller_config(robot="PandaOmron"),
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=False,
            ignore_done=True,
        )
        if split == "target":
            env_kwargs["obj_instance_split"] = "target"
            env_kwargs["layout_and_style_ids"] = list(zip(range(1, 11), range(1, 11)))
        elif split == "pretrain":
            env_kwargs["obj_instance_split"] = "pretrain"
            env_kwargs["layout_ids"] = -2
            env_kwargs["style_ids"] = -2
        elif split == "all":
            env_kwargs["layout_ids"] = -3
            env_kwargs["style_ids"] = -3
        elif split is not None:
            raise ValueError("split must be one of None, 'pretrain', 'target', or 'all'")
        if layout is not None:
            env_kwargs["layout_ids"] = [layout]
            env_kwargs.pop("layout_and_style_ids", None)
        if style is not None:
            env_kwargs["style_ids"] = [style]
            env_kwargs.pop("layout_and_style_ids", None)
        if seed is not None:
            env_kwargs["seed"] = seed
        if update_fxtr_cfg_dict is not None:
            env_kwargs["update_fxtr_cfg_dict"] = update_fxtr_cfg_dict
        if trajectory_object_names is not None:
            env_kwargs["trajectory_object_names"] = trajectory_object_names
        if trajectory_object_types is not None:
            env_kwargs["trajectory_object_types"] = trajectory_object_types
        if trajectory_object_specs is not None:
            env_kwargs["trajectory_object_specs"] = trajectory_object_specs

        self.env = make(**env_kwargs)
        # Wrap env to make enclosing walls translucent (required for room_view
        # free camera to see through walls, same as two_robot_video_sample.py)
        self.env = EnclosingWallRenderWrapper(self.env, alpha=0.1, enabled=True)
        self.env.reset()
        self._robot_colors = self._normalize_robot_colors(robot_colors)
        self._apply_robot_colors()

        self.render_width = render_width
        self.render_height = render_height

        # Build camera list: per-robot cameras + room_view
        if camera_names is None:
            self.camera_names = self._build_camera_list()
        else:
            self.camera_names = camera_names

        _ = placement  # Grid is the only supported placement backend today.
        self._refresh_scene_dependent_helpers()

    def _refresh_scene_dependent_helpers(self) -> None:
        # Build fixture index and the active occupancy-grid placement helpers.
        self._fixtures: dict[str, Fixture] = dict(self.env.fixtures)
        self._occupancy_grid = OccupancyGrid(self._fixtures, **self._grid_kwargs)

        base_room_cam_config = CamUtils.LAYOUT_CAMS.get(
            self.env.layout_id, CamUtils.DEFAULT_LAYOUT_CAM
        )
        self._room_cam_config = self._compute_room_cam_config(
            dict(base_room_cam_config)
        )
        self._top_cam_config = self._compute_top_cam_config(self._room_cam_config)

        self._scene = None
        self._object_locations = {}
        self._last_placement_diagnostics = None

    def reset_scene(self) -> None:
        self.env.reset()
        self._apply_robot_colors()
        self._refresh_scene_dependent_helpers()

    def _placement_debug_enabled(self) -> bool:
        return os.environ.get(_SWEEP_VERBOSE_ENV_VAR, "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    # ------------------------------------------------------------------
    # Robot positioning
    # ------------------------------------------------------------------

    def _get_robot_position(self, robot_idx: int) -> np.ndarray:
        """Return the current (x, y, z) world position of a robot's mobile base.

        Note: ``robot.robot_model.root_body`` (e.g. ``robot0_base``) is a
        fixed anchor at (10, 10, 0) — **not** the real base.  The actual
        movable platform is the ``mobilebase{idx}_base`` body.
        """
        body_name = f"mobilebase{robot_idx}_base"
        body_id = self.env.sim.model.body_name2id(body_name)
        return self.env.sim.data.body_xpos[body_id].copy()

    def _robots_too_close(self, idx_a: int, idx_b: int) -> bool:
        """Return True if two robots are closer than MIN_ROBOT_SEPARATION."""
        pa = self._get_robot_position(idx_a)
        pb = self._get_robot_position(idx_b)
        return float(np.linalg.norm(pa[:2] - pb[:2])) < MIN_ROBOT_SEPARATION

    def _lookup_named_world_xy(self, name: str) -> np.ndarray | None:
        """Return the world XY position of a named site, geom, or body."""
        sim = self.env.sim
        model = sim.model
        data = sim.data

        for id_lookup, pos_array in (
            (model.site_name2id, data.site_xpos),
            (model.geom_name2id, data.geom_xpos),
            (model.body_name2id, data.body_xpos),
        ):
            try:
                idx = id_lookup(name)
            except Exception:
                continue
            if idx is None or idx < 0:
                continue
            return np.asarray(pos_array[idx][:2], dtype=float).copy()
        return None

    def _scan_fixture_named_world_xy(
        self,
        fixture: Fixture,
        name_filter,
    ) -> list[np.ndarray]:
        """Return XY positions of named sim elements associated with *fixture*."""
        sim = self.env.sim
        model = sim.model
        data = sim.data
        prefixes = []
        for prefix in (
            getattr(fixture, "naming_prefix", None),
            getattr(fixture, "name", None),
        ):
            if isinstance(prefix, str) and prefix:
                prefixes.append(prefix)

        def _matches_prefix(name: str) -> bool:
            return any(name.startswith(prefix) for prefix in prefixes)

        matches: list[np.ndarray] = []
        for count, id_to_name, pos_array in (
            (model.nsite, model.site_id2name, data.site_xpos),
            (model.ngeom, model.geom_id2name, data.geom_xpos),
            (model.nbody, model.body_id2name, data.body_xpos),
        ):
            for idx in range(count):
                name = id_to_name(idx)
                if not isinstance(name, str) or not _matches_prefix(name):
                    continue
                if not name_filter(name.lower()):
                    continue
                matches.append(np.asarray(pos_array[idx][:2], dtype=float).copy())
        return matches

    def _get_fixture_front_target_xy(self, fixture_id: str) -> np.ndarray | None:
        """Return the fixture's front working target, preferring handle geometry."""
        fixture = self._fixtures.get(fixture_id)
        if fixture is None:
            return None

        # Coffee-machine rotation does not reliably identify the usable side.
        # Its pouring site and start button are the actual task-facing geometry.
        interaction_positions: list[np.ndarray] = []
        pouring_site = getattr(fixture, "_receptacle_pouring_site", None)
        if pouring_site is not None:
            pouring_site_name = pouring_site.get("name")
            if isinstance(pouring_site_name, str):
                pouring_position = self._lookup_named_world_xy(pouring_site_name)
                if pouring_position is not None:
                    interaction_positions.append(pouring_position)
        start_button_names = getattr(fixture, "_start_button_names", None)
        naming_prefix = getattr(fixture, "naming_prefix", "")
        if isinstance(start_button_names, (list, tuple)):
            for button_name in start_button_names:
                if not isinstance(button_name, str):
                    continue
                full_name = (
                    button_name
                    if button_name.startswith(str(naming_prefix))
                    else f"{naming_prefix}{button_name}"
                )
                button_position = self._lookup_named_world_xy(full_name)
                if button_position is not None:
                    interaction_positions.append(button_position)
        if interaction_positions:
            return np.mean(np.stack(interaction_positions), axis=0)

        explicit_handle_names: list[str] = []
        for attr_name in ("left_handle_name", "right_handle_name", "handle_name"):
            try:
                candidate = getattr(fixture, attr_name)
            except Exception:
                candidate = None
            if isinstance(candidate, str):
                explicit_handle_names.append(candidate)

        handle_positions = [
            pos
            for pos in (
                self._lookup_named_world_xy(name) for name in explicit_handle_names
            )
            if pos is not None
        ]
        if handle_positions:
            return np.mean(np.stack(handle_positions), axis=0)

        handle_positions = self._scan_fixture_named_world_xy(
            fixture,
            lambda name: "handle" in name
            and ("main" in name or name.endswith("_handle")),
        )
        if handle_positions:
            return np.mean(np.stack(handle_positions), axis=0)

        handle_positions = self._scan_fixture_named_world_xy(
            fixture,
            lambda name: "handle" in name,
        )
        if handle_positions:
            return np.mean(np.stack(handle_positions), axis=0)

        try:
            door_name = getattr(fixture, "door_name")
        except Exception:
            door_name = None
        if isinstance(door_name, str):
            door_pos = self._lookup_named_world_xy(door_name)
            if door_pos is not None:
                return door_pos

        return None

    def _resolve_fixture_target_pos(
        self,
        fixture_id: str,
        ref_object_id: str | None = None,
        ref_pos_override: np.ndarray | None = None,
        require_front: bool = False,
    ) -> np.ndarray | None:
        """Resolve the placement target for a fixture interaction."""
        if ref_pos_override is not None:
            return np.asarray(ref_pos_override, dtype=float)[:2]
        if require_front:
            return self._get_fixture_front_target_xy(fixture_id)
        return self._resolve_ref_object_pos(ref_object_id)

    def _get_kitchen_aabb(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (min_xy, max_xy) bounding box of all fixture positions."""
        if not hasattr(self, "_kitchen_aabb"):
            pts = []
            for fxtr in self._fixtures.values():
                if hasattr(fxtr, "pos") and fxtr.pos is not None:
                    pts.append(np.asarray(fxtr.pos, dtype=float)[:2])
            if pts:
                arr = np.stack(pts)
                # Small margin for robot standoff — tight enough to reject
                # positions outside walls, loose enough for the robot to stand
                # in front of wall-adjacent fixtures.
                margin = 0.3
                self._kitchen_aabb = (
                    arr.min(axis=0) - margin,
                    arr.max(axis=0) + margin,
                )
            else:
                self._kitchen_aabb = (np.array([-100, -100]), np.array([100, 100]))
        return self._kitchen_aabb

    def _is_valid_robot_position(self, pos_xy: np.ndarray) -> bool:
        """Check whether a 2D position is inside the reachable kitchen floor."""
        pos = np.asarray(pos_xy, dtype=float)[:2]
        # Hard boundary: must be within fixture AABB + small margin
        aabb_min, aabb_max = self._get_kitchen_aabb()
        if pos[0] < aabb_min[0] or pos[0] > aabb_max[0]:
            return False
        if pos[1] < aabb_min[1] or pos[1] > aabb_max[1]:
            return False
        # Grid check: must be on a reachable free cell
        if self._occupancy_grid is not None:
            return self._occupancy_grid.is_free(pos)
        return True  # no placement system available — assume valid

    def _get_kitchen_center(self) -> np.ndarray:
        """Return the approximate center of the kitchen floor (from fixtures)."""
        positions = []
        for fxtr in self._fixtures.values():
            if hasattr(fxtr, "pos") and fxtr.pos is not None:
                positions.append(np.asarray(fxtr.pos, dtype=float)[:2])
        if positions:
            return np.mean(np.stack(positions), axis=0)
        return np.array([0.0, 0.0])

    def _rescue_robot_to_kitchen(self, robot_idx: int) -> bool:
        """If the robot is outside the kitchen, move it to a safe position.

        Tries the room center first, then spirals outward to find a free cell.
        Returns True if the robot was rescued (or was already valid).
        """
        pos = self._get_robot_position(robot_idx)[:2]
        if self._is_valid_robot_position(pos):
            return True

        center = self._get_kitchen_center()
        if self._is_valid_robot_position(center):
            yaw = 0.0
            self._set_robot_pose(robot_idx, center, yaw)
            print(
                f"[rescue] robot{robot_idx} was outside kitchen at "
                f"({pos[0]:.2f}, {pos[1]:.2f}), moved to center "
                f"({center[0]:.2f}, {center[1]:.2f})"
            )
            return True

        # Spiral search from center for a free cell
        if self._occupancy_grid is not None:
            grid = self._occupancy_grid
            seed_r, seed_c = grid._world_to_grid(center)
            for radius in range(1, max(grid._rows, grid._cols)):
                for dr in range(-radius, radius + 1):
                    for dc in range(-radius, radius + 1):
                        if abs(dr) != radius and abs(dc) != radius:
                            continue  # only check perimeter
                        r, c = seed_r + dr, seed_c + dc
                        if 0 <= r < grid._rows and 0 <= c < grid._cols:
                            if not grid._grid[r, c]:
                                safe_pos = grid._grid_to_world(r, c)
                                self._set_robot_pose(robot_idx, safe_pos, 0.0)
                                print(
                                    f"[rescue] robot{robot_idx} was outside kitchen at "
                                    f"({pos[0]:.2f}, {pos[1]:.2f}), moved to "
                                    f"({safe_pos[0]:.2f}, {safe_pos[1]:.2f})"
                                )
                                return True

        print(
            f"[rescue] WARNING: robot{robot_idx} is outside kitchen at "
            f"({pos[0]:.2f}, {pos[1]:.2f}) and no safe position found"
        )
        return False

    def _offset_robot_beside_other(
        self,
        robot_idx: int,
        fixture_id: str,
        ref_object_id: str | None = None,
        ref_pos_override: np.ndarray | None = None,
        require_front: bool = False,
    ):
        """Re-place *robot_idx* beside the other robot at the same fixture.

        Re-queries placement with both the other robot's position and the
        current (colliding) position excluded.
        """
        if fixture_id not in self._fixtures:
            return
        fxtr = self._fixtures[fixture_id]
        ref_pos = self._resolve_fixture_target_pos(
            fixture_id,
            ref_object_id=ref_object_id,
            ref_pos_override=ref_pos_override,
            require_front=require_front,
        )

        robot_cells = []
        robot_positions = []
        for other_idx in range(self._num_robots):
            if other_idx != robot_idx:
                other_pos = self._get_robot_position(other_idx)[:2]
                robot_cells.append(self._occupancy_grid._world_to_grid(other_pos))
                robot_positions.append(other_pos)
        my_pos = self._get_robot_position(robot_idx)[:2]
        my_cell = self._occupancy_grid._world_to_grid(my_pos)
        if my_cell not in robot_cells:
            robot_cells.append(my_cell)
        result = self._occupancy_grid.find_placement(
            fxtr,
            robot_cells,
            ref_pos,
            robot_positions=robot_positions,
            require_front=require_front,
            prohibited_working_fixtures=self._exclusive_children_for_parent(fxtr),
        )

        if result is not None:
            pos_xy, yaw = result
            self._set_robot_pose(robot_idx, pos_xy, yaw)

    def _resolve_ref_object_pos(self, ref_object_id: str | None) -> np.ndarray | None:
        """Return the 2D world position of a reference object, or None."""
        if ref_object_id is None:
            return None
        try:
            body_id = self.env.obj_body_id.get(ref_object_id)
            if body_id is not None:
                return self.env.sim.data.body_xpos[body_id][:2].copy()
        except Exception:
            pass
        return None

    def _move_robot_near_fixture(
        self,
        robot_idx: int,
        fixture_id: str,
        ref_object_id: str | None = None,
        ref_pos_override: np.ndarray | None = None,
        require_front: bool = False,
    ) -> bool:
        """Teleport a robot's base near the given fixture, facing it.

        Uses occupancy-grid placement. If, after placement, another robot is
        too close (< MIN_ROBOT_SEPARATION), the moved robot is re-placed to an
        alternative position.

        Args:
            ref_pos_override: explicit 2D position to prefer (takes precedence
                over ``ref_object_id``).  Used when the target position is
                known before the object is actually placed (e.g.,
                ``place_on_surface`` pre-computes the landing spot).
            require_front: If True, robot must approach from the fixture's
                front face (for interactive fixtures like fridge, cabinet).
                If False, pick the closest valid position on any face.

        Returns:
            True if the robot was successfully placed, False if no valid
            position was found (robot stays at current position).
        """
        pre_pos = self._get_robot_position(robot_idx)[:2]
        _ = (  # debug pre-position info available if needed
            f"[move_robot] robot{robot_idx} -> {fixture_id}: "
            f"pre=({pre_pos[0]:.2f}, {pre_pos[1]:.2f}), "
            f"valid={self._is_valid_robot_position(pre_pos)}"
        )
        if fixture_id not in self._fixtures:
            return False
        fxtr = self._fixtures[fixture_id]
        ref_pos = self._resolve_fixture_target_pos(
            fixture_id,
            ref_object_id=ref_object_id,
            ref_pos_override=ref_pos_override,
            require_front=require_front,
        )

        result = self._move_robot_grid(robot_idx, fxtr, ref_pos, require_front)

        placed = False
        if result is not None:
            pos_xy, yaw = result
            self._set_robot_pose(robot_idx, pos_xy, yaw)
            placed = True

        # --- If overlapping another robot, shift to its side ---
        if placed:
            for other_idx in range(self._num_robots):
                if other_idx == robot_idx:
                    continue
                if self._robots_too_close(robot_idx, other_idx):
                    self._offset_robot_beside_other(
                        robot_idx,
                        fixture_id,
                        ref_object_id=ref_object_id,
                        ref_pos_override=ref_pos,
                        require_front=require_front,
                    )
                    break  # only one correction needed for 2-robot setups

        # --- Final safety: ensure robot is inside the kitchen ---
        # Placement or offset logic may have pushed the robot outside walls.
        post_pos = self._get_robot_position(robot_idx)[:2]
        valid = self._is_valid_robot_position(post_pos)
        if not valid:
            self._rescue_robot_to_kitchen(robot_idx)

        return placed

    def _move_robot_grid(
        self,
        robot_idx: int,
        fixture: Fixture,
        ref_pos: np.ndarray | None,
        require_front: bool = False,
    ) -> tuple[np.ndarray, float] | None:
        """Find placement using the occupancy grid."""
        robot_cells = []
        robot_positions = []
        for other_idx in range(self._num_robots):
            if other_idx == robot_idx:
                continue
            other_pos = self._get_robot_position(other_idx)[:2]
            robot_cells.append(self._occupancy_grid._world_to_grid(other_pos))
            robot_positions.append(other_pos)
        return self._occupancy_grid.find_placement(
            fixture,
            robot_cells,
            ref_pos,
            robot_positions=robot_positions,
            require_front=require_front,
            prohibited_working_fixtures=self._exclusive_children_for_parent(fixture),
        )

    def _exclusive_children_for_parent(self, parent_fixture: Fixture) -> list[Fixture]:
        """Return exclusive child fixtures whose workspaces a parent may not borrow."""
        if os.environ.get("ROBOCASA_RESERVE_EXCLUSIVE_CHILD_WORKSPACES", "1") == "0":
            return []
        scene = self.get_scene_description()
        fixtures = scene.get("fixtures") or {}
        parent_id = next(
            (fixture_id for fixture_id, candidate in self._fixtures.items()
             if candidate is parent_fixture),
            None,
        )
        if parent_id is None:
            return []
        children: list[Fixture] = []
        for fixture_id, state in fixtures.items():
            if not isinstance(state, dict) or state.get("parent_fixture") != parent_id:
                continue
            if str(state.get("fixture_type", "")).lower() not in EXCLUSIVE_FIXTURE_TYPES:
                continue
            child = self._fixtures.get(fixture_id)
            if child is not None:
                children.append(child)
        return children

    def _find_nearest_surface(self, pos_2d: np.ndarray) -> str | None:
        """Find the nearest counter / placeable surface to a 2-D position."""
        best_id, best_dist = None, float("inf")
        for fid, fxtr in self._fixtures.items():
            ftype = _classify_fixture(fxtr)
            if ftype is None or "counter" not in ftype:
                continue
            d = float(np.linalg.norm(np.array(fxtr.pos[:2]) - pos_2d))
            if d < best_dist:
                best_dist = d
                best_id = fid
        return best_id

    def give_space(
        self,
        robot_idx: int,
        fixture_id: str,
        min_distance: float = 1.5,
    ):
        """Move *robot_idx* to open floor space away from *fixture_id*.

        Grid mode: finds the nearest free cell that is ≥ *min_distance* from
        the fixture center and not occupied by another robot.

        Continuous mode: samples positions on a circle around the fixture at
        increasing radii until a standable, collision-free position is found.
        """
        if fixture_id not in self._fixtures:
            return
        fxtr = self._fixtures[fixture_id]
        fxtr_pos = np.asarray(fxtr.pos[:2], dtype=float)

        # Collect other robot positions
        other_positions = []
        for other_idx in range(self._num_robots):
            if other_idx == robot_idx:
                continue
            other_positions.append(self._get_robot_position(other_idx)[:2])

        best_pos = None
        best_yaw = None
        robot_pos = self._get_robot_position(robot_idx)[:2]

        def _is_valid_candidate(pos):
            """Check a candidate is collision-free and standable."""
            for rp in other_positions:
                if float(np.linalg.norm(pos - rp)) < MIN_ROBOT_SEPARATION:
                    return False
            if self._occupancy_grid is not None and not self._occupancy_grid.is_standable(pos):
                return False
            return True

        def _try_grid():
            """Scan all free cells for the closest one far enough from fixture."""
            nonlocal best_pos, best_yaw
            if self._occupancy_grid is None:
                return
            grid = self._occupancy_grid
            best_dist_to_robot = float("inf")

            for r in range(grid._rows):
                for c in range(grid._cols):
                    if grid._grid[r, c]:
                        continue  # occupied cell
                    cell_pos = grid._grid_to_world(r, c)
                    dist_to_fixture = float(np.linalg.norm(cell_pos - fxtr_pos))
                    if dist_to_fixture < min_distance:
                        continue
                    if not _is_valid_candidate(cell_pos):
                        continue
                    dist_to_robot = float(np.linalg.norm(cell_pos - robot_pos))
                    if dist_to_robot < best_dist_to_robot:
                        best_dist_to_robot = dist_to_robot
                        best_pos = cell_pos
                        delta = fxtr_pos - cell_pos
                        best_yaw = float(np.arctan2(delta[1], delta[0]))

        _try_grid()

        if best_pos is not None:
            self._set_robot_pose(robot_idx, best_pos, best_yaw)

    def hand_off_object(
        self,
        robot_idx: int,
        object_id: str,
        to_robot_idx: int,
    ):
        """Place *object_id* in front of the receiving robot and teleport the
        delivering robot to stand beside them.

        1. Find the nearest counter/surface to *to_robot_idx*'s position.
        2. Teleport the object onto that surface.
        3. Teleport *robot_idx* next to *to_robot_idx* (side-offset).
        """
        receiving_pos = self._get_robot_position(to_robot_idx)[:2]
        surface = self._find_nearest_surface(receiving_pos)
        if surface is None:
            return  # nothing we can place on

        # Place the object on that surface
        if object_id:
            self.move_object(object_id, surface)

        # Teleport delivering robot beside receiving robot at the same fixture
        self._move_robot_near_fixture(robot_idx, surface)
        # _move_robot_near_fixture already handles offset if too close

    def _set_robot_yaw(self, robot_idx: int, yaw: float | None):
        """Set the mobile base yaw joint for the robot."""
        yaw_jnt = f"mobilebase{robot_idx}_joint_mobile_yaw"
        try:
            addr = self.env.sim.model.get_joint_qpos_addr(yaw_jnt)
            if yaw is not None:
                # ori from compute_robot_base_placement_pose is an euler [0,0,yaw]
                # The yaw joint is relative to the robot's anchor orientation
                anchor_ori = getattr(
                    self.env, "init_robot_base_ori_anchors", [None] * (robot_idx + 1)
                )[robot_idx]
                if anchor_ori is not None:
                    self.env.sim.data.qpos[addr] = yaw - anchor_ori[2]
                else:
                    self.env.sim.data.qpos[addr] = yaw
            else:
                self.env.sim.data.qpos[addr] = 0.0
            self.env.sim.forward()
        except Exception:
            pass

    def _set_robot_pose(self, robot_idx: int, pos_xy: np.ndarray, yaw: float | None):
        """Set robot position and yaw atomically.

        The yaw joint rotates the body offset relative to the joint anchor,
        which shifts body_xpos.  To achieve the desired world position *and*
        yaw, we: (1) set yaw, (2) set position, (3) correct for the
        yaw-induced position drift by re-setting position.
        """
        self._set_robot_yaw(robot_idx, yaw)
        pos_3d = np.array([pos_xy[0], pos_xy[1], 0.0])
        EnvUtils.set_robot_to_position(self.env, pos_3d, robot_idx=robot_idx)
        # Correct for yaw-induced body offset: read back actual position
        # and adjust if it doesn't match the target.
        actual = self._get_robot_position(robot_idx)[:2]
        error = pos_xy - actual
        if np.linalg.norm(error) > 0.01:
            corrected = pos_3d.copy()
            corrected[:2] += error
            EnvUtils.set_robot_to_position(self.env, corrected, robot_idx=robot_idx)

    def _get_step_fixture(self, action: str, args: dict) -> str | None:
        """Extract the fixture a step interacts with."""
        if action == "move_object":
            return args.get("to")
        elif action == "interact":
            return args.get("fixture") or args.get("fixture_id")
        elif action == "navigate":
            return args.get("fixture") or args.get("fixture_id")
        elif action == "move_away":
            return None  # hand_off_object handles its own positioning
        elif action == "give_space":
            return None  # give_space handles its own positioning
        return None

    # ------------------------------------------------------------------
    # Scene description
    # ------------------------------------------------------------------

    def get_scene_description(self) -> dict:
        """Extract a JSON-serializable description of the scene for the LLM."""
        if self._scene is not None:
            return self._scene

        fixtures_info = {}
        fixture_positions = {}

        for name, fxtr in self._fixtures.items():
            ftype = _classify_fixture(fxtr)
            if ftype is None:
                continue

            pos = (
                fxtr.pos.tolist()
                if hasattr(fxtr, "pos") and fxtr.pos is not None
                else [0, 0, 0]
            )
            fixture_positions[name] = np.array(pos[:2])

            interactions = _get_interactions(fxtr)
            can_place = any(
                fixture_is_type(fxtr, ft)
                for ft in _PLACEABLE_FIXTURE_TYPES
                if ft in FixtureType.__members__.values()
            )

            fixtures_info[name] = FixtureInfo(
                fixture_id=name,
                fixture_type=ftype,
                position=[round(p, 3) for p in pos],
                interactions=interactions,
                can_place_objects=can_place,
            )

        # Compute nearby fixtures (within 1.0m)
        for name, info in fixtures_info.items():
            if name not in fixture_positions:
                continue
            pos_a = fixture_positions[name]
            nearby = []
            for other_name, pos_b in fixture_positions.items():
                if other_name == name:
                    continue
                if np.linalg.norm(pos_a - pos_b) < 1.0:
                    nearby.append(other_name)
            info.nearby_fixtures = nearby

        # Compute parent fixture. Countertop appliances use the horizontal
        # surface physically underneath them; cabinets/drawers retain the
        # existing paired-workspace containment relationship.
        counter_fixtures = {
            name: fxtr
            for name, fxtr in self._fixtures.items()
            if any(
                fixture_is_type(fxtr, ft)
                for ft in _PLACEABLE_FIXTURE_TYPES
                if ft in FixtureType.__members__.values()
            )
        }
        for name, info in fixtures_info.items():
            if info.fixture_type in _HORIZONTAL_SUPPORT_TYPES:
                continue  # counters don't have parent counters
            fxtr = self._fixtures.get(name)
            if fxtr is None or not hasattr(fxtr, "pos"):
                continue
            if info.fixture_type in _COUNTERTOP_APPLIANCE_TYPES:
                info.parent_fixture = _countertop_support_parent(
                    name,
                    fxtr,
                    self._fixtures,
                    {fixture_name: fixture_info.fixture_type
                     for fixture_name, fixture_info in fixtures_info.items()},
                )
                continue
            for cname, cfxtr in counter_fixtures.items():
                if cname == name:
                    continue
                if OU.point_in_fixture(fxtr.pos, cfxtr, only_2d=True):
                    info.parent_fixture = cname
                    break

        # Objects — use ep_meta for rich type info when available
        objects_info = {}
        obj_cfg_map = {}
        if hasattr(self.env, "get_ep_meta"):
            ep_meta = self.env.get_ep_meta()
            for cfg in ep_meta.get("object_cfgs", []):
                obj_cfg_map[cfg["name"]] = cfg

        # Ground truth: fixture refs and object placements from the task
        fixture_obj_to_id = {id(fxtr): fid for fid, fxtr in self._fixtures.items()}

        fixture_refs_info = {}
        if hasattr(self.env, "fixture_refs"):
            for role, ref_value in self.env.fixture_refs.items():
                fxtr_obj = ref_value[0] if isinstance(ref_value, tuple) else ref_value
                fxtr_id = fixture_obj_to_id.get(id(fxtr_obj))
                if fxtr_id is not None:
                    fixture_refs_info[role] = fxtr_id

        object_placements = {}
        if hasattr(self.env, "object_cfgs"):
            for cfg in self.env.object_cfgs:
                obj_name = cfg.get("name")
                placement_fxtr = (cfg.get("placement") or {}).get("fixture")
                if obj_name and placement_fxtr is not None:
                    fxtr_id = fixture_obj_to_id.get(id(placement_fxtr))
                    if fxtr_id is not None:
                        object_placements[obj_name] = fxtr_id

        if hasattr(self.env, "objects") and self.env.objects:
            for obj_name in self.env.objects:
                obj_pos = self.env.sim.data.body_xpos[
                    self.env.obj_body_id[obj_name]
                ]
                support_object = self._find_support_object(obj_name)
                location = (
                    support_object
                    or object_placements.get(obj_name)
                    or self._find_object_fixture(obj_pos)
                )

                cfg = obj_cfg_map.get(obj_name, {})
                info = cfg.get("info", {})
                obj_type = str(info.get("cat", obj_name))

                objects_info[obj_name] = ObjectInfo(
                    object_id=obj_name,
                    object_type=obj_type,
                    location=location,
                )
                self._object_locations[obj_name] = location

        # Robots
        robots_info = []
        for i, robot in enumerate(self.env.robots):
            robot_pos = self.env.sim.data.body_xpos[
                self.env.sim.model.body_name2id(robot.robot_model.root_body)
            ]
            robots_info.append(
                {
                    "robot_id": f"agent_{i}",
                    "position": [round(float(p), 3) for p in robot_pos],
                }
            )

        # Task instruction
        task_lang = None
        if hasattr(self.env, "get_ep_meta"):
            task_lang = self.env.get_ep_meta().get("lang")

        # Robot spawn fixture (ground truth from the task definition)
        init_robot_base_ref_id = None
        if (
            hasattr(self.env, "init_robot_base_ref")
            and self.env.init_robot_base_ref is not None
        ):
            ref = self.env.init_robot_base_ref
            if isinstance(ref, str):
                init_robot_base_ref_id = ref
            else:
                init_robot_base_ref_id = fixture_obj_to_id.get(id(ref))

        self._scene = {
            "task": task_lang,
            "fixtures": {k: v.to_dict() for k, v in fixtures_info.items()},
            "objects": {k: v.to_dict() for k, v in objects_info.items()},
            "fixture_refs": fixture_refs_info,
            "object_placements": object_placements,
            "init_robot_base_ref": init_robot_base_ref_id,
            "cameras": self.camera_names,
            "robots": robots_info,
        }
        return self._scene

    def _find_support_object(self, obj_name: str) -> str | None:
        """Return the supported object id when an object is resting on another object.

        This preserves object-on-object semantics in the extracted scene instead
        of flattening everything to a fixture location.
        """
        if not hasattr(self.env, "objects") or obj_name not in self.env.objects:
            return None
        if obj_name not in self.env.obj_body_id:
            return None

        obj = self.env.objects[obj_name]
        obj_pos = np.asarray(self.env.sim.data.body_xpos[self.env.obj_body_id[obj_name]], dtype=float)
        candidates: list[tuple[float, str]] = []

        for other_name, other_obj in self.env.objects.items():
            if other_name == obj_name or other_name not in self.env.obj_body_id:
                continue
            try:
                in_contact = bool(self.env.check_contact(obj, other_obj))
            except Exception:
                in_contact = False
            if not in_contact:
                continue

            other_pos = np.asarray(
                self.env.sim.data.body_xpos[self.env.obj_body_id[other_name]],
                dtype=float,
            )
            max_xy = float(getattr(other_obj, "horizontal_radius", 0.0)) * 1.05
            if max_xy > 0.0 and float(np.linalg.norm(obj_pos[:2] - other_pos[:2])) > max_xy:
                continue
            # Prefer closest supporting object in XY.
            candidates.append((float(np.linalg.norm(obj_pos[:2] - other_pos[:2])), other_name))

        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    def _find_object_fixture(self, obj_pos: np.ndarray) -> str:
        """Find the most plausible support fixture for an object's position."""

        containing_placeable: list[tuple[float, str]] = []
        nearest_placeable: list[tuple[float, str]] = []
        nearest_any: list[tuple[float, str]] = []

        for name, fxtr in self._fixtures.items():
            if not hasattr(fxtr, "pos") or fxtr.pos is None:
                continue
            fxtr_pos = np.asarray(fxtr.pos[:2], dtype=float)
            dist = float(np.linalg.norm(obj_pos[:2] - fxtr_pos))
            nearest_any.append((dist, name))

            is_placeable = any(fixture_is_type(fxtr, ft) for ft in _PLACEABLE_FIXTURE_TYPES)
            if is_placeable:
                nearest_placeable.append((dist, name))
                try:
                    if OU.point_in_fixture(obj_pos, fxtr, only_2d=True):
                        containing_placeable.append((dist, name))
                        continue
                except Exception:
                    pass

            try:
                if not OU.point_in_fixture(obj_pos, fxtr, only_2d=True):
                    continue
            except Exception:
                continue

            # If the object lies inside a non-placeable accessory or appliance,
            # recover the supporting surface the fixture itself sits on.
            for parent_name, parent_fxtr in self._fixtures.items():
                if parent_name == name or not hasattr(parent_fxtr, "pos") or parent_fxtr.pos is None:
                    continue
                if not any(fixture_is_type(parent_fxtr, ft) for ft in _PLACEABLE_FIXTURE_TYPES):
                    continue
                try:
                    if OU.point_in_fixture(np.asarray(fxtr.pos, dtype=float), parent_fxtr, only_2d=True):
                        containing_placeable.append((dist, parent_name))
                        break
                except Exception:
                    continue

        if containing_placeable:
            return min(containing_placeable)[1]
        if nearest_placeable:
            return min(nearest_placeable)[1]
        if nearest_any:
            return min(nearest_any)[1]
        return "unknown"

    def _set_object_location(self, object_id: str, fixture_id: str):
        """Update cached fixture grounding for an object."""
        self._object_locations[object_id] = fixture_id
        if self._scene is None:
            return
        object_info = self._scene.get("objects", {}).get(object_id)
        if isinstance(object_info, dict):
            object_info["location"] = fixture_id

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self) -> dict[str, np.ndarray]:
        """Render all configured cameras and return {camera_name: image}."""
        images = {}
        for cam_name in self.camera_names:
            if cam_name == "room_view":
                images[cam_name] = self._render_room_view()
                continue
            if cam_name == "top_view":
                images[cam_name] = self._render_top_view()
                continue
            try:
                frame = self.env.sim.render(
                    height=self.render_height,
                    width=self.render_width,
                    camera_name=cam_name,
                )[
                    ::-1
                ]  # flip vertical (MuJoCo convention)
                images[cam_name] = frame
            except Exception:
                continue
        return images

    # ------------------------------------------------------------------
    # Primitives
    # ------------------------------------------------------------------

    def _fixture_local_to_world(
        self,
        target_fxtr: Fixture,
        local_offset: np.ndarray,
    ) -> np.ndarray:
        """Convert a fixture-local offset into a world position."""
        offset = np.asarray(local_offset, dtype=float)
        if hasattr(target_fxtr, "rot") and target_fxtr.rot is not None:
            angle = float(target_fxtr.rot)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rot_offset = np.array(
                [
                    cos_a * offset[0] - sin_a * offset[1],
                    sin_a * offset[0] + cos_a * offset[1],
                    offset[2],
                ],
                dtype=float,
            )
        else:
            rot_offset = offset
        return np.asarray(target_fxtr.pos, dtype=float) + rot_offset

    def _world_to_fixture_local(
        self,
        target_fxtr: Fixture,
        world_xy: np.ndarray,
    ) -> np.ndarray:
        """Project a world XY position into a fixture's local frame."""
        world_xy = np.asarray(world_xy, dtype=float)
        offset = world_xy - np.asarray(target_fxtr.pos[:2], dtype=float)
        if not hasattr(target_fxtr, "rot") or target_fxtr.rot is None:
            return offset
        angle = float(target_fxtr.rot)
        cos_a, sin_a = np.cos(angle), np.sin(angle)
        return np.array(
            [
                cos_a * offset[0] + sin_a * offset[1],
                -sin_a * offset[0] + cos_a * offset[1],
            ],
            dtype=float,
        )

    def _sample_axis_values(self, center: float, half_span: float) -> np.ndarray:
        """Sample positions along one fixture-local axis."""
        if half_span <= 1e-6:
            return np.array([center], dtype=float)

        axis_min = center - half_span
        axis_max = center + half_span
        span = axis_max - axis_min
        approx_count = int(np.floor(span / _OBJECT_PLACEMENT_STEP)) + 1
        count = min(_OBJECT_PLACEMENT_MAX_AXIS_SAMPLES, max(2, approx_count))
        return np.linspace(axis_min, axis_max, num=count, dtype=float)

    def _get_object_placement_metadata(
        self, object_id: str
    ) -> dict[str, np.ndarray | float]:
        """Return bbox-derived placement metadata for an object."""
        obj = self.env.objects[object_id]
        qpos = self.env.sim.data.get_joint_qpos(obj.joints[0]).copy()
        quat_wxyz = qpos[3:7]
        quat_xyzw = T.convert_quat(quat_wxyz, to="xyzw")
        bbox_points = np.asarray(
            obj.get_bbox_points(trans=np.zeros(3, dtype=float), rot=quat_xyzw),
            dtype=float,
        )
        span = np.max(bbox_points, axis=0) - np.min(bbox_points, axis=0)
        try:
            xy_radius = float(obj.horizontal_radius)
        except Exception:
            xy_radius = 0.5 * float(max(span[0], span[1]))
        return {
            "quat_wxyz": quat_wxyz,
            "quat_xyzw": quat_xyzw,
            "size": span,
            "xy_radius": max(xy_radius, 0.5 * float(max(span[0], span[1]))),
        }

    def _get_object_fixture_xy_half_extent(
        self,
        target_fxtr: Fixture,
        object_id: str,
        *,
        metadata: dict[str, np.ndarray | float] | None = None,
    ) -> np.ndarray:
        """Return fixture-aligned half extents for the object's current orientation."""
        metadata = (
            metadata
            if isinstance(metadata, dict)
            else self._get_object_placement_metadata(object_id)
        )
        if "quat_xyzw" not in metadata:
            xy_radius = float(metadata["xy_radius"])
            return np.array([xy_radius, xy_radius], dtype=float)
        obj = self.env.objects[object_id]
        quat_xyzw = np.asarray(metadata["quat_xyzw"], dtype=float)
        try:
            bbox_points = np.asarray(
                obj.get_bbox_points(
                    trans=np.zeros(3, dtype=float),
                    rot=quat_xyzw,
                ),
                dtype=float,
            )
            world_xy = bbox_points[:, :2]
            if not hasattr(target_fxtr, "rot") or target_fxtr.rot is None:
                local_xy = world_xy
            else:
                angle = float(target_fxtr.rot)
                cos_a, sin_a = np.cos(angle), np.sin(angle)
                local_xy = np.stack(
                    (
                        cos_a * world_xy[:, 0] + sin_a * world_xy[:, 1],
                        -sin_a * world_xy[:, 0] + cos_a * world_xy[:, 1],
                    ),
                    axis=1,
                )
            spans = np.max(local_xy, axis=0) - np.min(local_xy, axis=0)
            return 0.5 * np.asarray(spans, dtype=float)
        except Exception:
            xy_radius = float(metadata["xy_radius"])
            return np.array([xy_radius, xy_radius], dtype=float)

    def _get_fixture_reset_regions(
        self,
        target_fxtr: Fixture,
        min_size: np.ndarray | None = None,
        site_id: str | None = None,
    ) -> list[dict]:
        """Return placement regions for a fixture, filtered by minimum size."""
        regions: list[dict] = []
        try:
            all_regions = target_fxtr.get_reset_regions(env=self.env)
        except Exception:
            all_regions = None

        if isinstance(all_regions, dict):
            for region_name, region in all_regions.items():
                if isinstance(site_id, str) and region_name != site_id:
                    continue
                region_size = np.asarray(region.get("size", (0.1, 0.1)), dtype=float)
                region_height = region.get("height")
                # For an explicitly requested site, trust the authored region
                # and let later collision checks decide feasibility. Compact
                # interior sites such as toaster racks and blender bowls are
                # often smaller than the placed object's footprint metadata.
                if min_size is not None and not isinstance(site_id, str):
                    if min_size[0] > max(region_size) and min_size[1] > max(region_size):
                        continue
                    if (
                        region_height is not None
                        and len(min_size) == 3
                        and min_size[2] > float(region_height)
                    ):
                        continue
                region_dict = dict(region)
                region_dict["name"] = region_name
                regions.append(region_dict)

        if regions:
            return regions

        if isinstance(site_id, str):
            return []

        try:
            fallback = target_fxtr.sample_reset_region(env=self.env, min_size=min_size)
        except Exception:
            fallback = {"offset": (0.0, 0.0, 0.0), "size": (0.1, 0.1)}
        fallback_region = dict(fallback)
        fallback_region.setdefault("name", "fallback")
        return [fallback_region]

    def _iter_object_target_candidates(
        self,
        target_fxtr: Fixture,
        object_id: str,
        preferred_xy: np.ndarray | None = None,
        rng_offset: np.ndarray | None = None,
        site_id: str | None = None,
    ) -> list[np.ndarray]:
        """Generate candidate world positions for placing an object on a fixture."""
        metadata = self._get_object_placement_metadata(object_id)
        object_size = np.asarray(metadata["size"], dtype=float)
        footprint_half_xy = self._get_object_fixture_xy_half_extent(
            target_fxtr,
            object_id,
            metadata=metadata,
        )
        min_size = np.array(
            [
                2.0 * footprint_half_xy[0],
                2.0 * footprint_half_xy[1],
                object_size[2],
            ],
            dtype=float,
        ) + 2.0 * _OBJECT_PLACEMENT_MARGIN

        # Compute the Z lift so the object's bottom rests on the surface,
        # matching native placement_samplers.py behaviour.
        obj = self.env.objects[object_id]
        z_lift = -obj.bottom_offset[-1]  # positive when bottom is below origin

        preferred_local = None
        if preferred_xy is not None:
            preferred_local = self._world_to_fixture_local(target_fxtr, preferred_xy)

        candidates: list[np.ndarray] = []
        seen: set[tuple[float, float, float]] = set()

        for region in self._get_fixture_reset_regions(
            target_fxtr,
            min_size=min_size,
            site_id=site_id,
        ):
            offset = np.asarray(region.get("offset", (0.0, 0.0, 0.0)), dtype=float)
            size = np.asarray(region.get("size", (0.1, 0.1)), dtype=float)[:2]
            site_half = 0.5 * size
            usable_half = site_half - (
                footprint_half_xy + _OBJECT_PLACEMENT_MARGIN
            )
            if isinstance(site_id, str):
                # For explicit authored sites, search across the site itself
                # instead of shrinking the search range by the object's full
                # footprint. Compact sites like toaster racks and blender
                # bowls often rely on partial overhang, and later collision
                # checks are the real feasibility gate.
                search_half = np.maximum(site_half, 0.0)
            else:
                if np.any(usable_half < -1e-6):
                    continue
                search_half = np.maximum(usable_half, 0.0)

            local_candidates = [offset.copy()]

            if rng_offset is not None:
                jittered = offset.copy()
                jittered[:2] += np.asarray(rng_offset[:2], dtype=float)
                if np.all(np.abs(jittered[:2] - offset[:2]) <= search_half + 1e-6):
                    local_candidates.append(jittered)

            if preferred_local is not None:
                projected = offset.copy()
                projected[0] = float(np.clip(
                    preferred_local[0],
                    offset[0] - search_half[0],
                    offset[0] + search_half[0],
                ))
                projected[1] = float(np.clip(
                    preferred_local[1],
                    offset[1] - search_half[1],
                    offset[1] + search_half[1],
                ))
                local_candidates.append(projected)

            x_values = self._sample_axis_values(float(offset[0]), float(search_half[0]))
            y_values = self._sample_axis_values(float(offset[1]), float(search_half[1]))
            for x in x_values:
                for y in y_values:
                    local = offset.copy()
                    local[0] = float(x)
                    local[1] = float(y)
                    local_candidates.append(local)

            for local in local_candidates:
                key = tuple(np.round(local, 4))
                if key in seen:
                    continue
                seen.add(key)
                world = self._fixture_local_to_world(target_fxtr, local)
                world[2] += z_lift
                candidates.append(world)

        if candidates:
            return candidates

        if isinstance(site_id, str):
            return []

        fallback = self._fixture_local_to_world(
            target_fxtr,
            np.array([0.0, 0.0, 0.0], dtype=float),
        )
        fallback[2] += z_lift
        return [fallback]

    def _build_object_collision_context(
        self,
        object_id: str,
        target_fixture_id: str,
        ignored_object_ids: set[str] | None = None,
        ignored_fixture_ids: set[str] | None = None,
    ) -> dict:
        """Pre-compute static scene geometry used during one placement search."""
        metadata = self._get_object_placement_metadata(object_id)
        obj = self.env.objects[object_id]
        ignored = set() if ignored_object_ids is None else set(ignored_object_ids)
        ignored_fixture_ids = (
            set() if ignored_fixture_ids is None else set(ignored_fixture_ids)
        )
        target_bounds = _fixture_extents_3d(self._fixtures[target_fixture_id])
        target_surface_z = (
            float(target_bounds[1][2]) if target_bounds is not None else None
        )

        object_obstacles = []
        object_obstacle_ids: list[str] = []
        for other_id, other_obj in self.env.objects.items():
            if other_id == object_id or other_id in ignored:
                continue
            other_qpos = self.env.sim.data.get_joint_qpos(other_obj.joints[0]).copy()
            other_pos = other_qpos[:3]
            try:
                other_radius = float(other_obj.horizontal_radius)
            except Exception:
                other_radius = 0.0
            object_obstacles.append(
                (
                    other_obj,
                    other_pos,
                    T.convert_quat(other_qpos[3:7], to="xyzw"),
                    other_radius,
                )
            )
            object_obstacle_ids.append(str(other_id))

        fixture_obstacles = []
        fixture_obstacle_ids: list[str] = []
        for fixture_id, fixture in self._fixtures.items():
            if fixture_id == target_fixture_id or fixture_id in ignored_fixture_ids:
                continue
            # Drawers, cabinets, and other structural components entirely
            # below a support surface cannot intersect an object resting on
            # top of that surface. Some fixture assets expose broad collision
            # envelopes that otherwise create false positive intersections
            # across the entire countertop.
            if target_surface_z is not None:
                fixture_bounds = _fixture_extents_3d(fixture)
                if (
                    fixture_bounds is not None
                    and float(fixture_bounds[1][2]) <= target_surface_z + 1e-3
                ):
                    continue
            fixture_pos = np.asarray(fixture.pos, dtype=float)
            try:
                fixture_radius = float(fixture.horizontal_radius)
            except Exception:
                fixture_radius = 0.0
            fixture_obstacles.append((fixture, fixture_pos, fixture_radius))
            fixture_obstacle_ids.append(str(fixture_id))

        return {
            "object": obj,
            "obj_quat": metadata["quat_xyzw"],
            "obj_radius": float(metadata["xy_radius"]),
            "object_obstacles": object_obstacles,
            "object_obstacle_ids": object_obstacle_ids,
            "fixture_obstacles": fixture_obstacles,
            "fixture_obstacle_ids": fixture_obstacle_ids,
        }

    def _candidate_overlaps_scene(
        self,
        candidate_pos: np.ndarray,
        collision_context: dict,
    ) -> bool:
        """Return True if the candidate intersects precomputed scene geometry."""
        obj = collision_context["object"]
        obj_quat = collision_context["obj_quat"]
        obj_radius = float(collision_context["obj_radius"])

        for other_obj, other_pos, other_quat, other_radius in collision_context[
            "object_obstacles"
        ]:
            if (
                np.linalg.norm(other_pos[:2] - candidate_pos[:2])
                > obj_radius + other_radius + 0.30
            ):
                continue
            try:
                if OU.objs_intersect(
                    obj,
                    candidate_pos,
                    obj_quat,
                    other_obj,
                    other_pos,
                    other_quat,
                ):
                    return True
            except Exception:
                continue

        for fixture, fixture_pos, fixture_radius in collision_context[
            "fixture_obstacles"
        ]:
            if (
                np.linalg.norm(fixture_pos[:2] - candidate_pos[:2])
                > obj_radius + fixture_radius + 0.30
            ):
                continue
            try:
                if OU.objs_intersect(
                    obj,
                    candidate_pos,
                    obj_quat,
                    fixture,
                    fixture_pos,
                    None,
                ):
                    return True
            except Exception:
                continue

        return False

    def _candidate_scene_overlap_labels(
        self,
        candidate_pos: np.ndarray,
        collision_context: dict,
    ) -> list[str]:
        """Return labeled obstacles intersecting a candidate for diagnostics."""
        obj = collision_context["object"]
        obj_quat = collision_context["obj_quat"]
        obj_radius = float(collision_context["obj_radius"])
        labels: list[str] = []
        object_ids = collision_context.get("object_obstacle_ids") or []
        for index, (other_obj, other_pos, other_quat, other_radius) in enumerate(
            collision_context["object_obstacles"]
        ):
            if np.linalg.norm(other_pos[:2] - candidate_pos[:2]) > obj_radius + other_radius + 0.30:
                continue
            try:
                if OU.objs_intersect(obj, candidate_pos, obj_quat, other_obj, other_pos, other_quat):
                    obstacle_id = object_ids[index] if index < len(object_ids) else str(index)
                    labels.append(f"object:{obstacle_id}")
            except Exception:
                continue
        fixture_ids = collision_context.get("fixture_obstacle_ids") or []
        for index, (fixture, fixture_pos, fixture_radius) in enumerate(
            collision_context["fixture_obstacles"]
        ):
            if np.linalg.norm(fixture_pos[:2] - candidate_pos[:2]) > obj_radius + fixture_radius + 0.30:
                continue
            try:
                if OU.objs_intersect(obj, candidate_pos, obj_quat, fixture, fixture_pos, None):
                    obstacle_id = fixture_ids[index] if index < len(fixture_ids) else str(index)
                    labels.append(f"fixture:{obstacle_id}")
            except Exception:
                continue
        return labels

    def _compute_object_target_pos(
        self,
        target_fxtr: Fixture,
        rng_offset: np.ndarray | None = None,
        object_id: str | None = None,
        preferred_xy: np.ndarray | None = None,
        ignored_object_ids: set[str] | None = None,
        ignored_fixture_ids: set[str] | None = None,
        site_id: str | None = None,
    ) -> np.ndarray:
        """Compute a collision-aware world position on *target_fxtr*'s surface."""
        self._last_placement_diagnostics = None
        if object_id is None:
            try:
                region = target_fxtr.sample_reset_region(env=self.env)
            except Exception:
                region = {"offset": (0.0, 0.0, 0.0), "size": (0.1, 0.1)}

            offset = np.array(region["offset"], dtype=float)
            if rng_offset is not None:
                offset[0] += rng_offset[0]
                offset[1] += rng_offset[1]

            target_pos = self._fixture_local_to_world(target_fxtr, offset)
            target_pos[2] += 0.02
            return target_pos

        preferred = (
            None if preferred_xy is None else np.asarray(preferred_xy, dtype=float)[:2]
        )
        candidates = self._iter_object_target_candidates(
            target_fxtr,
            object_id,
            preferred_xy=preferred,
            rng_offset=rng_offset,
            site_id=site_id,
        )
        collision_context = self._build_object_collision_context(
            object_id,
            target_fxtr.name,
            ignored_object_ids=ignored_object_ids,
            ignored_fixture_ids=ignored_fixture_ids,
        )

        target_xy = (
            preferred
            if preferred is not None
            else np.asarray(target_fxtr.pos[:2], dtype=float)
        )
        valid: list[tuple[float, np.ndarray]] = []
        rejected_outside_fixture = 0
        rejected_scene_overlap = 0
        require_fixture_containment = not isinstance(site_id, str)
        for candidate in candidates:
            if require_fixture_containment and not self._validate_object_on_fixture(
                candidate, target_fxtr.name
            ):
                rejected_outside_fixture += 1
                continue
            if self._candidate_overlaps_scene(candidate, collision_context):
                rejected_scene_overlap += 1
                continue
            score = float(np.linalg.norm(candidate[:2] - target_xy))
            valid.append((score, candidate.copy()))

        self._last_placement_diagnostics = {
            "object_id": object_id,
            "fixture_id": target_fxtr.name,
            "site_id": site_id,
            "preferred_xy": preferred.tolist() if preferred is not None else None,
            "ignored_fixture_ids": sorted(ignored_fixture_ids or ()),
            "candidate_count": len(candidates),
            "valid_count": len(valid),
            "rejected_outside_fixture": rejected_outside_fixture,
            "rejected_scene_overlap": rejected_scene_overlap,
        }

        if valid:
            return min(valid, key=lambda item: item[0])[1]

        if self._placement_debug_enabled():
            print(
                "[placement] no valid candidate:",
                json.dumps(self._last_placement_diagnostics, sort_keys=True),
            )

        raise RuntimeError(
            f"No collision-free placement candidate found for {object_id!r} on {target_fxtr.name!r}."
        )

    def _validate_object_on_fixture(
        self,
        obj_pos: np.ndarray,
        fixture_id: str,
    ) -> bool:
        """Return True if *obj_pos* is geometrically inside *fixture_id*."""
        fxtr = self._fixtures.get(fixture_id)
        if fxtr is None:
            return False
        try:
            return OU.point_in_fixture(obj_pos, fxtr, only_2d=True)
        except Exception:
            # Fallback: accept if within 0.3 m of the fixture centre
            return float(np.linalg.norm(obj_pos[:2] - fxtr.pos[:2])) < 0.3

    def _find_contained_objects(self, container_id: str) -> list[str]:
        """Find objects physically inside/on top of *container_id*."""
        contained = []
        body_id = self.env.obj_body_id.get(container_id)
        if body_id is None:
            return contained
        container_pos = self.env.sim.data.body_xpos[body_id].copy()
        container_obj = self.env.objects[container_id]
        radius = getattr(container_obj, "horizontal_radius", 0.10) * 1.2
        for other_id in self.env.objects:
            if other_id == container_id:
                continue
            other_body_id = self.env.obj_body_id.get(other_id)
            if other_body_id is None:
                continue
            other_pos = self.env.sim.data.body_xpos[other_body_id].copy()
            xy_dist = float(np.linalg.norm(other_pos[:2] - container_pos[:2]))
            z_diff = other_pos[2] - container_pos[2]
            if xy_dist < radius and -0.05 < z_diff < 0.20:
                contained.append(other_id)
        return contained

    def move_object(
        self,
        object_id: str,
        to_fixture: str,
        target_pos: np.ndarray | None = None,
        preferred_xy: np.ndarray | None = None,
    ):
        """Teleport an object to a fixture's surface.

        If the object contains other objects (e.g. a bowl with slices),
        those are moved along with it.

        Placement is collision-aware: candidates are sampled from fixture
        reset regions, filtered against existing objects / nearby fixtures,
        and ranked by distance to ``preferred_xy`` when provided.
        """
        if object_id not in self.env.objects:
            raise ValueError(
                f"Unknown object '{object_id}'. "
                f"Available: {list(self.env.objects.keys())}"
            )
        if to_fixture not in self._fixtures:
            raise ValueError(
                f"Unknown fixture '{to_fixture}'. "
                f"Available: {list(self._fixtures.keys())}"
            )

        target_fxtr = self._fixtures[to_fixture]
        obj = self.env.objects[object_id]
        current_qpos = self.env.sim.data.get_joint_qpos(obj.joints[0])
        current_quat = current_qpos[3:7]
        old_pos = current_qpos[:3].copy()

        # Snapshot contained objects before moving
        contained = self._find_contained_objects(object_id)
        if target_pos is None:
            target_pos = self._compute_object_target_pos(
                target_fxtr,
                object_id=object_id,
                preferred_xy=preferred_xy,
                ignored_object_ids=set(contained),
            )

        self.env.sim.data.set_joint_qpos(
            obj.joints[0],
            np.concatenate([target_pos, current_quat]),
        )

        # Move contained objects by the same delta
        if contained:
            delta = target_pos - old_pos
            for child_id in contained:
                child_obj = self.env.objects[child_id]
                child_qpos = self.env.sim.data.get_joint_qpos(child_obj.joints[0])
                child_qpos[:3] += delta
                self.env.sim.data.set_joint_qpos(child_obj.joints[0], child_qpos)

        self.env.sim.forward()

        self._set_object_location(object_id, to_fixture)
        for child_id in contained:
            self._set_object_location(child_id, to_fixture)

    def interact(self, fixture_id: str, action: str):
        """Change a fixture's state: open, close, turn_on, turn_off."""
        if fixture_id not in self._fixtures:
            raise ValueError(
                f"Unknown fixture '{fixture_id}'. "
                f"Available: {list(self._fixtures.keys())}"
            )

        fxtr = self._fixtures[fixture_id]

        if action == "open":
            fxtr.open_door(env=self.env)
        elif action == "close":
            fxtr.close_door(env=self.env)
        elif action == "turn_on":
            if hasattr(fxtr, "_joint_infos"):
                non_door = [
                    j
                    for j in fxtr._joint_infos
                    if "door" not in j.lower() and "drawer" not in j.lower()
                ]
                if non_door:
                    fxtr.set_joint_state(
                        min=0.9, max=1.0, env=self.env, joint_names=non_door
                    )
        elif action == "turn_off":
            if hasattr(fxtr, "_joint_infos"):
                non_door = [
                    j
                    for j in fxtr._joint_infos
                    if "door" not in j.lower() and "drawer" not in j.lower()
                ]
                if non_door:
                    fxtr.set_joint_state(
                        min=0.0, max=0.0, env=self.env, joint_names=non_door
                    )
        else:
            raise ValueError(
                f"Unknown action '{action}'. "
                f"Supported: open, close, turn_on, turn_off"
            )

        self.env.sim.forward()

    # ------------------------------------------------------------------
    # Trajectory execution
    # ------------------------------------------------------------------

    def run(self, trajectory: dict | list) -> TrajectoryResult:
        """
        Execute a trajectory and capture visual observations.

        For each step:
          1. Teleport the acting robot near the target fixture
          2. Render before images
          3. Execute the primitive (move_object / interact / communicate)
          4. Render after images
        """
        if isinstance(trajectory, dict):
            steps = trajectory.get("steps", [])
        else:
            steps = trajectory

        scene = self.get_scene_description()

        # Do a zero-action step to fully initialize the render context
        # (required for free camera rendering, same as two_robot_video_sample)
        low, _ = self.env.action_spec
        self.env.step(np.zeros_like(low))

        initial_obs = self.render()
        step_results = []

        for i, step in enumerate(steps):
            agent_id = step.get("agent_id", "agent_0")
            action = step.get("action") or step.get("tool_name", "")
            args = step.get("args") or step.get("tool_args", {})
            move_target_pos = None

            # Parse robot index from agent_id
            robot_idx = int(agent_id.replace("agent_", ""))

            # Teleport robot near the fixture it will interact with.
            # For move_object, pre-compute the drop position so the robot
            # stands near where the object will actually land.
            target_fixture = self._get_step_fixture(action, args)
            if target_fixture is not None:
                ref_override = None
                if action == "move_object" and target_fixture in self._fixtures:
                    move_target_pos = self._compute_object_target_pos(
                        self._fixtures[target_fixture],
                        object_id=args["object"],
                    )
                    ref_override = move_target_pos[:2]
                self._move_robot_near_fixture(
                    robot_idx,
                    target_fixture,
                    ref_pos_override=ref_override,
                )

            # Render before
            before = self.render()

            # Execute primitive
            if action == "move_object":
                self.move_object(
                    object_id=args["object"],
                    to_fixture=args["to"],
                    target_pos=move_target_pos,
                )
            elif action == "interact":
                fixture_id = args.get("fixture") or args.get("fixture_id")
                act = args.get("action")
                if fixture_id is None or act is None:
                    raise ValueError(
                        f"Step {i}: 'interact' requires 'fixture' and 'action' in args"
                    )
                self.interact(fixture_id, act)
            elif action == "navigate":
                # Robot was already teleported above via _move_robot_near_fixture
                pass
            elif action == "move_away":
                to_agent = args.get("to_agent", "")
                obj_id = args.get("object", "")
                to_robot = (
                    int(to_agent.replace("agent_", "")) if to_agent else (1 - robot_idx)
                )
                self.hand_off_object(robot_idx, obj_id, to_robot)
            elif action == "give_space":
                fixture_id = args.get("fixture") or args.get("fixture_id", "")
                self.give_space(robot_idx, fixture_id)
            elif action == "communicate":
                pass  # no sim change
            elif action == "wait":
                pass  # no sim change, just observe
            else:
                raise ValueError(
                    f"Step {i}: unknown action '{action}'. "
                    f"Expected: move_object, interact, navigate, give_space, communicate, wait"
                )

            # Render after
            after = self.render()

            step_results.append(
                StepResult(
                    step_index=i,
                    agent_id=agent_id,
                    action=action,
                    args=step.get("args") or step.get("tool_args", {}),
                    before=before,
                    after=after,
                )
            )

        return TrajectoryResult(
            initial_obs=initial_obs,
            steps=step_results,
            scene=scene,
        )

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def close(self):
        """Close the environment."""
        self.env.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
