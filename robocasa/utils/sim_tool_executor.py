"""
Executable simulator tool layer for symbolic kitchen manipulation.

This is a pragmatic executor over the existing ``TrajectoryRunner`` substrate:
robot navigation uses the runner's working-pose placement, object transport is
represented by a held-object cache plus explicit object pose updates, and
fixture manipulation is performed by direct simulator state changes.

These tools are "real" in the sense that they mutate a live RoboCasa
environment. They are not low-level controller policies.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
from copy import deepcopy
import re
from typing import Any, Sequence

import imageio
import numpy as np
import robosuite.utils.transform_utils as T

from robotalk.tasks.shared.constants import (
    EXCLUSIVE_FIXTURE_TYPES,
)

from robocasa.models.fixtures.blender import Blender
from robocasa.models.fixtures.coffee_machine import CoffeeMachine
from robocasa.models.fixtures.electric_kettle import ElectricKettle
from robocasa.models.fixtures.microwave import Microwave
from robocasa.models.fixtures.sink import Sink
from robocasa.models.fixtures.toaster import Toaster
import robocasa.utils.object_utils as OU
from robocasa.models.fixtures import FixtureType
from robocasa.models.fixtures.fixture_utils import fixture_is_type
from robocasa.utils.placement import (
    MAX_FRONT_WORKING_LATERAL_OFFSET,
    get_face_order,
    get_front_alignment_metrics,
    get_fixture_aabb,
)
from robocasa.utils.sim_tool_executor_execution import SimToolExecutorExecutionMixin
from robocasa.utils.sim_tool_executor_inspection import (
    SimToolExecutorInspectionMixin,
)
from robocasa.utils.sim_tool_executor_planning import SimToolExecutorPlanningMixin
from robocasa.utils.sim_tool_executor_state_loading import (
    SimToolExecutorStateLoadingMixin,
)
from robocasa.utils.sim_tool_specs import SIM_TOOL_SPEC_BY_NAME
from robocasa.utils.trajectory_runner import TrajectoryRunner

# Large fixtures with a door/opening that MUST be approached from the front.
# For these fixtures, require_front=True ensures the robot lines up with the
# opening rather than standing on a blocked side.
_REQUIRE_FRONT_TYPES = {
    FixtureType.FRIDGE,
    FixtureType.MICROWAVE,
    FixtureType.OVEN,
    FixtureType.DISHWASHER,
    FixtureType.TOP_DRAWER,
    FixtureType.DRAWER,
}

# Small countertop appliances: robot should approach the fixture center (not
# an object inside it), but does NOT need require_front — any face is fine.
# Their AABBs are tiny, so front-face-only sampling often lands the robot in
# a bad spot on the counter.
_COUNTERTOP_APPLIANCE_TYPES = {
    FixtureType.TOASTER,
    FixtureType.TOASTER_OVEN,
    FixtureType.COFFEE_MACHINE,
    FixtureType.BLENDER,
    FixtureType.STAND_MIXER,
    FixtureType.ELECTRIC_KETTLE,
    FixtureType.STOVE,
}
_COUNTERTOP_APPLIANCE_TYPE_NAMES = frozenset(
    {
        "toaster",
        "toaster_oven",
        "coffee_machine",
        "blender",
        "stand_mixer",
        "electric_kettle",
        "stove",
        "stovetop",
        "cooktop",
    }
)
# Appliances that should keep extra clearance when something is placed next
# to them (hot surfaces, pouring spouts, or just bulky footprints).
_HIGH_CLEARANCE_REFERENCE_TYPE_NAMES = frozenset(
    {
        "electric_kettle",
        "coffee_machine",
        "toaster",
        "toaster_oven",
        "microwave",
        "stand_mixer",
    }
)
_SEAT_REFERENCE_TYPE_NAMES = frozenset({"stool", "chair"})
_HIGH_CLEARANCE_APPLIANCE_BONUS = 0.08

# Union: all fixtures where the robot targets the fixture center, not an
# object's position inside it.
_APPROACH_CENTER_TYPES = _REQUIRE_FRONT_TYPES | _COUNTERTOP_APPLIANCE_TYPES

_FRONT_READY_MIN_GAP = 0.05
_FRONT_READY_MAX_GAP = 0.75
_FRONT_READY_MAX_CENTER_DISTANCE = 1.0
_SWEEP_VERBOSE_ENV_VAR = "ROBOCASA_SWEEP_VERBOSE"
_SETTLE_STEPS = 2
_PLACEMENT_SETTLE_STEPS = 6
_FIXTURE_RUNTIME_ATTR_NAMES = (
    "_turned_on",
    "_state",
    "_num_steps_on",
    "_cooldown",
    "_cooldown_time",
    "_last_lid_update",
    "_target_lid_angle",
    "_lid",
    "_lid_speed",
    "water_temp_state",
)
_SITE_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SITE_STOPWORDS = frozenset(
    {"support", "site", "surface", "region", "placement", "place", "burner"}
)
_STRUCTURAL_FIXTURE_ID_TOKENS = frozenset({"wall", "backing", "box", "housing"})
_AMBIGUOUS_SURFACE_DISTANCE_DELTA = 0.15
_COUNTERLIKE_FIXTURE_TYPES = frozenset(
    {"counter", "counter_non_dining", "counter_non_corner", "dining_counter", "island"}
)
_RECEPTACLE_FIXTURE_TYPES = frozenset(
    {
        "cabinet",
        "cabinet_single_door",
        "cabinet_double_door",
        "cabinet_with_door",
        "drawer",
        "top_drawer",
        "fridge",
        "sink",
        "blender",
        "stand_mixer",
        "dish_rack",
        "dishwasher",
        "microwave",
        "oven",
    }
)
_SURFACE_FIXTURE_TYPES = frozenset(
    {
        "counter",
        "counter_non_dining",
        "counter_non_corner",
        "dining_counter",
        "island",
        "stove",
    }
)
_SITE_REQUIRED_FIXTURE_TYPES = frozenset(
    {
        "cabinet",
        "cabinet_single_door",
        "cabinet_double_door",
        "cabinet_with_door",
        "drawer",
        "top_drawer",
        "fridge",
        "sink",
        "stove",
        "blender",
        "stand_mixer",
        "dish_rack",
        "dishwasher",
        "microwave",
        "oven",
    }
)
_PARTITIONED_SITE_TOKENS = frozenset(
    {"shelf", "burner", "basin", "slot", "tray", "rack", "bowl", "left", "right"}
)
_INTERIOR_SITE_TOKENS = frozenset(
    {"shelf", "basin", "slot", "tray", "rack", "bowl"}
)
_FRONT_BIASED_INTERIOR_FIXTURE_TYPES = frozenset(
    {
        "cabinet",
        "cabinet_single_door",
        "cabinet_double_door",
        "cabinet_with_door",
        "fridge",
        "microwave",
        "oven",
        "toaster",
        "toaster_oven",
        "dishwasher",
    }
)
_VERTICAL_INSERT_SUPPORT_TOKENS = frozenset({"cup", "mug", "glass", "pot"})
_VERTICAL_INSERT_OBJECT_TOKENS = frozenset({"straw", "skewer", "stirrer"})
_DEEP_RECEPTACLE_SUPPORT_TOKENS = frozenset(
    {"bowl", "cup", "mug", "glass", "pot", "jug", "pitcher"}
)
_SHALLOW_RECEPTACLE_SUPPORT_TOKENS = frozenset({"plate", "tray", "pan"})
# Supports whose bounding box includes a handle sticking out along the long
# axis.  Restrict candidate XY to the short-axis square to keep items on the
# usable surface instead of on the handle.
_HANDLED_SUPPORT_TOKENS = frozenset({"pan", "skillet", "saucepan", "pot"})
_DIRECTIONAL_SURFACE_OBJECT_TOKENS = frozenset(
    {
        "baguette",
        "bun",
        "fork",
        "hotdog",
        "knife",
        "spoon",
        "straw",
        "sausage",
        "skewer",
        "tongs",
    }
)
_FLAT_UTENSIL_OBJECT_TOKENS = frozenset({"fork", "knife", "spoon"})
_PARTIAL_DRAWER_OPEN_FRACTION = 0.55
_SURFACE_FLAT_OBJECT_TOKENS = frozenset(
    {"baguette", "bun", "fork", "hotdog", "knife", "sausage", "skewer", "spoon", "tongs"}
)
_SURFACE_UPRIGHT_OBJECT_TOKENS = frozenset(
    {
        "bottle",
        "can",
        "jar",
        "jug",
        "pitcher",
        "cup",
        "mug",
        "glass",
        "pot",
        "bowl",
        "shaker",
        "thermos",
        "vase",
        "candle",
        "kettle",
        "yogurt",
        "condiment",
        "ketchup",
        "mayonnaise",
        "mustard",
        "syrup",
    }
)
_TOP_UP_FLAT_ON_PLATE_OBJECT_TOKENS = frozenset({"bun", "hotdog", "sausage"})
_TRAY_BULKY_PAIR_OBJECT_TOKENS = frozenset(
    {
        "bottle",
        "cup",
        "glass",
        "jar",
        "kettle",
        "mug",
        "pitcher",
        "teapot",
        "thermos",
    }
)
_TRAY_KETTLE_MUG_PAIR_OBJECT_TOKENS = frozenset({"kettle", "mug"})
_COMPACT_SURFACE_DUPLICATE_TOKENS = frozenset({"cube", "sugar", "ice"})
_POSE_STABLE_SUPPORT_TOKENS = frozenset({"bowl", "board", "cutting"})
_NO_SETTLE_CHILD_SUPPORT_TOKENS = frozenset({"bowl", "board", "cutting", "pancake"})
_TOP_VISIBLE_SUPPORT_TOKENS = frozenset({"pancake"})
_DISPENSER_SITE_TOKENS = frozenset(
    {"dispenser", "dispense", "spout", "nozzle", "outlet"}
)
_RECEPTACLE_OBJECT_TOKENS = frozenset(
    {"bowl", "plate", "tray", "pan", "pot", "skillet"}
)
_FOOD_OBJECT_TOKENS = frozenset(
    {
        "bread",
        "bun",
        "cake",
        "carrot",
        "cheese",
        "chicken",
        "cucumber",
        "fish",
        "hotdog",
        "meat",
        "pancake",
        "sausage",
        "steak",
        "vegetable",
    }
)
_TOOL_OBJECT_TOKENS = frozenset(
    {"fork", "knife", "ladle", "spatula", "spoon", "tongs", "utensil"}
)


def _sim_tool_debug_enabled() -> bool:
    """Return whether sweep-level simulator debug prints should be shown."""

    return os.environ.get(_SWEEP_VERBOSE_ENV_VAR, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _is_approach_center(fixture) -> bool:
    """Return True if the robot should approach the fixture center, not an object inside it."""
    try:
        return any(fixture_is_type(fixture, ft) for ft in _APPROACH_CENTER_TYPES)
    except Exception:
        return False


def _is_countertop_appliance(fixture) -> bool:
    """Return whether occupancy-derived aisle-facing approach semantics apply."""
    try:
        return any(
            fixture_is_type(fixture, fixture_type)
            for fixture_type in _COUNTERTOP_APPLIANCE_TYPES
        )
    except Exception:
        return False


def _require_front(fixture) -> bool:
    """Return True if the robot must approach from the fixture's front face."""
    try:
        return any(fixture_is_type(fixture, ft) for ft in _REQUIRE_FRONT_TYPES)
    except Exception:
        return False


@dataclass
class ToolResult:
    tool_name: str
    success: bool
    details: dict[str, Any]


class SimToolExecutor(
    SimToolExecutorExecutionMixin,
    SimToolExecutorInspectionMixin,
    SimToolExecutorPlanningMixin,
    SimToolExecutorStateLoadingMixin,
):
    """Dispatch simulator tool calls against a live RoboCasa environment."""

    _HELD_Z_OFFSET = 0.0
    _NON_MUTATING_TOOL_NAMES = frozenset({"get_image", "communicate", "wait"})
    _DEMO_TASK_BY_NAME = {
        "cooperative_hotdog_setup": "HotDogSetup",
        "sandwich_station": "PrepareSandwichStation",
    }
    # Runtime-only fixture fields that must be restored when reusing one live
    # simulator across multiple trajectories. MuJoCo dynamic state snapshots do
    # not include Python-side counters, toggle flags, or cached control values.
    _FIXTURE_RUNTIME_ATTR_NAMES = {
        "_button_contact_prev_timestep",
        "_button_head_lock",
        "_cooldown",
        "_cooldown_time",
        "_doneness",
        "_door",
        "_door_target",
        "_function",
        "_head_value",
        "_last_lid_update",
        "_last_time_update",
        "_lid",
        "_lid_on_blender",
        "_lid_speed",
        "_num_steps_on",
        "_rack",
        "_state",
        "_target_lid_angle",
        "_temperature",
        "_time",
        "_timer",
        "_tray",
        "_turned_on",
        "_speed_dial_knob_value",
    }

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
        map_dpi: int = 300,
        map_renderer: str = "legacy",
        gl_backend: str = "osmesa",
        placement: str = "grid",
        cell_size: float = 0.05,
        align_to_wall: bool = True,
        standoff: float = 0.40,
        sample_spacing: float = 0.08,
        robot_radius: float = 0.18,
        robot_spawn: str = "trajectory",
        full_scene_view: bool = True,
        robot_colors: Sequence[Sequence[float]] | None = None,
        update_fxtr_cfg_dict: dict[str, dict[str, Any]] | None = None,
        trajectory_object_names: list[str] | tuple[str, ...] | None = None,
        trajectory_object_types: list[str] | tuple[str, ...] | None = None,
        trajectory_object_specs: dict | list | None = None,
    ):
        if map_dpi < 1:
            raise ValueError("map_dpi must be at least 1.")
        if map_renderer not in {"legacy", "raster"}:
            raise ValueError("map_renderer must be either 'legacy' or 'raster'.")
        os.environ["MUJOCO_GL"] = gl_backend
        self.runner = TrajectoryRunner(
            task_name=task_name,
            robots=robots,
            layout=layout,
            style=style,
            seed=seed,
            split=split,
            camera_names=camera_names,
            render_width=render_width,
            render_height=render_height,
            gl_backend=gl_backend,
            placement=placement,
            cell_size=cell_size,
            align_to_wall=align_to_wall,
            standoff=standoff,
            sample_spacing=sample_spacing,
            robot_radius=robot_radius,
            full_scene_view=full_scene_view,
            robot_colors=robot_colors,
            update_fxtr_cfg_dict=update_fxtr_cfg_dict,
            trajectory_object_names=trajectory_object_names,
            trajectory_object_types=trajectory_object_types,
            trajectory_object_specs=trajectory_object_specs,
        )
        self.env = self.runner.env
        self._task_name = task_name
        self._held_objects: dict[int, str] = {}
        self._held_object_offsets: dict[int, np.ndarray] = {}
        self._support_parents: dict[str, str] = {}
        # Single-use context for drawer flows:
        # robot -> drawer fixture most recently opened by open_sliding_part.
        # Used to avoid immediate re-navigation on the follow-up pickup.
        self._recent_opened_sliding_fixture: dict[int, str] = {}
        self._clean_map_labels: bool = True
        self._map_dpi: int = int(map_dpi)
        self._map_renderer: str = map_renderer
        self._robot_spawn: str = robot_spawn

        # Place all robots at the task's init_robot_base_ref (ground truth
        # starting position).  The env spawns robots at (10,10,0) by default;
        # this moves them to the fixture the task designates as the start.
        # In "trajectory" mode this is still called so the initial_map.png
        # shows sim ground truth; robots are repositioned later in
        # load_initial_state when trajectory agent locations are applied.
        self._place_robots_at_spawn()
        self._initialize_support_graph_from_scene()
        self._capture_baseline_state()

    def reset_scene(self) -> None:
        """Reset the underlying task env and refresh executor caches."""
        self.runner.reset_scene()
        self.env = self.runner.env
        self._held_objects.clear()
        self._held_object_offsets.clear()
        self._support_parents.clear()
        self._recent_opened_sliding_fixture.clear()
        self._place_robots_at_spawn()
        self._initialize_support_graph_from_scene()
        self._capture_baseline_state()
        invalidate_visual_cache = getattr(self, "_invalidate_visual_cache", None)
        if callable(invalidate_visual_cache):
            invalidate_visual_cache()

    def _place_robots_at_spawn(self):
        """Move all robots to init_robot_base_ref — the task's ground truth spawn."""
        scene = self.get_scene_description()
        spawn_fixture = scene.get("init_robot_base_ref")
        if not spawn_fixture or spawn_fixture not in self.runner._fixtures:
            return
        for i in range(self.runner._num_robots):
            self.runner._move_robot_near_fixture(i, spawn_fixture)
        # Belt-and-suspenders: verify every robot ended up inside kitchen
        for i in range(self.runner._num_robots):
            self.runner._rescue_robot_to_kitchen(i)

    def _snapshot_fixture_runtime_state(self) -> dict[str, dict[str, Any]]:
        """Capture Python-side fixture fields that live outside MuJoCo state."""
        runtime_state: dict[str, dict[str, Any]] = {}
        for fixture_id, fixture in self.runner._fixtures.items():
            fixture_state = {}
            for attr_name in self._FIXTURE_RUNTIME_ATTR_NAMES:
                if hasattr(fixture, attr_name):
                    fixture_state[attr_name] = deepcopy(getattr(fixture, attr_name))
            if fixture_state:
                runtime_state[fixture_id] = fixture_state
        return runtime_state

    def _restore_fixture_runtime_state(self) -> None:
        """Restore cached fixture runtime fields before update_state syncs visuals."""
        for fixture_id, fixture_state in self._baseline_fixture_runtime_state.items():
            fixture = self.runner._fixtures.get(fixture_id)
            if fixture is None:
                continue
            for attr_name, attr_value in fixture_state.items():
                setattr(fixture, attr_name, deepcopy(attr_value))

    def _capture_baseline_state(self) -> None:
        """Snapshot the clean post-construction simulator state for reuse."""
        self.env.sim.forward()
        self._baseline_sim_state = np.array(
            self.env.sim.get_state().flatten(),
            copy=True,
        )
        self._baseline_scene = deepcopy(self.runner.get_scene_description())
        self._baseline_object_locations = dict(self.runner._object_locations)
        self._baseline_fixture_runtime_state = self._snapshot_fixture_runtime_state()
        self._baseline_model_site_rgba = np.array(
            self.env.sim.model.site_rgba,
            copy=True,
        )
        self._baseline_model_site_size = np.array(
            self.env.sim.model.site_size,
            copy=True,
        )
        self._baseline_model_geom_rgba = np.array(
            self.env.sim.model.geom_rgba,
            copy=True,
        )

    def restore_baseline_state(self) -> None:
        """Return the live simulator to its clean post-construction baseline."""
        self.env.sim.set_state_from_flattened(self._baseline_sim_state.copy())
        self.env.sim.forward()
        self.env.sim.model.site_rgba[:] = self._baseline_model_site_rgba
        self.env.sim.model.site_size[:] = self._baseline_model_site_size
        self.env.sim.model.geom_rgba[:] = self._baseline_model_geom_rgba
        self._restore_fixture_runtime_state()
        self._held_objects.clear()
        self.runner._object_locations = dict(self._baseline_object_locations)
        self.runner._scene = deepcopy(self._baseline_scene)
        if hasattr(self.env, "update_sites"):
            self.env.update_sites()
        if hasattr(self.env, "update_state"):
            self.env.update_state()
        self.env.sim.forward()

    def _settle_scene(self, steps: int = _SETTLE_STEPS) -> None:
        """Advance a few zero-action physics steps after teleports/joint edits."""
        self.env.sim.forward()
        for _ in range(max(0, int(steps))):
            try:
                self.env.sim.step()
            except Exception:
                break
        self.env.sim.forward()

    def _initialize_support_graph_from_scene(self) -> None:
        """Seed explicit support parents from the current scene cache."""
        self._support_parents.clear()
        scene = self.get_scene_description()
        for object_id, object_info in (scene.get("objects") or {}).items():
            if not isinstance(object_id, str) or not isinstance(object_info, dict):
                continue
            location = object_info.get("location")
            if isinstance(location, str):
                self._support_parents[object_id] = location

    def _snapshot_fixture_runtime_state(self) -> dict[str, dict[str, Any]]:
        """Capture Python-side fixture fields that live outside MuJoCo state."""
        runtime_state: dict[str, dict[str, Any]] = {}
        for fixture_id, fixture in self.runner._fixtures.items():
            fixture_state: dict[str, Any] = {}
            for attr_name in _FIXTURE_RUNTIME_ATTR_NAMES:
                if hasattr(fixture, attr_name):
                    fixture_state[attr_name] = deepcopy(getattr(fixture, attr_name))
            if fixture_state:
                runtime_state[fixture_id] = fixture_state
        return runtime_state

    def _restore_fixture_runtime_state(self) -> None:
        """Restore cached fixture runtime fields before update_state syncs visuals."""
        for fixture_id, fixture_state in self._baseline_fixture_runtime_state.items():
            fixture = self.runner._fixtures.get(fixture_id)
            if fixture is None:
                continue
            for attr_name, attr_value in fixture_state.items():
                setattr(fixture, attr_name, deepcopy(attr_value))

    def restore_baseline_state(self) -> None:
        """Return the live simulator to its clean post-construction baseline."""
        self.env.sim.set_state_from_flattened(self._baseline_sim_state.copy())
        self.env.sim.forward()
        self.env.sim.model.site_rgba[:] = self._baseline_model_site_rgba
        self.env.sim.model.site_size[:] = self._baseline_model_site_size
        self.env.sim.model.geom_rgba[:] = self._baseline_model_geom_rgba
        self._restore_fixture_runtime_state()
        held_objects = getattr(self, "_held_objects", None)
        if hasattr(held_objects, "clear"):
            held_objects.clear()
        held_object_offsets = getattr(self, "_held_object_offsets", None)
        if hasattr(held_object_offsets, "clear"):
            held_object_offsets.clear()
        recent_opened_sliding_fixture = getattr(
            self,
            "_recent_opened_sliding_fixture",
            None,
        )
        if hasattr(recent_opened_sliding_fixture, "clear"):
            recent_opened_sliding_fixture.clear()
        self.runner._object_locations = dict(self._baseline_object_locations)
        self.runner._scene = deepcopy(self._baseline_scene)
        support_parents = getattr(self, "_support_parents", None)
        if not isinstance(support_parents, dict):
            support_parents = {}
            self._support_parents = support_parents
        support_parents.clear()
        for object_id, object_info in (self._baseline_scene.get("objects") or {}).items():
            if not isinstance(object_id, str) or not isinstance(object_info, dict):
                continue
            location = object_info.get("location")
            if isinstance(location, str):
                support_parents[object_id] = location
        if hasattr(self.env, "update_sites"):
            self.env.update_sites()
        if hasattr(self.env, "update_state"):
            self.env.update_state()
        self.env.sim.forward()
        invalidate_visual_cache = getattr(self, "_invalidate_visual_cache", None)
        if callable(invalidate_visual_cache):
            invalidate_visual_cache()

    def _set_support_parent(self, object_id: str, parent_id: str | None) -> None:
        if parent_id is None:
            self._support_parents.pop(object_id, None)
        else:
            self._support_parents[object_id] = parent_id

    def _iter_direct_supported_children(self, parent_id: str) -> list[str]:
        support_parents = getattr(self, "_support_parents", None)
        if not isinstance(support_parents, dict):
            return []
        return sorted(
            object_id
            for object_id, support_parent in support_parents.items()
            if support_parent == parent_id
        )

    def _iter_supported_descendants(self, parent_id: str) -> list[str]:
        descendants: list[str] = []
        seen = {parent_id}
        pending = [parent_id]
        while pending:
            current_parent = pending.pop()
            for child_id in self._iter_direct_supported_children(current_parent):
                if child_id in seen:
                    continue
                seen.add(child_id)
                descendants.append(child_id)
                pending.append(child_id)
        return descendants

    @staticmethod
    def _normalize_site_signature(token: str) -> tuple[str, ...]:
        normalized_tokens: list[str] = []
        for raw_token in _SITE_TOKEN_RE.findall(str(token).lower()):
            alias = "rear" if raw_token in {"back", "top"} else raw_token
            if alias == "bottom":
                alias = "front"
            if alias in _SITE_STOPWORDS or not alias:
                continue
            if alias not in normalized_tokens:
                normalized_tokens.append(alias)
        if normalized_tokens:
            return tuple(normalized_tokens)
        lowered = str(token).strip().lower()
        return (lowered,) if lowered else ()

    def _normalize_site_id(self, site_id: str | None) -> str | None:
        if not isinstance(site_id, str):
            return None
        lowered = "_".join(_SITE_TOKEN_RE.findall(site_id.lower()))
        if not lowered:
            return None
        for suffix, canonical in (
            ("_basin", "basin"),
            ("_bowl", "bowl"),
            ("_rack", "rack"),
            ("_tray", "tray"),
            ("_shelf", "shelf"),
            ("_burner", "burner"),
        ):
            if lowered.endswith(suffix):
                return canonical
        if lowered.startswith("level") and lowered[5:].isdigit():
            return f"shelf_{int(lowered[5:])}"
        if lowered.startswith("rack") and lowered[4:].isdigit():
            return f"rack_{int(lowered[4:])}"
        if lowered.startswith("tray") and lowered[4:].isdigit():
            return f"tray_{int(lowered[4:])}"
        if lowered == "slotl":
            return "slot_left"
        if lowered == "slotr":
            return "slot_right"
        if lowered.startswith("sidel_slot"):
            return f"slot_pair_0_{'left' if lowered.endswith('slotl') else 'right'}"
        if lowered.startswith("sider_slot"):
            return f"slot_pair_1_{'left' if lowered.endswith('slotl') else 'right'}"
        return lowered.replace("back", "rear")

    def _raw_support_site_to_external(self, raw_site_id: str) -> str:
        normalized = self._normalize_site_id(raw_site_id)
        return normalized if isinstance(normalized, str) else raw_site_id

    def _get_fixture_reset_regions(self, fixture_id: str) -> dict[str, dict[str, Any]]:
        fixture = self._require_fixture(fixture_id)
        try:
            reset_regions = fixture.get_reset_regions(env=self.env)
        except Exception:
            reset_regions = None
        if not isinstance(reset_regions, dict):
            return {}
        return {
            str(site_id): dict(region)
            for site_id, region in reset_regions.items()
            if isinstance(site_id, str) and isinstance(region, dict)
        }

    def _get_fixture_type_name(self, fixture_id: str) -> str | None:
        scene = self.get_scene_description()
        fixture_info = (scene.get("fixtures") or {}).get(fixture_id, {})
        fixture_type = fixture_info.get("fixture_type")
        return str(fixture_type) if isinstance(fixture_type, str) else None

    def _fixture_is_cabinet(self, fixture_id: str) -> bool:
        return (self._get_fixture_type_name(fixture_id) or "").lower() in {
            "cab",
            "cabinet",
            "cabinet_single_door",
            "cabinet_double_door",
            "cabinet_with_door",
        }

    def _fixture_parent_id(self, fixture_id: str) -> str | None:
        state = (self.get_scene_description().get("fixtures") or {}).get(
            fixture_id, {}
        )
        parent = state.get("parent_fixture")
        return parent if isinstance(parent, str) and parent else None

    def _canonical_navigation_fixture_id(self, fixture_id: str) -> str:
        if self._fixture_is_cabinet(fixture_id):
            return self._fixture_parent_id(fixture_id) or fixture_id
        return fixture_id

    def _robot_at_exclusive_child_of_parent(
        self, robot_idx: int, parent_fixture_id: str
    ) -> bool:
        scene = self.get_scene_description().get("fixtures") or {}
        for child_id, state in scene.items():
            if not isinstance(state, dict) or state.get("parent_fixture") != parent_fixture_id:
                continue
            child_type = str(state.get("fixture_type") or "").lower()
            if child_type not in EXCLUSIVE_FIXTURE_TYPES:
                continue
            if self._robot_near_fixture(robot_idx, child_id):
                return True
        return False

    def _fixture_id_is_structural_obstacle(self, fixture_id: str) -> bool:
        lowered = str(fixture_id).lower()
        if any(token in lowered for token in _STRUCTURAL_FIXTURE_ID_TOKENS):
            return True
        fixture = getattr(self.runner, "_fixtures", {}).get(fixture_id)
        if fixture is None:
            return False
        return type(fixture).__name__.lower() == "box"

    def _surface_fixture_prefers_front_approach(self, fixture_id: str) -> bool:
        if self._fixture_placement_semantics(fixture_id) != "surface":
            return False
        fixture_type = (self._get_fixture_type_name(fixture_id) or "").lower()
        fixture_name = str(fixture_id).lower()
        fixture = getattr(self.runner, "_fixtures", {}).get(fixture_id)
        raw_fixture_name = getattr(fixture, "name", "")
        if isinstance(raw_fixture_name, str):
            fixture_name = f"{fixture_name} {raw_fixture_name.lower()}"
        return "corner" in fixture_name or fixture_type == "counter_corner"

    def _fixture_requires_front_approach(self, fixture_id: str) -> bool:
        fixture = getattr(self.runner, "_fixtures", {}).get(fixture_id)
        return fixture is not None and (
            _require_front(fixture) or isinstance(fixture, CoffeeMachine)
        )

    def _fixture_is_drawer(self, fixture_id: str) -> bool:
        fixture_type = (self._get_fixture_type_name(fixture_id) or "").lower()
        if fixture_type in {"drawer", "top_drawer"}:
            return True
        return "drawer" in str(fixture_id).lower()

    def _fixture_placement_semantics(self, fixture_id: str) -> str:
        fixture_type = (self._get_fixture_type_name(fixture_id) or "").lower()
        if fixture_type in _RECEPTACLE_FIXTURE_TYPES:
            return "receptacle"
        if fixture_type in _SURFACE_FIXTURE_TYPES:
            return "surface"

        support_sites = [site_id.lower() for site_id in self.get_support_sites(fixture_id)]
        if any(
            any(token in site_id for token in ("shelf", "basin", "slot", "tray", "rack", "bowl"))
            for site_id in support_sites
        ):
            return "receptacle"
        return "surface"

    def _fixture_requires_explicit_site(self, fixture_id: str) -> bool:
        support_sites = [site_id.lower() for site_id in self.get_support_sites(fixture_id)]
        if len(support_sites) <= 1:
            return False
        fixture_type = (self._get_fixture_type_name(fixture_id) or "").lower()
        if fixture_type in _SITE_REQUIRED_FIXTURE_TYPES:
            return True
        return any(
            any(token in site_id for token in _PARTITIONED_SITE_TOKENS)
            for site_id in support_sites
        )

    def _require_explicit_site_if_needed(
        self,
        fixture_id: str,
        site_id: str | None,
        *,
        action_name: str,
    ) -> None:
        if site_id is None and self._fixture_requires_explicit_site(fixture_id):
            raise ValueError(
                f"{action_name} requires an explicit site for fixture {fixture_id!r}."
            )

    def _normalize_target_site_id_for_placement(
        self,
        fixture_id: str,
        site_id: str | None,
    ) -> str | None:
        if not isinstance(site_id, str):
            return None
        external_site_id = site_id
        try:
            resolved_site_id = self._resolve_fixture_site_id(fixture_id, site_id)
        except Exception:
            resolved_site_id = site_id
        if isinstance(resolved_site_id, str):
            external_site_id = self._raw_support_site_to_external(resolved_site_id)
        if self._fixture_placement_semantics(fixture_id) == "surface":
            lowered = external_site_id.lower()
            if lowered == "geom" or lowered.startswith("geom_"):
                return None
        return external_site_id

    def _default_support_site_for_unspecified_fixture(
        self,
        fixture_id: str,
        *,
        incoming_object_id: str | None = None,
    ) -> str | None:
        reset_regions = self._get_fixture_reset_regions(fixture_id)
        if not reset_regions:
            return None

        fixture_type = (self._get_fixture_type_name(fixture_id) or "").lower()
        prefer_bottom_for_cabinet = fixture_type.startswith("cabinet")
        if prefer_bottom_for_cabinet:
            # If the cabinet exposes exactly one shelf site, use it directly.
            # This avoids ambiguous ranking and keeps repeated placements stable.
            cabinet_shelf_sites: list[str] = []
            for raw_site_id in reset_regions:
                site_id = self._normalize_target_site_id_for_placement(
                    fixture_id,
                    raw_site_id,
                )
                if not isinstance(site_id, str):
                    continue
                site_tokens = set(self._normalize_site_signature(site_id))
                raw_tokens = set(_SITE_TOKEN_RE.findall(raw_site_id.lower()))
                if "shelf" in (site_tokens | raw_tokens):
                    cabinet_shelf_sites.append(site_id)
            unique_shelf_sites = sorted(dict.fromkeys(cabinet_shelf_sites))
            if len(unique_shelf_sites) == 1:
                return unique_shelf_sites[0]

        ranked_sites: list[tuple[int, int, float, str]] = []
        exclude = {incoming_object_id} if isinstance(incoming_object_id, str) else set()
        for raw_site_id, region in reset_regions.items():
            site_id = self._normalize_target_site_id_for_placement(fixture_id, raw_site_id)
            if not isinstance(site_id, str):
                continue
            area = float(
                np.prod(np.asarray(region.get("size", (0.0, 0.0)), dtype=float)[:2])
            )
            existing_count = len(
                self._objects_on_fixture_site(
                    fixture_id,
                    site_id=raw_site_id,
                    exclude=exclude,
                )
            )
            site_tokens = set(self._normalize_site_signature(site_id))
            raw_site_tokens = set(_SITE_TOKEN_RE.findall(raw_site_id.lower()))
            bottom_priority = 0
            if prefer_bottom_for_cabinet:
                # Cabinet default when site is unspecified: bias toward bottom shelf.
                if {"bottom", "lower", "low"} & (site_tokens | raw_site_tokens):
                    bottom_priority = -3
                elif "middle" in (site_tokens | raw_site_tokens):
                    bottom_priority = -2
                elif {"top", "upper", "up"} & (site_tokens | raw_site_tokens):
                    bottom_priority = -1
            ranked_sites.append((existing_count, bottom_priority, -area, site_id))

        if not ranked_sites:
            return None
        ranked_sites.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
        return ranked_sites[0][3]

    def _surface_fixture_slot_positions(
        self,
        fixture_id: str,
        *,
        count: int,
    ) -> list[np.ndarray]:
        bounds = self._fixture_local_bounds(fixture_id, site_id=None)
        if bounds is None:
            return []
        local_min, local_max = bounds
        size = np.asarray(local_max, dtype=float) - np.asarray(local_min, dtype=float)
        axis_is_x = float(size[0]) >= float(size[1])
        major = max(float(size[0]), float(size[1]), 1e-6)
        minor = max(min(float(size[0]), float(size[1])), 1e-6)
        elongated = (major / minor) >= 1.35
        primary_half = (0.42 if elongated else 0.34) * major
        secondary_half = (0.14 if elongated else 0.22) * minor
        slot_offsets = self._slot_offsets_from_half_extents(
            count=max(int(count), 1),
            primary_half=primary_half,
            secondary_half=secondary_half,
            axis_is_x=axis_is_x,
            elongated=elongated,
        )
        center = 0.5 * (np.asarray(local_min, dtype=float) + np.asarray(local_max, dtype=float))
        fixture = self._require_fixture(fixture_id)
        slot_positions: list[np.ndarray] = []
        clip_margin = np.minimum(0.04, 0.25 * size)
        for local_offset in slot_offsets:
            local_xy = center + np.asarray(local_offset, dtype=float)
            local_xy = np.clip(local_xy, local_min + clip_margin, local_max - clip_margin)
            world = self.runner._fixture_local_to_world(
                fixture,
                np.array([local_xy[0], local_xy[1], 0.0], dtype=float),
            )
            slot_positions.append(np.asarray(world[:2], dtype=float))
        return slot_positions

    def _fixture_surface_blocker_positions(
        self,
        fixture_id: str,
    ) -> list[np.ndarray]:
        scene_fixtures = (self.get_scene_description().get("fixtures") or {})
        blocker_positions: list[np.ndarray] = []
        seen_positions: set[tuple[float, float]] = set()
        for child_fixture_id, fixture_info in scene_fixtures.items():
            if child_fixture_id == fixture_id:
                continue
            if fixture_info.get("parent_fixture") != fixture_id:
                continue
            fixture_type = str(fixture_info.get("fixture_type") or "").lower()
            if fixture_type not in _COUNTERTOP_APPLIANCE_TYPE_NAMES:
                continue
            child_fixture = self.runner._fixtures.get(child_fixture_id)
            if child_fixture is None or getattr(child_fixture, "pos", None) is None:
                continue
            blocker_xy = np.asarray(child_fixture.pos[:2], dtype=float)
            key = tuple(np.round(blocker_xy, 4))
            if key in seen_positions:
                continue
            seen_positions.add(key)
            blocker_positions.append(blocker_xy)
        return blocker_positions

    def _default_fixture_surface_preference(
        self,
        fixture_id: str,
        incoming_object_id: str,
    ) -> np.ndarray | None:
        existing_object_ids = self._objects_on_fixture_site(
            fixture_id,
            exclude={incoming_object_id},
        )
        existing_positions = [
            self._get_object_pose(object_id)[0][:2].copy()
            for object_id in existing_object_ids
        ]
        if existing_positions:
            incoming_tokens = self._scene_object_tokens(incoming_object_id)
        else:
            incoming_tokens = set()
        if existing_positions and incoming_tokens & _COMPACT_SURFACE_DUPLICATE_TOKENS:
            same_family_positions = [
                self._get_object_pose(object_id)[0][:2].copy()
                for object_id in existing_object_ids
                if self._scene_object_tokens(object_id)
                & _COMPACT_SURFACE_DUPLICATE_TOKENS
            ]
            anchor_positions = same_family_positions
            if not anchor_positions:
                anchor_positions = [
                    self._get_object_pose(object_id)[0][:2].copy()
                    for object_id in existing_object_ids
                    if self._scene_object_tokens(object_id)
                    & {"cake", "container", "plate"}
                ]
            if anchor_positions:
                fixture = self._require_fixture(fixture_id)
                fixture_angle = float(getattr(fixture, "rot", 0.0) or 0.0)
                local_x_world = np.array(
                    [np.cos(fixture_angle), np.sin(fixture_angle)],
                    dtype=float,
                )
                offset_scale = len(same_family_positions) if same_family_positions else 1
                offset = local_x_world * min(0.08 * offset_scale, 0.16)
                return self._project_xy_onto_fixture(
                    fixture_id,
                    np.asarray(anchor_positions[-1], dtype=float) + offset,
                )
        blocker_positions = self._fixture_surface_blocker_positions(fixture_id)
        slot_positions = self._surface_fixture_slot_positions(
            fixture_id,
            count=len(existing_positions) + len(blocker_positions) + 1,
        )
        center_xy = self._preferred_xy_for_fixture_target(fixture_id)
        if center_xy is None and slot_positions:
            center_xy = slot_positions[0]
        if center_xy is None:
            return None
        return self._choose_slot_position(
            slot_positions,
            existing_positions=[*existing_positions, *blocker_positions],
            center_xy=np.asarray(center_xy, dtype=float),
        )

    def _infer_object_support_site(
        self,
        object_id: str,
        fixture_id: str,
    ) -> str | None:
        reset_regions = self._get_fixture_reset_regions(fixture_id)
        if not reset_regions:
            return None
        try:
            object_pos, _ = self._get_object_pose(object_id)
        except Exception:
            return None
        fixture = self._require_fixture(fixture_id)
        local_xy = self.runner._world_to_fixture_local(fixture, object_pos[:2])
        matches: list[str] = []
        for raw_site_id, region in reset_regions.items():
            offset = np.asarray(region.get("offset", (0.0, 0.0, 0.0)), dtype=float)
            size = np.asarray(region.get("size", (0.0, 0.0, 0.0)), dtype=float)
            half = 0.5 * size[:2]
            if np.all(np.abs(local_xy[:2] - offset[:2]) <= half + 1e-6):
                matches.append(raw_site_id)
        if len(matches) == 1:
            return matches[0]
        return None

    def _resolve_fixture_site_id(
        self,
        fixture_id: str,
        requested_site_id: str | None,
    ) -> str | None:
        if requested_site_id is None:
            return None

        reset_regions = self._get_fixture_reset_regions(fixture_id)
        if not reset_regions:
            raise ValueError(f"Fixture {fixture_id!r} does not expose support sites.")

        if requested_site_id in reset_regions:
            return requested_site_id

        requested_normalized = self._normalize_site_id(requested_site_id)
        requested_signature = self._normalize_site_signature(requested_site_id)
        matches: list[str] = []
        for raw_site_id in reset_regions:
            raw_normalized = self._raw_support_site_to_external(raw_site_id)
            if requested_normalized in {raw_site_id.lower(), raw_normalized}:
                matches.append(raw_site_id)
                continue
            if requested_signature == self._normalize_site_signature(raw_normalized):
                matches.append(raw_site_id)

        if not matches and isinstance(requested_normalized, str):
            token_matches = []
            for raw_site_id in reset_regions:
                raw_normalized = self._raw_support_site_to_external(raw_site_id)
                raw_signature = self._normalize_site_signature(raw_normalized)
                if requested_normalized in raw_signature:
                    token_matches.append(raw_site_id)
            matches = sorted(dict.fromkeys(token_matches))

        if not matches and set(requested_signature) & _DISPENSER_SITE_TOKENS:
            dispenser_matches = []
            for raw_site_id in reset_regions:
                raw_normalized = self._raw_support_site_to_external(raw_site_id)
                raw_signature = set(self._normalize_site_signature(raw_normalized))
                if raw_normalized == "bottom" or "front" in raw_signature:
                    dispenser_matches.append(raw_site_id)
            dispenser_matches = sorted(dict.fromkeys(dispenser_matches))
            if len(dispenser_matches) == 1:
                return dispenser_matches[0]

        if not matches:
            requested_signature_set = set(requested_signature)
            support_family_tokens = (
                "shelf",
                "rack",
                "basin",
                "slot",
                "tray",
                "burner",
                "bowl",
                "geom",
                "int",
            )
            requested_family = next(
                (
                    token
                    for token in support_family_tokens
                    if token in requested_signature_set
                ),
                None,
            )
            if requested_family is not None:
                family_matches = []
                for raw_site_id in reset_regions:
                    raw_normalized = self._raw_support_site_to_external(raw_site_id)
                    raw_signature = set(self._normalize_site_signature(raw_normalized))
                    if requested_family in raw_signature:
                        family_matches.append(raw_site_id)
                family_matches = sorted(dict.fromkeys(family_matches))
                if len(family_matches) == 1:
                    return family_matches[0]

        matches = sorted(dict.fromkeys(matches))
        if len(matches) == 1:
            return matches[0]
        if matches:
            raise ValueError(
                f"Ambiguous site {requested_site_id!r} for fixture {fixture_id!r}: "
                f"{[self._raw_support_site_to_external(site_id) for site_id in matches]}"
            )
        raise ValueError(
            f"Unknown site {requested_site_id!r} for fixture {fixture_id!r}. "
            f"Available: {self.get_support_sites(fixture_id)}"
        )

    def _score_support_target_site_match(
        self,
        support_id: str,
        fixture_id: str,
    ) -> int:
        requested_tokens = set(_SITE_TOKEN_RE.findall(str(support_id).lower()))
        if not requested_tokens:
            return 0

        fixture_tokens = set(_SITE_TOKEN_RE.findall(str(fixture_id).lower()))
        fixture_type = (self._get_fixture_type_name(fixture_id) or "").lower()
        fixture_tokens.update(_SITE_TOKEN_RE.findall(fixture_type))
        fixture = getattr(self.runner, "_fixtures", {}).get(fixture_id)
        fixture_name = getattr(fixture, "name", None)
        if isinstance(fixture_name, str):
            fixture_tokens.update(_SITE_TOKEN_RE.findall(fixture_name.lower()))
        return len(requested_tokens & fixture_tokens)

    def _fixture_local_bounds(
        self,
        fixture_id: str,
        site_id: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        reset_regions = self._get_fixture_reset_regions(fixture_id)
        regions: list[dict[str, Any]] = []
        if site_id is not None:
            raw_site_id = self._resolve_fixture_site_id(fixture_id, site_id)
            region = reset_regions.get(raw_site_id)
            if isinstance(region, dict):
                regions = [region]
        else:
            regions = list(reset_regions.values())

        if regions:
            minima = []
            maxima = []
            for region in regions:
                offset = np.asarray(region.get("offset", (0.0, 0.0, 0.0)), dtype=float)
                size = np.asarray(region.get("size", (0.0, 0.0, 0.0)), dtype=float)
                minima.append(offset[:2] - 0.5 * size[:2])
                maxima.append(offset[:2] + 0.5 * size[:2])
            return np.min(np.stack(minima), axis=0), np.max(np.stack(maxima), axis=0)

        fixture = self._require_fixture(fixture_id)
        aabb = get_fixture_aabb(fixture)
        if aabb is None:
            return None
        corners = [
            self.runner._world_to_fixture_local(fixture, np.asarray(corner, dtype=float)[:2])
            for corner in (
                np.array([aabb[0][0], aabb[0][1]]),
                np.array([aabb[0][0], aabb[1][1]]),
                np.array([aabb[1][0], aabb[0][1]]),
                np.array([aabb[1][0], aabb[1][1]]),
            )
        ]
        return np.min(np.stack(corners), axis=0), np.max(np.stack(corners), axis=0)

    def _relative_direction_vector(
        self,
        relative_position: str | None,
    ) -> np.ndarray | None:
        if not isinstance(relative_position, str):
            return None
        tokens = {
            "rear" if token in {"back", "top"} else "front" if token == "bottom" else token
            for token in _SITE_TOKEN_RE.findall(relative_position.lower())
        }
        vector = np.zeros(2, dtype=float)
        if "left" in tokens:
            vector[0] -= 1.0
        if "right" in tokens:
            vector[0] += 1.0
        if "front" in tokens:
            vector[1] -= 1.0
        if "rear" in tokens:
            vector[1] += 1.0
        if np.linalg.norm(vector) <= 1e-9:
            return np.zeros(2, dtype=float) if "center" in tokens else None
        return vector / max(np.linalg.norm(vector), 1.0)

    def _preferred_xy_for_fixture_target(
        self,
        fixture_id: str,
        *,
        site_id: str | None = None,
        relative_position: str | None = None,
    ) -> np.ndarray | None:
        bounds = self._fixture_local_bounds(fixture_id, site_id=site_id)
        if bounds is None:
            return None
        local_min, local_max = bounds
        center = 0.5 * (local_min + local_max)
        fixture_type = (self._get_fixture_type_name(fixture_id) or "").lower()
        if fixture_type in _FRONT_BIASED_INTERIOR_FIXTURE_TYPES and relative_position is None:
            front_face = self._fixture_front_face(fixture_id)
            if front_face is not None:
                inward_local = np.zeros(2, dtype=float)
                axis_idx = 1
                if front_face == "neg_y":
                    inward_local = np.array([0.0, 1.0], dtype=float)
                    axis_idx = 1
                elif front_face == "pos_y":
                    inward_local = np.array([0.0, -1.0], dtype=float)
                    axis_idx = 1
                elif front_face == "neg_x":
                    inward_local = np.array([1.0, 0.0], dtype=float)
                    axis_idx = 0
                elif front_face == "pos_x":
                    inward_local = np.array([-1.0, 0.0], dtype=float)
                    axis_idx = 0
                half_extent = 0.5 * (local_max - local_min)
                inward_bias = min(0.35 * float(half_extent[axis_idx]), 0.055)
                center = center + inward_local * inward_bias
        direction = self._relative_direction_vector(relative_position)
        if direction is not None:
            half_extent = np.maximum(0.5 * (local_max - local_min) - 0.03, 0.0)
            center = center + direction * (0.7 * half_extent)
        center = np.clip(center, local_min + 0.02, local_max - 0.02)
        fixture = self._require_fixture(fixture_id)
        world_pos = self.runner._fixture_local_to_world(
            fixture,
            np.array([center[0], center[1], 0.0], dtype=float),
        )
        return np.asarray(world_pos[:2], dtype=float)

    def _project_xy_onto_fixture(
        self,
        fixture_id: str,
        world_xy: np.ndarray,
        *,
        site_id: str | None = None,
    ) -> np.ndarray:
        bounds = self._fixture_local_bounds(fixture_id, site_id=site_id)
        if bounds is None:
            return np.asarray(world_xy, dtype=float)[:2]
        local_min, local_max = bounds
        fixture = self._require_fixture(fixture_id)
        local_xy = self.runner._world_to_fixture_local(
            fixture,
            np.asarray(world_xy, dtype=float)[:2],
        )
        clipped_local = np.clip(local_xy[:2], local_min, local_max)
        world_pos = self.runner._fixture_local_to_world(
            fixture,
            np.array([clipped_local[0], clipped_local[1], 0.0], dtype=float),
        )
        return np.asarray(world_pos[:2], dtype=float)

    # ------------------------------------------------------------------
    # Scene / state helpers
    # ------------------------------------------------------------------

    def _require_fixture(self, fixture_id: str):
        if fixture_id not in self.runner._fixtures:
            raise ValueError(f"Unknown fixture_id: {fixture_id!r}")
        return self.runner._fixtures[fixture_id]

    def _require_object(self, object_id: str):
        if object_id not in self.env.objects:
            raise ValueError(f"Unknown object_id: {object_id!r}")
        return self.env.objects[object_id]

    def _get_object_pose(self, object_id: str) -> tuple[np.ndarray, np.ndarray]:
        obj = self._require_object(object_id)
        qpos = self.env.sim.data.get_joint_qpos(obj.joints[0]).copy()
        return qpos[:3], qpos[3:7]

    def _get_object_placement_metadata(self, object_id: str) -> dict[str, Any]:
        return self.runner._get_object_placement_metadata(object_id)

    def _get_scene_object_type(self, object_id: str) -> str:
        object_info = (self.get_scene_description().get("objects") or {}).get(
            object_id,
            {},
        )
        object_type = object_info.get("object_type")
        if isinstance(object_type, str):
            return object_type.lower()
        return str(object_id).lower()

    def _scene_object_tokens(self, object_id: str) -> set[str]:
        return self._object_tokens(
            self._get_scene_object_type(object_id)
        ) | self._object_tokens(object_id)

    def _default_relative_position_next_to_object(
        self,
        object_id: str,
        reference_object_id: str,
    ) -> str | None:
        object_tokens = self._scene_object_tokens(object_id)
        reference_tokens = self._scene_object_tokens(reference_object_id)
        if "plate" not in reference_tokens:
            return None
        if "fork" in object_tokens:
            return "left"
        if object_tokens & {"knife", "spoon"}:
            return "right"
        return None

    def _support_object_tokens(self, support_object_id: str) -> set[str]:
        return self._scene_object_tokens(support_object_id)

    @staticmethod
    def _object_tokens(text: str | None) -> set[str]:
        if not isinstance(text, str):
            return set()
        return {token for token in _SITE_TOKEN_RE.findall(text.lower()) if token}

    def _object_bbox_metadata_for_quat(
        self,
        object_id: str,
        quat_wxyz: np.ndarray,
    ) -> dict[str, Any]:
        obj = self._require_object(object_id)
        quat_xyzw = T.convert_quat(np.asarray(quat_wxyz, dtype=float), to="xyzw")
        bbox_points = np.asarray(
            obj.get_bbox_points(trans=np.zeros(3, dtype=float), rot=quat_xyzw),
            dtype=float,
        )
        spans = np.max(bbox_points, axis=0) - np.min(bbox_points, axis=0)
        bottom_z = float(np.min(bbox_points[:, 2]))
        xy_radius = 0.5 * float(np.linalg.norm(spans[:2]))
        half_spans_xy = 0.5 * np.asarray(spans[:2], dtype=float)
        return {
            "bbox_points": bbox_points,
            "spans": spans,
            "bottom_z": bottom_z,
            "xy_radius": max(xy_radius, 1e-6),
            "half_spans_xy": np.maximum(half_spans_xy, 1e-6),
        }

    @staticmethod
    def _slot_templates(count: int, *, elongated: bool) -> list[tuple[float, float]]:
        if count <= 1:
            return [(0.0, 0.0)]
        if elongated:
            if count == 2:
                return [(-0.55, 0.0), (0.55, 0.0)]
            if count == 3:
                return [(-0.78, 0.0), (0.0, 0.0), (0.78, 0.0)]
            if count == 4:
                return [(-0.9, 0.0), (-0.3, 0.0), (0.3, 0.0), (0.9, 0.0)]
            return [
                (-0.92, 0.0),
                (-0.46, 0.0),
                (0.0, 0.0),
                (0.46, 0.0),
                (0.92, 0.0),
                (-0.7, 0.42),
                (0.7, 0.42),
                (-0.7, -0.42),
                (0.7, -0.42),
            ]

        if count == 2:
            return [(-0.48, 0.0), (0.48, 0.0)]
        if count == 3:
            return [(-0.55, 0.0), (0.55, 0.0), (0.0, 0.0)]
        if count == 4:
            return [(-0.48, -0.34), (-0.48, 0.34), (0.48, -0.34), (0.48, 0.34)]
        return [
            (-0.52, -0.36),
            (-0.52, 0.36),
            (0.52, -0.36),
            (0.52, 0.36),
            (0.0, 0.0),
            (0.0, -0.48),
            (0.0, 0.48),
        ]

    @classmethod
    def _slot_offsets_from_half_extents(
        cls,
        *,
        count: int,
        primary_half: float,
        secondary_half: float,
        axis_is_x: bool,
        elongated: bool,
    ) -> list[np.ndarray]:
        offsets: list[np.ndarray] = []
        for primary_norm, secondary_norm in cls._slot_templates(
            count,
            elongated=elongated,
        ):
            if axis_is_x:
                offset = np.array(
                    [primary_norm * primary_half, secondary_norm * secondary_half],
                    dtype=float,
                )
            else:
                offset = np.array(
                    [secondary_norm * secondary_half, primary_norm * primary_half],
                    dtype=float,
                )
            offsets.append(offset)
        return offsets

    @staticmethod
    def _choose_slot_position(
        slot_positions: Sequence[np.ndarray],
        *,
        existing_positions: Sequence[np.ndarray],
        center_xy: np.ndarray,
    ) -> np.ndarray | None:
        if not slot_positions:
            return None
        center_xy = np.asarray(center_xy, dtype=float)[:2]
        if not existing_positions:
            return min(
                slot_positions,
                key=lambda position: float(
                    np.linalg.norm(np.asarray(position, dtype=float)[:2] - center_xy)
                ),
            )

        best_position: np.ndarray | None = None
        best_score: tuple[float, float] | None = None
        for position in slot_positions:
            pos_xy = np.asarray(position, dtype=float)[:2]
            min_existing_distance = min(
                float(np.linalg.norm(pos_xy - np.asarray(existing, dtype=float)[:2]))
                for existing in existing_positions
            )
            center_distance = float(np.linalg.norm(pos_xy - center_xy))
            score = (min_existing_distance, -center_distance)
            if best_score is None or score > best_score:
                best_score = score
                best_position = pos_xy
        return best_position

    def _support_object_geometry(self, support_object_id: str) -> dict[str, Any]:
        support_obj = self._require_object(support_object_id)
        support_body_id = self.env.obj_body_id[support_object_id]
        support_pos = self.env.sim.data.body_xpos[support_body_id].copy()
        support_quat_xyzw = T.convert_quat(
            self.env.sim.data.body_xquat[support_body_id].copy(),
            to="xyzw",
        )
        support_rot = T.quat2mat(support_quat_xyzw)
        local_points = np.asarray(
            support_obj.get_bbox_points(trans=np.zeros(3, dtype=float), rot=None),
            dtype=float,
        )
        local_min = np.min(local_points, axis=0)
        local_max = np.max(local_points, axis=0)
        local_center = 0.5 * (local_min + local_max)
        local_size = local_max - local_min
        world_points = np.asarray(
            support_obj.get_bbox_points(trans=support_pos, rot=support_quat_xyzw),
            dtype=float,
        )
        world_center = support_pos + support_rot @ local_center
        max_xy = float(max(local_size[0], local_size[1]))
        support_tokens = self._support_object_tokens(support_object_id)
        geometry_is_concave = (
            max_xy > 0.0 and float(local_size[2]) / max_xy >= 0.3
        )
        if support_tokens & _DEEP_RECEPTACLE_SUPPORT_TOKENS:
            placement_family = "deep_receptacle"
        elif support_tokens & _SHALLOW_RECEPTACLE_SUPPORT_TOKENS:
            placement_family = "shallow_receptacle"
        elif geometry_is_concave:
            placement_family = "deep_receptacle"
        else:
            placement_family = "surface"
        return {
            "world_center": world_center,
            "center_xy": np.asarray(world_center[:2], dtype=float),
            "top_z": float(np.max(world_points[:, 2])),
            "bottom_z": float(np.min(world_points[:, 2])),
            "extent_x": float(local_size[0]),
            "extent_y": float(local_size[1]),
            "extent_z": float(local_size[2]),
            "max_xy": max_xy,
            "is_concave": geometry_is_concave,
            "support_tokens": support_tokens,
            "placement_family": placement_family,
            "uses_interior_floor": placement_family != "surface",
            "rot_xy": np.asarray(support_rot[:2, :2], dtype=float),
        }

    def _support_object_anchor_axes_xy(
        self,
        support_object_id: str,
        *,
        support_geometry: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (lateral_xy, inward_xy) axes for support-object-relative placement.

        For tray-like supports, prefer the anchor fixture frame so left/right is
        scene-consistent (counter-local), not bbox-local.
        """
        if support_geometry is None:
            support_geometry = self._support_object_geometry(support_object_id)
        default_lateral = np.asarray(support_geometry["rot_xy"], dtype=float)[:, 0]
        default_lateral = default_lateral / max(float(np.linalg.norm(default_lateral)), 1e-9)
        default_inward = np.asarray(support_geometry["rot_xy"], dtype=float)[:, 1]
        default_inward = default_inward / max(float(np.linalg.norm(default_inward)), 1e-9)

        try:
            anchor_fixture_id = self._resolve_object_anchor_fixture(
                support_object_id,
            )
        except Exception:
            anchor_fixture_id = None
        if not isinstance(anchor_fixture_id, str):
            return default_lateral, default_inward
        try:
            # Use fixture local X explicitly as lateral (left-right) for
            # support-object pair layout.
            lateral_world = self._fixture_world_axis(anchor_fixture_id, "x")
            lateral_xy = np.asarray(lateral_world[:2], dtype=float)
            lateral_norm = float(np.linalg.norm(lateral_xy))
            if lateral_norm <= 1e-9:
                return default_lateral, default_inward
            lateral_xy = lateral_xy / lateral_norm

            inward_world = self._fixture_inward_world_axis(anchor_fixture_id, "y")
            inward_xy = np.asarray(inward_world[:2], dtype=float)
            inward_norm = float(np.linalg.norm(inward_xy))
            if inward_norm <= 1e-9:
                inward_xy = np.array([-lateral_xy[1], lateral_xy[0]], dtype=float)
            else:
                inward_xy = inward_xy / inward_norm
                # Re-orthogonalize so depth checks are measured in the same frame.
                inward_xy = inward_xy - lateral_xy * float(np.dot(inward_xy, lateral_xy))
                inward_norm2 = float(np.linalg.norm(inward_xy))
                if inward_norm2 <= 1e-9:
                    inward_xy = np.array([-lateral_xy[1], lateral_xy[0]], dtype=float)
                else:
                    inward_xy = inward_xy / inward_norm2
            return lateral_xy, inward_xy
        except Exception:
            return default_lateral, default_inward

    def _support_object_placement_family(
        self,
        support_object_id: str,
        support_geometry: dict[str, Any] | None = None,
    ) -> str:
        if support_geometry is None:
            support_geometry = self._support_object_geometry(support_object_id)
        family = support_geometry.get("placement_family")
        if family in {"deep_receptacle", "shallow_receptacle", "surface"}:
            return str(family)
        support_tokens = support_geometry.get("support_tokens")
        if not isinstance(support_tokens, set):
            support_tokens = self._support_object_tokens(support_object_id)
        if support_tokens & _DEEP_RECEPTACLE_SUPPORT_TOKENS:
            return "deep_receptacle"
        if support_tokens & _SHALLOW_RECEPTACLE_SUPPORT_TOKENS:
            return "shallow_receptacle"
        if support_geometry.get("is_concave"):
            return "deep_receptacle"
        return "surface"

    def _support_object_uses_interior_floor(
        self,
        support_object_id: str,
        support_geometry: dict[str, Any] | None = None,
    ) -> bool:
        return (
            self._support_object_placement_family(
                support_object_id,
                support_geometry,
            )
            == "deep_receptacle"
        )

    def _support_object_xy_scale(
        self,
        support_object_id: str,
        support_geometry: dict[str, Any] | None = None,
    ) -> float:
        family = self._support_object_placement_family(
            support_object_id,
            support_geometry,
        )
        if family == "deep_receptacle":
            return 0.22
        if family == "shallow_receptacle":
            return 0.46
        return 0.48

    def _fixture_major_axis(
        self,
        fixture_id: str,
        *,
        site_id: str | None = None,
    ) -> str | None:
        bounds = self._fixture_local_bounds(fixture_id, site_id=site_id)
        if bounds is None:
            return None
        local_min, local_max = bounds
        size = np.asarray(local_max, dtype=float) - np.asarray(local_min, dtype=float)
        return "x" if float(size[0]) >= float(size[1]) else "y"

    def _fixture_world_axis(self, fixture_id: str, axis: str) -> np.ndarray:
        fixture = self._require_fixture(fixture_id)
        angle = float(getattr(fixture, "rot", 0.0) or 0.0)
        if axis == "x":
            return np.array([np.cos(angle), np.sin(angle), 0.0], dtype=float)
        if axis == "y":
            return np.array([-np.sin(angle), np.cos(angle), 0.0], dtype=float)
        raise ValueError(f"Unsupported fixture axis {axis!r}")

    def _fixture_front_face(self, fixture_id: str) -> str | None:
        fixture = self._require_fixture(fixture_id)
        front_target_xy = None
        get_front_target = getattr(self.runner, "_get_fixture_front_target_xy", None)
        if callable(get_front_target):
            front_target_xy = get_front_target(fixture_id)
        try:
            return get_face_order(fixture, front_target_xy=front_target_xy)[0]
        except Exception:
            return None

    def _fixture_inward_world_axis(self, fixture_id: str, axis: str) -> np.ndarray:
        front_face = self._fixture_front_face(fixture_id)
        if axis == "y":
            if front_face == "neg_y":
                return self._fixture_world_axis(fixture_id, "y")
            if front_face == "pos_y":
                return -self._fixture_world_axis(fixture_id, "y")
        if axis == "x":
            if front_face == "neg_x":
                return self._fixture_world_axis(fixture_id, "x")
            if front_face == "pos_x":
                return -self._fixture_world_axis(fixture_id, "x")
        return -self._fixture_world_axis(fixture_id, axis)

    def _support_object_radial_scales(
        self,
        support_object_id: str,
        *,
        existing_on_support: Sequence[str],
        support_geometry: dict[str, Any] | None = None,
    ) -> tuple[float, ...]:
        support_tokens = (
            support_geometry.get("support_tokens")
            if isinstance(support_geometry, dict)
            else None
        )
        if not isinstance(support_tokens, set):
            support_tokens = self._support_object_tokens(support_object_id)
        family = self._support_object_placement_family(
            support_object_id,
            support_geometry,
        )
        if family == "deep_receptacle":
            return (0.0, 0.16, 0.3) if existing_on_support else (0.0, 0.12, 0.24)
        if family == "shallow_receptacle":
            if "tray" in support_tokens:
                return (0.34, 0.6, 0.82) if existing_on_support else (0.0, 0.42, 0.7)
            return (0.18, 0.36, 0.54) if existing_on_support else (0.0, 0.22, 0.4)
        return (0.28, 0.55, 0.8) if existing_on_support else (0.0, 0.35, 0.65)

    def _support_object_slot_positions(
        self,
        support_object_id: str,
        *,
        count: int,
    ) -> list[np.ndarray]:
        geometry = self._support_object_geometry(support_object_id)
        family = self._support_object_placement_family(
            support_object_id,
            geometry,
        )
        support_tokens = geometry.get("support_tokens") or set()
        axis_is_x = geometry["extent_x"] >= geometry["extent_y"]
        major = max(geometry["extent_x"], geometry["extent_y"])
        minor = max(min(geometry["extent_x"], geometry["extent_y"]), 1e-6)
        elongated = (major / minor) >= 1.35
        if "tray" in support_tokens and count == 2:
            # Keep bulky pairs separated without pushing either item onto the
            # tray rim.  The collision check below handles the exact footprint.
            primary_half = 0.22 * major
            if axis_is_x:
                slot_offsets = [
                    np.array([-primary_half, 0.0], dtype=float),
                    np.array([primary_half, 0.0], dtype=float),
                ]
            else:
                slot_offsets = [
                    np.array([0.0, -primary_half], dtype=float),
                    np.array([0.0, primary_half], dtype=float),
                ]
            center_xy = geometry["center_xy"]
            rot_xy = geometry["rot_xy"]
            return [
                center_xy + rot_xy @ np.asarray(offset, dtype=float)
                for offset in slot_offsets
            ]
        if family == "deep_receptacle":
            primary_half = 0.32 * major
            secondary_half = 0.18 * minor
        elif family == "shallow_receptacle":
            if "tray" in support_tokens:
                primary_half = 0.52 * major
                secondary_half = (0.2 if elongated else 0.26) * minor
            else:
                primary_half = 0.46 * major
                secondary_half = (0.16 if elongated else 0.22) * minor
        elif elongated:
            primary_half = 0.5 * major
            secondary_half = 0.22 * minor
        elif geometry["is_concave"]:
            primary_half = 0.42 * major
            secondary_half = 0.25 * minor
        else:
            primary_half = 0.5 * major
            secondary_half = 0.34 * minor

        slot_offsets = self._slot_offsets_from_half_extents(
            count=count,
            primary_half=primary_half,
            secondary_half=secondary_half,
            axis_is_x=axis_is_x,
            elongated=elongated,
        )
        center_xy = geometry["center_xy"]
        rot_xy = geometry["rot_xy"]
        return [
            center_xy + rot_xy @ np.asarray(offset, dtype=float)
            for offset in slot_offsets
        ]

    def _fixture_site_slot_positions(
        self,
        fixture_id: str,
        site_id: str,
        *,
        count: int,
    ) -> list[np.ndarray]:
        fixture = self._require_fixture(fixture_id)
        regions = self.runner._get_fixture_reset_regions(
            fixture,
            min_size=None,
            site_id=site_id,
        )
        if not isinstance(regions, (list, tuple)) or not regions:
            return []

        region = max(
            regions,
            key=lambda item: float(
                np.prod(np.asarray(item.get("size", (0.1, 0.1)), dtype=float)[:2])
            ),
        )
        offset = np.asarray(region.get("offset", (0.0, 0.0, 0.0)), dtype=float)
        size = np.asarray(region.get("size", (0.1, 0.1)), dtype=float)[:2]
        axis_is_x = float(size[0]) >= float(size[1])
        major = max(float(size[0]), float(size[1]))
        minor = max(min(float(size[0]), float(size[1])), 1e-6)
        elongated = (major / minor) >= 1.35
        primary_half = 0.34 * major
        secondary_half = (0.12 if elongated else 0.2) * minor
        slot_offsets = self._slot_offsets_from_half_extents(
            count=count,
            primary_half=primary_half,
            secondary_half=secondary_half,
            axis_is_x=axis_is_x,
            elongated=elongated,
        )
        fixture_type = (self._get_fixture_type_name(fixture_id) or "").lower()
        front_bias = (
            min(0.18 * float(size[1]), 0.03)
            if fixture_type in _FRONT_BIASED_INTERIOR_FIXTURE_TYPES
            else 0.0
        )
        slot_positions: list[np.ndarray] = []
        for local_offset in slot_offsets:
            local = offset.copy()
            local[:2] += np.asarray(local_offset, dtype=float)
            if front_bias > 0.0:
                local[1] -= front_bias
            world = self.runner._fixture_local_to_world(fixture, local)
            slot_positions.append(np.asarray(world[:2], dtype=float))
        return slot_positions

    def _incoming_support_slot_preference(
        self,
        support_object_id: str,
        incoming_object_id: str,
    ) -> np.ndarray | None:
        existing_object_ids = [
            child_id
            for child_id in self._iter_direct_supported_children(support_object_id)
            if child_id != incoming_object_id
        ]
        if not existing_object_ids:
            existing_object_ids = self._find_objects_on_support(
                support_object_id,
                exclude={incoming_object_id},
            )
        support_tokens = self._support_object_tokens(support_object_id)
        incoming_tokens = self._scene_object_tokens(incoming_object_id)
        hotdog_pair_slot = self._portion_hotdogs_plate_slot_preference(
            support_object_id,
            incoming_object_id,
            existing_object_ids=existing_object_ids,
        )
        if hotdog_pair_slot is not None:
            return hotdog_pair_slot
        if (
            not existing_object_ids
            and "tray" in support_tokens
            and incoming_tokens & _TRAY_BULKY_PAIR_OBJECT_TOKENS
        ):
            slot_positions = self._support_object_slot_positions(
                support_object_id,
                count=2,
            )
            if slot_positions:
                return np.asarray(slot_positions[0], dtype=float)[:2]
        if "tray" in support_tokens and not existing_object_ids:
            if incoming_tokens & {"kettle"}:
                # Deterministic anchor for tea setup: kettle goes to tray-right.
                geometry = self._support_object_geometry(support_object_id)
                lateral_xy, _inward_xy = self._support_object_anchor_axes_xy(
                    support_object_id,
                    support_geometry=geometry,
                )
                side_half = 0.58 * max(float(geometry["extent_x"]), 1e-6)
                return geometry["center_xy"] + lateral_xy * side_half
        if "tray" in support_tokens and len(existing_object_ids) == 1:
            existing_object_id = existing_object_ids[0]
            existing_tokens = self._scene_object_tokens(existing_object_id)
            pair_tokens = (
                existing_tokens | incoming_tokens
            ) & _TRAY_KETTLE_MUG_PAIR_OBJECT_TOKENS
            if len(pair_tokens) == 2:
                # Keep kettle+mug pairs farther apart on trays. The kettle handle
                # often occupies usable area near the center even when simple
                # collision checks pass.
                geometry = self._support_object_geometry(support_object_id)
                lateral_xy, _inward_xy = self._support_object_anchor_axes_xy(
                    support_object_id,
                    support_geometry=geometry,
                )
                # Deterministic directional assignment:
                # kettle on tray-right (local +X), mug on tray-left (local -X).
                side_half = 0.58 * max(float(geometry["extent_x"]), 1e-6)
                desired_local_x = side_half
                if incoming_tokens & {"mug"}:
                    desired_local_x = -side_half
                elif incoming_tokens & {"kettle"}:
                    desired_local_x = side_half
                else:
                    # Fallback for naming drift: place opposite side of existing object.
                    existing_pos = self._get_object_pose(existing_object_id)[0][:2].copy()
                    existing_delta = np.asarray(existing_pos, dtype=float) - geometry["center_xy"]
                    existing_side = float(np.dot(existing_delta, lateral_xy))
                    desired_local_x = -side_half if existing_side >= 0.0 else side_half
                return geometry["center_xy"] + lateral_xy * desired_local_x
        slot_positions = self._support_object_slot_positions(
            support_object_id,
            count=len(existing_object_ids) + 1,
        )
        existing_positions = [
            self._get_object_pose(object_id)[0][:2].copy()
            for object_id in existing_object_ids
        ]
        geometry = self._support_object_geometry(support_object_id)
        return self._choose_slot_position(
            slot_positions,
            existing_positions=existing_positions,
            center_xy=geometry["center_xy"],
        )

    def _is_portion_hotdogs_plate_pair(
        self,
        support_object_id: str,
        object_ids: Sequence[str],
    ) -> bool:
        if str(getattr(self, "_task_name", "") or "").strip().lower() != "portionhotdogs":
            return False
        support_tokens = self._support_object_tokens(support_object_id)
        if "plate" not in support_tokens:
            return False
        if not object_ids:
            return False
        pair_tokens = {"bun", "hotdog", "sausage"}
        for object_id in object_ids:
            object_tokens = self._scene_object_tokens(object_id)
            if not (object_tokens & pair_tokens):
                return False
        return True

    def _portion_hotdogs_plate_slot_preference(
        self,
        support_object_id: str,
        incoming_object_id: str,
        *,
        existing_object_ids: Sequence[str],
    ) -> np.ndarray | None:
        relevant_object_ids = list(existing_object_ids) + [incoming_object_id]
        if not self._is_portion_hotdogs_plate_pair(
            support_object_id,
            relevant_object_ids,
        ):
            return None

        geometry = self._support_object_geometry(support_object_id)
        lateral_xy, inward_xy = self._support_object_anchor_axes_xy(
            support_object_id,
            support_geometry=geometry,
        )
        major = max(float(geometry["extent_x"]), float(geometry["extent_y"]), 1e-6)
        minor = max(min(float(geometry["extent_x"]), float(geometry["extent_y"])), 1e-6)
        side_half = 0.16 * major
        depth_bias = 0.05 * minor
        center_xy = np.asarray(geometry["center_xy"], dtype=float)

        slot_positions = [
            center_xy - np.asarray(lateral_xy, dtype=float) * side_half
            - np.asarray(inward_xy, dtype=float) * depth_bias,
            center_xy + np.asarray(lateral_xy, dtype=float) * side_half
            + np.asarray(inward_xy, dtype=float) * depth_bias,
        ]
        existing_positions = [
            self._get_object_pose(object_id)[0][:2].copy()
            for object_id in existing_object_ids
        ]
        return self._choose_slot_position(
            slot_positions,
            existing_positions=existing_positions,
            center_xy=center_xy,
        )

    def _incoming_fixture_site_preference(
        self,
        fixture_id: str,
        site_id: str,
        *,
        incoming_object_id: str,
    ) -> np.ndarray | None:
        existing_object_ids = self._objects_on_fixture_site(
            fixture_id,
            site_id=site_id,
            exclude={incoming_object_id},
        )
        slot_positions = self._fixture_site_slot_positions(
            fixture_id,
            site_id,
            count=len(existing_object_ids) + 1,
        )
        if not slot_positions:
            return None
        existing_positions = [
            self._get_object_pose(object_id)[0][:2].copy()
            for object_id in existing_object_ids
        ]
        center_xy = self._preferred_xy_for_fixture_target(
            fixture_id,
            site_id=site_id,
        )
        if center_xy is None:
            center_xy = slot_positions[0]
        return self._choose_slot_position(
            slot_positions,
            existing_positions=existing_positions,
            center_xy=np.asarray(center_xy, dtype=float),
        )

    def _support_object_prefers_upright(
        self,
        support_object_id: str,
        object_id: str,
    ) -> bool:
        support_tokens = self._support_object_tokens(support_object_id)
        if not (support_tokens & _VERTICAL_INSERT_SUPPORT_TOKENS):
            return False
        object_tokens = self._scene_object_tokens(object_id)
        if object_tokens & _VERTICAL_INSERT_OBJECT_TOKENS:
            return True
        metadata = self._get_object_placement_metadata(object_id)
        spans = np.sort(np.asarray(metadata["size"], dtype=float))
        if spans.shape[0] != 3:
            return False
        return float(spans[2]) >= 1.35 * max(float(spans[1]), 1e-6)

    def _support_object_floor_clearance(
        self,
        support_object_id: str,
        support_geometry: dict[str, Any],
    ) -> float:
        support_tokens = self._support_object_tokens(support_object_id)
        extent_z = float(support_geometry["extent_z"])
        if support_tokens & frozenset(
            {"bowl", "plate", "tray", "pan", "pot", "cup", "mug", "glass"}
        ):
            return min(max(0.04 * extent_z, 0.001), 0.006)
        return min(max(0.08 * extent_z, 0.002), 0.01)

    def _support_object_should_keep_pose(self, support_object_id: str) -> bool:
        return bool(
            self._support_object_tokens(support_object_id)
            & _POSE_STABLE_SUPPORT_TOKENS
        )

    def _support_object_accepts_child_without_settle(
        self,
        support_object_id: str,
    ) -> bool:
        if (
            str(getattr(self, "_task_name", "") or "").strip().lower()
            == "portionhotdogs"
            and "plate" in self._support_object_tokens(support_object_id)
        ):
            return True
        return bool(
            self._support_object_tokens(support_object_id)
            & _NO_SETTLE_CHILD_SUPPORT_TOKENS
        )

    def _snapshot_stable_support_poses(
        self,
    ) -> dict[str, tuple[np.ndarray, np.ndarray, list[str]]]:
        snapshots: dict[str, tuple[np.ndarray, np.ndarray, list[str]]] = {}
        for support_object_id in getattr(self.env, "objects", {}):
            if not self._support_object_should_keep_pose(support_object_id):
                continue
            child_ids = self._iter_direct_supported_children(support_object_id)
            if not child_ids:
                continue
            try:
                pos, quat = self._get_object_pose(support_object_id)
            except Exception:
                continue
            snapshots[support_object_id] = (pos.copy(), quat.copy(), list(child_ids))
        return snapshots

    def _restore_stable_support_poses(
        self,
        snapshots: dict[str, tuple[np.ndarray, np.ndarray, list[str]]],
    ) -> None:
        for support_object_id, (support_pos, support_quat, child_ids) in snapshots.items():
            if support_object_id not in self.env.objects:
                continue
            self._set_object_pose(support_object_id, support_pos, support_quat)
            for child_id in child_ids:
                if child_id not in self.env.objects or self._held_by_robot(child_id) is not None:
                    continue
                self._set_support_parent(child_id, support_object_id)
                self.runner._set_object_location(child_id, support_object_id)
                self._place_on_object_center(child_id, support_object_id)

    def _support_object_contact_plane_z(
        self,
        support_object_id: str,
        support_geometry: dict[str, Any],
    ) -> float:
        family = self._support_object_placement_family(
            support_object_id,
            support_geometry,
        )
        extent_z = float(support_geometry["extent_z"])
        bottom_z = float(support_geometry["bottom_z"])
        top_z = float(support_geometry["top_z"])
        if family == "deep_receptacle":
            base_height = min(max(0.14 * extent_z, 0.008), 0.02)
            return (
                bottom_z
                + base_height
                + self._support_object_floor_clearance(
                    support_object_id,
                    support_geometry,
                )
            )
        if family == "shallow_receptacle":
            rim_inset = min(max(0.22 * extent_z, 0.002), 0.012)
            return max(bottom_z, top_z - rim_inset)
        return top_z

    def _settle_and_reseat_supported_object(
        self,
        object_id: str,
        support_object_id: str,
        *,
        relative_position: str | None = None,
        steps: int | None = None,
    ) -> None:
        settle_steps = (
            max(_PLACEMENT_SETTLE_STEPS * 2, 12)
            if steps is None
            else max(int(steps), 1)
        )
        support_geometry: dict[str, Any] | None = None
        prefers_upright = self._support_object_prefers_upright(
            support_object_id,
            object_id,
        )

        def _as_explicit_bool(value: Any) -> bool:
            return bool(value) if isinstance(value, (bool, np.bool_)) else False

        self._set_support_parent(object_id, support_object_id)
        self.runner._set_object_location(object_id, support_object_id)
        if self._support_object_accepts_child_without_settle(support_object_id):
            self._place_on_object_center(
                object_id,
                support_object_id,
                relative_position=relative_position,
            )
            child_ids = self._iter_direct_supported_children(support_object_id)
            if self._supported_children_should_balance_slots(support_object_id, child_ids):
                self._repack_supported_children(support_object_id, child_ids)
            return

        self._settle_scene(steps=settle_steps)
        settled_pos, _settled_quat = self._get_object_pose(object_id)
        self._place_on_object_center(
            object_id,
            support_object_id,
            relative_position=relative_position,
        )
        reseated_pos, reseated_quat = self._get_object_pose(object_id)
        if float(reseated_pos[2]) > float(settled_pos[2]) + 1e-6:
            clamped_pos = reseated_pos.copy()
            clamped_pos[2] = float(settled_pos[2])
            self._set_object_pose(object_id, clamped_pos, reseated_quat)
        self._set_support_parent(object_id, support_object_id)
        self.runner._set_object_location(object_id, support_object_id)
        self._settle_scene(steps=settle_steps)
        if prefers_upright:
            final_pos, final_quat = self._get_object_pose(object_id)
            if not self._object_is_upright(object_id, final_quat):
                self._place_on_object_center(
                    object_id,
                    support_object_id,
                    relative_position=relative_position,
                )
                self._set_support_parent(object_id, support_object_id)
                self.runner._set_object_location(object_id, support_object_id)
        else:
            support_geometry = self._support_object_geometry(support_object_id)
            uses_interior_floor = _as_explicit_bool(
                self._support_object_uses_interior_floor(
                    support_object_id,
                    support_geometry,
                )
            )
            if not uses_interior_floor:
                for _ in range(3):
                    gap = self._supported_object_vertical_gap(
                        object_id,
                        support_object_id,
                        support_geometry=support_geometry,
                    )
                    if gap <= 0.006:
                        break
                    final_pos, final_quat = self._get_object_pose(object_id)
                    adjusted_pos = np.asarray(final_pos, dtype=float).copy()
                    adjusted_pos[2] -= gap
                    self._set_object_pose(object_id, adjusted_pos, final_quat)
                    self._set_support_parent(object_id, support_object_id)
                    self.runner._set_object_location(object_id, support_object_id)
                    self._settle_scene(steps=max(_PLACEMENT_SETTLE_STEPS, 4))
            else:
                self._repair_supported_children_vertical_gaps(
                    support_object_id,
                    [object_id],
                    support_geometry=support_geometry,
                )
        child_ids = self._iter_direct_supported_children(support_object_id)
        if self._supported_children_should_balance_slots(support_object_id, child_ids):
            if support_geometry is None:
                support_geometry = self._support_object_geometry(support_object_id)
            self._repack_supported_children(support_object_id, child_ids)
            self._settle_scene(steps=max(_PLACEMENT_SETTLE_STEPS, 4))
            self._repair_supported_children_vertical_gaps(
                support_object_id,
                child_ids,
                support_geometry=support_geometry,
            )
        if self._supported_children_need_repack(support_object_id, child_ids):
            if support_geometry is None:
                support_geometry = self._support_object_geometry(support_object_id)
            # Prefer to move only the newly-placed child: existing children
            # that the user/scene already arranged correctly should not jump
            # around just because a sibling was added.  Only if re-placing
            # the new child fails to resolve collisions do we fall back to
            # repacking everything.
            self._place_on_object_center(
                object_id,
                support_object_id,
                relative_position=relative_position,
            )
            self._set_support_parent(object_id, support_object_id)
            self.runner._set_object_location(object_id, support_object_id)
            self._settle_scene(steps=max(_PLACEMENT_SETTLE_STEPS, 4))
            if self._supported_children_need_repack(support_object_id, child_ids):
                self._repack_supported_children(support_object_id, child_ids)
                self._settle_scene(steps=max(_PLACEMENT_SETTLE_STEPS, 4))
            self._repair_supported_children_vertical_gaps(
                support_object_id,
                child_ids,
                support_geometry=support_geometry,
            )
        final_child_ids = self._iter_direct_supported_children(support_object_id)
        if not final_child_ids:
            return
        if support_geometry is None:
            support_geometry = self._support_object_geometry(support_object_id)
        uses_interior_floor = _as_explicit_bool(
            self._support_object_uses_interior_floor(
                support_object_id,
                support_geometry,
            )
        )
        if uses_interior_floor:
            if final_child_ids:
                for _ in range(2):
                    self._settle_scene(steps=max(_PLACEMENT_SETTLE_STEPS, 4))
                    self._repair_supported_children_vertical_gaps(
                        support_object_id,
                        final_child_ids,
                        support_geometry=support_geometry,
                        positive_tolerance=0.003,
                    )

    @staticmethod
    def _quat_align_vectors(source: np.ndarray, target: np.ndarray) -> np.ndarray:
        source_vec = np.asarray(source, dtype=float)
        target_vec = np.asarray(target, dtype=float)
        source_norm = float(np.linalg.norm(source_vec))
        target_norm = float(np.linalg.norm(target_vec))
        if source_norm <= 1e-9 or target_norm <= 1e-9:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        source_unit = source_vec / source_norm
        target_unit = target_vec / target_norm
        dot = float(np.clip(np.dot(source_unit, target_unit), -1.0, 1.0))
        if dot >= 1.0 - 1e-6:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        if dot <= -1.0 + 1e-6:
            fallback_axes = (
                np.array([1.0, 0.0, 0.0], dtype=float),
                np.array([0.0, 1.0, 0.0], dtype=float),
                np.array([0.0, 0.0, 1.0], dtype=float),
            )
            for axis in fallback_axes:
                orthogonal = np.cross(source_unit, axis)
                if float(np.linalg.norm(orthogonal)) > 1e-6:
                    return T.axisangle2quat(
                        orthogonal / np.linalg.norm(orthogonal) * np.pi
                    )
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        axis = np.cross(source_unit, target_unit)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm <= 1e-9:
            return np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
        angle = float(np.arccos(dot))
        return T.axisangle2quat(axis / axis_norm * angle)

    def _dominant_object_local_axis(self, object_id: str) -> np.ndarray:
        obj = self._require_object(object_id)
        local_bbox_points = np.asarray(
            obj.get_bbox_points(
                trans=np.zeros(3, dtype=float),
                rot=np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
            ),
            dtype=float,
        )
        local_spans = np.max(local_bbox_points, axis=0) - np.min(local_bbox_points, axis=0)
        return np.eye(3, dtype=float)[int(np.argmax(local_spans))]

    def _thinnest_object_local_axis(self, object_id: str) -> np.ndarray:
        obj = self._require_object(object_id)
        local_bbox_points = np.asarray(
            obj.get_bbox_points(
                trans=np.zeros(3, dtype=float),
                rot=np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
            ),
            dtype=float,
        )
        local_spans = np.max(local_bbox_points, axis=0) - np.min(local_bbox_points, axis=0)
        return np.eye(3, dtype=float)[int(np.argmin(local_spans))]

    def _flat_alignment_up_axis(self, object_id: str) -> np.ndarray:
        up_axis = self._thinnest_object_local_axis(object_id)
        object_tokens = self._scene_object_tokens(object_id)
        if object_tokens & _FLAT_UTENSIL_OBJECT_TOKENS:
            return -up_axis
        return up_axis

    def _flat_alignment_long_axis(self, object_id: str) -> np.ndarray:
        return self._dominant_object_local_axis(object_id)

    def _object_prefers_upright_on_surface(self, object_id: str) -> bool:
        object_tokens = self._scene_object_tokens(object_id)
        if object_tokens & _SURFACE_FLAT_OBJECT_TOKENS:
            return False
        if object_tokens & _SURFACE_UPRIGHT_OBJECT_TOKENS:
            return True
        metadata = self._get_object_placement_metadata(object_id)
        spans = np.sort(np.asarray(metadata["size"], dtype=float))
        if spans.shape[0] != 3:
            return False
        return float(spans[2]) >= 1.5 * max(float(spans[1]), 1e-6) and float(
            spans[1]
        ) <= 1.25 * max(float(spans[0]), 1e-6)

    def _object_has_directional_long_axis(self, object_id: str) -> bool:
        object_tokens = self._scene_object_tokens(object_id)
        if object_tokens & _DIRECTIONAL_SURFACE_OBJECT_TOKENS:
            return True
        metadata = self._get_object_placement_metadata(object_id)
        spans = np.sort(np.asarray(metadata["size"], dtype=float))
        if spans.shape[0] != 3:
            return False
        return float(spans[2]) >= 1.15 * max(float(spans[1]), 1e-6)

    def _aligned_flat_quat_for_world_axis(
        self,
        object_id: str,
        target_axis_world: np.ndarray,
        *,
        preserve_direction: bool = False,
        allow_target_sign_flip: bool = True,
    ) -> np.ndarray | None:
        if self._object_prefers_upright_on_surface(object_id):
            return None
        if not self._object_has_directional_long_axis(object_id):
            return None

        target_axis = np.asarray(target_axis_world, dtype=float)
        if float(np.linalg.norm(target_axis[:2])) <= 1e-9:
            return None
        target_axis_xy = target_axis[:2] / max(float(np.linalg.norm(target_axis[:2])), 1e-9)
        original_quat_wxyz = self._get_object_pose(object_id)[1]
        base_quat_xyzw = T.convert_quat(np.asarray(original_quat_wxyz, dtype=float), to="xyzw")
        base_rot = T.quat2mat(base_quat_xyzw)

        thin_local_axis = self._flat_alignment_up_axis(object_id)
        world_thin_axis = base_rot @ thin_local_axis
        align_up_quat_xyzw = self._quat_align_vectors(
            world_thin_axis,
            np.array([0.0, 0.0, 1.0], dtype=float),
        )
        flat_quat_xyzw = T.quat_multiply(align_up_quat_xyzw, base_quat_xyzw)

        dominant_local_axis = self._flat_alignment_long_axis(object_id)
        flat_world_axis = T.quat2mat(flat_quat_xyzw) @ dominant_local_axis
        flat_world_xy = np.asarray(flat_world_axis[:2], dtype=float)
        if float(np.linalg.norm(flat_world_xy)) <= 1e-9:
            return T.convert_quat(flat_quat_xyzw, to="wxyz")
        flat_world_xy = flat_world_xy / float(np.linalg.norm(flat_world_xy))

        if (
            allow_target_sign_flip
            and not preserve_direction
            and float(np.dot(flat_world_xy, target_axis_xy)) < 0.0
        ):
            target_axis_xy = -target_axis_xy
        cross_z = float(
            flat_world_xy[0] * target_axis_xy[1] - flat_world_xy[1] * target_axis_xy[0]
        )
        dot = float(np.clip(np.dot(flat_world_xy, target_axis_xy), -1.0, 1.0))
        yaw = float(np.arctan2(cross_z, dot))
        yaw_quat_xyzw = T.axisangle2quat(np.array([0.0, 0.0, yaw], dtype=float))
        aligned_quat_xyzw = T.quat_multiply(yaw_quat_xyzw, flat_quat_xyzw)
        return T.convert_quat(aligned_quat_xyzw, to="wxyz")

    def _aligned_upright_quat_for_world_axis(
        self,
        object_id: str,
        target_axis_world: np.ndarray,
        *,
        preserve_direction: bool = False,
        allow_target_sign_flip: bool = True,
    ) -> np.ndarray | None:
        if not self._object_prefers_upright_on_surface(object_id):
            return None
        target_axis = np.asarray(target_axis_world, dtype=float)
        if float(np.linalg.norm(target_axis[:2])) <= 1e-9:
            return None
        target_axis_xy = target_axis[:2] / max(float(np.linalg.norm(target_axis[:2])), 1e-9)

        original_quat_wxyz = self._get_object_pose(object_id)[1]
        base_quat_xyzw = T.convert_quat(np.asarray(original_quat_wxyz, dtype=float), to="xyzw")
        base_rot = T.quat2mat(base_quat_xyzw)

        obj = self._require_object(object_id)
        local_bbox_points = np.asarray(
            obj.get_bbox_points(
                trans=np.zeros(3, dtype=float),
                rot=np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
            ),
            dtype=float,
        )
        spans = np.max(local_bbox_points, axis=0) - np.min(local_bbox_points, axis=0)
        axis_idx = 0 if float(spans[0]) >= float(spans[1]) else 1
        local_axis = np.zeros(3, dtype=float)
        local_axis[axis_idx] = 1.0
        world_axis = base_rot @ local_axis
        world_axis_xy = np.asarray(world_axis[:2], dtype=float)
        if float(np.linalg.norm(world_axis_xy)) <= 1e-9:
            return None
        world_axis_xy = world_axis_xy / float(np.linalg.norm(world_axis_xy))

        if (
            allow_target_sign_flip
            and not preserve_direction
            and float(np.dot(world_axis_xy, target_axis_xy)) < 0.0
        ):
            target_axis_xy = -target_axis_xy
        cross_z = float(
            world_axis_xy[0] * target_axis_xy[1]
            - world_axis_xy[1] * target_axis_xy[0]
        )
        dot = float(np.clip(np.dot(world_axis_xy, target_axis_xy), -1.0, 1.0))
        yaw = float(np.arctan2(cross_z, dot))
        yaw_quat_xyzw = T.axisangle2quat(np.array([0.0, 0.0, yaw], dtype=float))
        aligned_quat_xyzw = T.quat_multiply(yaw_quat_xyzw, base_quat_xyzw)
        return T.convert_quat(aligned_quat_xyzw, to="wxyz")

    def _object_is_upright(
        self,
        object_id: str,
        quat_wxyz: np.ndarray,
        *,
        min_alignment: float = 0.8,
    ) -> bool:
        dominant_local_axis = self._dominant_object_local_axis(object_id)
        quat_xyzw = T.convert_quat(np.asarray(quat_wxyz, dtype=float), to="xyzw")
        world_axis = T.quat2mat(quat_xyzw) @ dominant_local_axis
        return abs(float(world_axis[2])) >= float(min_alignment)

    def _object_is_flat_on_surface(
        self,
        object_id: str,
        quat_wxyz: np.ndarray,
        *,
        min_alignment: float = 0.85,
    ) -> bool:
        thin_local_axis = self._flat_alignment_up_axis(object_id)
        quat_xyzw = T.convert_quat(np.asarray(quat_wxyz, dtype=float), to="xyzw")
        world_axis = T.quat2mat(quat_xyzw) @ thin_local_axis
        return abs(float(world_axis[2])) >= float(min_alignment)

    def _object_is_flat_top_up_on_surface(
        self,
        object_id: str,
        quat_wxyz: np.ndarray,
        *,
        min_alignment: float = 0.7,
    ) -> bool:
        thin_local_axis = self._flat_alignment_up_axis(object_id)
        quat_xyzw = T.convert_quat(np.asarray(quat_wxyz, dtype=float), to="xyzw")
        world_axis = T.quat2mat(quat_xyzw) @ thin_local_axis
        return float(world_axis[2]) >= float(min_alignment)

    def _object_requires_top_up_flat_on_support(
        self,
        object_id: str,
        support_object_id: str,
    ) -> bool:
        object_tokens = self._scene_object_tokens(object_id)
        if not (object_tokens & _TOP_UP_FLAT_ON_PLATE_OBJECT_TOKENS):
            return False
        support_tokens = self._support_object_tokens(support_object_id)
        return bool(support_tokens & {"plate"})

    def _object_long_axis_matches_surface(
        self,
        object_id: str,
        quat_wxyz: np.ndarray,
        target_axis_world: np.ndarray,
        *,
        min_alignment: float = 0.7,
    ) -> bool:
        dominant_local_axis = self._flat_alignment_long_axis(object_id)
        quat_xyzw = T.convert_quat(np.asarray(quat_wxyz, dtype=float), to="xyzw")
        world_axis = T.quat2mat(quat_xyzw) @ dominant_local_axis
        world_xy = np.asarray(world_axis[:2], dtype=float)
        target_xy = np.asarray(target_axis_world[:2], dtype=float)
        if float(np.linalg.norm(world_xy)) <= 1e-9 or float(np.linalg.norm(target_xy)) <= 1e-9:
            return False
        world_xy /= float(np.linalg.norm(world_xy))
        target_xy /= float(np.linalg.norm(target_xy))
        return abs(float(np.dot(world_xy, target_xy))) >= float(min_alignment)

    def _surface_pose_needs_reset(
        self,
        object_id: str,
        fixture_id: str,
        *,
        site_id: str | None = None,
        quat_wxyz: np.ndarray | None = None,
    ) -> bool:
        if self._fixture_placement_semantics(fixture_id) != "surface":
            return False
        if quat_wxyz is None:
            _pos, quat_wxyz = self._get_object_pose(object_id)
        quat = np.asarray(quat_wxyz, dtype=float)
        if self._object_prefers_upright_on_surface(object_id):
            return not self._object_is_upright(object_id, quat)
        if not self._object_has_directional_long_axis(object_id):
            return False
        if not self._object_is_flat_on_surface(object_id, quat):
            return True
        major_axis = self._fixture_major_axis(fixture_id, site_id=site_id)
        if major_axis not in {"x", "y"}:
            return False
        return not self._object_long_axis_matches_surface(
            object_id,
            quat,
            self._fixture_world_axis(fixture_id, major_axis),
        )

    def _fixture_pose_needs_reset(
        self,
        object_id: str,
        fixture_id: str,
        *,
        site_id: str | None = None,
        quat_wxyz: np.ndarray | None = None,
    ) -> bool:
        if quat_wxyz is None:
            _pos, quat_wxyz = self._get_object_pose(object_id)
        quat = np.asarray(quat_wxyz, dtype=float)
        if self._surface_pose_needs_reset(
            object_id,
            fixture_id,
            site_id=site_id,
            quat_wxyz=quat,
        ):
            return True
        if self._fixture_placement_semantics(fixture_id) != "receptacle":
            return False
        if not self._object_prefers_upright_on_surface(object_id):
            return False
        return not self._object_is_upright(object_id, quat)

    def _supported_object_vertical_gap(
        self,
        object_id: str,
        support_object_id: str,
        *,
        support_geometry: dict[str, Any] | None = None,
    ) -> float:
        if support_geometry is None:
            support_geometry = self._support_object_geometry(support_object_id)
        obj_pos, obj_quat_wxyz = self._get_object_pose(object_id)
        obj = self._require_object(object_id)
        obj_points = np.asarray(
            obj.get_bbox_points(
                trans=np.asarray(obj_pos, dtype=float),
                rot=T.convert_quat(np.asarray(obj_quat_wxyz, dtype=float), to="xyzw"),
            ),
            dtype=float,
        )
        object_bottom_z = float(np.min(obj_points[:, 2]))
        support_plane_z = self._support_object_contact_plane_z(
            support_object_id,
            support_geometry,
        )
        return object_bottom_z - float(support_plane_z)

    def _repair_supported_children_vertical_gaps(
        self,
        support_object_id: str,
        child_ids: Sequence[str],
        *,
        support_geometry: dict[str, Any] | None = None,
        negative_tolerance: float = -0.01,
        positive_tolerance: float = 0.008,
    ) -> None:
        if support_geometry is None:
            support_geometry = self._support_object_geometry(support_object_id)
        if not self._support_object_uses_interior_floor(
            support_object_id,
            support_geometry,
        ):
            return
        for child_id in child_ids:
            gap = self._supported_object_vertical_gap(
                child_id,
                support_object_id,
                support_geometry=support_geometry,
            )
            if gap <= positive_tolerance:
                continue
            child_pos, child_quat = self._get_object_pose(child_id)
            adjusted_pos = np.asarray(child_pos, dtype=float).copy()
            adjusted_pos[2] -= float(gap)
            self._set_object_pose(child_id, adjusted_pos, child_quat)
            self._set_support_parent(child_id, support_object_id)
            self.runner._set_object_location(child_id, support_object_id)

    def _support_object_candidate_quaternions(
        self,
        object_id: str,
        support_object_id: str,
    ) -> list[np.ndarray]:
        original_quat_wxyz = self._get_object_pose(object_id)[1]
        base_quat_xyzw = T.convert_quat(original_quat_wxyz, to="xyzw")
        candidates: list[tuple[float, float, np.ndarray]] = []
        seen: set[tuple[float, float, float, float]] = set()
        base_rot = T.quat2mat(base_quat_xyzw)
        upright_support = self._support_object_prefers_upright(
            support_object_id,
            object_id,
        )
        require_top_up_flat = self._object_requires_top_up_flat_on_support(
            object_id,
            support_object_id,
        )

        def _append_candidate(candidate_quat_xyzw: np.ndarray) -> None:
            candidate_quat_wxyz = T.convert_quat(candidate_quat_xyzw, to="wxyz")
            if (
                require_top_up_flat
                and not self._object_is_flat_top_up_on_surface(
                    object_id,
                    candidate_quat_wxyz,
                )
            ):
                return
            key = tuple(np.round(candidate_quat_wxyz, 5))
            if key in seen:
                return
            seen.add(key)
            bbox_metadata = self._object_bbox_metadata_for_quat(
                object_id,
                candidate_quat_wxyz,
            )
            spans = bbox_metadata["spans"]
            candidates.append(
                (
                    float(spans[2]),
                    float(max(spans[0], spans[1])),
                    candidate_quat_wxyz,
                )
            )

        if upright_support:
            alignment_axis = self._dominant_object_local_axis(object_id)
            aligned_quat_candidates_xyzw: list[np.ndarray] = []
            for sign in (1.0, -1.0):
                world_axis = base_rot @ (alignment_axis * sign)
                align_quat_xyzw = self._quat_align_vectors(
                    world_axis,
                    np.array([0.0, 0.0, 1.0], dtype=float),
                )
                aligned_quat_xyzw = T.quat_multiply(
                    align_quat_xyzw,
                    base_quat_xyzw,
                )
                for yaw in (0.0, 0.5 * np.pi, np.pi, -0.5 * np.pi):
                    if abs(yaw) <= 1e-9:
                        candidate_quat_xyzw = aligned_quat_xyzw
                    else:
                        yaw_quat_xyzw = T.axisangle2quat(
                            np.array([0.0, 0.0, yaw], dtype=float)
                        )
                        candidate_quat_xyzw = T.quat_multiply(
                            yaw_quat_xyzw,
                            aligned_quat_xyzw,
                        )
                    aligned_quat_candidates_xyzw.append(candidate_quat_xyzw)
            tilt_pairs = (
                (0.0, 0.0),
                (0.5 * np.pi, 0.0),
                (-0.5 * np.pi, 0.0),
                (0.0, 0.5 * np.pi),
                (0.0, -0.5 * np.pi),
                (0.5 * np.pi, 0.5 * np.pi),
                (0.5 * np.pi, -0.5 * np.pi),
                (-0.5 * np.pi, 0.5 * np.pi),
                (-0.5 * np.pi, -0.5 * np.pi),
            )
            for candidate_quat_xyzw in aligned_quat_candidates_xyzw:
                _append_candidate(candidate_quat_xyzw)
            for tilt_x, tilt_y in tilt_pairs:
                if abs(tilt_x) <= 1e-9 and abs(tilt_y) <= 1e-9:
                    _append_candidate(base_quat_xyzw)
                    continue
                candidate_quat_xyzw = base_quat_xyzw
                if abs(tilt_x) > 1e-9:
                    tilt_x_quat_xyzw = T.axisangle2quat(
                        np.array([tilt_x, 0.0, 0.0], dtype=float)
                    )
                    candidate_quat_xyzw = T.quat_multiply(
                        tilt_x_quat_xyzw,
                        candidate_quat_xyzw,
                    )
                if abs(tilt_y) > 1e-9:
                    tilt_y_quat_xyzw = T.axisangle2quat(
                        np.array([0.0, tilt_y, 0.0], dtype=float)
                    )
                    candidate_quat_xyzw = T.quat_multiply(
                        tilt_y_quat_xyzw,
                        candidate_quat_xyzw,
                    )
                _append_candidate(candidate_quat_xyzw)
        elif self._object_prefers_upright_on_surface(object_id):
            return [original_quat_wxyz]
        else:
            support_tokens = self._support_object_tokens(support_object_id)
            if (
                support_tokens & {"plate", "bowl", "tray"}
                and self._object_has_directional_long_axis(object_id)
            ):
                support_geometry = self._support_object_geometry(support_object_id)
                local_axis = (
                    np.array([1.0, 0.0], dtype=float)
                    if float(support_geometry["extent_x"]) >= float(support_geometry["extent_y"])
                    else np.array([0.0, 1.0], dtype=float)
                )
                world_axis_xy = np.asarray(support_geometry["rot_xy"], dtype=float) @ local_axis
                aligned_quat_wxyz = self._aligned_flat_quat_for_world_axis(
                    object_id,
                    np.array([world_axis_xy[0], world_axis_xy[1], 0.0], dtype=float),
                )
                if aligned_quat_wxyz is not None:
                    _append_candidate(
                        T.convert_quat(
                            np.asarray(aligned_quat_wxyz, dtype=float),
                            to="xyzw",
                        )
                    )
            alignment_axis = self._thinnest_object_local_axis(object_id)
            signs = (1.0,) if require_top_up_flat else (1.0, -1.0)
            for sign in signs:
                world_axis = base_rot @ (alignment_axis * sign)
                align_quat_xyzw = self._quat_align_vectors(
                    world_axis,
                    np.array([0.0, 0.0, 1.0], dtype=float),
                )
                aligned_quat_xyzw = T.quat_multiply(
                    align_quat_xyzw,
                    base_quat_xyzw,
                )
                for yaw in (0.0, 0.5 * np.pi, np.pi, -0.5 * np.pi):
                    if abs(yaw) <= 1e-9:
                        candidate_quat_xyzw = aligned_quat_xyzw
                    else:
                        yaw_quat_xyzw = T.axisangle2quat(
                            np.array([0.0, 0.0, yaw], dtype=float)
                        )
                        candidate_quat_xyzw = T.quat_multiply(
                            yaw_quat_xyzw,
                            aligned_quat_xyzw,
                        )
                    _append_candidate(candidate_quat_xyzw)
            if not require_top_up_flat:
                _append_candidate(base_quat_xyzw)

        if not candidates:
            return [original_quat_wxyz]
        if upright_support:
            max_vertical_span = max(candidate[0] for candidate in candidates)
            filtered_candidates = [
                candidate
                for candidate in candidates
                if candidate[0] >= 0.85 * max_vertical_span
            ]
            filtered_candidates.sort(key=lambda item: (-item[0], item[1]))
        else:
            filtered_candidates = sorted(candidates, key=lambda item: (item[0], item[1]))
        return [quat for _vertical_span, _horizontal_span, quat in filtered_candidates]

    def _set_object_pose(
        self,
        object_id: str,
        pos: np.ndarray | list[float],
        quat: np.ndarray | list[float] | None = None,
    ):
        obj = self._require_object(object_id)
        old_pos, current_quat = self._get_object_pose(object_id)
        contained = self._iter_direct_supported_children(object_id)
        quat_arr = current_quat if quat is None else np.asarray(quat, dtype=float)
        target_pos = np.asarray(pos, dtype=float)
        self.env.sim.data.set_joint_qpos(
            obj.joints[0],
            np.concatenate([target_pos, quat_arr]),
        )
        self.env.sim.forward()

        if contained:
            delta = target_pos - old_pos
            if np.linalg.norm(delta) > 1e-9:
                for child_id in contained:
                    child_pos, child_quat = self._get_object_pose(child_id)
                    self._set_object_pose(child_id, child_pos + delta, child_quat)

    def _get_robot_eef_pos(self, robot_idx: int) -> np.ndarray:
        site_id = self.env.robots[robot_idx].eef_site_id["right"]
        return self.env.sim.data.site_xpos[site_id].copy()

    def _find_contained_objects(self, container_id: str) -> list[str]:
        """Find objects physically inside/on top of *container_id*.

        Only returns objects that are smaller than (or equal to)
        *container_id* so that picking up an item inside a bowl/tray
        does NOT drag the bowl/tray along with it.
        """
        contained = []
        container_pos, _ = self._get_object_pose(container_id)
        container_obj = self.env.objects[container_id]
        container_radius = getattr(container_obj, "horizontal_radius", 0.10)
        search_radius = container_radius * 1.2
        for other_id in self.env.objects:
            if other_id == container_id:
                continue
            other_obj = self.env.objects[other_id]
            other_radius = getattr(other_obj, "horizontal_radius", 0.10)
            # Skip objects that are larger — they are parents, not children.
            if other_radius > container_radius:
                continue
            other_pos, _ = self._get_object_pose(other_id)
            xy_dist = float(np.linalg.norm(other_pos[:2] - container_pos[:2]))
            z_diff = other_pos[2] - container_pos[2]
            # Object must be close horizontally and roughly at the same height
            # as the container.  Allows slight negative z_diff for objects
            # inside concave containers (slices resting at the bottom of a bowl).
            if xy_dist < search_radius and -0.05 <= z_diff < 0.20:
                contained.append(other_id)
        return contained

    def _sync_held_object(self, robot_idx: int):
        object_id = self._held_objects.get(robot_idx)
        if object_id is None:
            return

        eef_pos = self._get_robot_eef_pos(robot_idx)
        offset = np.asarray(
            getattr(self, "_held_object_offsets", {}).get(
                robot_idx,
                np.zeros(3, dtype=float),
            ),
            dtype=float,
        )
        held_pos = eef_pos.copy() + offset
        held_pos[2] += self._HELD_Z_OFFSET
        if _sim_tool_debug_enabled():
            print(
                f"[_sync_held] robot{robot_idx} holds {object_id}: "
                f"eef=({eef_pos[0]:.3f}, {eef_pos[1]:.3f}, {eef_pos[2]:.3f}) "
                f"-> held=({held_pos[0]:.3f}, {held_pos[1]:.3f}, {held_pos[2]:.3f})"
            )
        # _set_object_pose already moves contained objects (e.g. slices
        # inside a bowl) by the same delta — no extra handling needed.
        self._set_object_pose(object_id, held_pos)

    def _held_pose_offset(self, object_id: str) -> np.ndarray:
        """Return object-center offset from EEF for a natural held pose."""

        if not (self._scene_object_tokens(object_id) & _HANDLED_SUPPORT_TOKENS):
            return np.zeros(3, dtype=float)
        return self._handled_object_center_offset_from_grasp(object_id)

    def _handled_object_center_offset_from_grasp(self, object_id: str) -> np.ndarray:
        """Bias handled cookware grasps to the handle instead of bbox center."""

        obj = self._require_object(object_id)
        pos, quat_wxyz = self._get_object_pose(object_id)
        del pos
        local_bbox_points = np.asarray(
            obj.get_bbox_points(
                trans=np.zeros(3, dtype=float),
                rot=np.array([0.0, 0.0, 0.0, 1.0], dtype=float),
            ),
            dtype=float,
        )
        if local_bbox_points.ndim != 2 or local_bbox_points.shape[1] < 3:
            return np.zeros(3, dtype=float)

        local_spans = np.max(local_bbox_points, axis=0) - np.min(
            local_bbox_points,
            axis=0,
        )
        axis_idx = int(np.argmax(local_spans))
        local_min = float(np.min(local_bbox_points[:, axis_idx]))
        local_max = float(np.max(local_bbox_points[:, axis_idx]))
        handle_extent = local_max if abs(local_max) >= abs(local_min) else local_min
        if abs(handle_extent) <= 1e-6:
            return np.zeros(3, dtype=float)

        # Grasp slightly in from the bbox extreme so the EEF targets the handle,
        # not the very tip.
        handle_local = np.zeros(3, dtype=float)
        handle_local[axis_idx] = 0.85 * handle_extent
        quat_xyzw = T.convert_quat(np.asarray(quat_wxyz, dtype=float), to="xyzw")
        handle_vector_world = T.quat2mat(quat_xyzw) @ handle_local
        return -handle_vector_world

    def _held_by_robot(self, object_id: str) -> int | None:
        for robot_idx, held_object in self._held_objects.items():
            if held_object == object_id:
                return robot_idx
        return None

    def _candidate_overlaps_supported_siblings(
        self,
        object_id: str,
        candidate_pos: np.ndarray,
        candidate_quat_wxyz: np.ndarray,
        sibling_ids: Sequence[str],
    ) -> bool:
        obj = self._require_object(object_id)
        obj_radius = float(getattr(obj, "horizontal_radius", 0.05))
        candidate_quat_xyzw = T.convert_quat(candidate_quat_wxyz, to="xyzw")
        for sibling_id in sibling_ids:
            sibling_obj = self._require_object(sibling_id)
            sibling_radius = float(getattr(sibling_obj, "horizontal_radius", 0.05))
            sibling_pos, sibling_quat_wxyz = self._get_object_pose(sibling_id)
            if (
                float(np.linalg.norm(sibling_pos[:2] - candidate_pos[:2]))
                > obj_radius + sibling_radius + 0.03
            ):
                continue
            try:
                if OU.objs_intersect(
                    obj,
                    candidate_pos,
                    candidate_quat_xyzw,
                    sibling_obj,
                    sibling_pos,
                    T.convert_quat(sibling_quat_wxyz, to="xyzw"),
                ):
                    return True
            except Exception:
                if (
                    float(np.linalg.norm(sibling_pos[:2] - candidate_pos[:2]))
                    < obj_radius + sibling_radius + 0.005
                    and abs(float(sibling_pos[2] - candidate_pos[2])) < 0.08
                ):
                    return True
        return False

    def _repack_supported_children(
        self,
        support_object_id: str,
        child_ids: Sequence[str],
    ) -> None:
        ordered_child_ids = sorted(
            (
                child_id
                for child_id in child_ids
                if child_id in self.env.objects and self._held_by_robot(child_id) is None
            ),
            key=lambda child_id: float(
                getattr(self._require_object(child_id), "horizontal_radius", 0.05)
            ),
            reverse=True,
        )
        for child_id in ordered_child_ids:
            self._place_on_object_center(child_id, support_object_id)
            self._set_support_parent(child_id, support_object_id)
            self.runner._set_object_location(child_id, support_object_id)

    def _place_on_object_center(
        self,
        object_id: str,
        support_object_id: str,
        *,
        relative_position: str | None = None,
    ) -> None:
        support_geometry = self._support_object_geometry(support_object_id)
        support_family = self._support_object_placement_family(
            support_object_id,
            support_geometry,
        )
        support_tokens_base = support_geometry.get("support_tokens") or set()
        if support_tokens_base & _HANDLED_SUPPORT_TOKENS:
            # For handled containers (pan, skillet, pot) the bbox center is
            # biased toward the handle.  Use the body origin as the target
            # center instead — it aligns with the usable head of the vessel.
            support_body_id = self.env.obj_body_id[support_object_id]
            support_body_pos = self.env.sim.data.body_xpos[support_body_id].copy()
            target_pos_base = np.array(
                [float(support_body_pos[0]), float(support_body_pos[1]), 0.0],
                dtype=float,
            )
        else:
            target_pos_base = np.array(
                [
                    support_geometry["center_xy"][0],
                    support_geometry["center_xy"][1],
                    0.0,
                ],
                dtype=float,
            )
        direction = self._relative_direction_vector(relative_position)
        existing_on_support = [
            child_id
            for child_id in self._iter_direct_supported_children(support_object_id)
            if child_id != object_id
        ]
        if not existing_on_support:
            existing_on_support = self._find_objects_on_support(
                support_object_id, exclude={object_id}
            )
        enforce_kettle_mug_spacing = False
        kettle_mug_min_distance = 0.0
        kettle_mug_side_sign = 0.0
        if "tray" in support_tokens_base and len(existing_on_support) == 1:
            existing_tokens = self._scene_object_tokens(existing_on_support[0])
            incoming_tokens = self._scene_object_tokens(object_id)
            if len(
                (existing_tokens | incoming_tokens) & _TRAY_KETTLE_MUG_PAIR_OBJECT_TOKENS
            ) == 2:
                enforce_kettle_mug_spacing = True
                tray_lateral_xy, tray_inward_xy = self._support_object_anchor_axes_xy(
                    support_object_id,
                    support_geometry=support_geometry,
                )
                if incoming_tokens & {"kettle"}:
                    kettle_mug_side_sign = 1.0
                elif incoming_tokens & {"mug"}:
                    kettle_mug_side_sign = -1.0
                kettle_mug_min_distance = max(
                    0.24,
                    0.38 * max(float(support_geometry["extent_x"]), 1e-6),
                )
                kettle_mug_max_depth = max(
                    0.02,
                    0.10 * max(float(support_geometry["extent_y"]), 1e-6),
                )
            else:
                kettle_mug_max_depth = float("inf")
        else:
            kettle_mug_max_depth = float("inf")

        def _append_offset(
            destination: list[np.ndarray],
            offset: np.ndarray,
            usable_half_xy: np.ndarray,
            seen: set[tuple[float, float]],
        ) -> None:
            clipped = np.array(
                [
                    float(np.clip(offset[0], -usable_half_xy[0], usable_half_xy[0])),
                    float(np.clip(offset[1], -usable_half_xy[1], usable_half_xy[1])),
                ],
                dtype=float,
            )
            key = tuple(np.round(clipped, 4))
            if key in seen:
                return
            seen.add(key)
            destination.append(clipped)

        desired_offset = None
        if relative_position is None:
            preferred_slot_xy = self._incoming_support_slot_preference(
                support_object_id,
                object_id,
            )
            if preferred_slot_xy is not None:
                desired_offset = (
                    np.asarray(preferred_slot_xy, dtype=float)[:2]
                    - support_geometry["center_xy"]
                )

        direction_vectors = (
            np.array([1.0, 0.0], dtype=float),
            np.array([-1.0, 0.0], dtype=float),
            np.array([0.0, 1.0], dtype=float),
            np.array([0.0, -1.0], dtype=float),
            np.array([1.0, 1.0], dtype=float),
            np.array([1.0, -1.0], dtype=float),
            np.array([-1.0, 1.0], dtype=float),
            np.array([-1.0, -1.0], dtype=float),
        )
        best_clear_candidate: tuple[float, np.ndarray, np.ndarray] | None = None
        best_fallback_candidate: tuple[float, np.ndarray, np.ndarray] | None = None
        for candidate_quat_wxyz in self._support_object_candidate_quaternions(
            object_id,
            support_object_id,
        ):
            bbox_metadata = self._object_bbox_metadata_for_quat(
                object_id,
                candidate_quat_wxyz,
            )
            obj_bottom_z = float(bbox_metadata["bottom_z"])
            obj_half_xy = np.asarray(
                bbox_metadata["half_spans_xy"],
                dtype=float,
            )
            target_pos = target_pos_base.copy()
            target_pos[2] = (
                self._support_object_contact_plane_z(
                    support_object_id,
                    support_geometry,
                )
                - obj_bottom_z
            )
            if support_tokens_base & _TOP_VISIBLE_SUPPORT_TOKENS:
                target_pos[2] = max(
                    float(target_pos[2]),
                    float(support_geometry["top_z"]) - obj_bottom_z + 0.012,
                )

            support_scale = self._support_object_xy_scale(
                support_object_id,
                support_geometry,
            )
            support_extent_x = float(support_geometry["extent_x"])
            support_extent_y = float(support_geometry["extent_y"])
            support_tokens = support_geometry.get("support_tokens") or set()
            if support_tokens & _HANDLED_SUPPORT_TOKENS:
                # Collapse to a square based on the short axis so candidates
                # never land on the handle.
                short_extent = min(support_extent_x, support_extent_y)
                support_extent_x = short_extent
                support_extent_y = short_extent
            if (
                support_tokens & {"tray"}
                and self._scene_object_tokens(object_id)
                & _TRAY_BULKY_PAIR_OBJECT_TOKENS
            ):
                support_scale = max(support_scale, 0.52)
            usable_half_xy = np.array(
                [
                    max(support_extent_x * support_scale - obj_half_xy[0], 0.0),
                    max(support_extent_y * support_scale - obj_half_xy[1], 0.0),
                ],
                dtype=float,
            )

            candidate_offsets: list[np.ndarray] = []
            seen_offsets: set[tuple[float, float]] = set()
            _append_offset(
                candidate_offsets,
                np.zeros(2, dtype=float),
                usable_half_xy,
                seen_offsets,
            )

            if direction is not None:
                desired_offset = direction * usable_half_xy * (
                    0.7 if support_geometry["is_concave"] else 0.82
                )
            if desired_offset is not None:
                desired_offset_candidate = np.asarray(desired_offset, dtype=float)
                if support_family == "deep_receptacle":
                    desired_offset_candidate = np.array(
                        [
                            float(
                                np.clip(
                                    desired_offset_candidate[0],
                                    -0.6 * usable_half_xy[0],
                                    0.6 * usable_half_xy[0],
                                )
                            ),
                            float(
                                np.clip(
                                    desired_offset_candidate[1],
                                    -0.6 * usable_half_xy[1],
                                    0.6 * usable_half_xy[1],
                                )
                            ),
                        ],
                        dtype=float,
                    )
                _append_offset(
                    candidate_offsets,
                    desired_offset_candidate,
                    usable_half_xy,
                    seen_offsets,
                )
            if enforce_kettle_mug_spacing:
                # Force-side candidates for kettle/mug pair on tray.
                # We bias strongly to the required side and only allow small depth jitter.
                signed_lateral = kettle_mug_side_sign if kettle_mug_side_sign != 0.0 else 1.0
                if kettle_mug_side_sign == 0.0 and existing_on_support:
                    existing_pos = self._get_object_pose(existing_on_support[0])[0][:2]
                    existing_delta = np.asarray(existing_pos, dtype=float) - support_geometry["center_xy"]
                    existing_side = float(
                        np.dot(existing_delta, np.asarray(tray_lateral_xy, dtype=float))
                    )
                    signed_lateral = -1.0 if existing_side >= 0.0 else 1.0
                side_magnitude = max(
                    0.42 * max(float(support_geometry["extent_x"]), 1e-6),
                    kettle_mug_min_distance,
                )
                forced_base = np.asarray(tray_lateral_xy, dtype=float) * (
                    signed_lateral * side_magnitude
                )
                candidate_offsets = [
                    forced_base.copy(),
                    forced_base + np.asarray(tray_inward_xy, dtype=float) * 0.01,
                    forced_base - np.asarray(tray_inward_xy, dtype=float) * 0.01,
                ]

            radial_scales = self._support_object_radial_scales(
                support_object_id,
                existing_on_support=existing_on_support,
                support_geometry=support_geometry,
            )
            if not enforce_kettle_mug_spacing:
                for scale in radial_scales:
                    if scale == 0.0:
                        _append_offset(
                            candidate_offsets,
                            np.zeros(2, dtype=float),
                            usable_half_xy,
                            seen_offsets,
                        )
                        continue
                    for basis in direction_vectors:
                        normalized_basis = basis / max(np.linalg.norm(basis), 1.0)
                        offset = normalized_basis * usable_half_xy * scale
                        _append_offset(
                            candidate_offsets,
                            offset,
                            usable_half_xy,
                            seen_offsets,
                        )

            vertical_bonus = 0.0
            if self._support_object_prefers_upright(support_object_id, object_id):
                vertical_bonus = 0.1 * float(bbox_metadata["spans"][2])

            for offset in candidate_offsets:
                candidate_pos = target_pos.copy()
                candidate_pos[:2] += offset
                sibling_distances = [
                    float(
                        np.linalg.norm(
                            self._get_object_pose(child_id)[0][:2] - candidate_pos[:2]
                        )
                    )
                    for child_id in existing_on_support
                ]
                min_sibling_distance = (
                    min(sibling_distances) if sibling_distances else float("inf")
                )
                kettle_mug_horizontal_sep = float("inf")
                kettle_mug_depth_sep = 0.0
                if enforce_kettle_mug_spacing and existing_on_support:
                    existing_pos = self._get_object_pose(existing_on_support[0])[0][:2]
                    relative_world = np.asarray(candidate_pos[:2] - existing_pos, dtype=float)
                    kettle_mug_side_proj = float(
                        np.dot(relative_world, np.asarray(tray_lateral_xy, dtype=float))
                    )
                    kettle_mug_horizontal_sep = float(
                        abs(kettle_mug_side_proj)
                    )
                    kettle_mug_depth_sep = float(
                        abs(np.dot(relative_world, np.asarray(tray_inward_xy, dtype=float)))
                    )
                else:
                    kettle_mug_side_proj = 0.0
                if desired_offset is not None:
                    score = -float(np.linalg.norm(offset - desired_offset))
                elif sibling_distances:
                    score = min_sibling_distance - 0.2 * float(np.linalg.norm(offset))
                else:
                    score = -float(np.linalg.norm(offset))
                if enforce_kettle_mug_spacing and np.isfinite(min_sibling_distance):
                    spacing_shortfall = kettle_mug_min_distance - min_sibling_distance
                    if spacing_shortfall > 0.0:
                        score -= 25.0 * spacing_shortfall
                    horizontal_shortfall = kettle_mug_min_distance - kettle_mug_horizontal_sep
                    if horizontal_shortfall > 0.0:
                        score -= 35.0 * horizontal_shortfall
                    # Prefer left-right separation over front-back displacement.
                    score -= 12.0 * kettle_mug_depth_sep
                    depth_excess = kettle_mug_depth_sep - kettle_mug_max_depth
                    if depth_excess > 0.0:
                        score -= 50.0 * depth_excess
                score += vertical_bonus

                record = (score, candidate_pos.copy(), candidate_quat_wxyz.copy())
                if (
                    best_fallback_candidate is None
                    or score > best_fallback_candidate[0]
                ):
                    best_fallback_candidate = record
                if self._candidate_overlaps_supported_siblings(
                    object_id,
                    candidate_pos,
                    candidate_quat_wxyz,
                    existing_on_support,
                ):
                    continue
                if (
                    enforce_kettle_mug_spacing
                    and np.isfinite(min_sibling_distance)
                    and min_sibling_distance < kettle_mug_min_distance
                ):
                    continue
                if (
                    enforce_kettle_mug_spacing
                    and np.isfinite(kettle_mug_horizontal_sep)
                    and kettle_mug_horizontal_sep < kettle_mug_min_distance
                ):
                    continue
                if (
                    enforce_kettle_mug_spacing
                    and kettle_mug_side_sign != 0.0
                    and np.sign(kettle_mug_side_proj) != np.sign(kettle_mug_side_sign)
                ):
                    continue
                if (
                    enforce_kettle_mug_spacing
                    and np.isfinite(kettle_mug_depth_sep)
                    and kettle_mug_depth_sep > kettle_mug_max_depth
                ):
                    continue
                if (
                    best_clear_candidate is None
                    or score > best_clear_candidate[0]
                ):
                    best_clear_candidate = record

        chosen_pos, chosen_quat = (
            (best_clear_candidate[1], best_clear_candidate[2])
            if best_clear_candidate is not None
            else (best_fallback_candidate[1], best_fallback_candidate[2])
        )
        self._set_object_pose(object_id, chosen_pos, chosen_quat)

    def _find_objects_on_support(
        self, support_object_id: str, exclude: set[str] | None = None,
    ) -> list[str]:
        """Find objects currently resting on a support object."""
        exclude = exclude or set()
        support_body_id = self.env.obj_body_id[support_object_id]
        support_pos = self.env.sim.data.body_xpos[support_body_id].copy()
        support_obj = self.env.objects[support_object_id]
        support_radius = getattr(support_obj, "horizontal_radius", 0.10)
        support_geometry = self._support_object_geometry(support_object_id)
        support_top_z = float(support_geometry["top_z"])
        support_bottom_z = float(support_geometry["bottom_z"])
        support_extent_z = float(support_geometry["extent_z"])
        uses_interior_floor = self._support_object_uses_interior_floor(
            support_object_id,
            support_geometry,
        )
        result = []
        for other_id in self.env.objects:
            if other_id == support_object_id or other_id in exclude:
                continue
            other_obj = self.env.objects[other_id]
            other_radius = getattr(other_obj, "horizontal_radius", 0.10)
            if other_radius > support_radius:
                continue
            other_pos, _ = self._get_object_pose(other_id)
            xy_dist = float(np.linalg.norm(other_pos[:2] - support_pos[:2]))
            if xy_dist >= support_radius * 1.2:
                continue
            if uses_interior_floor:
                z_min = support_bottom_z - max(0.02, 0.2 * support_extent_z)
                z_max = support_top_z + max(0.05, 0.5 * support_extent_z)
                if z_min <= float(other_pos[2]) <= z_max:
                    result.append(other_id)
                continue
            z_diff = float(other_pos[2] - support_pos[2])
            if 0.0 <= z_diff < 0.15:
                result.append(other_id)
        return result

    def _rebalance_kettle_mug_on_tray(self, tray_object_id: str) -> bool:
        """Force a stable side-by-side layout for kettle+mug on trays."""
        if "tray" not in self._support_object_tokens(tray_object_id):
            return False
        child_ids = [
            child_id
            for child_id in self._iter_direct_supported_children(tray_object_id)
            if child_id != tray_object_id
        ]
        if not child_ids:
            child_ids = self._find_objects_on_support(tray_object_id)
        kettle_id = None
        mug_id = None
        for child_id in child_ids:
            tokens = self._scene_object_tokens(child_id)
            if kettle_id is None and "kettle" in tokens:
                kettle_id = child_id
            if mug_id is None and "mug" in tokens:
                mug_id = child_id
        if not (isinstance(kettle_id, str) and isinstance(mug_id, str)):
            return False

        geometry = self._support_object_geometry(tray_object_id)
        lateral_xy, inward_xy = self._support_object_anchor_axes_xy(
            tray_object_id,
            support_geometry=geometry,
        )
        side_mag = 0.40 * max(float(geometry["extent_x"]), 1e-6)
        depth_mag = 0.02 * max(float(geometry["extent_y"]), 1e-6)

        kettle_pos, kettle_quat = self._get_object_pose(kettle_id)
        mug_pos, mug_quat = self._get_object_pose(mug_id)
        kettle_target = np.asarray(kettle_pos, dtype=float).copy()
        mug_target = np.asarray(mug_pos, dtype=float).copy()
        kettle_target[:2] = (
            geometry["center_xy"]
            + np.asarray(lateral_xy, dtype=float) * side_mag
            + np.asarray(inward_xy, dtype=float) * depth_mag
        )
        mug_target[:2] = (
            geometry["center_xy"]
            - np.asarray(lateral_xy, dtype=float) * side_mag
            - np.asarray(inward_xy, dtype=float) * depth_mag
        )
        self._set_object_pose(kettle_id, kettle_target, kettle_quat)
        self._set_object_pose(mug_id, mug_target, mug_quat)
        self._set_support_parent(kettle_id, tray_object_id)
        self._set_support_parent(mug_id, tray_object_id)
        return True

    def _supported_children_need_repack(
        self,
        support_object_id: str,
        child_ids: Sequence[str],
    ) -> bool:
        active_child_ids = [
            child_id
            for child_id in child_ids
            if child_id in self.env.objects and self._held_by_robot(child_id) is None
        ]
        if len(active_child_ids) <= 1:
            return False
        for child_id in active_child_ids:
            sibling_ids = [
                sibling_id
                for sibling_id in active_child_ids
                if sibling_id != child_id
            ]
            child_pos, child_quat_wxyz = self._get_object_pose(child_id)
            if self._candidate_overlaps_supported_siblings(
                child_id,
                np.asarray(child_pos, dtype=float),
                np.asarray(child_quat_wxyz, dtype=float),
                sibling_ids,
            ):
                return True
        return False

    def _supported_children_should_balance_slots(
        self,
        support_object_id: str,
        child_ids: Sequence[str],
    ) -> bool:
        active_child_ids = [
            child_id
            for child_id in child_ids
            if child_id in self.env.objects and self._held_by_robot(child_id) is None
        ]
        if len(active_child_ids) != 2:
            return False
        if self._is_portion_hotdogs_plate_pair(support_object_id, active_child_ids):
            return True
        support_tokens = self._support_object_tokens(support_object_id)
        if "tray" not in support_tokens:
            return False
        return any(
            self._scene_object_tokens(child_id) & _TRAY_BULKY_PAIR_OBJECT_TOKENS
            for child_id in active_child_ids
        )

    def _current_support_object(self, object_id: str) -> str | None:
        support_parent = self._support_parents.get(object_id)
        if isinstance(support_parent, str) and support_parent in self.env.objects:
            return support_parent
        for support_object_id in self.env.objects:
            if support_object_id == object_id:
                continue
            if object_id in self._find_objects_on_support(support_object_id):
                return support_object_id
        return None

    def _set_named_joint(self, fixture, joint_name: str, value: float):
        fixture.set_joint_state(
            min=value,
            max=value,
            env=self.env,
            joint_names=[joint_name],
        )
        self._settle_scene()

    @staticmethod
    def _drawer_transient_obstacle_key(target_id: str, part_id: str) -> str:
        return f"drawer::{target_id}::{part_id}"

    @staticmethod
    def _drawer_slide_delta_from_aabbs(
        before_aabb: tuple[np.ndarray, np.ndarray] | None,
        after_aabb: tuple[np.ndarray, np.ndarray] | None,
    ) -> np.ndarray:
        if before_aabb is None or after_aabb is None:
            return np.zeros(3, dtype=float)
        before_min, before_max = before_aabb
        after_min, after_max = after_aabb
        delta_min = np.asarray(after_min, dtype=float)[:2] - np.asarray(before_min, dtype=float)[:2]
        delta_max = np.asarray(after_max, dtype=float)[:2] - np.asarray(before_max, dtype=float)[:2]
        axis_candidates = [
            (0, delta_max[0]),
            (0, delta_min[0]),
            (1, delta_max[1]),
            (1, delta_min[1]),
        ]
        axis, signed_delta = max(axis_candidates, key=lambda item: abs(float(item[1])))
        if abs(float(signed_delta)) <= 1e-6:
            return np.zeros(3, dtype=float)
        delta = np.zeros(3, dtype=float)
        delta[axis] = float(signed_delta)
        return delta

    def _translate_supported_fixture_children(
        self,
        fixture_id: str,
        delta: np.ndarray,
    ) -> list[dict[str, Any]]:
        delta_vec = np.asarray(delta, dtype=float)
        if float(np.linalg.norm(delta_vec[:2])) <= 1e-6:
            return []
        child_ids = self._iter_direct_supported_children(fixture_id)
        if not child_ids:
            return []
        moved_children: list[dict[str, Any]] = []
        for child_id in child_ids:
            if child_id not in self.env.objects or self._held_by_robot(child_id) is not None:
                continue
            child_pos, child_quat = self._get_object_pose(child_id)
            child_pos_before = np.asarray(child_pos, dtype=float).copy()
            self._set_object_pose(child_id, child_pos + delta_vec, child_quat)
            child_pos_after = self._get_object_pose(child_id)[0]
            self._set_support_parent(child_id, fixture_id)
            self.runner._set_object_location(child_id, fixture_id)
            moved_children.append(
                {
                    "object_id": child_id,
                    "delta": (np.asarray(child_pos_after, dtype=float) - child_pos_before).tolist(),
                    "before": child_pos_before.tolist(),
                    "after": np.asarray(child_pos_after, dtype=float).tolist(),
                }
            )
        self._settle_scene(steps=max(_PLACEMENT_SETTLE_STEPS, 4))
        return moved_children

    def _refresh_fixture_supported_children(
        self,
        fixture_id: str,
        *,
        anchor_aabb: tuple[np.ndarray, np.ndarray] | None = None,
    ) -> list[str]:
        """Ensure fixture-contained objects are attached in support graph.

        This is especially important for sliding fixtures (drawers), where
        support graph drift can prevent contents from translating with drawer
        motion even if the simulator scene location says they are in the drawer.
        """
        attached: list[str] = []
        for object_id in sorted(self.env.objects.keys()):
            if self._held_by_robot(object_id) is not None:
                continue
            if self._get_scene_object_location(object_id) != fixture_id:
                continue
            self._set_support_parent(object_id, fixture_id)
            self.runner._set_object_location(object_id, fixture_id)
            attached.append(object_id)
        return attached

    def _update_drawer_transient_obstacle(
        self,
        target_id: str,
        part_id: str,
        *,
        before_aabb: tuple[np.ndarray, np.ndarray] | None,
        after_aabb: tuple[np.ndarray, np.ndarray] | None,
        is_open: bool,
    ) -> None:
        occupancy_grid = getattr(self.runner, "_occupancy_grid", None)
        if occupancy_grid is None:
            return
        obstacle_key = self._drawer_transient_obstacle_key(target_id, part_id)
        if not is_open:
            clear_fn = getattr(occupancy_grid, "clear_world_aabb_occupied", None)
            if callable(clear_fn):
                clear_fn(obstacle_key)
            return
        if before_aabb is None or after_aabb is None:
            return
        union_min = np.minimum(
            np.asarray(before_aabb[0], dtype=float)[:2],
            np.asarray(after_aabb[0], dtype=float)[:2],
        )
        union_max = np.maximum(
            np.asarray(before_aabb[1], dtype=float)[:2],
            np.asarray(after_aabb[1], dtype=float)[:2],
        )
        mark_fn = getattr(occupancy_grid, "mark_world_aabb_occupied", None)
        if callable(mark_fn):
            mark_fn((union_min, union_max), obstacle_key)

    def _sliding_joint_open_fraction(
        self,
        joint_name: str,
        *,
        part_id: str | None = None,
    ) -> float:
        normalized_joint = str(joint_name).strip().lower().replace(" ", "_")
        normalized_part = str(part_id or "").strip().lower().replace(" ", "_")
        combined = "_".join(token for token in (normalized_joint, normalized_part) if token)
        if any(token in combined for token in ("rack", "tray")):
            return 1.0
        if any(token in combined for token in ("drawer", "slide", "sliding")):
            return _PARTIAL_DRAWER_OPEN_FRACTION
        return 1.0

    def _joint_qpos_scalar(self, joint_name: str) -> float | None:
        """Return scalar qpos for a named joint, or None when unavailable."""
        try:
            joint_id = int(self.env.sim.model.joint_name2id(joint_name))
            qpos_addr = int(self.env.sim.model.jnt_qposadr[joint_id])
            return float(self.env.sim.data.qpos[qpos_addr])
        except Exception:
            try:
                qpos_val = self.env.sim.data.get_joint_qpos(joint_name)
                if isinstance(qpos_val, (float, int)):
                    return float(qpos_val)
                qpos_arr = np.asarray(qpos_val, dtype=float).reshape(-1)
                if qpos_arr.size:
                    return float(qpos_arr[0])
            except Exception:
                return None
        return None

    def _sliding_joint_world_delta_from_qpos(
        self,
        joint_name: str,
        *,
        before_qpos: float | None,
        after_qpos: float | None,
    ) -> np.ndarray:
        """Estimate world translation from slide-joint qpos change."""
        if before_qpos is None or after_qpos is None:
            return np.zeros(3, dtype=float)
        dq = float(after_qpos - before_qpos)
        if abs(dq) <= 1e-8:
            return np.zeros(3, dtype=float)
        try:
            joint_id = int(self.env.sim.model.joint_name2id(joint_name))
            body_id = int(self.env.sim.model.jnt_bodyid[joint_id])
            local_axis = np.asarray(self.env.sim.model.jnt_axis[joint_id], dtype=float)
            body_xmat = np.asarray(self.env.sim.data.body_xmat[body_id], dtype=float).reshape(3, 3)
            world_axis = body_xmat @ local_axis
            return np.asarray(world_axis, dtype=float) * dq
        except Exception:
            return np.zeros(3, dtype=float)

    def _joint_token_candidates(self, token: str) -> tuple[str, ...]:
        lowered = str(token).strip().lower().replace(" ", "_")
        candidates = [lowered]
        alias_groups = (
            (
                {"time", "timer", "time_knob", "timer_knob"},
                ("time", "timer"),
            ),
            (
                {"temperature", "temp", "temperature_knob", "temp_knob", "heat_knob"},
                ("temperature", "temp"),
            ),
            (
                {"function", "mode", "function_knob", "mode_knob"},
                ("function", "mode"),
            ),
            (
                {
                    "doneness",
                    "toast",
                    "browning",
                    "doneness_knob",
                    "toast_knob",
                    "browning_knob",
                },
                ("doneness", "toast", "browning"),
            ),
        )
        for source_tokens, alias_tokens in alias_groups:
            if lowered not in source_tokens:
                continue
            for alias_token in alias_tokens:
                if alias_token not in candidates:
                    candidates.append(alias_token)
            break
        return tuple(candidates)

    def _resolve_joint_name(self, fixture, token: str) -> str:
        token_candidates = self._joint_token_candidates(token)

        if hasattr(fixture, "_joint_names"):
            for token_candidate in token_candidates:
                if token_candidate in fixture._joint_names:
                    return fixture._joint_names[token_candidate]

        if hasattr(fixture, "_joint_infos"):
            for token_candidate in token_candidates:
                for joint_name in fixture._joint_infos:
                    if token_candidate in joint_name.lower():
                        return joint_name

        raise ValueError(
            f"Unknown part/control {str(token).strip().lower()!r} for fixture {fixture.name!r}"
        )

    def _get_scene_object_location(self, object_id: str) -> str | None:
        current_object_id = object_id
        seen_object_ids: set[str] = set()
        scene = self.get_scene_description()
        scene_objects = scene.get("objects", {})
        scene_fixtures = scene.get("fixtures", {})
        support_parents = getattr(self, "_support_parents", {})

        while isinstance(current_object_id, str):
            if current_object_id in seen_object_ids:
                break
            seen_object_ids.add(current_object_id)

            support_parent = support_parents.get(current_object_id)
            if isinstance(support_parent, str):
                if support_parent in self.runner._fixtures:
                    return support_parent
                if support_parent in scene_objects:
                    current_object_id = support_parent
                    continue
                if support_parent.startswith("held_by_robot_"):
                    return None

            cached_location = self.runner._object_locations.get(current_object_id)
            if isinstance(cached_location, str):
                if cached_location in self.runner._fixtures:
                    return cached_location
                if cached_location in scene_objects:
                    current_object_id = cached_location
                    continue

            object_info = scene_objects.get(current_object_id, {})
            location = object_info.get("location")
            if isinstance(location, str):
                if location in scene_fixtures:
                    return location
                if location in scene_objects:
                    current_object_id = location
                    continue
                if location in self.runner._fixtures:
                    return location

        try:
            obj_pos, _ = self._get_object_pose(object_id)
        except Exception:
            return None
        inferred = self.runner._find_object_fixture(obj_pos)
        if inferred in self.runner._fixtures:
            return inferred
        return None

    def _clear_held_object_state(self, robot_idx: int) -> None:
        held_objects = getattr(self, "_held_objects", None)
        if isinstance(held_objects, dict):
            held_objects.pop(robot_idx, None)
        held_object_offsets = getattr(self, "_held_object_offsets", None)
        if isinstance(held_object_offsets, dict):
            held_object_offsets.pop(robot_idx, None)

    def _resolve_pick_source_target(
        self,
        source_id: str,
        source_site_id: str | None = None,
    ) -> tuple[str, str | None, str | None]:
        if source_id in self.env.objects:
            source_fixture_id = self._get_scene_object_location(source_id)
            if not isinstance(source_fixture_id, str):
                raise ValueError(
                    f"Could not resolve containing fixture for source object {source_id!r}"
                )
            resolved_site_id = None
            if isinstance(source_site_id, str):
                resolved_site_id = self._resolve_fixture_site_id(
                    source_fixture_id,
                    source_site_id,
                )
            return source_fixture_id, resolved_site_id, source_id

        source_fixture_id, resolved_site_id = self._resolve_support_target(
            source_id,
            source_site_id,
        )
        return source_fixture_id, resolved_site_id, None

    def _get_task_class_name(self) -> str:
        env = self.env
        while hasattr(env, "env"):
            env = env.env
        return env.__class__.__name__

    def _find_fixture_by_type(self, fixture_types: set[str]) -> str:
        scene = self.get_scene_description()
        for fixture_id, fixture_info in scene.get("fixtures", {}).items():
            if fixture_info.get("fixture_type") in fixture_types:
                return fixture_id
        raise ValueError(f"Could not find fixture with type in {sorted(fixture_types)}")

    # ------------------------------------------------------------------
    # Spatial relation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_dispenser_site_name(fixture) -> str | None:
        """Return the MuJoCo site name for a fixture's dispenser output, or None.

        This is the single place to register dispenser-type fixtures so that
        ``place_under`` can handle them uniformly.  To add a new dispenser,
        add an ``elif`` branch here.
        """
        if isinstance(fixture, CoffeeMachine):
            return f"{fixture.naming_prefix}receptacle_place_site"
        if isinstance(fixture, Sink) and hasattr(fixture, "water_site"):
            site_elem = fixture.water_site
            if site_elem is not None:
                return site_elem.get("name")
        return None

    def _resolve_reference_target(
        self,
        *,
        reference_id: str | None = None,
        reference_object_id: str | None = None,
        reference_fixture_id: str | None = None,
    ) -> tuple[str, bool]:
        candidates = [
            value
            for value in (reference_id, reference_object_id, reference_fixture_id)
            if isinstance(value, str)
        ]
        if len(candidates) != 1:
            raise ValueError(
                "Exactly one of reference_id, reference_object_id, or reference_fixture_id must be provided."
            )
        resolved_id = candidates[0]
        return resolved_id, resolved_id in self.runner._fixtures

    def _reference_extent_xy(self, reference_id: str, *, is_fixture: bool) -> np.ndarray:
        if is_fixture:
            fixture = self._require_fixture(reference_id)
            aabb = get_fixture_aabb(fixture)
            if aabb is None:
                return np.array([0.2, 0.2], dtype=float)
            return np.asarray(aabb[1][:2] - aabb[0][:2], dtype=float)

        reference_object = self._require_object(reference_id)
        body_id = self.env.obj_body_id[reference_id]
        body_pos = self.env.sim.data.body_xpos[body_id].copy()
        quat_xyzw = T.convert_quat(
            self.env.sim.data.body_xquat[body_id].copy(),
            to="xyzw",
        )
        bbox = reference_object.get_bbox_points(trans=body_pos, rot=quat_xyzw)
        extent_x = max(point[0] for point in bbox) - min(point[0] for point in bbox)
        extent_y = max(point[1] for point in bbox) - min(point[1] for point in bbox)
        return np.array([extent_x, extent_y], dtype=float)

    def _reference_extent_xy_in_support_local(
        self,
        support_fixture_id: str,
        reference_id: str,
        *,
        is_fixture: bool,
    ) -> np.ndarray:
        support_fixture = self._require_fixture(support_fixture_id)
        if is_fixture:
            fixture = self._require_fixture(reference_id)
            aabb = get_fixture_aabb(fixture)
            if aabb is None:
                return np.array([0.2, 0.2], dtype=float)
            min_xy = np.asarray(aabb[0][:2], dtype=float)
            max_xy = np.asarray(aabb[1][:2], dtype=float)
            world_points = np.asarray(
                [
                    [min_xy[0], min_xy[1]],
                    [min_xy[0], max_xy[1]],
                    [max_xy[0], min_xy[1]],
                    [max_xy[0], max_xy[1]],
                ],
                dtype=float,
            )
        else:
            reference_object = self._require_object(reference_id)
            body_id = self.env.obj_body_id[reference_id]
            body_pos = self.env.sim.data.body_xpos[body_id].copy()
            quat_xyzw = T.convert_quat(
                self.env.sim.data.body_xquat[body_id].copy(),
                to="xyzw",
            )
            world_points = np.asarray(
                reference_object.get_bbox_points(trans=body_pos, rot=quat_xyzw),
                dtype=float,
            )[:, :2]
        local_points = np.asarray(
            [
                self.runner._world_to_fixture_local(support_fixture, point)[:2]
                for point in world_points
            ],
            dtype=float,
        )
        return np.max(local_points, axis=0) - np.min(local_points, axis=0)

    def _candidate_xy_next_to_reference(
        self,
        *,
        support_fixture_id: str,
        reference_xy: np.ndarray,
        reference_extent_xy: np.ndarray,
        reference_id: str | None = None,
        reference_is_fixture: bool = False,
        target_site_id: str | None = None,
        relative_position: str | None = None,
    ) -> list[np.ndarray]:
        support_fixture = self._require_fixture(support_fixture_id)
        reference_local = np.asarray(
            self.runner._world_to_fixture_local(
                support_fixture,
                np.asarray(reference_xy, dtype=float)[:2],
            ),
            dtype=float,
        )[:2]
        local_reference_extent_xy = np.asarray(reference_extent_xy, dtype=float)
        if isinstance(reference_id, str):
            try:
                local_reference_extent_xy = self._reference_extent_xy_in_support_local(
                    support_fixture_id,
                    reference_id,
                    is_fixture=reference_is_fixture,
                )
            except Exception:
                local_reference_extent_xy = np.asarray(reference_extent_xy, dtype=float)
        bounds = self._fixture_local_bounds(support_fixture_id, site_id=target_site_id)
        direction = self._relative_direction_vector(relative_position)
        reference_fixture_type = None
        if reference_is_fixture and isinstance(reference_id, str):
            reference_fixture_type = (self._get_fixture_type_name(reference_id) or "").lower()
        if direction is None:
            if reference_fixture_type in _SEAT_REFERENCE_TYPE_NAMES:
                if bounds is not None:
                    local_min, local_max = bounds
                    reference_xy_local = np.asarray(reference_local, dtype=float)
                    low_violation = np.maximum(
                        np.asarray(local_min, dtype=float) - reference_xy_local,
                        0.0,
                    )
                    high_violation = np.maximum(
                        reference_xy_local - np.asarray(local_max, dtype=float),
                        0.0,
                    )
                    outside_magnitude = np.maximum(low_violation, high_violation)
                    if float(np.max(outside_magnitude)) > 1e-9:
                        axis_idx = int(np.argmax(outside_magnitude))
                        inward_direction = np.zeros(2, dtype=float)
                        inward_direction[axis_idx] = (
                            1.0 if low_violation[axis_idx] > 0.0 else -1.0
                        )
                    else:
                        support_center_local = 0.5 * (
                            np.asarray(local_min, dtype=float)
                            + np.asarray(local_max, dtype=float)
                        )
                        inward_direction = support_center_local - reference_xy_local
                else:
                    inward_direction = -np.asarray(reference_local, dtype=float)
                if float(np.linalg.norm(inward_direction)) > 1e-9:
                    inward_direction = inward_direction / float(
                        np.linalg.norm(inward_direction)
                    )
                    tangent_direction = np.array(
                        [-inward_direction[1], inward_direction[0]],
                        dtype=float,
                    )
                    candidate_directions = (
                        inward_direction,
                        tangent_direction,
                        -tangent_direction,
                        -inward_direction,
                    )
                else:
                    candidate_directions = (
                        np.array([0.0, 1.0], dtype=float),
                        np.array([1.0, 0.0], dtype=float),
                        np.array([-1.0, 0.0], dtype=float),
                    )
            elif reference_fixture_type in _COUNTERTOP_APPLIANCE_TYPE_NAMES:
                candidate_directions = (
                    np.array([1.0, 0.0], dtype=float),
                    np.array([-1.0, 0.0], dtype=float),
                )
            else:
                primary_axis = (
                    np.array([1.0, 0.0], dtype=float)
                    if local_reference_extent_xy[0] >= local_reference_extent_xy[1]
                    else np.array([0.0, 1.0], dtype=float)
                )
                secondary_axis = np.array([-primary_axis[1], primary_axis[0]], dtype=float)
                candidate_directions = (
                    primary_axis,
                    -primary_axis,
                    secondary_axis,
                    -secondary_axis,
                )
        else:
            candidate_directions = (direction,)

        offset_distance = max(float(np.max(local_reference_extent_xy)) * 0.5 + 0.06, 0.06)
        if reference_fixture_type in _SEAT_REFERENCE_TYPE_NAMES:
            offset_distance += 0.05
        if reference_fixture_type in _HIGH_CLEARANCE_REFERENCE_TYPE_NAMES:
            offset_distance += _HIGH_CLEARANCE_APPLIANCE_BONUS
        elif reference_fixture_type in _COUNTERTOP_APPLIANCE_TYPE_NAMES:
            offset_distance += 0.03
        if reference_fixture_type in _COUNTERTOP_APPLIANCE_TYPE_NAMES and direction is None:
            tangent_step = max(min(0.2 * offset_distance, 0.05), 0.02)
            distance_scales = (1.0, 1.25, 1.5)
            lateral_offsets = (0.0,)
        else:
            tangent_step = max(min(0.4 * offset_distance, 0.12), 0.04)
            distance_scales = (1.0, 1.35, 1.7)
            lateral_offsets = (0.0, -1.0, 1.0, -2.0, 2.0)
        preferred_world_positions: list[np.ndarray] = []
        seen_positions: set[tuple[float, float]] = set()
        for local_direction in candidate_directions:
            tangent = np.array([-local_direction[1], local_direction[0]], dtype=float)
            for distance_scale in distance_scales:
                base_local = reference_local + local_direction * offset_distance * distance_scale
                for lateral_scale in lateral_offsets:
                    candidate_local = base_local + tangent * tangent_step * lateral_scale
                    if bounds is not None:
                        local_min, local_max = bounds
                        candidate_local = np.clip(candidate_local, local_min, local_max)
                    candidate_world = self.runner._fixture_local_to_world(
                        support_fixture,
                        np.array([candidate_local[0], candidate_local[1], 0.0], dtype=float),
                    )
                    world_xy = np.asarray(candidate_world[:2], dtype=float)
                    key = tuple(np.round(world_xy, 4))
                    if key in seen_positions:
                        continue
                    seen_positions.add(key)
                    preferred_world_positions.append(world_xy)
        return preferred_world_positions

    def _find_placeable_surface_near_fixture(
        self,
        reference_fixture_id: str,
        *,
        robot_idx: int | None = None,
    ) -> str:
        """Find the nearest placeable surface to a reference fixture.

        If the reference fixture is itself placeable (e.g., a counter), returns it.
        Otherwise searches nearby fixtures and falls back to the closest counter.

        This indirection is needed because the reference in a spatial relation
        is often non-placeable (e.g., "near toaster_oven" — the toaster isn't a
        surface, but the counter it sits on is).
        """
        scene = self.get_scene_description()
        fixtures = scene.get("fixtures", {})
        ref_info = fixtures.get(reference_fixture_id)
        if ref_info is None:
            raise ValueError(f"Unknown fixture: {reference_fixture_id!r}")

        ref_fixture_type = str(ref_info.get("fixture_type") or "").lower()
        is_stove_like_reference = (
            "stove" in ref_fixture_type or "cooktop" in ref_fixture_type
        )
        if ref_info.get("can_place_objects", False) and not is_stove_like_reference:
            return reference_fixture_id

        # Use parent_fixture (containment-based) when available — this is
        # the counter the fixture actually sits on.
        parent_id = ref_info.get("parent_fixture")
        if parent_id and parent_id in fixtures:
            parent_info = fixtures[parent_id]
            parent_type = str(parent_info.get("fixture_type") or "").lower()
            if parent_info.get("can_place_objects", False) and (
                not is_stove_like_reference
                or parent_type in _COUNTERLIKE_FIXTURE_TYPES
            ):
                return parent_id

        nearby_placeable = [
            fixture_id
            for fixture_id in ref_info.get("nearby_fixtures", [])
            if fixtures.get(fixture_id, {}).get("can_place_objects", False)
        ]
        if is_stove_like_reference:
            counter_nearby = [
                fixture_id
                for fixture_id in nearby_placeable
                if fixtures.get(fixture_id, {}).get("fixture_type")
                in _COUNTERLIKE_FIXTURE_TYPES
            ]
            # Deterministic grounding-first disambiguation:
            # if a task-level fixture role "counter" is grounded and is a valid
            # nearby counter for this stove-like reference, use it.
            if counter_nearby:
                scene_fixture_refs = scene.get("fixture_refs", {})
                grounded_counter = scene_fixture_refs.get("counter")
                if (
                    isinstance(grounded_counter, str)
                    and grounded_counter in counter_nearby
                ):
                    return grounded_counter
            if len(counter_nearby) == 1:
                return counter_nearby[0]
            if counter_nearby:
                nearby_placeable = counter_nearby
        if len(nearby_placeable) == 1:
            return nearby_placeable[0]

        # Fallback: nearest placeable surface by center distance, but refuse
        # ties across counter-like surfaces because that produces unstable
        # grounding between dining counters and nearby prep counters.
        ref_pos = np.asarray(ref_info["position"][:2], dtype=float)
        ranked_candidates: list[tuple[float, str]] = []
        for fixture_id, info in fixtures.items():
            if not info.get("can_place_objects", False):
                continue
            dist = float(
                np.linalg.norm(np.asarray(info["position"][:2], dtype=float) - ref_pos)
            )
            ranked_candidates.append((dist, fixture_id))

        if not ranked_candidates:
            raise ValueError(
                f"No placeable surface found near {reference_fixture_id!r}"
            )
        if is_stove_like_reference:
            counter_candidates = [
                (dist, fixture_id)
                for dist, fixture_id in ranked_candidates
                if fixture_id != reference_fixture_id
                and fixtures.get(fixture_id, {}).get("fixture_type")
                in _COUNTERLIKE_FIXTURE_TYPES
            ]
            if counter_candidates:
                return min(counter_candidates)[1]
        ranked_candidates.sort()
        best_dist, best_id = ranked_candidates[0]
        close_candidates = [
            fixture_id
            for dist, fixture_id in ranked_candidates
            if dist <= best_dist + _AMBIGUOUS_SURFACE_DISTANCE_DELTA
        ]
        # Prefer a counter-like surface over a stove/cooktop when both are
        # within the tie window.  A "spice on counter" spec should not land
        # on counter_stove just because the stove happened to be 0.05 m
        # closer to the reference fixture.
        counter_close = [
            fixture_id
            for fixture_id in close_candidates
            if fixtures.get(fixture_id, {}).get("fixture_type")
            in _COUNTERLIKE_FIXTURE_TYPES
        ]
        if counter_close and fixtures.get(best_id, {}).get("fixture_type") == "stove":
            best_id = counter_close[0]
            close_candidates = counter_close
        if close_candidates:
            scene_fixture_refs = scene.get("fixture_refs", {})
            grounded_counter = scene_fixture_refs.get("counter")
            if (
                isinstance(grounded_counter, str)
                and grounded_counter in close_candidates
                and fixtures.get(grounded_counter, {}).get("fixture_type")
                in _COUNTERLIKE_FIXTURE_TYPES
            ):
                return grounded_counter
        if len(close_candidates) > 1 and all(
            fixtures.get(fixture_id, {}).get("fixture_type") in _COUNTERLIKE_FIXTURE_TYPES
            for fixture_id in close_candidates
        ):
            # Keep deterministic behavior across runs even when still ambiguous.
            # Grounding-first choice above handles the common intended case.
            return sorted(close_candidates)[0]
        return best_id

    def _ignored_fixture_ids_for_adjacent_reference(
        self,
        *,
        support_fixture_id: str,
        support_site_id: str | None,
        reference_fixture_id: str | None,
    ) -> set[str]:
        ignored_fixture_ids = set(
            self._ignored_fixture_ids_for_placement(
                support_fixture_id,
                support_site_id,
            )
        )
        if not isinstance(reference_fixture_id, str):
            return ignored_fixture_ids
        if reference_fixture_id == support_fixture_id:
            return ignored_fixture_ids
        reference_fixture_type = (self._get_fixture_type_name(reference_fixture_id) or "").lower()
        if reference_fixture_type in _COUNTERTOP_APPLIANCE_TYPE_NAMES:
            ignored_fixture_ids.add(reference_fixture_id)
        return ignored_fixture_ids


    def _find_nearest_fixture_for_object(
        self,
        object_id: str,
        preferred_fixture_types: list[str] | set[str] | tuple[str, ...] | None = None,
        require_placeable: bool = False,
    ) -> str:
        object_pos, _ = self._get_object_pose(object_id)
        scene = self.get_scene_description()
        fixtures = scene.get("fixtures", {})
        preferred_fixture_types = self._normalize_preferred_fixture_types(
            preferred_fixture_types
        )

        def candidate_ids() -> list[str]:
            candidates = []
            for fixture_id, fixture_info in fixtures.items():
                if (
                    preferred_fixture_types is not None
                    and fixture_info.get("fixture_type") not in preferred_fixture_types
                ):
                    continue
                if require_placeable and not fixture_info.get(
                    "can_place_objects", False
                ):
                    continue
                candidates.append(fixture_id)
            return candidates

        candidates = candidate_ids()
        if not candidates:
            raise ValueError(
                f"Could not find candidate fixtures for {object_id!r} "
                f"with types {sorted(preferred_fixture_types or set())}"
            )

        return min(
            candidates,
            key=lambda fixture_id: float(
                np.linalg.norm(
                    object_pos[:2]
                    - np.asarray(fixtures[fixture_id]["position"][:2], dtype=float)
                )
            ),
        )

    def _resolve_object_anchor_fixture(
        self,
        object_id: str,
        preferred_fixture_types: list[str] | set[str] | tuple[str, ...] | None = None,
        require_placeable: bool = False,
    ) -> str:
        scene = self.get_scene_description()
        fixtures = scene.get("fixtures", {})
        object_pos, _ = self._get_object_pose(object_id)
        preferred_fixture_types = self._normalize_preferred_fixture_types(
            preferred_fixture_types
        )

        def matches(fixture_id: str) -> bool:
            fixture_info = fixtures.get(fixture_id)
            if fixture_info is None:
                return False
            if (
                preferred_fixture_types is not None
                and fixture_info.get("fixture_type") not in preferred_fixture_types
            ):
                return False
            if require_placeable and not fixture_info.get("can_place_objects", False):
                return False
            return True

        explicit_location = self._get_scene_object_location(object_id)
        if explicit_location is not None and matches(explicit_location):
            return explicit_location

        candidate_ids = []
        if explicit_location is not None:
            candidate_ids.extend(
                fixtures.get(explicit_location, {}).get("nearby_fixtures", [])
            )
        candidate_ids.extend(fixtures.keys())

        deduped_candidate_ids = []
        seen = set()
        for fixture_id in candidate_ids:
            if fixture_id in seen or not matches(fixture_id):
                continue
            seen.add(fixture_id)
            deduped_candidate_ids.append(fixture_id)

        if not deduped_candidate_ids:
            return self._find_nearest_fixture_for_object(
                object_id,
                preferred_fixture_types=preferred_fixture_types,
                require_placeable=require_placeable,
            )

        def candidate_score(fixture_id: str) -> tuple[int, int, float]:
            fixture = self.runner._fixtures.get(fixture_id)
            contains_object = False
            if fixture is not None:
                try:
                    contains_object = bool(
                        OU.point_in_fixture(object_pos, fixture, only_2d=True)
                    )
                except Exception:
                    contains_object = False

            fixture_type = fixtures[fixture_id].get("fixture_type")
            type_priority = self._get_fixture_type_priority(
                fixture_type,
                preferred_fixture_types,
            )
            distance = float(
                np.linalg.norm(
                    object_pos[:2]
                    - np.asarray(fixtures[fixture_id]["position"][:2], dtype=float)
                )
            )
            return (
                0 if contains_object else 1,
                type_priority,
                distance,
            )

        return min(deduped_candidate_ids, key=candidate_score)

    def _normalize_preferred_fixture_types(
        self,
        preferred_fixture_types: list[str] | set[str] | tuple[str, ...] | None,
    ) -> list[str] | None:
        if preferred_fixture_types is None:
            return None
        if isinstance(preferred_fixture_types, list):
            return preferred_fixture_types
        return list(preferred_fixture_types)

    def _get_fixture_type_priority(
        self,
        fixture_type: str | None,
        preferred_fixture_types: list[str] | None,
    ) -> int:
        if preferred_fixture_types is None or fixture_type is None:
            return 0
        try:
            return preferred_fixture_types.index(fixture_type)
        except ValueError:
            return len(preferred_fixture_types)

    def _infer_source_fixture(
        self,
        object_id: str,
        preferred_fixture_types: list[str] | set[str] | tuple[str, ...] | None = None,
    ) -> str:
        return self._resolve_object_anchor_fixture(
            object_id,
            preferred_fixture_types=preferred_fixture_types,
        )

    def _move_robot_near_object_anchor(
        self,
        robot_idx: int,
        object_id: str,
        preferred_fixture_types: list[str] | set[str] | tuple[str, ...] | None = None,
        require_placeable: bool = False,
    ) -> str:
        fixture_id = self._resolve_object_anchor_fixture(
            object_id,
            preferred_fixture_types=preferred_fixture_types,
            require_placeable=require_placeable,
        )
        fixture = self.runner._fixtures[fixture_id]
        if _is_approach_center(fixture):
            self.runner._move_robot_near_fixture(
                robot_idx,
                fixture_id,
                require_front=True,
            )
        else:
            self.runner._move_robot_near_fixture(
                robot_idx,
                fixture_id,
                ref_object_id=object_id,
            )
        self._sync_held_object(robot_idx)
        return fixture_id

    # ------------------------------------------------------------------
    # Primitive tools
    # ------------------------------------------------------------------

    def get_image(
        self,
        views: list[str] | tuple[str, ...] | str | None = None,
        image_paths: list[str] | tuple[str, ...] | str | None = None,
        agent_id: str | int | None = None,
        robot_idx: int = 0,
        view: str | None = None,
        image_path: str | None = None,
    ) -> ToolResult:
        if views is None:
            if view is None:
                raise ValueError("get_image requires views or view.")
            view_names = [str(view)]
        elif isinstance(views, str):
            view_names = [views]
        else:
            view_names = [str(view_name) for view_name in views]

        if not view_names:
            raise ValueError("get_image requires at least one view.")

        if image_paths is None:
            if image_path is None:
                raise ValueError("get_image requires image_paths or image_path.")
            requested_paths = [str(image_path)]
        elif isinstance(image_paths, (str, Path)):
            requested_paths = [str(image_paths)]
        else:
            requested_paths = [str(path) for path in image_paths]

        if len(requested_paths) != len(view_names):
            raise ValueError(
                "get_image requires image_paths to match the number of requested views."
            )

        resolved_agent_id = agent_id if agent_id is not None else f"agent_{robot_idx}"
        resolved_robot_idx = self._parse_agent_idx(resolved_agent_id)
        saved_paths: list[str] = []
        camera_names: list[str] = []

        normalized_view_names: list[str] = []
        for view_name, requested_path in zip(view_names, requested_paths):
            normalized_view = str(view_name).strip().lower()
            normalized_view_names.append(normalized_view)
            if normalized_view == "map":
                saved_path = self._save_map_image(
                    requested_path,
                    clean_labels=getattr(self, "_clean_map_labels", True),
                )
                camera_name = "map"
            elif normalized_view in {"room_view", "top_view"}:
                camera_name = normalized_view
                image = self._render_camera(camera_name)
                saved_path = self._save_image(image, requested_path)
            else:
                _, camera_name = self._camera_name_for_agent_view(
                    resolved_agent_id,
                    normalized_view,
                )
                image = self._render_camera(camera_name)
                saved_path = self._save_image(image, requested_path)

            camera_names.append(camera_name)
            saved_paths.append(str(saved_path))

        details = {
            "robot_idx": resolved_robot_idx,
            "agent_id": str(resolved_agent_id),
            "views": normalized_view_names,
            "camera_names": camera_names,
            "image_paths": saved_paths,
        }
        if len(normalized_view_names) == 1:
            details["view"] = normalized_view_names[0]
            details["camera_name"] = camera_names[0]
            details["image_path"] = saved_paths[0]
        return ToolResult("get_image", True, details)

    def navigate_to_fixture(self, fixture_id: str, robot_idx: int = 0) -> ToolResult:
        requested_fixture_id = fixture_id
        fixture_id = self._canonical_navigation_fixture_id(fixture_id)
        fixture = self._require_fixture(fixture_id)
        placed = self._move_robot_near_fixture_with_retries(
            robot_idx,
            fixture_id,
            require_front=self._fixture_requires_front_approach(fixture_id)
            or self._surface_fixture_prefers_front_approach(fixture_id),
        )
        self._sync_held_object(robot_idx)
        return ToolResult(
            tool_name="navigate_to_fixture",
            success=placed,
            details={
                "fixture_id": requested_fixture_id,
                "resolved_workspace_fixture_id": fixture_id,
                "robot_idx": robot_idx,
            },
        )

    def _normalize_robot_facing_for_stove_workstation(
        self,
        robot_idx: int,
        fixture_id: str,
    ) -> None:
        """Back robot away from stove and orient toward it during init spawn only."""
        scene_fixtures = (self.get_scene_description().get("fixtures") or {})
        fixture_info = scene_fixtures.get(fixture_id)
        if not isinstance(fixture_info, dict):
            return
        fixture_type = str(fixture_info.get("fixture_type") or "").lower()
        if "stove" not in fixture_type:
            return
        focus_fixture = self.runner._fixtures.get(fixture_id)
        setter = getattr(self.runner, "_set_robot_pose", None)
        if focus_fixture is None or not callable(setter):
            return
        get_pos = getattr(self.runner, "_get_robot_position", None)
        is_valid = getattr(self.runner, "_is_valid_robot_position", None)
        if not callable(get_pos):
            return
        robot_xy = np.asarray(get_pos(robot_idx)[:2], dtype=float)
        focus_xy = np.asarray(focus_fixture.pos[:2], dtype=float)

        num_robots = len(getattr(self.env, "robots", []))
        anchor_robot_idx = 1 if num_robots >= 2 else None
        placed_xy = robot_xy

        # Formation rule for stove init: place the non-anchor robot directly
        # behind robot1 along the "away from stove" direction.
        if (
            isinstance(anchor_robot_idx, int)
            and robot_idx != anchor_robot_idx
            and callable(get_pos)
        ):
            anchor_xy = np.asarray(get_pos(anchor_robot_idx)[:2], dtype=float)
            away = anchor_xy - focus_xy
            if float(np.linalg.norm(away)) <= 1e-6:
                away = robot_xy - focus_xy
            if float(np.linalg.norm(away)) > 1e-9:
                away = away / float(np.linalg.norm(away))
                lateral = np.array([-away[1], away[0]], dtype=float)
                desired_gap = 0.46
                min_anchor_sep = 0.38
                # Keep the primary target "directly behind", but allow tiny
                # lateral nudges as validity fallback.
                for lateral_offset in (0.0, 0.08, -0.08, 0.14, -0.14):
                    candidate_xy = (
                        anchor_xy
                        + away * desired_gap
                        + lateral * float(lateral_offset)
                    )
                    if callable(is_valid) and not bool(is_valid(candidate_xy)):
                        continue
                    if float(np.linalg.norm(candidate_xy - anchor_xy)) < min_anchor_sep:
                        continue
                    placed_xy = candidate_xy
                    break
        else:
            # Single-robot fallback: move slightly away from the stove center.
            outward = robot_xy - focus_xy
            if float(np.linalg.norm(outward)) <= 1e-6:
                inward_axis = np.asarray(
                    self._fixture_inward_world_axis(fixture_id, "y")[:2],
                    dtype=float,
                )
                outward = -inward_axis
            outward_norm = float(np.linalg.norm(outward))
            if outward_norm <= 1e-9:
                return
            outward = outward / outward_norm
            for retreat in (0.34, 0.26, 0.18, 0.10, 0.06):
                candidate_xy = robot_xy + outward * float(retreat)
                if callable(is_valid) and not bool(is_valid(candidate_xy)):
                    continue
                placed_xy = candidate_xy
                break

        facing = focus_xy - placed_xy
        if float(np.linalg.norm(facing)) <= 1e-6:
            return
        yaw = float(np.arctan2(facing[1], facing[0]))
        setter(robot_idx, placed_xy, yaw)

    def open_hinged_part(
        self,
        target_id: str,
        part_id: str,
        robot_idx: int = 0,
    ) -> ToolResult:
        fixture = self._require_fixture(target_id)
        if not self._robot_near_fixture(robot_idx, target_id):
            moved = self._move_robot_near_fixture_with_retries(
                robot_idx,
                target_id,
                require_front=True,
            )
            if moved:
                self._sync_held_object(robot_idx)
        if isinstance(fixture, Blender) and part_id == "lid":
            return ToolResult(
                "open_hinged_part", True, {"target_id": target_id, "part_id": part_id}
            )
        if part_id == "hinged" and hasattr(fixture, "open_door"):
            fixture.open_door(env=self.env)
            self._settle_scene()
        else:
            joint_name = self._resolve_joint_name(fixture, part_id)
            self._set_named_joint(fixture, joint_name, 1.0)
        return ToolResult(
            "open_hinged_part", True, {"target_id": target_id, "part_id": part_id}
        )

    def close_hinged_part(
        self,
        target_id: str,
        part_id: str,
        robot_idx: int = 0,
    ) -> ToolResult:
        fixture = self._require_fixture(target_id)
        if not self._robot_near_fixture(robot_idx, target_id):
            moved = self._move_robot_near_fixture_with_retries(
                robot_idx,
                target_id,
                require_front=True,
            )
            if moved:
                self._sync_held_object(robot_idx)
        if isinstance(fixture, Blender) and part_id == "lid":
            return ToolResult(
                "close_hinged_part", True, {"target_id": target_id, "part_id": part_id}
            )
        if part_id == "hinged" and hasattr(fixture, "close_door"):
            fixture.close_door(env=self.env)
            self._settle_scene()
        else:
            joint_name = self._resolve_joint_name(fixture, part_id)
            self._set_named_joint(fixture, joint_name, 0.0)
        return ToolResult(
            "close_hinged_part", True, {"target_id": target_id, "part_id": part_id}
        )

    def open_sliding_part(
        self,
        target_id: str,
        part_id: str,
        robot_idx: int = 0,
    ) -> ToolResult:
        fixture = self._require_fixture(target_id)
        # Always attempt a placement refresh for sliding parts (drawers). The
        # generic near-check can be too permissive in corner layouts and skip
        # needed movement.
        moved = self._move_robot_near_fixture_with_retries(
            robot_idx,
            target_id,
            require_front=False,
        )
        if moved:
            self._sync_held_object(robot_idx)
        before_aabb = get_fixture_aabb(fixture)
        joint_name = self._resolve_joint_name(
            fixture, part_id if part_id != "sliding" else "slide"
        )
        before_qpos = self._joint_qpos_scalar(joint_name)
        self._set_named_joint(
            fixture,
            joint_name,
            self._sliding_joint_open_fraction(joint_name, part_id=part_id),
        )
        after_qpos = self._joint_qpos_scalar(joint_name)
        after_aabb = get_fixture_aabb(fixture)
        refreshed_children = self._refresh_fixture_supported_children(
            target_id,
            anchor_aabb=before_aabb,
        )
        drawer_delta = self._drawer_slide_delta_from_aabbs(before_aabb, after_aabb)
        delta_source = "aabb"
        if float(np.linalg.norm(np.asarray(drawer_delta, dtype=float)[:2])) <= 1e-6:
            qpos_delta = self._sliding_joint_world_delta_from_qpos(
                joint_name,
                before_qpos=before_qpos,
                after_qpos=after_qpos,
            )
            if float(np.linalg.norm(np.asarray(qpos_delta, dtype=float)[:2])) > 1e-6:
                drawer_delta = qpos_delta
                delta_source = "joint_qpos"
        moved_children = self._translate_supported_fixture_children(target_id, drawer_delta)
        self._update_drawer_transient_obstacle(
            target_id,
            part_id,
            before_aabb=before_aabb,
            after_aabb=after_aabb,
            is_open=True,
        )
        self._retreat_and_face_open_drawer(
            target_id=target_id,
            robot_idx=robot_idx,
            drawer_delta=drawer_delta,
            drawer_aabb=after_aabb,
        )
        recent_opened_sliding_fixture = getattr(
            self,
            "_recent_opened_sliding_fixture",
            None,
        )
        if not isinstance(recent_opened_sliding_fixture, dict):
            self._recent_opened_sliding_fixture = {}
            recent_opened_sliding_fixture = self._recent_opened_sliding_fixture
        recent_opened_sliding_fixture[robot_idx] = target_id
        occupancy_grid = getattr(self.runner, "_occupancy_grid", None)
        if occupancy_grid is not None:
            robot_pos = self.runner._get_robot_position(robot_idx)[:2]
            if not occupancy_grid.is_free(robot_pos):
                self.runner._move_robot_near_fixture(
                    robot_idx,
                    target_id,
                    require_front=True,
                )
                self._sync_held_object(robot_idx)
        return ToolResult(
            "open_sliding_part",
            True,
            {
                "target_id": target_id,
                "part_id": part_id,
                "joint_name": joint_name,
                "drawer_delta": np.asarray(drawer_delta, dtype=float).tolist(),
                "drawer_delta_source": delta_source,
                "joint_qpos_before": before_qpos,
                "joint_qpos_after": after_qpos,
                "refreshed_supported_children": refreshed_children,
                "moved_supported_children": moved_children,
            },
        )

    def close_sliding_part(
        self,
        target_id: str,
        part_id: str,
        robot_idx: int = 0,
    ) -> ToolResult:
        fixture = self._require_fixture(target_id)
        if not self._robot_near_fixture(robot_idx, target_id):
            moved = self._move_robot_near_fixture_with_retries(
                robot_idx,
                target_id,
                require_front=True,
            )
            if moved:
                self._sync_held_object(robot_idx)
        before_aabb = get_fixture_aabb(fixture)
        joint_name = self._resolve_joint_name(
            fixture, part_id if part_id != "sliding" else "slide"
        )
        before_qpos = self._joint_qpos_scalar(joint_name)
        self._set_named_joint(fixture, joint_name, 0.0)
        after_qpos = self._joint_qpos_scalar(joint_name)
        after_aabb = get_fixture_aabb(fixture)
        self._refresh_fixture_supported_children(target_id, anchor_aabb=before_aabb)
        drawer_delta = self._drawer_slide_delta_from_aabbs(before_aabb, after_aabb)
        if float(np.linalg.norm(np.asarray(drawer_delta, dtype=float)[:2])) <= 1e-6:
            qpos_delta = self._sliding_joint_world_delta_from_qpos(
                joint_name,
                before_qpos=before_qpos,
                after_qpos=after_qpos,
            )
            if float(np.linalg.norm(np.asarray(qpos_delta, dtype=float)[:2])) > 1e-6:
                drawer_delta = qpos_delta
        self._translate_supported_fixture_children(target_id, drawer_delta)
        self._update_drawer_transient_obstacle(
            target_id,
            part_id,
            before_aabb=before_aabb,
            after_aabb=after_aabb,
            is_open=False,
        )
        return ToolResult(
            "close_sliding_part", True, {"target_id": target_id, "part_id": part_id}
        )

    def _retreat_and_face_open_drawer(
        self,
        target_id: str,
        robot_idx: int,
        drawer_delta: np.ndarray | Sequence[float] | None,
        drawer_aabb: tuple[np.ndarray, np.ndarray] | None,
    ) -> None:
        """Move robot slightly away from opened drawer and face it."""
        fixture = self.runner._fixtures.get(target_id)
        if fixture is None:
            return
        setter = getattr(self.runner, "_set_robot_pose", None)
        checker = getattr(self.runner, "_is_valid_robot_position", None)
        if not callable(setter):
            return

        center_xy = np.asarray(fixture.pos[:2], dtype=float)
        if drawer_aabb is not None:
            center_xy = np.asarray(
                0.5 * (np.asarray(drawer_aabb[0])[:2] + np.asarray(drawer_aabb[1])[:2]),
                dtype=float,
            )
        current_xy = np.asarray(self.runner._get_robot_position(robot_idx)[:2], dtype=float)

        radial = current_xy - center_xy
        retreat_dir = None
        radial_norm = float(np.linalg.norm(radial))
        if radial_norm > 1e-6:
            retreat_dir = radial / radial_norm
        else:
            delta = np.asarray(drawer_delta, dtype=float).reshape(-1) if drawer_delta is not None else np.zeros(2)
            delta = delta[:2] if delta.size >= 2 else np.zeros(2)
            delta_norm = float(np.linalg.norm(delta))
            if delta_norm > 1e-6:
                retreat_dir = delta / delta_norm
        if retreat_dir is None:
            retreat_dir = np.array([1.0, 0.0], dtype=float)

        def _inside_drawer_bbox(xy: np.ndarray, margin: float = 0.16) -> bool:
            if drawer_aabb is None:
                return False
            min_xy = np.asarray(drawer_aabb[0][:2], dtype=float) - float(margin)
            max_xy = np.asarray(drawer_aabb[1][:2], dtype=float) + float(margin)
            return bool(min_xy[0] <= xy[0] <= max_xy[0] and min_xy[1] <= xy[1] <= max_xy[1])

        grid = getattr(self.runner, "_occupancy_grid", None)
        placed = False
        for step in (0.30, 0.45, 0.60):
            candidate = current_xy + retreat_dir * step
            if _inside_drawer_bbox(candidate):
                continue
            if callable(checker):
                try:
                    if not bool(checker(candidate)):
                        continue
                except Exception:
                    pass
            if grid is not None:
                try:
                    if hasattr(grid, "is_standable") and not bool(grid.is_standable(candidate)):
                        continue
                except Exception:
                    pass
            facing = center_xy - candidate
            if float(np.linalg.norm(facing)) <= 1e-6:
                continue
            yaw = float(np.arctan2(facing[1], facing[0]))
            setter(robot_idx, candidate, yaw)
            self._sync_held_object(robot_idx)
            placed = True
            break

        if not placed:
            # At minimum keep the current XY and orient toward the opened drawer.
            facing = center_xy - current_xy
            if float(np.linalg.norm(facing)) > 1e-6:
                yaw = float(np.arctan2(facing[1], facing[0]))
                setter(robot_idx, current_xy, yaw)
                self._sync_held_object(robot_idx)

    def _robot_near_fixture(
        self, robot_idx: int, fixture_id: str, threshold: float = 1.5
    ) -> bool:
        """Return True if the robot is ready to interact with the fixture."""
        fixture_id = self._canonical_navigation_fixture_id(fixture_id)
        fxtr = self.runner._fixtures.get(fixture_id)
        if fxtr is None:
            return False
        pos = self.runner._get_robot_position(robot_idx)[:2]
        if _is_approach_center(fxtr):
            target_xy = self.runner._get_fixture_front_target_xy(fixture_id)
            # Countertop appliance mesh rotation is not a reliable declaration
            # of its usable aisle-facing side. Placement already derives that
            # side from occupancy; readiness must use the identical face.
            front_face = None
            if _is_countertop_appliance(fxtr) and not isinstance(
                fxtr, CoffeeMachine
            ):
                front_face = (
                    self.runner._occupancy_grid.preferred_reachable_approach_face(
                        fxtr
                    )
                )
            metrics = get_front_alignment_metrics(
                fxtr,
                pos,
                target_xy=target_xy,
                front_face=front_face,
            )
            if metrics is None:
                return False
            fixture_center = np.asarray(fxtr.pos[:2], dtype=float)
            return bool(
                metrics["on_front_face"]
                and metrics["within_span"]
                and metrics["lateral_offset"] <= MAX_FRONT_WORKING_LATERAL_OFFSET
                and _FRONT_READY_MIN_GAP <= metrics["front_gap"] <= _FRONT_READY_MAX_GAP
                and float(np.linalg.norm(pos - fixture_center))
                <= _FRONT_READY_MAX_CENTER_DISTANCE
            )
        if self._fixture_is_drawer(fixture_id):
            # Drawers are better handled with bbox proximity than center distance.
            aabb = get_fixture_aabb(fxtr)
            if aabb is not None:
                min_xy = np.asarray(aabb[0][:2], dtype=float)
                max_xy = np.asarray(aabb[1][:2], dtype=float)
                inside = bool(
                    min_xy[0] <= pos[0] <= max_xy[0]
                    and min_xy[1] <= pos[1] <= max_xy[1]
                )
                dx = max(min_xy[0] - pos[0], 0.0, pos[0] - max_xy[0])
                dy = max(min_xy[1] - pos[1], 0.0, pos[1] - max_xy[1])
                outside_dist = float(np.hypot(dx, dy))
                # Near enough when outside but close to the drawer envelope.
                return (not inside) and outside_dist <= 0.95
        fxtr_pos = np.asarray(fxtr.pos[:2], dtype=float)
        return float(np.linalg.norm(pos - fxtr_pos)) < threshold

    def _safe_compute_object_target_pos(
        self,
        support_id: str,
        object_id: str,
        preferred_xy: np.ndarray | None = None,
        support_site_id: str | None = None,
        ignored_fixture_ids: set[str] | None = None,
        ignored_object_ids: set[str] | None = None,
    ) -> np.ndarray:
        """Compute a placement target using only validated fixture candidates."""
        support_fxtr = self.runner._fixtures[support_id]
        ignored_object_ids = set() if ignored_object_ids is None else set(ignored_object_ids)
        ignored_object_ids.update(self._iter_supported_descendants(object_id))
        ignored_object_ids.update(self._find_contained_objects(object_id))
        try:
            return self.runner._compute_object_target_pos(
                support_fxtr,
                object_id=object_id,
                preferred_xy=preferred_xy,
                ignored_object_ids=ignored_object_ids,
                ignored_fixture_ids=ignored_fixture_ids,
                site_id=support_site_id,
            )
        except RuntimeError:
            if support_site_id is not None:
                if preferred_xy is None:
                    if len(self.get_support_sites(support_id) or ()) != 1:
                        raise
                    return self.runner._compute_object_target_pos(
                        support_fxtr,
                        object_id=object_id,
                        preferred_xy=None,
                        ignored_object_ids=ignored_object_ids,
                        ignored_fixture_ids=ignored_fixture_ids,
                        site_id=None,
                    )
                try:
                    return self.runner._compute_object_target_pos(
                        support_fxtr,
                        object_id=object_id,
                        preferred_xy=None,
                        ignored_object_ids=ignored_object_ids,
                        ignored_fixture_ids=ignored_fixture_ids,
                        site_id=support_site_id,
                    )
                except RuntimeError:
                    if len(self.get_support_sites(support_id) or ()) != 1:
                        raise
                    return self.runner._compute_object_target_pos(
                        support_fxtr,
                        object_id=object_id,
                        preferred_xy=preferred_xy,
                        ignored_object_ids=ignored_object_ids,
                        ignored_fixture_ids=ignored_fixture_ids,
                        site_id=None,
                    )
            if preferred_xy is None:
                raise
            return self.runner._compute_object_target_pos(
                support_fxtr,
                object_id=object_id,
                preferred_xy=None,
                ignored_object_ids=ignored_object_ids,
                ignored_fixture_ids=ignored_fixture_ids,
                site_id=None,
            )

    def _place_object_on_fixture(
        self,
        object_id: str,
        fixture_id: str,
        *,
        preferred_xy: np.ndarray | None = None,
        target_site_id: str | None = None,
        settle: bool = True,
    ) -> np.ndarray:
        target_site_id = self._normalize_target_site_id_for_placement(
            fixture_id,
            target_site_id,
        )
        if (
            target_site_id is None
            and self._fixture_requires_explicit_site(fixture_id)
        ):
            target_site_id = self._default_support_site_for_unspecified_fixture(
                fixture_id,
                incoming_object_id=object_id,
            )
        if preferred_xy is None and self._fixture_placement_semantics(fixture_id) == "surface":
            preferred_xy = self._default_fixture_surface_preference(
                fixture_id,
                object_id,
            )
        fixture_type = (self._get_fixture_type_name(fixture_id) or "").lower()
        ignored_fixture_ids = self._ignored_fixture_ids_for_placement(
            fixture_id,
            target_site_id,
        )
        try:
            target_pos, target_quat = self._compute_object_target_pose(
                fixture_id,
                object_id,
                preferred_xy=preferred_xy,
                support_site_id=target_site_id,
                ignored_fixture_ids=ignored_fixture_ids,
                allow_yaw_search=(
                    isinstance(target_site_id, str)
                    or fixture_type in _FRONT_BIASED_INTERIOR_FIXTURE_TYPES
                ),
            )
        except RuntimeError:
            if not isinstance(target_site_id, str):
                raise
            slot_target = self._find_fixture_site_slot_target(
                fixture_id,
                target_site_id,
                object_id=object_id,
                current_preferred_xy=preferred_xy,
                ignored_fixture_ids=ignored_fixture_ids,
            )
            if slot_target is None:
                raise
            target_pos, target_quat = slot_target
        self._set_object_pose(object_id, target_pos, target_quat)
        self._set_support_parent(object_id, fixture_id)
        self.runner._set_object_location(object_id, fixture_id)
        if settle:
            self._settle_scene(steps=_PLACEMENT_SETTLE_STEPS)
        return target_pos

    def _compute_object_target_pose(
        self,
        fixture_id: str,
        object_id: str,
        *,
        preferred_xy: np.ndarray | None = None,
        support_site_id: str | None = None,
        ignored_fixture_ids: set[str] | None = None,
        ignored_object_ids: set[str] | None = None,
        allow_yaw_search: bool = False,
        preferred_long_axis: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        original_pos, original_quat_wxyz = self._get_object_pose(object_id)
        fixture_type = (self._get_fixture_type_name(fixture_id) or "").lower()
        fixture_semantics = self._fixture_placement_semantics(fixture_id)
        orientation_candidates: list[np.ndarray] = [np.asarray(original_quat_wxyz, dtype=float)]
        if preferred_long_axis is None and (
            isinstance(support_site_id, str)
            or fixture_type in _FRONT_BIASED_INTERIOR_FIXTURE_TYPES
            or (
                fixture_semantics == "surface"
                and self._object_has_directional_long_axis(object_id)
            )
        ):
            preferred_long_axis = self._fixture_major_axis(
                fixture_id,
                site_id=support_site_id,
            )
        if isinstance(preferred_long_axis, str):
            aligned_quat = self._aligned_flat_quat_for_world_axis(
                object_id,
                self._fixture_world_axis(fixture_id, preferred_long_axis),
            )
            if aligned_quat is not None:
                orientation_candidates.insert(0, np.asarray(aligned_quat, dtype=float))
        if allow_yaw_search:
            expanded_candidates: list[np.ndarray] = []
            for base_quat_wxyz in orientation_candidates:
                expanded_candidates.append(np.asarray(base_quat_wxyz, dtype=float))
                base_quat_xyzw = T.convert_quat(np.asarray(base_quat_wxyz, dtype=float), to="xyzw")
                for angle in (30, -30, 60, -60, 90, -90, 120, -120, 150, -150, 180):
                    yaw_quat_xyzw = T.axisangle2quat(
                        np.array([0.0, 0.0, np.deg2rad(angle)], dtype=float)
                    )
                    candidate_quat_xyzw = T.quat_multiply(yaw_quat_xyzw, base_quat_xyzw)
                    expanded_candidates.append(
                        T.convert_quat(candidate_quat_xyzw, to="wxyz")
                    )
            seen_orientation_keys: set[tuple[float, float, float, float]] = set()
            deduped_candidates: list[np.ndarray] = []
            for candidate in expanded_candidates:
                key = tuple(np.round(np.asarray(candidate, dtype=float), 5))
                if key in seen_orientation_keys:
                    continue
                seen_orientation_keys.add(key)
                deduped_candidates.append(np.asarray(candidate, dtype=float))
            orientation_candidates = deduped_candidates

        last_error: RuntimeError | None = None
        for candidate_quat_wxyz in orientation_candidates:
            self._set_object_pose(object_id, original_pos, candidate_quat_wxyz)
            try:
                target_pos = self._safe_compute_object_target_pos(
                    fixture_id,
                    object_id,
                    preferred_xy=preferred_xy,
                    support_site_id=support_site_id,
                    ignored_fixture_ids=ignored_fixture_ids,
                    ignored_object_ids=ignored_object_ids,
                )
                return target_pos, candidate_quat_wxyz
            except RuntimeError as exc:
                last_error = exc

        self._set_object_pose(object_id, original_pos, original_quat_wxyz)
        if last_error is None:
            raise RuntimeError(
                f"No collision-free placement candidate found for {object_id!r} on {fixture_id!r}."
            )
        raise last_error

    def _objects_on_fixture_site(
        self,
        fixture_id: str,
        *,
        site_id: str | None = None,
        exclude: set[str] | None = None,
    ) -> list[str]:
        exclude = set() if exclude is None else set(exclude)
        object_ids: list[str] = []
        for candidate_id in self.env.objects:
            if candidate_id in exclude:
                continue
            if self._held_by_robot(candidate_id) is not None:
                continue
            if self._get_scene_object_location(candidate_id) != fixture_id:
                continue
            if self._current_support_object(candidate_id) is not None:
                continue
            if site_id is not None:
                try:
                    if self._infer_object_support_site(candidate_id, fixture_id) != site_id:
                        continue
                except Exception:
                    continue
            object_ids.append(candidate_id)
        return object_ids

    def _find_fixture_site_slot_target(
        self,
        fixture_id: str,
        site_id: str,
        *,
        object_id: str,
        current_preferred_xy: np.ndarray | None = None,
        ignored_fixture_ids: set[str] | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        existing_object_ids = self._objects_on_fixture_site(
            fixture_id,
            site_id=site_id,
            exclude={object_id},
        )
        slot_positions = self._fixture_site_slot_positions(
            fixture_id,
            site_id,
            count=len(existing_object_ids) + 1,
        )
        if not slot_positions:
            return None

        existing_positions = [
            self._get_object_pose(existing_object_id)[0][:2].copy()
            for existing_object_id in existing_object_ids
        ]
        center_xy = (
            np.asarray(current_preferred_xy, dtype=float)[:2]
            if current_preferred_xy is not None
            else self._preferred_xy_for_fixture_target(fixture_id, site_id=site_id)
        )
        if center_xy is None:
            center_xy = slot_positions[0]
        ranked_slot_positions = sorted(
            slot_positions,
            key=lambda position: (
                -min(
                    (
                        float(
                            np.linalg.norm(
                                np.asarray(position, dtype=float)[:2]
                                - np.asarray(existing_position, dtype=float)[:2]
                            )
                        )
                        for existing_position in existing_positions
                    ),
                    default=float("inf"),
                ),
                float(
                    np.linalg.norm(
                        np.asarray(position, dtype=float)[:2]
                        - np.asarray(center_xy, dtype=float)[:2]
                    )
                ),
            ),
        )
        for preferred_xy in ranked_slot_positions:
            try:
                return self._compute_object_target_pose(
                    fixture_id,
                    object_id,
                    preferred_xy=np.asarray(preferred_xy, dtype=float),
                    support_site_id=site_id,
                    ignored_fixture_ids=ignored_fixture_ids,
                    allow_yaw_search=True,
                )
            except RuntimeError:
                continue
        return None

    def _object_z_lift(self, object_id: str) -> float:
        obj = self._require_object(object_id)
        return float(-obj.bottom_offset[-1])

    def _compute_region_center_target_pos(
        self,
        fixture_id: str,
        object_id: str,
        *,
        site_id: str,
    ) -> np.ndarray:
        raw_site_id = self._resolve_fixture_site_id(fixture_id, site_id)
        reset_regions = self._get_fixture_reset_regions(fixture_id)
        region = reset_regions.get(raw_site_id)
        if not isinstance(region, dict):
            raise RuntimeError(
                f"Fixture {fixture_id!r} does not expose region {site_id!r}."
            )
        offset = np.asarray(region.get("offset", (0.0, 0.0, 0.0)), dtype=float)
        fixture = self._require_fixture(fixture_id)
        target_pos = self.runner._fixture_local_to_world(fixture, offset)
        target_pos[2] += self._object_z_lift(object_id)
        return np.asarray(target_pos, dtype=float)

    def _fixture_clearance_radius(self, fixture_id: str) -> float:
        """Return a radius around the fixture where teammates likely block access."""
        fixture = self.runner._fixtures.get(fixture_id)
        if fixture is None:
            return 1.5
        aabb = get_fixture_aabb(fixture)
        if aabb is None:
            return 1.5
        half_extent = 0.5 * np.linalg.norm(aabb[1] - aabb[0])
        return max(1.5, float(half_extent + 0.8))

    def _clear_fixture_blockers(self, robot_idx: int, fixture_id: str) -> bool:
        """Move nearby teammate robots away from a fixture before retrying placement."""
        fixture = self.runner._fixtures.get(fixture_id)
        if fixture is None:
            return False
        fixture_pos = np.asarray(fixture.pos[:2], dtype=float)
        clearance_radius = self._fixture_clearance_radius(fixture_id)
        moved_any = False
        for other_idx in range(self.runner._num_robots):
            if other_idx == robot_idx:
                continue
            other_pos = self.runner._get_robot_position(other_idx)[:2]
            if float(np.linalg.norm(other_pos - fixture_pos)) >= clearance_radius:
                continue
            self.runner.give_space(other_idx, fixture_id)
            self._sync_held_object(other_idx)
            moved_any = True
        return moved_any

    def _preferred_xy_for_support_site(
        self,
        fixture_id: str,
        support_site_id: str,
    ) -> np.ndarray | None:
        raw_site_id = self._resolve_fixture_site_id(fixture_id, support_site_id)
        return self._preferred_xy_for_fixture_target(
            fixture_id,
            site_id=raw_site_id,
        )

    def _ignored_fixture_ids_for_placement(
        self,
        fixture_id: str,
        support_site_id: str | None = None,
    ) -> set[str]:
        ignored_fixture_ids: set[str] = set()
        scene_fixture = (self.get_scene_description().get("fixtures") or {}).get(
            fixture_id,
            {},
        )
        parent_fixture_id = scene_fixture.get("parent_fixture")
        if isinstance(parent_fixture_id, str):
            ignored_fixture_ids.add(parent_fixture_id)

        for other_fixture_id in self.runner._fixtures:
            other_fixture_id = str(other_fixture_id)
            if self._fixture_id_is_structural_obstacle(other_fixture_id):
                ignored_fixture_ids.add(other_fixture_id)

        site_tokens: set[str] = set()
        if isinstance(support_site_id, str):
            site_tokens = {
                token
                for token in _SITE_TOKEN_RE.findall(support_site_id.lower())
                if token
            }

        if (
            self._fixture_placement_semantics(fixture_id) == "receptacle"
            or bool(site_tokens & _INTERIOR_SITE_TOKENS)
        ):
            for other_fixture_id in self.runner._fixtures:
                other_fixture_id = str(other_fixture_id)
                if self._fixture_id_is_structural_obstacle(other_fixture_id):
                    ignored_fixture_ids.add(str(other_fixture_id))
        return ignored_fixture_ids

    def _resolve_support_target(
        self,
        support_id: str,
        support_site_id: str | None = None,
    ) -> tuple[str, str | None]:
        if support_id in self.runner._fixtures:
            return support_id, self._resolve_fixture_site_id(support_id, support_site_id)

        matching_targets: list[tuple[str, str]] = []
        for fixture_id in self.runner._fixtures:
            try:
                resolved_site_id = self._resolve_fixture_site_id(fixture_id, support_id)
            except ValueError:
                continue
            if isinstance(resolved_site_id, str):
                matching_targets.append((fixture_id, resolved_site_id))

        if len(matching_targets) == 1:
            return matching_targets[0]
        if matching_targets:
            scored_targets = [
                (
                    self._score_support_target_site_match(support_id, fixture_id),
                    fixture_id,
                    site_id,
                )
                for fixture_id, site_id in matching_targets
            ]
            best_score = max(score for score, _fixture_id, _site_id in scored_targets)
            if best_score > 0:
                best_targets = [
                    (fixture_id, site_id)
                    for score, fixture_id, site_id in scored_targets
                    if score == best_score
                ]
                if len(best_targets) == 1:
                    return best_targets[0]
            matching_fixture_ids = sorted(
                fixture_id for fixture_id, _support_site_id in matching_targets
            )
            raise ValueError(
                f"Ambiguous support site {support_id!r}; matches fixtures {matching_fixture_ids}"
            )
        raise ValueError(f"Unknown support fixture/site {support_id!r}")

    def _move_robot_near_fixture_with_retries(
        self,
        robot_idx: int,
        fixture_id: str,
        ref_object_id: str | None = None,
        ref_pos_override: np.ndarray | None = None,
        require_front: bool = False,
    ) -> bool:
        """Place a robot near a fixture, clearing teammate blockers if needed."""
        if self._fixture_is_cabinet(fixture_id):
            require_front = False
        fixture_id = self._canonical_navigation_fixture_id(fixture_id)
        if self._robot_at_exclusive_child_of_parent(robot_idx, fixture_id):
            return True
        placed = self.runner._move_robot_near_fixture(
            robot_idx,
            fixture_id,
            ref_object_id=ref_object_id,
            ref_pos_override=ref_pos_override,
            require_front=require_front,
        )
        if placed or not require_front:
            return placed
        if not self._clear_fixture_blockers(robot_idx, fixture_id):
            return placed
        return self.runner._move_robot_near_fixture(
            robot_idx,
            fixture_id,
            ref_object_id=ref_object_id,
            ref_pos_override=ref_pos_override,
            require_front=require_front,
        )

    def pick_up_object(
        self,
        object_id: str,
        source_id: str,
        source_site_id: str | None = None,
        robot_idx: int = 0,
    ) -> ToolResult:
        self._require_object(object_id)
        actual_fixture_id = self._get_scene_object_location(object_id)
        try:
            source_fixture_id, source_site_id, source_object_id = (
                self._resolve_pick_source_target(source_id, source_site_id)
            )
        except ValueError:
            fallback_fixture_id = actual_fixture_id
            if not isinstance(fallback_fixture_id, str):
                for candidate_object_id in (object_id, source_id):
                    if candidate_object_id not in self.env.objects:
                        continue
                    try:
                        candidate_pos, _candidate_quat = self._get_object_pose(
                            candidate_object_id
                        )
                    except Exception:
                        continue
                    inferred_fixture_id = self.runner._find_object_fixture(candidate_pos)
                    if inferred_fixture_id in self.runner._fixtures:
                        fallback_fixture_id = inferred_fixture_id
                        break
            if source_id not in self.env.objects or not isinstance(fallback_fixture_id, str):
                raise
            source_fixture_id = fallback_fixture_id
            source_object_id = source_id
            if source_site_id is None:
                source_site_id = self._infer_object_support_site(
                    object_id,
                    source_fixture_id,
                )
        if (
            isinstance(actual_fixture_id, str)
            and self._fixture_is_drawer(actual_fixture_id)
            and self._fixture_is_drawer(source_fixture_id)
            and actual_fixture_id != source_fixture_id
        ):
            # Keep original drawer selection when symbolic source is generic.
            source_fixture_id = actual_fixture_id
            if source_site_id is None:
                source_site_id = self._infer_object_support_site(object_id, source_fixture_id)

        current_holder = self._held_by_robot(object_id)
        if current_holder is not None and current_holder != robot_idx:
            raise ValueError(
                f"Object {object_id!r} is already held by robot {current_holder}"
            )

        recent_opened_sliding_fixture = getattr(
            self,
            "_recent_opened_sliding_fixture",
            None,
        )
        if not isinstance(recent_opened_sliding_fixture, dict):
            self._recent_opened_sliding_fixture = {}
            recent_opened_sliding_fixture = self._recent_opened_sliding_fixture
        recent_opened_drawer = recent_opened_sliding_fixture.get(robot_idx)
        skip_renav_after_drawer_open = bool(
            isinstance(recent_opened_drawer, str)
            and recent_opened_drawer == source_fixture_id
            and self._fixture_is_drawer(source_fixture_id)
        )
        # Single-use behavior: keep context only for the immediate matching pickup.
        if skip_renav_after_drawer_open or isinstance(recent_opened_drawer, str):
            recent_opened_sliding_fixture.pop(robot_idx, None)

        # Skip navigation only if the robot is already in a usable working
        # pose for the fixture.
        # Match navigate_to_fixture: countertop appliances such as stoves
        # target the appliance center but do not require a front-face pose.
        requires_front_approach = (
            self._fixture_requires_front_approach(source_fixture_id)
            or self._surface_fixture_prefers_front_approach(source_fixture_id)
        )
        if (
            not skip_renav_after_drawer_open
            and (
                requires_front_approach
                or not self._robot_near_fixture(robot_idx, source_fixture_id)
            )
        ):
            source_fxtr = self.runner._fixtures[source_fixture_id]
            if _is_approach_center(source_fxtr):
                # Approach-center fixtures use the same front constraint as
                # explicit navigation to that fixture.
                moved = self._move_robot_near_fixture_with_retries(
                    robot_idx,
                    source_fixture_id,
                    require_front=requires_front_approach,
                )
            else:
                # Surface — pre-compute object position, pick closest face
                ref_pos = self._preferred_xy_for_fixture_target(
                    source_fixture_id,
                    site_id=source_site_id,
                )
                if ref_pos is None:
                    obj_pos, _ = self._get_object_pose(object_id)
                    ref_pos = obj_pos[:2].copy()
                moved = self._move_robot_near_fixture_with_retries(
                    robot_idx,
                    source_fixture_id,
                    ref_pos_override=ref_pos,
                    require_front=requires_front_approach,
                )
            if not moved:
                return ToolResult(
                    "pick_up_object",
                    False,
                    {
                        "object_id": object_id,
                        "source_id": source_id,
                        "resolved_source_fixture_id": source_fixture_id,
                        "resolved_source_site_id": source_site_id,
                        "resolved_source_object_id": source_object_id,
                        "robot_idx": robot_idx,
                    },
                )
            self._sync_held_object(robot_idx)
        obj_pos_before, _ = self._get_object_pose(object_id)
        if _sim_tool_debug_enabled():
            print(
                f"[pick_up] robot{robot_idx} picking {object_id} from {source_id}: "
                f"obj_before=({obj_pos_before[0]:.3f}, {obj_pos_before[1]:.3f}, {obj_pos_before[2]:.3f})"
            )
        self._held_objects[robot_idx] = object_id
        self._held_object_offsets[robot_idx] = self._held_pose_offset(object_id)
        self._set_support_parent(object_id, f"held_by_robot_{robot_idx}")
        self._sync_held_object(robot_idx)
        obj_pos_after, _ = self._get_object_pose(object_id)
        if _sim_tool_debug_enabled():
            print(
                f"[pick_up] after sync: obj=({obj_pos_after[0]:.3f}, {obj_pos_after[1]:.3f}, {obj_pos_after[2]:.3f})"
            )
        return ToolResult(
            "pick_up_object",
            True,
            {
                "object_id": object_id,
                "source_id": source_id,
                "resolved_source_fixture_id": source_fixture_id,
                "resolved_source_site_id": (
                    self._raw_support_site_to_external(source_site_id)
                    if isinstance(source_site_id, str)
                    else None
                ),
                "resolved_source_object_id": source_object_id,
                "robot_idx": robot_idx,
            },
        )

    def place_on_surface(
        self,
        object_id: str,
        support_id: str | None = None,
        target_id: str | None = None,
        target_site_id: str | None = None,
        relative_position: str | None = None,
        robot_idx: int = 0,
    ) -> ToolResult:
        self._require_object(object_id)
        holder = self._held_by_robot(object_id)
        if holder not in {None, robot_idx}:
            raise ValueError(f"Object {object_id!r} is held by robot {holder}")

        target_name = target_id or support_id
        if not isinstance(target_name, str):
            raise ValueError("place_on_surface requires target_id or support_id.")
        support_fixture_id, support_site_id = self._resolve_support_target(
            target_name,
            target_site_id,
        )
        self._require_explicit_site_if_needed(
            support_fixture_id,
            support_site_id,
            action_name="place_on_surface",
        )
        if self._fixture_placement_semantics(support_fixture_id) != "surface":
            raise ValueError(
                f"Fixture {support_fixture_id!r} has receptacle/interior semantics; "
                "use place_in_receptacle instead."
            )
        if support_site_id is None and relative_position is None:
            preferred_xy = self._default_fixture_surface_preference(
                support_fixture_id,
                object_id,
            )
        else:
            preferred_xy = None
        if preferred_xy is None:
            preferred_xy = self._preferred_xy_for_fixture_target(
                support_fixture_id,
                site_id=support_site_id,
                relative_position=relative_position,
            )

        ignored_fixture_ids = self._ignored_fixture_ids_for_placement(
            support_fixture_id,
            support_site_id,
        )
        target_pos, target_quat = self._compute_object_target_pose(
            support_fixture_id,
            object_id,
            preferred_xy=preferred_xy,
            support_site_id=support_site_id,
            ignored_fixture_ids=ignored_fixture_ids,
            allow_yaw_search=isinstance(support_site_id, str),
        )
        self.runner._move_robot_near_fixture(
            robot_idx,
            support_fixture_id,
            ref_pos_override=target_pos[:2],
            require_front=self._surface_fixture_prefers_front_approach(
                support_fixture_id
            ),
        )
        self._sync_held_object(robot_idx)
        self._set_object_pose(object_id, target_pos, target_quat)
        self._set_support_parent(object_id, support_fixture_id)
        self.runner._set_object_location(object_id, support_fixture_id)
        self._clear_held_object_state(robot_idx)
        self._settle_scene()
        return ToolResult(
            "place_on_surface",
            True,
            {
                "object_id": object_id,
                "support_id": support_id,
                "target_id": target_id,
                "resolved_support_fixture_id": support_fixture_id,
                "resolved_support_site_id": (
                    self._raw_support_site_to_external(support_site_id)
                    if isinstance(support_site_id, str)
                    else None
                ),
                "robot_idx": robot_idx,
            },
        )

    def place_in_receptacle(
        self,
        object_id: str,
        receptacle_id: str | None = None,
        target_id: str | None = None,
        target_site_id: str | None = None,
        relative_position: str | None = None,
        robot_idx: int = 0,
    ) -> ToolResult:
        self._require_object(object_id)
        holder = self._held_by_robot(object_id)
        if holder not in {None, robot_idx}:
            raise ValueError(f"Object {object_id!r} is held by robot {holder}")

        receptacle_name = target_id or receptacle_id
        if not isinstance(receptacle_name, str):
            raise ValueError("place_in_receptacle requires target_id or receptacle_id.")

        final_settle_steps = _PLACEMENT_SETTLE_STEPS
        force_tray_kettle_mug_rebalance = False
        force_tray_object_id: str | None = None
        resolved_support_target = None
        if receptacle_name not in self.env.objects:
            if receptacle_name in self.runner._fixtures:
                resolved_support_target = self._resolve_support_target(
                    receptacle_name,
                    target_site_id,
                )
            else:
                try:
                    resolved_support_target = self._resolve_support_target(
                        receptacle_name,
                        target_site_id,
                    )
                except ValueError:
                    resolved_support_target = None

        if resolved_support_target is not None:
            support_fixture_id, support_site_id = resolved_support_target
            if (
                receptacle_name not in self.runner._fixtures
                and target_id is None
                and receptacle_id is not None
            ):
                receptacle_id = support_fixture_id
            support_fixture_id, support_site_id = self._resolve_support_target(
                support_fixture_id,
                support_site_id,
            )
            if support_site_id is None:
                # Generic deterministic fallback for interior placements: if
                # no explicit site was provided, bind to a default support site
                # so we do not solve against an unconstrained multi-shelf interior.
                default_site_id = self._default_support_site_for_unspecified_fixture(
                    support_fixture_id,
                    incoming_object_id=object_id,
                )
                if isinstance(default_site_id, str):
                    support_site_id = self._resolve_fixture_site_id(
                        support_fixture_id,
                        default_site_id,
                    )
            self._require_explicit_site_if_needed(
                support_fixture_id,
                support_site_id,
                action_name="place_in_receptacle",
            )
            if self._fixture_placement_semantics(support_fixture_id) != "receptacle":
                raise ValueError(
                    f"Fixture {support_fixture_id!r} has exposed surface semantics; "
                    "use place_on_surface instead."
                )
            preferred_xy = self._preferred_xy_for_fixture_target(
                support_fixture_id,
                site_id=support_site_id,
                relative_position=relative_position,
            )
            if isinstance(support_site_id, str) and relative_position is None:
                site_preferred_xy = self._incoming_fixture_site_preference(
                    support_fixture_id,
                    support_site_id,
                    incoming_object_id=object_id,
                )
                if site_preferred_xy is not None:
                    preferred_xy = np.asarray(site_preferred_xy, dtype=float)
            ignored_fixture_ids = self._ignored_fixture_ids_for_placement(
                support_fixture_id,
                support_site_id,
            )
            try:
                target_pos, target_quat = self._compute_object_target_pose(
                    support_fixture_id,
                    object_id,
                    preferred_xy=preferred_xy,
                    support_site_id=support_site_id,
                    ignored_fixture_ids=ignored_fixture_ids,
                    allow_yaw_search=isinstance(support_site_id, str),
                )
            except RuntimeError:
                if not isinstance(support_site_id, str):
                    raise
                slot_target = self._find_fixture_site_slot_target(
                    support_fixture_id,
                    support_site_id,
                    object_id=object_id,
                    current_preferred_xy=preferred_xy,
                    ignored_fixture_ids=ignored_fixture_ids,
                )
                if slot_target is None:
                    raise
                target_pos, target_quat = slot_target
            require_front = self._fixture_requires_front_approach(
                support_fixture_id
            ) or self._surface_fixture_prefers_front_approach(support_fixture_id)
            self._move_robot_near_fixture_with_retries(
                robot_idx,
                support_fixture_id,
                ref_pos_override=target_pos[:2],
                require_front=require_front,
            )
            self._sync_held_object(robot_idx)
            self._set_object_pose(object_id, target_pos, target_quat)
            self._set_support_parent(object_id, support_fixture_id)
            self.runner._set_object_location(object_id, support_fixture_id)
        else:
            self._require_object(receptacle_name)
            object_tokens = self._scene_object_tokens(object_id)
            receptacle_tokens = self._support_object_tokens(receptacle_name)
            if (
                "tray" in receptacle_tokens
                and object_tokens & {"kettle", "mug"}
            ):
                force_tray_kettle_mug_rebalance = True
                force_tray_object_id = receptacle_name
            if (
                object_tokens & _RECEPTACLE_OBJECT_TOKENS
                and receptacle_tokens & {"plate"}
            ):
                redirected = self.place_next_to(
                    object_id,
                    reference_object_id=receptacle_name,
                    robot_idx=robot_idx,
                )
                return ToolResult(
                    "place_in_receptacle",
                    redirected.success,
                    {
                        **redirected.details,
                        "object_id": object_id,
                        "receptacle_id": receptacle_id,
                        "target_id": target_id,
                        "redirected_to": "place_next_to",
                        "redirect_reason": "large_receptacle_not_nested",
                    },
                )
            anchor_fixture_id = self._get_scene_object_location(receptacle_name)
            container_pos, _ = self._get_object_pose(receptacle_name)
            if anchor_fixture_id is not None:
                self.runner._move_robot_near_fixture(
                    robot_idx,
                    anchor_fixture_id,
                    ref_pos_override=container_pos[:2],
                )
            self._sync_held_object(robot_idx)
            self._place_on_object_center(
                object_id,
                receptacle_name,
                relative_position=relative_position,
            )
            self._settle_and_reseat_supported_object(
                object_id,
                receptacle_name,
                relative_position=relative_position,
            )
            if (
                force_tray_kettle_mug_rebalance
                and isinstance(force_tray_object_id, str)
                and self._rebalance_kettle_mug_on_tray(force_tray_object_id)
            ):
                final_settle_steps = min(final_settle_steps, 1)
            if (
                self._support_object_prefers_upright(receptacle_name, object_id)
                or self._support_object_accepts_child_without_settle(receptacle_name)
            ):
                final_settle_steps = 0

        self._clear_held_object_state(robot_idx)
        if final_settle_steps > 0:
            self._settle_scene(steps=final_settle_steps)
        if (
            force_tray_kettle_mug_rebalance
            and isinstance(force_tray_object_id, str)
            and self._rebalance_kettle_mug_on_tray(force_tray_object_id)
        ):
            self._settle_scene(steps=1)
        return ToolResult(
            "place_in_receptacle",
            True,
            {
                "object_id": object_id,
                "receptacle_id": receptacle_id,
                "target_id": target_id,
                "robot_idx": robot_idx,
            },
        )

    def place_next_to(
        self,
        object_id: str,
        reference_object_id: str | None = None,
        reference_id: str | None = None,
        reference_fixture_id: str | None = None,
        target_site_id: str | None = None,
        relative_position: str | None = None,
        robot_idx: int = 0,
    ) -> ToolResult:
        """Place an object adjacent to another object or nearby fixture."""
        self._require_object(object_id)
        holder = self._held_by_robot(object_id)
        if holder not in {None, robot_idx}:
            raise ValueError(f"Object {object_id!r} is held by robot {holder}")

        reference_target_id, ref_is_fixture = self._resolve_reference_target(
            reference_id=reference_id,
            reference_object_id=reference_object_id,
            reference_fixture_id=reference_fixture_id,
        )

        if ref_is_fixture:
            fixture = self.runner._fixtures[reference_target_id]
            ref_pos = np.asarray(fixture.pos, dtype=float)
            ref_extent_xy = self._reference_extent_xy(reference_target_id, is_fixture=True)
            support_fixture_id = self._find_placeable_surface_near_fixture(
                reference_target_id,
                robot_idx=robot_idx,
            )
            reference_fixture_type = (
                self._get_fixture_type_name(reference_target_id) or ""
            ).lower()
        else:
            self._require_object(reference_target_id)
            ref_pos, _ = self._get_object_pose(reference_target_id)
            ref_extent_xy = self._reference_extent_xy(reference_target_id, is_fixture=False)
            support_fixture_id = self._get_scene_object_location(reference_target_id)
            if support_fixture_id is None:
                support_fixture_id = self._resolve_object_anchor_fixture(
                    reference_target_id,
                    require_placeable=True,
                )
            reference_fixture_type = None

        if relative_position is None and not ref_is_fixture:
            relative_position = self._default_relative_position_next_to_object(
                object_id,
                reference_target_id,
            )

        self._require_fixture(support_fixture_id)
        resolved_target_site_id = (
            self._resolve_fixture_site_id(support_fixture_id, target_site_id)
            if isinstance(target_site_id, str)
            else None
        )
        preferred_positions = self._candidate_xy_next_to_reference(
            support_fixture_id=support_fixture_id,
            reference_xy=ref_pos[:2],
            reference_extent_xy=ref_extent_xy,
            reference_id=reference_target_id,
            reference_is_fixture=ref_is_fixture,
            target_site_id=target_site_id,
            relative_position=relative_position,
        )

        scored_targets: list[tuple[float, int, np.ndarray]] = []
        last_error: RuntimeError | None = None
        ignored_fixture_ids = self._ignored_fixture_ids_for_adjacent_reference(
            support_fixture_id=support_fixture_id,
            support_site_id=resolved_target_site_id,
            reference_fixture_id=reference_target_id if ref_is_fixture else None,
        )
        support_fixture = self._require_fixture(support_fixture_id)
        reference_local = self.runner._world_to_fixture_local(
            support_fixture,
            np.asarray(ref_pos[:2], dtype=float),
        )
        appliance_lateral_band = None
        if reference_fixture_type in _COUNTERTOP_APPLIANCE_TYPE_NAMES:
            appliance_lateral_band = max(
                0.5 * float(np.asarray(ref_extent_xy, dtype=float)[1]) + 0.05,
                0.08,
            )
        for preferred_idx, preferred_xy in enumerate(preferred_positions):
            try:
                target_pos = self._safe_compute_object_target_pos(
                    support_fixture_id,
                    object_id,
                    preferred_xy=preferred_xy,
                    support_site_id=resolved_target_site_id,
                    ignored_fixture_ids=ignored_fixture_ids,
                )
            except RuntimeError as exc:
                last_error = exc
                continue
            if appliance_lateral_band is not None:
                target_local = self.runner._world_to_fixture_local(
                    support_fixture,
                    np.asarray(target_pos[:2], dtype=float),
                )
                if abs(float(target_local[1] - reference_local[1])) > appliance_lateral_band:
                    continue
            score = float(np.linalg.norm(target_pos[:2] - preferred_xy[:2]))
            scored_targets.append((score, preferred_idx, target_pos))

        if not scored_targets:
            if last_error is not None:
                raise last_error
            raise RuntimeError(
                f"No collision-free placement candidate found for {object_id!r} next to {reference_target_id!r}."
            )
        target_pos = min(scored_targets, key=lambda item: (item[0], item[1]))[2]
        target_quat = None
        target_local = self.runner._world_to_fixture_local(
            support_fixture,
            np.asarray(target_pos[:2], dtype=float),
        )
        local_delta = np.asarray(target_local[:2], dtype=float) - np.asarray(
            reference_local[:2], dtype=float
        )
        if float(np.linalg.norm(local_delta)) > 1e-6:
            support_axis = "y" if abs(float(local_delta[0])) >= abs(float(local_delta[1])) else "x"
            alignment_axis_world = -self._fixture_inward_world_axis(
                support_fixture_id,
                support_axis,
            )
            target_quat = self._aligned_flat_quat_for_world_axis(
                object_id,
                alignment_axis_world,
                preserve_direction=False,
                allow_target_sign_flip=False,
            )

        self.runner._move_robot_near_fixture(
            robot_idx,
            support_fixture_id,
            ref_pos_override=target_pos[:2],
            require_front=self._surface_fixture_prefers_front_approach(
                support_fixture_id
            ),
        )
        self._sync_held_object(robot_idx)
        self._set_object_pose(object_id, target_pos, target_quat)
        self._set_support_parent(object_id, support_fixture_id)
        self.runner._set_object_location(object_id, support_fixture_id)

        self._clear_held_object_state(robot_idx)
        self._settle_scene()
        return ToolResult(
            "place_next_to",
            True,
            {
                "object_id": object_id,
                "reference_id": reference_id,
                "reference_object_id": reference_object_id,
                "reference_fixture_id": reference_fixture_id,
                "support_fixture_id": support_fixture_id,
                "robot_idx": robot_idx,
            },
        )

    def place_under(
        self,
        object_id: str,
        reference_fixture_id: str | None = None,
        target_id: str | None = None,
        target_site_id: str | None = None,
        robot_idx: int = 0,
    ) -> ToolResult:
        """Place an object directly beneath a reference fixture.

        For dispenser-type fixtures (CoffeeMachine, Sink) the object is
        positioned at the dispenser output site.  For all other fixtures
        (e.g. cabinet, hood) the object is placed on the nearest surface
        below, with XY aligned to the reference fixture.
        """
        self._require_object(object_id)
        reference_fixture_id = target_id or reference_fixture_id
        if not isinstance(reference_fixture_id, str):
            raise ValueError("place_under requires target_id or reference_fixture_id.")
        fixture = self._require_fixture(reference_fixture_id)
        holder = self._held_by_robot(object_id)
        if holder not in {None, robot_idx}:
            raise ValueError(f"Object {object_id!r} is held by robot {holder}")
        settle_after_placement = True

        self.runner._move_robot_near_fixture(
            robot_idx,
            reference_fixture_id,
            require_front=self._fixture_requires_front_approach(
                reference_fixture_id
            ),
        )
        self._sync_held_object(robot_idx)

        # Resolve dispenser site for fixtures that have one.
        dispenser_site_name = self._get_dispenser_site_name(fixture)

        if isinstance(fixture, Sink):
            if isinstance(target_site_id, str):
                resolved_target_site_id = self._resolve_fixture_site_id(
                    reference_fixture_id,
                    target_site_id,
                )
                target_pos = self._compute_region_center_target_pos(
                    reference_fixture_id,
                    object_id,
                    site_id=resolved_target_site_id,
                )
            else:
                water_site_name = self._get_dispenser_site_name(fixture)
                if water_site_name is None:
                    raise RuntimeError(
                        f"Sink fixture {reference_fixture_id!r} has no water site."
                    )
                site_id = self.env.sim.model.site_name2id(water_site_name)
                target_pos = self.env.sim.data.site_xpos[site_id].copy()
                resolved_target_site_id = None
            self._set_object_pose(object_id, target_pos)
            self._set_support_parent(object_id, reference_fixture_id)
            self.runner._set_object_location(object_id, reference_fixture_id)
            support_fixture_id = reference_fixture_id
            settle_after_placement = False
        elif dispenser_site_name is not None:
            resolved_target_site_id = None
            target_quat = None
            explicit_target_site_requested = isinstance(target_site_id, str)
            if explicit_target_site_requested:
                resolved_target_site_id = self._resolve_fixture_site_id(
                    reference_fixture_id,
                    target_site_id,
                )
            elif isinstance(fixture, CoffeeMachine):
                default_site_id = self._default_support_site_for_unspecified_fixture(
                    reference_fixture_id,
                    incoming_object_id=object_id,
                )
                if isinstance(default_site_id, str):
                    resolved_target_site_id = self._resolve_fixture_site_id(
                        reference_fixture_id,
                        default_site_id,
                    )
            site_id = self.env.sim.model.site_name2id(dispenser_site_name)
            site_pos = self.env.sim.data.site_xpos[site_id].copy()
            if explicit_target_site_requested and isinstance(resolved_target_site_id, str):
                target_pos = self._compute_region_center_target_pos(
                    reference_fixture_id,
                    object_id,
                    site_id=resolved_target_site_id,
                )
            else:
                target_pos = site_pos.copy()
            object_tokens = self._scene_object_tokens(object_id)
            if isinstance(fixture, CoffeeMachine) and object_tokens & {"mug"}:
                inward_axis = self._fixture_inward_world_axis(reference_fixture_id, "y")
                lateral_axis = np.array(
                    [-float(inward_axis[1]), float(inward_axis[0]), 0.0],
                    dtype=float,
                )
                if float(np.linalg.norm(lateral_axis[:2])) > 1e-9:
                    target_quat = self._aligned_upright_quat_for_world_axis(
                        object_id,
                        lateral_axis,
                    )
            self._set_object_pose(object_id, target_pos, target_quat)
            self._set_support_parent(object_id, reference_fixture_id)
            self.runner._set_object_location(object_id, reference_fixture_id)
            support_fixture_id = reference_fixture_id
            settle_after_placement = False
        else:
            # Generic: project fixture XY, find the surface below.
            fxtr_pos = np.asarray(fixture.pos, dtype=float)
            support_fixture_id = self._find_placeable_surface_near_fixture(
                reference_fixture_id
            )
            self._require_explicit_site_if_needed(
                support_fixture_id,
                target_site_id,
                action_name="place_under",
            )
            resolved_target_site_id = (
                self._resolve_fixture_site_id(support_fixture_id, target_site_id)
                if isinstance(target_site_id, str)
                else None
            )
            preferred_xy = (
                self._project_xy_onto_fixture(
                    support_fixture_id,
                    fxtr_pos[:2],
                    site_id=resolved_target_site_id,
                )
                if isinstance(target_site_id, str)
                else self._project_xy_onto_fixture(support_fixture_id, fxtr_pos[:2])
            )
            target_pos = self._safe_compute_object_target_pos(
                support_fixture_id,
                object_id,
                preferred_xy=preferred_xy,
                support_site_id=resolved_target_site_id,
                ignored_fixture_ids=self._ignored_fixture_ids_for_placement(
                    support_fixture_id,
                    resolved_target_site_id,
                ),
            )
            self._set_object_pose(object_id, target_pos)
            self._set_support_parent(object_id, support_fixture_id)
            self.runner._set_object_location(object_id, support_fixture_id)

        self._clear_held_object_state(robot_idx)
        if settle_after_placement:
            self._settle_scene()
        return ToolResult(
            "place_under",
            True,
            {
                "object_id": object_id,
                "target_id": target_id,
                "reference_fixture_id": reference_fixture_id,
                "support_fixture_id": support_fixture_id,
                "robot_idx": robot_idx,
            },
        )

    def place_on_object(
        self,
        object_id: str,
        support_object_id: str | None = None,
        target_id: str | None = None,
        anchor_fixture_id: str | None = None,
        relative_position: str | None = None,
        robot_idx: int = 0,
    ) -> ToolResult:
        self._require_object(object_id)
        support_object_id = target_id or support_object_id
        if not isinstance(support_object_id, str):
            raise ValueError("place_on_object requires target_id or support_object_id.")
        self._require_object(support_object_id)
        holder = self._held_by_robot(object_id)
        if holder not in {None, robot_idx}:
            raise ValueError(f"Object {object_id!r} is held by robot {holder}")

        object_tokens = self._scene_object_tokens(object_id)
        support_tokens = self._scene_object_tokens(support_object_id)
        if object_tokens & _TOOL_OBJECT_TOKENS and support_tokens & _FOOD_OBJECT_TOKENS:
            parent_id = self._current_support_object(support_object_id)
            if isinstance(parent_id, str) and parent_id in self.env.objects:
                support_object_id = parent_id

        # Pre-compute the support object's position — this is where the
        # object will land, so the robot should stand near it (not the
        # fixture center).  Mirrors how place_on_surface pre-computes the
        # drop position via ref_pos_override.
        support_pos, _ = self._get_object_pose(support_object_id)
        ref_pos = support_pos[:2].copy()

        if anchor_fixture_id is None:
            anchor_fixture_id = self._resolve_object_anchor_fixture(
                support_object_id,
                preferred_fixture_types=[
                    "dining_counter",
                    "island",
                    "counter_non_dining",
                ],
                require_placeable=True,
            )
        self._require_fixture(anchor_fixture_id)
        self.runner._move_robot_near_fixture(
            robot_idx,
            anchor_fixture_id,
            ref_pos_override=ref_pos,
        )
        self._sync_held_object(robot_idx)
        self._place_on_object_center(
            object_id,
            support_object_id,
            relative_position=relative_position,
        )
        self._settle_and_reseat_supported_object(
            object_id,
            support_object_id,
            relative_position=relative_position,
        )
        self._clear_held_object_state(robot_idx)
        if not self._support_object_accepts_child_without_settle(support_object_id):
            self._settle_scene(steps=_PLACEMENT_SETTLE_STEPS)
        return ToolResult(
            "place_on_object",
            True,
            {
                "object_id": object_id,
                "target_id": target_id,
                "support_object_id": support_object_id,
                "anchor_fixture_id": anchor_fixture_id,
                "robot_idx": robot_idx,
            },
        )

    def press_button(
        self,
        target_id: str,
        control_id: str,
        robot_idx: int = 0,
    ) -> ToolResult:
        fixture = self._require_fixture(target_id)
        moved = self._move_robot_near_fixture_with_retries(
            robot_idx,
            target_id,
            require_front=self._fixture_requires_front_approach(target_id),
        )
        if moved:
            self._sync_held_object(robot_idx)

        if isinstance(fixture, Microwave):
            fixture._turned_on = control_id == "start_button"
        elif isinstance(fixture, CoffeeMachine):
            fixture._turned_on = True
        elif isinstance(fixture, ElectricKettle):
            if control_id == "lid_button":
                fixture.set_lid(self.env, lid_val=1.0, gradual=False)
            else:
                raise ValueError(f"Unsupported kettle button {control_id!r}")
        else:
            joint_name = self._resolve_joint_name(fixture, control_id)
            self._set_named_joint(fixture, joint_name, 1.0)

        if hasattr(fixture, "update_state"):
            fixture.update_state(self.env)
        self._settle_scene()
        return ToolResult(
            "press_button", True, {"target_id": target_id, "control_id": control_id}
        )

    def press_lever(
        self,
        target_id: str,
        control_id: str,
        robot_idx: int = 0,
    ) -> ToolResult:
        fixture = self._require_fixture(target_id)
        moved = self._move_robot_near_fixture_with_retries(
            robot_idx,
            target_id,
            require_front=_require_front(fixture),
        )
        if moved:
            self._sync_held_object(robot_idx)

        if isinstance(fixture, Toaster):
            slot_pair = 0
            if "_" in control_id:
                _, slot_pair_str = control_id.rsplit("_", 1)
                slot_pair = int(slot_pair_str)
            fixture.set_lever(self.env, slot_pair=slot_pair, value=1.0)
        elif isinstance(fixture, ElectricKettle):
            fixture.set_power_state(self.env, power_on=True)
        else:
            joint_name = self._resolve_joint_name(fixture, control_id)
            self._set_named_joint(fixture, joint_name, 1.0)

        if hasattr(fixture, "update_state"):
            fixture.update_state(self.env)
        self._settle_scene()
        return ToolResult(
            "press_lever", True, {"target_id": target_id, "control_id": control_id}
        )

    def set_rotary_control(
        self,
        target_id: str,
        control_id: str,
        goal: str,
        robot_idx: int = 0,
    ) -> ToolResult:
        fixture = self._require_fixture(target_id)
        moved = self._move_robot_near_fixture_with_retries(
            robot_idx,
            target_id,
            require_front=True,
        )
        if moved:
            self._sync_held_object(robot_idx)

        goal_lower = str(goal).lower()
        if goal_lower in {"on", "open", "high", "max", "1", "true"}:
            value = 1.0
        elif goal_lower in {"off", "close", "low", "min", "0", "false"}:
            value = 0.0
        else:
            value = float(goal)

        control_token = str(control_id).strip().lower().replace(" ", "_")

        if isinstance(fixture, Toaster) and control_token.startswith("knob"):
            slot_pair = 0
            if "_" in control_token:
                _, slot_pair_str = control_token.rsplit("_", 1)
                slot_pair = int(slot_pair_str)
            fixture.set_doneness_knob(self.env, slot_pair=slot_pair, value=value)
        else:
            joint_name = self._resolve_joint_name(fixture, control_token)
            self._set_named_joint(fixture, joint_name, value)

        if hasattr(fixture, "update_state"):
            fixture.update_state(self.env)
        self._settle_scene()
        return ToolResult(
            "set_rotary_control",
            True,
            {"target_id": target_id, "control_id": control_id, "goal": goal},
        )

    def communicate(
        self,
        to: str,
        message: str,
        robot_idx: int = 0,
        releases: str | None = None,
        coordination_phase: str | None = None,
    ) -> ToolResult:
        # `releases` is symbolic bookkeeping: it names the resource this message
        # hands over, which is what wakes a partner blocked in wait_for_signal.
        # A message moves nothing in MuJoCo, so there is nothing to execute --
        # but the argument has to be ACCEPTED, or every tick trajectory dies at
        # its first handover with "unexpected keyword argument 'releases'".
        # Recorded in the result so the executed trace still shows the handover.
        return ToolResult(
            "communicate",
            True,
            {
                "robot_idx": robot_idx,
                "to": to,
                "message": message,
                "releases": releases,
                "coordination_phase": coordination_phase,
            },
        )

    def give_space(self, fixture_id: str, robot_idx: int = 0) -> ToolResult:
        """Move the robot to open floor space away from *fixture_id*.

        Delegates to ``TrajectoryRunner.give_space`` which finds a free grid
        cell that is ≥1.5 m from the fixture.
        """
        requested_fixture_id = fixture_id
        fixture_id = self._canonical_navigation_fixture_id(fixture_id)
        self._require_fixture(fixture_id)
        self.runner.give_space(robot_idx, fixture_id)
        self._sync_held_object(robot_idx)
        return ToolResult(
            "give_space",
            True,
            {
                "fixture_id": requested_fixture_id,
                "resolved_workspace_fixture_id": fixture_id,
                "robot_idx": robot_idx,
            },
        )

    def wait(self, robot_idx: int = 0) -> ToolResult:
        return ToolResult(
            "wait",
            True,
            {"robot_idx": robot_idx},
        )

    def wait_for_signal(
        self,
        about: str,
        robot_idx: int = 0,
        **kwargs,
    ) -> ToolResult:
        """Record a blocking wait; a no-op for the physics.

        Blocking is a scheduling concern and this replay path is already
        ordered, so there is nothing to suspend here -- the live-sim harness is
        what actually parks the agent. ``from`` arrives via kwargs because it is
        a Python keyword.
        """
        return ToolResult(
            "wait_for_signal",
            True,
            {
                "robot_idx": robot_idx,
                "from": kwargs.get("from"),
                "about": about,
            },
        )

    # ------------------------------------------------------------------
    # Generic dispatch
    # ------------------------------------------------------------------

    def execute(self, tool_name: str, robot_idx: int = 0, **kwargs) -> ToolResult:
        if tool_name not in SIM_TOOL_SPEC_BY_NAME:
            raise ValueError(f"Unknown tool {tool_name!r}")

        method = getattr(self, tool_name, None)
        if method is None:
            raise NotImplementedError(f"No executor method defined for {tool_name!r}")
        # Observation and coordination tools do not mutate any visual state.
        # Preserving their caches avoids regenerating byte-identical placement
        # maps after every communication turn.
        if tool_name not in self._NON_MUTATING_TOOL_NAMES:
            invalidate_visual_cache = getattr(self, "_invalidate_visual_cache", None)
            if callable(invalidate_visual_cache):
                invalidate_visual_cache()
        return method(robot_idx=robot_idx, **kwargs)


def _main():
    parser = argparse.ArgumentParser(
        description="Execute simulator tools and save scene frames / videos."
    )
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--robots", type=int, default=2)
    parser.add_argument(
        "--layout",
        type=int,
        default=None,
        help=(
            "Kitchen layout id. If omitted, uses trajectory scene_parameters.layout "
            "when present, otherwise defaults to 11."
        ),
    )
    parser.add_argument(
        "--style",
        type=int,
        default=None,
        help=(
            "Kitchen style id. If omitted, uses trajectory scene_parameters.style "
            "when present, otherwise defaults to 34."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Environment seed. If omitted, uses trajectory scene_parameters.seed "
            "when present, otherwise defaults to 42."
        ),
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument(
        "--map-dpi",
        type=int,
        default=300,
        help="Placement-map export DPI. Default: 300 (6000x4800 pixels).",
    )
    parser.add_argument(
        "--map-renderer",
        choices=("legacy", "raster"),
        default="legacy",
        help="Placement-grid renderer. Legacy is pixel-compatible; raster is faster.",
    )
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument(
        "--plan",
        type=str,
        default=None,
        help="Path to a JSON tool plan. If omitted, only current-scene frames are saved.",
    )
    parser.add_argument(
        "--demo-plan",
        type=str,
        default=None,
        help=(
            "Name of a built-in tool plan. Available: cooperative_hotdog_setup, sandwich_station. "
            "Mutually exclusive with --plan and --trajectory."
        ),
    )
    parser.add_argument(
        "--trajectory",
        type=str,
        default=None,
        help=(
            "Path to a full external trajectory JSON. Mutually exclusive with "
            "--plan and --demo-plan."
        ),
    )
    parser.add_argument("--fps", type=int, default=2)
    parser.add_argument(
        "--skip-videos",
        action="store_true",
        default=False,
        help="Skip MP4 video generation to reduce output size.",
    )
    parser.add_argument(
        "--raw-map-labels",
        action="store_true",
        default=False,
        help="Show full raw fixture ids on the map instead of cleaned-up labels.",
    )
    parser.add_argument(
        "--robot-spawn",
        choices=["sim", "trajectory"],
        default="trajectory",
        help=(
            "Robot initial placement source. 'trajectory' (default): place each robot "
            "at the location specified in the trajectory's initial_state. 'sim': place "
            "all robots at init_robot_base_ref (task ground truth)."
        ),
    )
    parser.add_argument("--gl-backend", type=str, default="osmesa")
    parser.add_argument(
        "--placement",
        choices=["grid"],
        default="grid",
        help="Robot placement strategy. Only occupancy-grid placement is supported.",
    )
    parser.add_argument(
        "--cell-size",
        type=float,
        default=0.05,
        help="Grid cell size in meters (only used when --placement=grid). Default: 0.05",
    )
    parser.add_argument(
        "--align-to-wall",
        action="store_true",
        default=False,
        help="Align grid origin so cell boundaries fall on y=0 and x=0 (wall edges).",
    )
    parser.add_argument(
        "--standoff",
        type=float,
        default=0.30,
        help="Distance from fixture face to robot center (meters). Default: 0.30",
    )
    parser.add_argument(
        "--sample-spacing",
        type=float,
        default=0.12,
        help="Spacing between candidate samples along fixture faces. Default: 0.12",
    )
    parser.add_argument(
        "--robot-radius",
        type=float,
        default=0.18,
        help="Robot collision radius for placement (meters). Default: 0.18",
    )
    args = parser.parse_args()

    if args.map_dpi < 1:
        parser.error("--map-dpi must be at least 1.")

    provided_inputs = [
        args.plan is not None,
        args.demo_plan is not None,
        args.trajectory is not None,
    ]
    if sum(provided_inputs) > 1:
        parser.error("Use only one of --plan, --demo-plan, or --trajectory.")

    trajectory_payload = None
    if args.trajectory is not None:
        with open(args.trajectory, "r") as f:
            trajectory_payload = json.load(f)

    def _resolve_scene_parameter(name: str, fallback: int) -> int:
        explicit_value = getattr(args, name)
        if explicit_value is not None:
            return explicit_value
        if isinstance(trajectory_payload, dict):
            scene_parameters = trajectory_payload.get("scene_parameters")
            if isinstance(scene_parameters, dict):
                scene_value = scene_parameters.get(name)
                if scene_value is not None:
                    return int(scene_value)
            top_level_value = trajectory_payload.get(name)
            if top_level_value is not None:
                return int(top_level_value)
        return fallback

    layout = _resolve_scene_parameter("layout", 11)
    style = _resolve_scene_parameter("style", 34)
    seed = _resolve_scene_parameter("seed", 42)

    if args.task is None:
        if args.demo_plan is not None:
            demo_key = args.demo_plan.strip().lower().replace("-", "_")
            task_name = SimToolExecutor._DEMO_TASK_BY_NAME.get(demo_key)
            if task_name is None:
                parser.error(
                    "Unknown demo plan. Available: cooperative_hotdog_setup, sandwich_station"
                )
        elif (
            trajectory_payload is not None
            and trajectory_payload.get("composite_task") is not None
        ):
            task_name = trajectory_payload["composite_task"]
        else:
            task_name = "MicrowaveThawing"
    else:
        task_name = args.task

    executor = SimToolExecutor(
        task_name=task_name,
        robots=args.robots,
        layout=layout,
        style=style,
        seed=seed,
        render_width=args.width,
        render_height=args.height,
        map_dpi=args.map_dpi,
        map_renderer=args.map_renderer,
        gl_backend=args.gl_backend,
        placement=args.placement,
        cell_size=args.cell_size,
        align_to_wall=args.align_to_wall,
        standoff=args.standoff,
        sample_spacing=args.sample_spacing,
        robot_radius=args.robot_radius,
        robot_spawn=args.robot_spawn,
    )
    # --raw-map-labels disables fixture label cleanup on the map
    executor._clean_map_labels = not args.raw_map_labels

    try:
        if args.plan is None and args.demo_plan is None and args.trajectory is None:
            # No trajectory/plan — save map and frames now
            map_path = executor.save_placement_map(
                args.output_dir,
                prefix="initial",
                clean_labels=executor._clean_map_labels,
            )
            print(f"Placement map: {map_path}")
            saved = executor.save_scene_frames(args.output_dir, prefix="initial")
            print(json.dumps({k: str(v) for k, v in saved.items()}, indent=2))
        else:
            if args.trajectory is not None:
                from robocasa.utils.trajectory_adapter import execute_trajectory
                import shutil

                # Copy original trajectory JSON to output dir
                out_path = Path(args.output_dir)
                out_path.mkdir(parents=True, exist_ok=True)
                shutil.copy2(args.trajectory, out_path / "original_trajectory.json")

                metadata = execute_trajectory(
                    executor=executor,
                    trajectory=trajectory_payload,
                    output_dir=args.output_dir,
                    fps=args.fps,
                    skip_videos=args.skip_videos,
                )
                # The adapter already saves initial_map.png right after
                # load_initial_state (before trajectory steps run), so the
                # map reflects the true initial robot/object positions.
                # Do NOT re-save here — that would overwrite with post-
                # execution positions.
            elif args.demo_plan is not None:
                tool_calls = executor.build_demo_plan(args.demo_plan)
                metadata = executor.run_tool_plan(
                    tool_calls=tool_calls,
                    output_dir=args.output_dir,
                    fps=args.fps,
                )
            else:
                with open(args.plan, "r") as f:
                    tool_calls = json.load(f)
                metadata = executor.run_tool_plan(
                    tool_calls=tool_calls,
                    output_dir=args.output_dir,
                    fps=args.fps,
                )
            # Save map after execution for demo_plan/plan paths
            if args.trajectory is None:
                map_path = executor.save_placement_map(
                    args.output_dir,
                    prefix="initial",
                    clean_labels=executor._clean_map_labels,
                )
                print(f"Placement map: {map_path}")
            print(json.dumps(metadata, indent=2))
    finally:
        executor.close()


__all__ = ["SimToolExecutor", "ToolResult"]


if __name__ == "__main__":
    _main()
