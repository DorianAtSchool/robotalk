"""Initial-state loading helpers for ``SimToolExecutor``."""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np

from robotalk.tasks.shared.constants import EXCLUSIVE_FIXTURE_TYPES

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
_PLACEMENT_SETTLE_STEPS = 6
_SWEEP_VERBOSE_ENV_VAR = "ROBOCASA_SWEEP_VERBOSE"


def _sim_tool_debug_enabled() -> bool:
    value = os.environ.get(_SWEEP_VERBOSE_ENV_VAR, "")
    return value.strip().lower() in {"1", "true", "yes", "on", "debug"}


class SimToolExecutorStateLoadingMixin:
    def load_initial_state(
        self, initial_state: dict[str, Any] | None
    ) -> dict[str, Any]:
        """Apply a normalized initial state to the live simulator."""
        if not initial_state:
            return {"loaded": False}

        fixtures = initial_state.get("fixtures", {})
        objects = initial_state.get("objects", {})
        agents = initial_state.get("agents", {})
        machine_state = initial_state.get("machine_state", {})
        held_assignments: dict[int, str] = {}

        if not hasattr(self, "_held_objects") or not isinstance(self._held_objects, dict):
            self._held_objects = {}
        if not hasattr(self, "_held_object_offsets") or not isinstance(
            self._held_object_offsets,
            dict,
        ):
            self._held_object_offsets = {}
        if not hasattr(self, "_support_parents") or not isinstance(self._support_parents, dict):
            self._support_parents = {}
        if not hasattr(self, "_recent_opened_sliding_fixture") or not isinstance(
            self._recent_opened_sliding_fixture,
            dict,
        ):
            self._recent_opened_sliding_fixture = {}

        self._held_objects.clear()
        self._held_object_offsets.clear()
        self._support_parents.clear()
        self._recent_opened_sliding_fixture.clear()

        for fixture_id, fixture_cfg in fixtures.items():
            fixture = self.runner._fixtures.get(fixture_id)
            if fixture is None:
                continue
            for part_id, part_cfg in fixture_cfg.get("parts", {}).items():
                desired_state = part_cfg.get("state")
                if desired_state == "closed":
                    if part_id == "hinged" and hasattr(fixture, "close_door"):
                        fixture.close_door(env=self.env)
                    else:
                        try:
                            joint_name = self._resolve_joint_name(fixture, part_id)
                            self._set_named_joint(fixture, joint_name, 0.0)
                        except ValueError:
                            pass
                elif desired_state == "open":
                    if part_id == "hinged" and hasattr(fixture, "open_door"):
                        fixture.open_door(env=self.env)
                    else:
                        try:
                            joint_name = self._resolve_joint_name(fixture, part_id)
                            self._set_named_joint(
                                fixture,
                                joint_name,
                                self._sliding_joint_open_fraction(
                                    joint_name,
                                    part_id=part_id,
                                ),
                            )
                        except ValueError:
                            pass
        self.env.sim.forward()

        for fixture_id, machine_cfg in machine_state.items():
            if not isinstance(machine_cfg, dict):
                continue
            if "started" in machine_cfg:
                self._set_fixture_machine_state(
                    fixture_id, bool(machine_cfg["started"])
                )

        held_object_ids = set()
        for agent_id, agent_state in agents.items():
            held_object = agent_state.get("held_object")
            if held_object is None:
                continue
            robot_idx = self._parse_agent_idx(agent_id)
            held_assignments[robot_idx] = held_object
            held_object_ids.add(held_object)

        scene = self.get_scene_description()
        sim_object_placements = scene.get("object_placements", {})

        current_scene_objects = scene.get("objects", {})
        placement_events: list[dict[str, Any]] = []

        def _resolve_fixture_location_target(
            object_state: dict[str, Any],
        ) -> tuple[str, str | None] | None:
            location = object_state.get("location")
            if not isinstance(location, str) or location in objects:
                return None
            target_site_id = object_state.get("target_site_id")
            if not isinstance(target_site_id, str):
                target_site_id = None
            return self._resolve_support_target(location, target_site_id)

        contained_children_by_parent: dict[str, list[str]] = {}
        for child_object_id, child_state in objects.items():
            child_location = child_state.get("location")
            if isinstance(child_location, str) and child_location in objects:
                contained_children_by_parent.setdefault(child_location, []).append(
                    child_object_id
                )

        contained_object_ids = {
            object_id
            for object_id, object_state in objects.items()
            if isinstance(object_state.get("location"), str)
            and object_state.get("location") in objects
        }

        for object_id, object_state in objects.items():
            location = object_state.get("location")
            if isinstance(location, str) and location in objects:
                self._set_support_parent(object_id, location)
            else:
                self._set_support_parent(object_id, None)

        def _containment_depth(object_id: str, stack: set[str] | None = None) -> int:
            object_state = objects.get(object_id, {})
            location = object_state.get("location")
            if not isinstance(location, str) or location not in objects:
                return 0
            if stack is None:
                stack = set()
            if object_id in stack:
                return 0
            return 1 + _containment_depth(location, stack | {object_id})

        def _preserve_pose_requested(object_state: dict[str, Any]) -> bool:
            flag = object_state.get("preserve_pose")
            if not isinstance(flag, bool):
                return False
            return flag

        for object_id, object_state in objects.items():
            if object_id in held_object_ids:
                continue
            resolved_target = _resolve_fixture_location_target(object_state)
            if resolved_target is not None:
                location = object_state.get("location")
                target_fixture_id, target_site_id = resolved_target
                explicit_target_site_requested = isinstance(
                    object_state.get("target_site_id"), str
                )
                current_scene_location = (
                    current_scene_objects.get(object_id, {}) or {}
                ).get("location")
                current_scene_site_id = (
                    self._infer_object_support_site(object_id, target_fixture_id)
                    if current_scene_location == target_fixture_id
                    else None
                )
                current_support_object = self._current_support_object(object_id)
                if (
                    _preserve_pose_requested(object_state)
                    and current_scene_location == target_fixture_id
                    and (
                        target_site_id is None
                        or current_scene_site_id == target_site_id
                    )
                ):
                    placement_events.append(
                        {
                            "object_id": object_id,
                            "target_fixture_id": target_fixture_id,
                            "target_site_id": (
                                self._raw_support_site_to_external(target_site_id)
                                if isinstance(target_site_id, str)
                                else None
                            ),
                            "skipped": True,
                            "reason": "preserve_pose",
                        }
                    )
                    self._set_support_parent(object_id, target_fixture_id)
                    self.runner._set_object_location(object_id, target_fixture_id)
                    continue
                if (
                    not explicit_target_site_requested
                    and current_support_object is None
                    and target_site_id is None
                    and current_scene_location == target_fixture_id
                    and sim_object_placements.get(object_id) == target_fixture_id
                    and (
                        self._fixture_placement_semantics(target_fixture_id) == "receptacle"
                        or any(
                            token in (
                                (self._get_fixture_type_name(target_fixture_id) or "")
                                .lower()
                            )
                            for token in ("cabinet", "fridge", "dishwasher", "drawer")
                        )
                    )
                ):
                    placement_events.append(
                        {
                            "object_id": object_id,
                            "target_fixture_id": target_fixture_id,
                            "target_site_id": None,
                            "skipped": True,
                            "reason": "preserve_original_spawn",
                        }
                    )
                    self._set_support_parent(object_id, target_fixture_id)
                    self.runner._set_object_location(object_id, target_fixture_id)
                    continue
                target_site_id = self._normalize_target_site_id_for_placement(
                    target_fixture_id,
                    target_site_id,
                )
                if target_site_id is None:
                    if self._fixture_placement_semantics(target_fixture_id) == "receptacle":
                        support_sites = self.get_support_sites(target_fixture_id)
                        if len(support_sites) == 1 and isinstance(support_sites[0], str):
                            target_site_id = support_sites[0]
                if (
                    target_site_id is None
                    and self._fixture_requires_explicit_site(target_fixture_id)
                ):
                    target_site_id = self._default_support_site_for_unspecified_fixture(
                        target_fixture_id,
                        incoming_object_id=object_id,
                    )
                current_scene_quat = None
                if current_scene_location == target_fixture_id:
                    try:
                        _current_scene_pos, current_scene_quat = self._get_object_pose(object_id)
                    except Exception:
                        current_scene_quat = None
                fixture_pose_requires_reset = (
                    current_scene_location == target_fixture_id
                    and current_scene_quat is not None
                    and self._fixture_pose_needs_reset(
                        object_id,
                        target_fixture_id,
                        site_id=target_site_id,
                        quat_wxyz=current_scene_quat,
                    )
                )
                if (
                    current_support_object is None
                    and target_site_id is None
                    and sim_object_placements.get(object_id) == target_fixture_id
                    and (
                        self._fixture_placement_semantics(target_fixture_id) == "receptacle"
                        or any(
                            token in (
                                (self._get_fixture_type_name(target_fixture_id) or "")
                                .lower()
                            )
                            for token in ("cabinet", "fridge", "dishwasher", "drawer")
                        )
                    )
                ):
                    placement_events.append(
                        {
                            "object_id": object_id,
                            "target_fixture_id": target_fixture_id,
                            "target_site_id": None,
                            "skipped": True,
                            "reason": "preserve_original_spawn",
                        }
                    )
                    self._set_support_parent(object_id, target_fixture_id)
                    self.runner._set_object_location(object_id, target_fixture_id)
                    continue
                if (
                    current_support_object is None
                    and not fixture_pose_requires_reset
                    and current_scene_location == target_fixture_id
                    and (
                        target_site_id is None
                        or current_scene_site_id == target_site_id
                    )
                ) or (
                    current_support_object is None
                    and not fixture_pose_requires_reset
                    and target_site_id is None
                    and sim_object_placements.get(object_id) == target_fixture_id
                ):
                    placement_events.append(
                        {
                            "object_id": object_id,
                            "target_fixture_id": target_fixture_id,
                            "target_site_id": (
                                self._raw_support_site_to_external(target_site_id)
                                if isinstance(target_site_id, str)
                                else None
                            ),
                            "skipped": True,
                            "reason": "already_on_target_fixture",
                        }
                    )
                    self._set_support_parent(object_id, target_fixture_id)
                    self.runner._set_object_location(object_id, target_fixture_id)
                    continue
                self._require_explicit_site_if_needed(
                    target_fixture_id,
                    target_site_id,
                    action_name="load_initial_state",
                )
                self._set_support_parent(object_id, target_fixture_id)
                if _sim_tool_debug_enabled():
                    print(
                        "[load_initial_state] placing",
                        json.dumps(
                            {
                                "object_id": object_id,
                                "requested_location": location,
                                "target_fixture_id": target_fixture_id,
                                "target_site_id": (
                                    self._raw_support_site_to_external(target_site_id)
                                    if isinstance(target_site_id, str)
                                    else None
                                ),
                            },
                            sort_keys=True,
                        ),
                    )
                preferred_xy = None
                if isinstance(target_site_id, str):
                    preferred_xy = self._incoming_fixture_site_preference(
                        target_fixture_id,
                        target_site_id,
                        incoming_object_id=object_id,
                    )
                    if preferred_xy is None:
                        preferred_xy = self._preferred_xy_for_fixture_target(
                            target_fixture_id,
                            site_id=target_site_id,
                        )
                else:
                    fixture_type = (self._get_fixture_type_name(target_fixture_id) or "").lower()
                    if self._fixture_placement_semantics(target_fixture_id) == "surface":
                        preferred_xy = self._default_fixture_surface_preference(
                            target_fixture_id,
                            object_id,
                        )
                    elif fixture_type in _FRONT_BIASED_INTERIOR_FIXTURE_TYPES:
                        preferred_xy = self._preferred_xy_for_fixture_target(
                            target_fixture_id
                        )
                try:
                    self._place_object_on_fixture(
                        object_id,
                        target_fixture_id,
                        preferred_xy=preferred_xy,
                        target_site_id=target_site_id,
                        settle=False,
                    )
                except Exception:
                    if _sim_tool_debug_enabled():
                        diagnostics = getattr(self.runner, "_last_placement_diagnostics", None)
                        if diagnostics is not None:
                            print(
                                "[load_initial_state] placement_diagnostics",
                                json.dumps(diagnostics, sort_keys=True),
                            )
                    raise
                placement_events.append(
                    {
                        "object_id": object_id,
                        "target_fixture_id": target_fixture_id,
                        "target_site_id": (
                            self._raw_support_site_to_external(target_site_id)
                            if isinstance(target_site_id, str)
                            else None
                        ),
                        "skipped": False,
                    }
                )

        _COINCIDENT_THRESHOLD = 0.03
        fixture_to_objects: dict[str, list[str]] = {}
        for object_id, object_state in objects.items():
            if object_id in held_object_ids:
                continue
            resolved_target = _resolve_fixture_location_target(object_state)
            if resolved_target is not None:
                fixture_id, _support_site_id = resolved_target
                fixture_to_objects.setdefault(fixture_id, []).append(object_id)

        for fixture_id, obj_ids in fixture_to_objects.items():
            if len(obj_ids) < 2:
                continue
            poses = {}
            for oid in obj_ids:
                try:
                    pos, _ = self._get_object_pose(oid)
                    obj = self.env.objects[oid]
                    radius = getattr(obj, "horizontal_radius", 0.05)
                    poses[oid] = (pos, radius)
                except Exception:
                    continue
            sorted_ids = sorted(poses.keys(), key=lambda o: poses[o][1])
            for i in range(len(sorted_ids)):
                for j in range(i + 1, len(sorted_ids)):
                    id_a, id_b = sorted_ids[i], sorted_ids[j]
                    pos_a, rad_a = poses[id_a]
                    pos_b, rad_b = poses[id_b]
                    xy_dist = float(np.linalg.norm(pos_a[:2] - pos_b[:2]))
                    if xy_dist >= _COINCIDENT_THRESHOLD:
                        continue
                    min_clearance = rad_a + rad_b + 0.005
                    fixture = self.runner._fixtures.get(fixture_id)
                    if fixture is not None and hasattr(fixture, "rot") and fixture.rot is not None:
                        angle = float(fixture.rot)
                        lateral = np.array([np.cos(angle), np.sin(angle)])
                    else:
                        lateral = np.array([1.0, 0.0])
                    placed = False
                    for sign in (1.0, -1.0):
                        candidate = pos_a.copy()
                        candidate[:2] += lateral * min_clearance * sign
                        if self.runner._validate_object_on_fixture(
                            candidate, fixture_id
                        ):
                            self._set_object_pose(id_a, candidate)
                            poses[id_a] = (candidate, rad_a)
                            placed = True
                            break
                    if not placed:
                        fallback = pos_a.copy()
                        fallback[:2] += lateral * min_clearance
                        self._set_object_pose(id_a, fallback)
                        poses[id_a] = (fallback, rad_a)

        for object_id in sorted(contained_object_ids, key=_containment_depth):
            object_state = objects[object_id]
            if object_id in held_object_ids:
                continue
            location = object_state.get("location")
            if isinstance(location, str) and location in objects:
                self._set_support_parent(object_id, location)
                if _preserve_pose_requested(object_state):
                    self.runner._set_object_location(object_id, location)
                    continue
                if self._current_support_object(object_id) == location:
                    self.runner._set_object_location(object_id, location)
                    continue
                try:
                    self._place_on_object_center(object_id, location)
                    self._settle_and_reseat_supported_object(object_id, location)
                except Exception as exc:
                    if _sim_tool_debug_enabled():
                        print(
                            f"[load_initial_state] Could not place {object_id} "
                            f"inside {location}: {exc}"
                        )

        support_children: dict[str, list[str]] = {}
        for object_id, object_state in objects.items():
            if object_id in held_object_ids:
                continue
            location = object_state.get("location")
            if isinstance(location, str) and location in objects:
                support_children.setdefault(location, []).append(object_id)

        for support_object_id, child_ids in support_children.items():
            if len(child_ids) <= 1:
                continue
            if self._support_object_should_keep_pose(support_object_id):
                continue
            if any(
                _preserve_pose_requested(objects[child_id])
                for child_id in child_ids
            ):
                continue
            if not self._supported_children_need_repack(support_object_id, child_ids):
                continue
            self._repack_supported_children(support_object_id, child_ids)
            self._settle_scene(steps=max(_PLACEMENT_SETTLE_STEPS, 4))

        if getattr(self, "_robot_spawn", "trajectory") == "trajectory":
            placement_requests = []
            for agent_id, agent_state in agents.items():
                location = agent_state.get("location")
                if isinstance(location, str) and location in self.runner._fixtures:
                    robot_idx = self._parse_agent_idx(agent_id)
                    location = self._canonical_navigation_fixture_id(location)
                    fixture_state = fixtures.get(location) or {}
                    fixture_type = str(
                        fixture_state.get("fixture_type") or ""
                    ).lower()
                    placement_requests.append(
                        (
                            0 if fixture_type in EXCLUSIVE_FIXTURE_TYPES else 1,
                            robot_idx,
                            agent_id,
                            location,
                        )
                    )

            # Default simulator poses are not part of the requested symbolic
            # initial state. Move every robot away before assigning positions,
            # otherwise a stale pose can falsely reserve the first target.
            set_pose = getattr(self.runner, "_set_robot_pose", None)
            if callable(set_pose):
                for robot_idx in range(self.runner._num_robots):
                    offset = float(robot_idx * 10)
                    set_pose(
                        robot_idx,
                        np.asarray([100.0 + offset, 100.0 + offset]),
                        0.0,
                    )
                self.env.sim.forward()

            # Constrained/exclusive targets claim their usable pose first;
            # shared workspaces then choose among the remaining valid poses.
            for _priority, robot_idx, agent_id, location in sorted(
                placement_requests,
                key=lambda request: (request[0], request[1]),
            ):
                require_front = self._fixture_requires_front_approach(
                    location
                ) or self._surface_fixture_prefers_front_approach(location)
                try:
                    placed = self.runner._move_robot_near_fixture(
                        robot_idx,
                        location,
                        require_front=require_front,
                    )
                except TypeError:
                    placed = self.runner._move_robot_near_fixture(robot_idx, location)
                if not placed:
                    raise RuntimeError(
                        f"Could not place {agent_id} at initial fixture {location!r}; "
                        "no valid unreserved working pose exists."
                    )
                self._normalize_robot_facing_for_stove_workstation(
                    robot_idx,
                    location,
                )
            for i in range(self.runner._num_robots):
                self.runner._rescue_robot_to_kitchen(i)

        for robot_idx, object_id in held_assignments.items():
            self._require_object(object_id)
            self._held_objects[robot_idx] = object_id
            self._held_object_offsets[robot_idx] = self._held_pose_offset(object_id)
            self._set_support_parent(object_id, f"held_by_robot_{robot_idx}")
            self._sync_held_object(robot_idx)
            agent_key = f"agent_{robot_idx}"
            agent_state = agents.get(agent_key, {})
            location = agent_state.get("location")
            if isinstance(location, str) and location in self.runner._fixtures:
                self.runner._set_object_location(object_id, location)

        stable_support_poses = self._snapshot_stable_support_poses()
        self._settle_scene(steps=max(_PLACEMENT_SETTLE_STEPS * 2, 10))
        self._restore_stable_support_poses(stable_support_poses)

        for support_object_id in list(self.env.objects.keys()):
            child_ids = self._iter_direct_supported_children(support_object_id)
            if not child_ids:
                continue
            if self._support_object_should_keep_pose(support_object_id):
                continue
            self._repair_supported_children_vertical_gaps(
                support_object_id,
                child_ids,
            )

        invalidate_visual_cache = getattr(self, "_invalidate_visual_cache", None)
        if callable(invalidate_visual_cache):
            invalidate_visual_cache()

        return {
            "loaded": True,
            "agents": sorted(agents.keys()),
            "objects": sorted(objects.keys()),
            "fixtures": sorted(fixtures.keys()),
            "placement_events": placement_events,
            "held_objects": {
                f"robot{robot_idx}": object_id
                for robot_idx, object_id in sorted(self._held_objects.items())
            },
        }
