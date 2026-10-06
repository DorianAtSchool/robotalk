"""Adapter for external full-trajectory JSON into executor tool calls.

Symbolic IDs in trajectory steps (e.g. "bun", "serving_surface") are resolved
to concrete sim IDs (e.g. "hotdog_bun", "dining_dining_group") via:

  1. **Sim ground truth** (preferred) — uses ``env.fixture_refs`` and
     ``env.object_cfgs`` placement info exposed in the scene description.
     Combined with the trajectory's ``initial_state`` type information,
     this resolves all symbols unambiguously.

  2. **Heuristic fallback** — any symbols not resolved by ground truth
     fall through to the existing type-match / token-match / ordinal heuristics.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import Any

from robotalk.tools.object_type_families import (
    OBJECT_TYPE_ALIASES,
    OBJECT_TYPE_FAMILIES,
    object_type_matches,
)

log = logging.getLogger(__name__)


@dataclass
class ResolutionRecord:
    entity_type: str
    requested_id: str
    resolved_id: str
    method: str
    confidence: float
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "entity_type": self.entity_type,
            "requested_id": self.requested_id,
            "resolved_id": self.resolved_id,
            "method": self.method,
            "confidence": self.confidence,
            "reason": self.reason,
        }


class TrajectoryAdapter:
    """Normalize and execute external trajectory JSON against SimToolExecutor."""

    _FIXTURE_TYPE_FAMILIES = {
        "cabinet": {
            "cabinet",
            "cabinet_single_door",
            "cabinet_double_door",
            "cabinet_with_door",
        },
        "counter": {
            "counter",
            "counter_non_dining",
            "counter_non_corner",
        },
        "drawer": {
            "drawer",
            "top_drawer",
        },
    }
    _OBJECT_TYPE_FAMILIES = OBJECT_TYPE_FAMILIES
    _FIXTURE_ARG_NAMES = {
        "anchor_fixture_id",
        "fixture_id",
        "reference_fixture_id",
        "target_id",
    }
    _OBJECT_ARG_NAMES = {
        "object_id",
        "reference_object_id",
        "support_object_id",
    }

    def __init__(
        self,
        executor,
        allow_approximate_ids: bool = True,
    ):
        self.executor = executor
        self.allow_approximate_ids = allow_approximate_ids
        self.scene = executor.get_scene_description()
        self._fixture_aliases: dict[str, str] = {}
        self._object_aliases: dict[str, str] = {}
        self._dispenser_aliases: dict[str, str] = {}
        self._resolution_log: list[ResolutionRecord] = []
        # Reverse maps: concrete env key → human-readable display name
        # Built during adapt() after resolution is complete.
        self._object_display_names: dict[str, str] = {}
        self._fixture_display_names: dict[str, str] = {}

    def _apply_sim_ground_truth(
        self,
        initial_state: dict[str, Any],
        grounding_symbols: dict[str, Any] | None = None,
    ) -> None:
        """Pre-populate alias caches from sim ground truth.

        Uses the trajectory's ``initial_state`` (object types and fixture
        types) combined with ``scene["object_placements"]`` and
        ``scene["fixture_refs"]`` to resolve symbolic IDs to concrete sim
        IDs with full confidence.

        ``grounding_symbols`` comes from ``trajectory["grounding_map"]["symbols"]``
        and carries richer resolver hints (e.g. ``anchor_fixture_symbol``,
        ``preferred_fixture_types``) that ``initial_state.fixtures`` doesn't have.
        """
        grounding_symbols = grounding_symbols or {}
        # object_placements: {env_obj_key: concrete_fixture_id} from env.object_cfgs
        object_placements = self.scene.get("object_placements", {})
        # fixture_refs: task-registered refs only (e.g. "coffee_machine", "cab"),
        # NOT all fixtures in the scene — just the ones the task explicitly registered
        # via register_fixture_ref() / get_fixture()
        fixture_refs = self.scene.get("fixture_refs", {})
        # scene_fixtures: ALL fixtures in the scene with positions, types, etc.
        # includes every counter, cabinet, appliance, etc. in the kitchen layout
        scene_fixtures = self.scene.get("fixtures", {})
        scene_objects = self.scene.get("objects", {})
        env_object_ids = set(self.scene.get("objects", {}).keys())

        if not object_placements and not fixture_refs and not env_object_ids:
            log.warning("No sim ground truth available; falling back to heuristics")
            return

        # --- Objects: match trajectory symbol → env.objects key by type ---
        traj_objects = initial_state.get("objects", {})
        traj_fixtures = initial_state.get("fixtures", {})
        for symbol, obj_state in traj_objects.items():
            grounding_spec = grounding_symbols.get(symbol) or {}
            explicit_index = grounding_spec.get("index")
            indexed_native_id = (
                f"obj_{explicit_index}"
                if grounding_spec.get("resolver") == "object_by_type"
                and isinstance(explicit_index, int)
                and explicit_index >= 0
                else None
            )
            if (
                indexed_native_id in env_object_ids
                and indexed_native_id not in self._object_aliases.values()
            ):
                self._object_aliases[symbol] = indexed_native_id
                self._resolution_log.append(
                    ResolutionRecord(
                        entity_type="object",
                        requested_id=symbol,
                        resolved_id=indexed_native_id,
                        method="explicit_grounding_index",
                        confidence=1.0,
                        reason=(
                            f"Grounding map index {explicit_index} selected native "
                            f"numbered object {indexed_native_id!r}."
                        ),
                    )
                )
                continue
            resolved = self._implicit_container_candidate_for_symbol(
                symbol,
                requested_object_state=obj_state,
                object_context=traj_objects,
            )
            if resolved is not None:
                obj_type = obj_state.get("object_type", "")
                self._object_aliases[symbol] = resolved
                self._resolution_log.append(
                    ResolutionRecord(
                        entity_type="object",
                        requested_id=symbol,
                        resolved_id=resolved,
                        method="implicit_container",
                        confidence=0.95,
                        reason=(
                            f"Mapped symbolic container {symbol!r} to native "
                            f"container object {resolved!r}."
                        ),
                    )
                )
                continue
            resolved = self._select_distinct_object_candidate(
                symbol,
                requested_object_state=obj_state,
                env_object_ids=env_object_ids,
                fixture_context=traj_fixtures,
            )
            if resolved is not None:
                obj_type = obj_state.get("object_type", "")
                self._object_aliases[symbol] = resolved
                self._resolution_log.append(ResolutionRecord(
                    entity_type="object",
                    requested_id=symbol,
                    resolved_id=resolved,
                    method="sim_ground_truth",
                    confidence=1.0,
                    reason=f"Matched object_type {obj_type!r} to env.objects[{resolved!r}]",
                ))

        # --- Fixtures: resolve via object placements and fixture refs ---
        # Build map: symbolic fixture → concrete ID by tracing object locations
        # If trajectory says object X is at fixture Y, and we resolved X to
        # env key K, then object_placements[K] gives the concrete fixture ID.
        traj_obj_locations = {}
        for symbol, obj_state in traj_objects.items():
            loc = obj_state.get("location")
            if isinstance(loc, str):
                traj_obj_locations.setdefault(loc, []).append(symbol)

        # --- Pass 1: strategies that don't depend on other fixtures ---
        for fixture_symbol, fixture_state in traj_fixtures.items():
            fixture_type = fixture_state.get("fixture_type", "")
            fixture_symbol_token = self._base_token(fixture_symbol)

            # Strategy 0: direct role binding from task fixture refs.
            # Prefer explicit task roles (e.g., "drawer", "cab", "stool")
            # over scene heuristics so symbolic fixture ids stay aligned
            # with task-defined references.
            for role, fxtr_id in fixture_refs.items():
                if not (isinstance(role, str) and isinstance(fxtr_id, str)):
                    continue
                role_token = self._base_token(role)
                if not (
                    fixture_symbol == role
                    or fixture_symbol_token == role_token
                ):
                    continue
                self._fixture_aliases[fixture_symbol] = fxtr_id
                self._resolution_log.append(ResolutionRecord(
                    entity_type="fixture",
                    requested_id=fixture_symbol,
                    resolved_id=fxtr_id,
                    method="sim_ground_truth",
                    confidence=1.0,
                    reason=f"fixture_refs direct role binding {role!r} -> {fxtr_id!r}",
                ))
                break

            if fixture_symbol in self._fixture_aliases:
                continue

            # Strategy 1: trace via object that lives at this fixture
            obj_symbols_here = traj_obj_locations.get(fixture_symbol, [])
            for obj_sym in obj_symbols_here:
                resolved_obj = self._object_aliases.get(obj_sym)
                concrete_fixture = None
                if resolved_obj and resolved_obj in object_placements:
                    concrete_fixture = object_placements[resolved_obj]
                elif resolved_obj:
                    scene_obj_info = scene_objects.get(resolved_obj) or {}
                    scene_location = scene_obj_info.get("location")
                    if isinstance(scene_location, str) and scene_location in scene_fixtures:
                        scene_fixture_type = str(
                            (scene_fixtures.get(scene_location) or {}).get(
                                "fixture_type",
                                "",
                            )
                        ).lower()
                        if self._fixture_type_matches(
                            str(fixture_type).lower(),
                            scene_fixture_type,
                        ):
                            concrete_fixture = scene_location
                if concrete_fixture:
                    self._fixture_aliases[fixture_symbol] = concrete_fixture
                    self._resolution_log.append(ResolutionRecord(
                        entity_type="fixture",
                        requested_id=fixture_symbol,
                        resolved_id=concrete_fixture,
                        method="sim_ground_truth",
                        confidence=1.0,
                        reason=(
                            f"Object {resolved_obj!r} is located at "
                            f"{concrete_fixture!r} in the simulator scene"
                        ),
                    ))
                    break

            if fixture_symbol in self._fixture_aliases:
                continue

            # Strategy 2: match fixture_refs (task-registered only) by type
            # Anchor-dependent symbols must reach pass 2. A generic role such
            # as ``counter`` is otherwise liable to steal a more precise
            # ``cabinet_parent_counter`` binding before its cabinet parent is
            # considered.
            has_anchor = isinstance(
                grounding_symbols.get(fixture_symbol, {}).get(
                    "anchor_fixture_symbol"
                ),
                str,
            )
            for role, fxtr_id in fixture_refs.items():
                if not (isinstance(role, str) and isinstance(fxtr_id, str)):
                    continue
                role_token = self._base_token(role)
                if not (
                    fixture_symbol == role
                    or fixture_symbol_token == role_token
                    or (
                        not has_anchor
                        and str(fixture_type).lower() == role.lower()
                    )
                ):
                    continue
                self._fixture_aliases[fixture_symbol] = fxtr_id
                self._resolution_log.append(ResolutionRecord(
                    entity_type="fixture",
                    requested_id=fixture_symbol,
                    resolved_id=fxtr_id,
                    method="sim_ground_truth",
                    confidence=1.0,
                    reason=f"fixture_refs[{role!r}] matched fixture_type {fixture_type!r}",
                ))
                break

        # --- Pass 2: anchor-dependent resolution (needs other fixtures resolved first) ---
        for fixture_symbol, fixture_state in traj_fixtures.items():
            if fixture_symbol in self._fixture_aliases:
                continue

            fixture_type = fixture_state.get("fixture_type", "")

            # Strategy 3: find the counter/surface that the anchor fixture sits on
            # (e.g. "staging_surface" = the counter the coffee_machine is placed on)
            # Uses parent_fixture from scene description (containment-based, same
            # logic as env.get_fixture(ref=...)). Falls back to nearest-center.
            # anchor/preferred info lives in grounding_map.symbols, not initial_state
            gm_entry = grounding_symbols.get(fixture_symbol, {})
            anchor_symbol = gm_entry.get("anchor_fixture_symbol")
            preferred_types = gm_entry.get("preferred_fixture_types", [])
            if anchor_symbol:
                # anchor must have been resolved in pass 1
                anchor_id = self._fixture_aliases.get(anchor_symbol)
                if anchor_id:
                    anchor_info = scene_fixtures.get(anchor_id, {})
                    match_types = set(preferred_types) | {fixture_type} if preferred_types else {fixture_type}

                    # Strategy 3a: parent_fixture — the counter the anchor sits ON
                    # (computed via point_in_fixture containment in get_scene_description)
                    parent_id = anchor_info.get("parent_fixture")
                    if parent_id and parent_id in scene_fixtures:
                        parent_type = scene_fixtures[parent_id].get("fixture_type", "")
                        if any(
                            self._fixture_type_matches(
                                str(match_type).lower(),
                                str(parent_type).lower(),
                            )
                            for match_type in match_types
                        ):
                            self._fixture_aliases[fixture_symbol] = parent_id
                            self._resolution_log.append(ResolutionRecord(
                                entity_type="fixture",
                                requested_id=fixture_symbol,
                                resolved_id=parent_id,
                                method="sim_ground_truth",
                                confidence=1.0,
                                reason=(
                                    f"Parent {parent_type!r} of anchor "
                                    f"{anchor_symbol!r} ({anchor_id!r}) via containment"
                                ),
                            ))
                            continue

                    # Strategy 3b: fallback — nearest scene fixture of matching type
                    anchor_pos = anchor_info.get("position", [0, 0, 0])
                    best_id = None
                    best_dist = float("inf")
                    for fid, finfo in scene_fixtures.items():
                        ftype = finfo.get("fixture_type", "")
                        if not any(
                            self._fixture_type_matches(
                                str(match_type).lower(),
                                str(ftype).lower(),
                            )
                            for match_type in match_types
                        ):
                            continue
                        fpos = finfo.get("position", [0, 0, 0])
                        dist = ((fpos[0] - anchor_pos[0]) ** 2 + (fpos[1] - anchor_pos[1]) ** 2) ** 0.5
                        if dist < best_dist:
                            best_dist = dist
                            best_id = fid
                    if best_id is not None:
                        self._fixture_aliases[fixture_symbol] = best_id
                        self._resolution_log.append(ResolutionRecord(
                            entity_type="fixture",
                            requested_id=fixture_symbol,
                            resolved_id=best_id,
                            method="sim_ground_truth",
                            confidence=0.8,
                            reason=(
                                f"Nearest {fixture_type!r} to anchor "
                                f"{anchor_symbol!r} ({anchor_id!r}) at dist {best_dist:.3f}m "
                                f"(parent_fixture unavailable, fell back to nearest-center)"
                            ),
                        ))

        resolved_count = len([r for r in self._resolution_log if r.method == "sim_ground_truth"])
        total_symbols = len(traj_objects) + len(traj_fixtures)
        if resolved_count == total_symbols:
            log.info("Sim ground truth resolved all %d symbols", resolved_count)
        else:
            log.warning(
                "Sim ground truth resolved %d/%d symbols; remaining will use heuristics",
                resolved_count, total_symbols,
            )

    def adapt(
        self,
        trajectory: dict[str, Any],
        output_dir: str | Path | None = None,
    ) -> dict[str, Any]:
        """Return a normalized executor-facing trajectory."""
        trajectory = deepcopy(trajectory)

        initial_state = trajectory.get("initial_state") or {}
        grounding_symbols = (trajectory.get("grounding_map") or {}).get("symbols") or {}
        self._apply_sim_ground_truth(initial_state, grounding_symbols)
        preserve_pose_object_ids = self._collect_reference_only_object_ids(trajectory)
        resolved_initial_state = self._adapt_initial_state(
            initial_state,
            preserve_pose_object_ids=preserve_pose_object_ids,
        )

        # Build display name maps: env key → human-readable name
        # Objects: use object_type from scene (sourced from info.cat)
        for env_key, obj_info in self.scene.get("objects", {}).items():
            self._object_display_names[env_key] = obj_info.get("object_type", env_key)
        # Also map from symbolic names to display names via aliases
        for symbol, env_key in self._object_aliases.items():
            if env_key not in self._object_display_names:
                self._object_display_names[env_key] = symbol
        # Fixtures: use symbolic name from trajectory as display name,
        # fall back to fixture_type from scene
        for symbol, env_key in self._fixture_aliases.items():
            self._fixture_display_names[env_key] = symbol
        for env_key, fxtr_info in self.scene.get("fixtures", {}).items():
            if env_key not in self._fixture_display_names:
                self._fixture_display_names[env_key] = fxtr_info.get("fixture_type", env_key)

        tool_calls = []

        for step in trajectory.get("steps", []):
            tool_calls.append(
                self._adapt_step(
                    step,
                    resolved_initial_state=resolved_initial_state,
                    output_dir=output_dir,
                )
            )

        return {
            "trajectory_id": trajectory.get("trajectory_id"),
            "composite_task": trajectory.get("composite_task"),
            "agents": deepcopy(trajectory.get("agents", [])),
            "initial_state": resolved_initial_state,
            "tool_calls": tool_calls,
            "resolution_log": [record.to_dict() for record in self._resolution_log],
            "display_names": {
                "objects": dict(self._object_display_names),
                "fixtures": dict(self._fixture_display_names),
            },
        }

    def execute(
        self,
        trajectory: dict[str, Any],
        output_dir: str | Path | None = None,
        fps: int = 2,
        skip_videos: bool = False,
        save_debug_frames: bool = False,
    ) -> dict[str, Any]:
        """Adapt, load initial state, then execute with frames and video.

        Delegates to ``executor.run_tool_plan()`` so that before/after frames
        and per-camera MP4 videos are generated automatically.
        """
        adapted = self.adapt(trajectory, output_dir=output_dir)

        # Save pre-initial-state frames (before doors are closed / objects moved).
        # Useful for debugging: confirms objects are spawned correctly by the sim
        # even if the trajectory's initial_state hides them (e.g. closes cabinet).
        if output_dir is not None and save_debug_frames:
            self.executor.save_scene_frames(output_dir, prefix="pre_initial_state")

        load_summary = self.executor.load_initial_state(adapted["initial_state"])

        # In "trajectory" spawn mode, robots were just repositioned by
        # load_initial_state.  Re-save the initial map so it reflects the
        # trajectory's agent locations rather than the sim default.
        robot_spawn = getattr(self.executor, "_robot_spawn", "sim")
        if output_dir is not None and robot_spawn == "trajectory":
            self.executor.save_placement_map(
                output_dir, prefix="initial",
                clean_labels=getattr(self.executor, "_clean_map_labels", True),
            )
            if save_debug_frames:
                # Save post-initial-state rendered frames only when explicitly
                # debugging spawn / initial-state alignment.
                self.executor.save_scene_frames(output_dir, prefix="initial")

        if output_dir is not None:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            with open(output_path / "adapted_trajectory.json", "w") as f:
                json.dump(adapted, f, indent=2)

        # run_tool_plan handles rendering, frame saving, and video generation.
        if hasattr(self.executor, "run_tool_plan"):
            plan_metadata = self.executor.run_tool_plan(
                tool_calls=adapted["tool_calls"],
                output_dir=output_dir or ".",
                fps=fps,
                skip_videos=skip_videos,
            )
        else:
            plan_metadata = {"steps": []}
            for tool_call in adapted["tool_calls"]:
                robot_idx = tool_call.get("robot_idx", 0)
                args = tool_call.get("args", {})
                result = self.executor.execute(
                    tool_call["tool"],
                    robot_idx=robot_idx,
                    **args,
                )
                plan_metadata["steps"].append(
                    {
                        "tool": tool_call["tool"],
                        "robot_idx": robot_idx,
                        "args": args,
                        "success": result.success,
                        "details": result.details,
                    }
                )

        metadata = {
            "trajectory_id": adapted.get("trajectory_id"),
            "composite_task": adapted.get("composite_task"),
            "load_initial_state": load_summary,
            "resolution_log": adapted["resolution_log"],
            **plan_metadata,
        }

        if output_dir is not None:
            with open(output_path / "trajectory_execution_metadata.json", "w") as f:
                json.dump(metadata, f, indent=2)

        return metadata

    def _collect_reference_only_object_ids(self, trajectory: dict[str, Any]) -> set[str]:
        """Collect object ids that are used only as static reference anchors."""

        reference_object_ids: set[str] = set()
        manipulated_object_ids: set[str] = set()
        for step in trajectory.get("steps", []) or []:
            if not isinstance(step, dict):
                continue
            args = step.get("args")
            if not isinstance(args, dict):
                continue
            reference_object_id = args.get("reference_object_id")
            if isinstance(reference_object_id, str):
                reference_object_ids.add(reference_object_id)
            reference_id = args.get("reference_id")
            if isinstance(reference_id, str):
                reference_object_ids.add(reference_id)
            object_id = args.get("object_id")
            if not isinstance(object_id, str):
                continue
            if step.get("tool") in {
                "pick_up_object",
                "place_on_surface",
                "place_in_receptacle",
                "place_on_object",
                "place_next_to",
                "place_under",
            }:
                manipulated_object_ids.add(object_id)

        return {
            object_id
            for object_id in reference_object_ids
            if object_id not in manipulated_object_ids
        }

    def _adapt_initial_state(
        self,
        initial_state: dict[str, Any],
        *,
        preserve_pose_object_ids: set[str] | None = None,
    ) -> dict[str, Any]:
        resolved_fixtures = {}
        resolved_objects = {}
        resolved_agents = {}
        resolved_machine_state = {}
        preserve_pose_object_ids = preserve_pose_object_ids or set()

        fixture_context = initial_state.get("fixtures", {})
        object_context = initial_state.get("objects", {})
        machine_state = initial_state.get("machine_state", {})

        for requested_fixture_id, fixture_state in fixture_context.items():
            resolved_fixture_id = self._resolve_fixture_id(
                requested_fixture_id,
                requested_fixture_state=fixture_state,
            )
            resolved_fixture_state = deepcopy(fixture_state)
            parts = {}
            for part_id, part_state in fixture_state.get("parts", {}).items():
                parts[self._normalize_part_id(part_id, part_state)] = deepcopy(
                    part_state
                )
            if parts:
                resolved_fixture_state["parts"] = parts
            # parent_fixture names a sibling fixture, so it has to travel through the
            # same alias path as the keys. Left raw it becomes a dangling symbolic id
            # that every concrete-id lookup misses silently.
            parent_fixture = fixture_state.get("parent_fixture")
            if isinstance(parent_fixture, str):
                resolved_fixture_state["parent_fixture"] = self._resolve_fixture_id(
                    parent_fixture,
                    requested_fixture_state=fixture_context.get(parent_fixture),
                )
            resolved_fixtures[resolved_fixture_id] = resolved_fixture_state

        for requested_fixture_id, machine_cfg in machine_state.items():
            # machine_state keys may be task-state namespaces (e.g.
            # "hot_dog_setup") rather than fixture references.  Only attempt
            # resolution when the key looks like a known fixture or alias.
            if (
                requested_fixture_id in self.scene.get("fixtures", {})
                or requested_fixture_id in self._fixture_aliases
                or requested_fixture_id in fixture_context
            ):
                resolved_fixture_id = self._resolve_fixture_id(
                    requested_fixture_id,
                    requested_fixture_state=fixture_context.get(requested_fixture_id),
                )
            else:
                resolved_fixture_id = requested_fixture_id
            resolved_machine_state[resolved_fixture_id] = deepcopy(machine_cfg)
            if not isinstance(machine_cfg, dict):
                continue
            dispenser_id = machine_cfg.get("dispenser_id")
            if isinstance(dispenser_id, str):
                self._dispenser_aliases[dispenser_id] = resolved_fixture_id

        for requested_object_id, object_state in object_context.items():
            resolved_object_id = self._resolve_object_id(
                requested_object_id,
                requested_object_state=object_state,
                fixture_context=fixture_context,
            )
            resolved_state = deepcopy(object_state)
            location = object_state.get("location")
            if isinstance(location, str):
                explicit_target_site = isinstance(object_state.get("target_site_id"), str)
                resolved_target_site_id: str | None = None
                # Location can be a fixture ("mug_source_fixture") or another
                # object ("ingredient_bowl" for slices inside a bowl).  Check
                # object aliases first to avoid sending object names through
                # fixture resolution, which would fall back to a random fixture.
                if location in self._object_aliases or location in object_context:
                    resolved_state["location"] = self._resolve_object_id(
                        location,
                        requested_object_state=object_context.get(location),
                    )
                elif self._is_known_support_site(location, fixture_context):
                    (
                        resolved_state["location"],
                        resolved_target_site_id,
                    ) = self._resolve_support_site_target(
                        location,
                        symbolic_fixture_context=fixture_context,
                    )
                else:
                    resolved_fixture_id = self._resolve_fixture_id(
                        location,
                        requested_fixture_state=fixture_context.get(location),
                    )
                    (
                        resolved_state["location"],
                        resolved_target_site_id,
                    ) = self._resolve_initial_fixture_location(
                        resolved_object_id=resolved_object_id,
                        explicit_target_site=explicit_target_site,
                        requested_location=location,
                        resolved_fixture_id=resolved_fixture_id,
                        symbolic_fixture_context=fixture_context,
                    )
                    scene_obj_info = (self.scene.get("objects") or {}).get(
                        resolved_object_id,
                        {},
                    )
                    if (
                        not explicit_target_site
                        and "preserve_pose" not in resolved_state
                        and (
                            scene_obj_info.get("location") == resolved_state["location"]
                            or requested_object_id in preserve_pose_object_ids
                            or resolved_object_id in preserve_pose_object_ids
                        )
                    ):
                        resolved_state["preserve_pose"] = True
                if isinstance(resolved_target_site_id, str):
                    resolved_state["target_site_id"] = resolved_target_site_id
                else:
                    resolved_state.pop("target_site_id", None)
            resolved_objects[resolved_object_id] = resolved_state

        for agent_id, agent_state in initial_state.get("agents", {}).items():
            resolved_state = deepcopy(agent_state)
            location = agent_state.get("location")
            if isinstance(location, str):
                resolved_state["location"] = self._resolve_fixture_id(
                    location,
                    requested_fixture_state=fixture_context.get(location),
                )
            held_object = agent_state.get("held_object")
            if isinstance(held_object, str):
                resolved_state["held_object"] = self._resolve_object_id(
                    held_object,
                    requested_object_state=object_context.get(held_object),
                    fixture_context=fixture_context,
                )
            resolved_agents[agent_id] = resolved_state

        return {
            "agents": resolved_agents,
            "objects": resolved_objects,
            "fixtures": resolved_fixtures,
            "machine_state": resolved_machine_state,
        }

    def _resolve_initial_fixture_location(
        self,
        *,
        resolved_object_id: str,
        explicit_target_site: bool,
        requested_location: str,
        resolved_fixture_id: str,
        symbolic_fixture_context: dict[str, Any],
    ) -> tuple[str, str | None]:
        """Prefer a concrete support site when one can be inferred conservatively."""
        scene_obj_info = (self.scene.get("objects") or {}).get(
            resolved_object_id,
            {},
        )
        current_scene_location = scene_obj_info.get("location")
        can_reuse_scene_site = current_scene_location == resolved_fixture_id

        symbolic_fixture_state = symbolic_fixture_context.get(requested_location)
        support_sites = (
            symbolic_fixture_state.get("support_sites")
            if isinstance(symbolic_fixture_state, dict)
            else None
        )
        if explicit_target_site and isinstance(support_sites, dict) and len(support_sites) == 1:
            symbolic_site_id = next(iter(support_sites))
            if isinstance(symbolic_site_id, str):
                return self._resolve_support_site_target(
                    symbolic_site_id,
                    parent_fixture_id=resolved_fixture_id,
                    symbolic_fixture_context=symbolic_fixture_context,
                )

        if can_reuse_scene_site and hasattr(self.executor, "_infer_object_support_site"):
            try:
                inferred_site_id = self.executor._infer_object_support_site(  # noqa: SLF001
                    resolved_object_id,
                    resolved_fixture_id,
                )
            except Exception:
                inferred_site_id = None
            if isinstance(inferred_site_id, str):
                if hasattr(self.executor, "_normalize_target_site_id_for_placement"):
                    try:
                        normalized_site_id = self.executor._normalize_target_site_id_for_placement(  # noqa: SLF001
                            resolved_fixture_id,
                            inferred_site_id,
                        )
                    except Exception:
                        normalized_site_id = inferred_site_id
                else:
                    normalized_site_id = inferred_site_id
                if not isinstance(normalized_site_id, str):
                    return resolved_fixture_id, None
                if hasattr(self.executor, "_raw_support_site_to_external"):
                    try:
                        return (
                            resolved_fixture_id,
                            self.executor._raw_support_site_to_external(  # noqa: SLF001
                                normalized_site_id
                            ),
                        )
                    except Exception:
                        return resolved_fixture_id, normalized_site_id
                return resolved_fixture_id, normalized_site_id

        if (
            hasattr(self.executor, "_fixture_requires_explicit_site")
            and hasattr(self.executor, "_default_support_site_for_unspecified_fixture")
        ):
            try:
                requires_site = self.executor._fixture_requires_explicit_site(  # noqa: SLF001
                    resolved_fixture_id
                )
            except Exception:
                requires_site = False
            if requires_site and (explicit_target_site or can_reuse_scene_site):
                try:
                    default_site_id = self.executor._default_support_site_for_unspecified_fixture(  # noqa: SLF001
                        resolved_fixture_id,
                        incoming_object_id=resolved_object_id,
                    )
                except Exception:
                    default_site_id = None
                if isinstance(default_site_id, str):
                    return resolved_fixture_id, default_site_id

        return resolved_fixture_id, None

    def _resolve_support_site_target(
        self,
        requested_site_id: str,
        *,
        parent_fixture_id: str | None = None,
        symbolic_fixture_context: dict[str, Any] | None = None,
    ) -> tuple[str, str | None]:
        """Resolve one support-site reference into fixture and site components."""

        resolved_parent_fixture_id = parent_fixture_id
        if resolved_parent_fixture_id is None and symbolic_fixture_context is not None:
            for symbolic_fixture_id, fixture_state in symbolic_fixture_context.items():
                if not isinstance(fixture_state, dict):
                    continue
                support_sites = fixture_state.get("support_sites")
                if (
                    isinstance(support_sites, dict)
                    and requested_site_id in support_sites
                ) or (
                    isinstance(support_sites, list)
                    and requested_site_id in support_sites
                ):
                    resolved_parent_fixture_id = self._resolve_fixture_id(
                        symbolic_fixture_id,
                        requested_fixture_state=fixture_state,
                    )
                    break

        if resolved_parent_fixture_id is None:
            resolved_parent_fixture_id = self._resolve_support_site_parent(
                requested_site_id
            )

        if (
            resolved_parent_fixture_id is not None
            and hasattr(self.executor, "_resolve_fixture_site_id")
            and hasattr(self.executor, "_raw_support_site_to_external")
        ):
            try:
                raw_site_id = self.executor._resolve_fixture_site_id(  # noqa: SLF001
                    resolved_parent_fixture_id,
                    requested_site_id,
                )
            except Exception:
                raw_site_id = None
            if isinstance(raw_site_id, str):
                try:
                    return (
                        resolved_parent_fixture_id,
                        self.executor._raw_support_site_to_external(raw_site_id),  # noqa: SLF001
                    )
                except Exception:
                    return resolved_parent_fixture_id, raw_site_id

        if resolved_parent_fixture_id is not None:
            return resolved_parent_fixture_id, requested_site_id

        return requested_site_id, None

    def _resolve_support_site_reference(
        self,
        requested_site_id: str,
        *,
        parent_fixture_id: str | None = None,
        symbolic_fixture_context: dict[str, Any] | None = None,
    ) -> str:
        """Resolve one symbolic support site into the concrete external site name."""
        _resolved_parent_fixture_id, resolved_site_id = self._resolve_support_site_target(
            requested_site_id,
            parent_fixture_id=parent_fixture_id,
            symbolic_fixture_context=symbolic_fixture_context,
        )
        return resolved_site_id if isinstance(resolved_site_id, str) else requested_site_id

    def _adapt_step(
        self,
        step: dict[str, Any],
        resolved_initial_state: dict[str, Any],
        output_dir: str | Path | None,
    ) -> dict[str, Any]:
        tool_name = str(step.get("tool", "")).strip()
        args = deepcopy(step.get("args", {}))
        agent_id = step.get("agent", "agent_0")
        robot_idx = self.executor._parse_agent_idx(agent_id)
        step_index = step.get("step")

        if tool_name == "place_under_dispenser":
            tool_name = "place_under"
            dispenser_id = args.pop("dispenser_id", None)
            if isinstance(dispenser_id, str):
                args["reference_fixture_id"] = self._resolve_dispenser_id(dispenser_id)

        if tool_name in {"get_env_image", "get_agent_image", "get_image"}:
            if "views" not in args and "view" in args:
                args["views"] = [args.pop("view")]
            elif isinstance(args.get("views"), str):
                args["views"] = [args["views"]]

            if tool_name == "get_agent_image":
                args.setdefault("agent_id", agent_id)

            image_paths = args.get("image_paths")
            if not (isinstance(image_paths, list) and image_paths):
                image_paths = step.get("image_paths")
            views = args.get("views") if isinstance(args.get("views"), list) else []
            if isinstance(image_paths, list) and image_paths:
                args["image_paths"] = [
                    str(
                        self._resolve_image_output_path(
                            image_path,
                            output_dir,
                            view_name=views[idx] if idx < len(views) else None,
                        )
                    )
                    for idx, image_path in enumerate(image_paths)
                ]
            else:
                image_path = args.pop("image_path", None)
                if not isinstance(image_path, str):
                    image_path = step.get("image_path")
                if isinstance(image_path, str):
                    args["image_paths"] = [
                        str(
                            self._resolve_image_output_path(
                                image_path,
                                output_dir,
                                view_name=views[0] if views else None,
                            )
                        )
                    ]
            tool_name = "get_image"

        args = self._resolve_step_args(
            tool_name,
            args,
            resolved_initial_state=resolved_initial_state,
        )

        # Build display_args: human-readable names for VLM consumption
        display_args = {}
        for arg_name, value in args.items():
            if not isinstance(value, str):
                continue
            if arg_name in self._OBJECT_ARG_NAMES:
                display_args[arg_name] = self._object_display_names.get(value, value)
            elif arg_name in self._FIXTURE_ARG_NAMES or arg_name == "receptacle_id":
                display_args[arg_name] = self._fixture_display_names.get(value, value)

        result = {
            "tool": tool_name,
            "robot_idx": robot_idx,
            "args": args,
            "metadata": {
                "step_index": step_index,
                "source_agent": agent_id,
                "reasoning": step.get("reasoning"),
                "image_path": step.get("image_path"),
                "image_paths": deepcopy(step.get("image_paths")),
            },
        }
        if display_args:
            result["display_args"] = display_args
        return result

    def _resolve_step_args(
        self,
        tool_name: str,
        args: dict[str, Any],
        resolved_initial_state: dict[str, Any],
    ) -> dict[str, Any]:
        resolved_args = deepcopy(args)
        known_objects = set(self._object_aliases) | set(self._object_aliases.values())
        known_objects.update(resolved_initial_state.get("objects", {}))
        known_objects.update(self.scene.get("objects", {}))
        for arg_name, value in list(resolved_args.items()):
            if not isinstance(value, str):
                continue
            # target_id is polymorphic in the public tool interface.  In
            # place_on_object it is always an object; in place_in_receptacle it
            # may be either a movable receptacle or a fixture.  Resolving every
            # target_id as a fixture turned symbolic plates into nearby
            # cabinets during rendering.
            if arg_name == "target_id" and (
                tool_name == "place_on_object"
                or (tool_name == "place_in_receptacle" and value in known_objects)
            ):
                resolved_args[arg_name] = self._resolve_object_id(
                    value,
                    requested_object_state=(
                        resolved_initial_state.get("objects", {}).get(value) or None
                    ),
                )
            elif arg_name in self._FIXTURE_ARG_NAMES:
                resolved_args[arg_name] = self._resolve_fixture_id(
                    value,
                    requested_fixture_state=(
                        resolved_initial_state.get("fixtures", {}).get(value) or None
                        ),
                    )
            elif arg_name in {"source_id", "support_id"}:
                if arg_name == "source_id" and (
                    value in self._object_aliases
                    or value in resolved_initial_state.get("objects", {})
                    or value in self.scene.get("objects", {})
                ):
                    resolved_args[arg_name] = self._resolve_object_id(
                        value,
                        requested_object_state=(
                            resolved_initial_state.get("objects", {}).get(value) or None
                        ),
                    )
                    continue
                if value in self._fixture_aliases or value in self.scene.get("fixtures", {}):
                    resolved_args[arg_name] = self._resolve_fixture_id(
                        value,
                        requested_fixture_state=(
                            resolved_initial_state.get("fixtures", {}).get(value) or None
                        ),
                    )
                    continue

                initial_fixtures = resolved_initial_state.get("fixtures", {})
                if isinstance(initial_fixtures, dict):
                    is_known_support_site = any(
                        isinstance(fixture_state, dict)
                        and (
                            value in (fixture_state.get("support_sites") or {})
                            or value in (fixture_state.get("support_sites") or [])
                        )
                        for fixture_state in initial_fixtures.values()
                    )
                    if is_known_support_site:
                        if arg_name == "source_id":
                            parent_fixture_id, site_id = self._resolve_support_site_target(
                                value,
                                symbolic_fixture_context=initial_fixtures,
                            )
                            resolved_args[arg_name] = parent_fixture_id
                            if isinstance(site_id, str):
                                resolved_args.setdefault("source_site_id", site_id)
                        else:
                            resolved_args[arg_name] = value
                        continue

                resolved_args[arg_name] = value
            elif arg_name in self._OBJECT_ARG_NAMES:
                # The value may actually be a fixture (e.g. "toaster_oven"
                # passed as reference_object_id in place_next_to).  Check
                # fixture aliases first to avoid a bad object fallback.
                if value in self._fixture_aliases or value in self.scene.get("fixtures", {}):
                    resolved_args[arg_name] = self._resolve_fixture_id(
                        value,
                        requested_fixture_state=(
                            resolved_initial_state.get("fixtures", {}).get(value) or None
                        ),
                    )
                else:
                    resolved_args[arg_name] = self._resolve_object_id(
                        value,
                        requested_object_state=(
                            resolved_initial_state.get("objects", {}).get(value) or None
                        ),
                    )
            elif arg_name == "receptacle_id":
                if (
                    value in self.scene.get("fixtures", {})
                    or value in self._fixture_aliases
                ):
                    resolved_args[arg_name] = self._resolve_fixture_id(value)
                elif self._is_known_support_site(
                    value,
                    resolved_initial_state.get("fixtures", {}),
                ):
                    resolved_args[arg_name] = self._resolve_support_site_reference(
                        value,
                        symbolic_fixture_context=resolved_initial_state.get("fixtures", {}),
                    )
                else:
                    resolved_args[arg_name] = self._resolve_object_id(value)
            elif arg_name in {"target_site_id", "source_site_id"}:
                parent_fixture_id = None
                for fixture_arg_name in (
                    "target_id",
                    "support_id",
                    "source_id",
                    "fixture_id",
                    "reference_fixture_id",
                ):
                    fixture_arg_value = resolved_args.get(fixture_arg_name)
                    if not isinstance(fixture_arg_value, str):
                        continue
                    if fixture_arg_value in self.scene.get("fixtures", {}):
                        parent_fixture_id = fixture_arg_value
                        break
                    if (
                        fixture_arg_value in self._fixture_aliases
                        or fixture_arg_value in self._fixture_aliases.values()
                    ):
                        parent_fixture_id = self._resolve_fixture_id(fixture_arg_value)
                        break
                if parent_fixture_id is not None:
                    requested_fixture_id = None
                    if value in self.scene.get("fixtures", {}) or value in self._fixture_aliases:
                        requested_fixture_id = self._resolve_fixture_id(value)
                    elif value in resolved_initial_state.get("fixtures", {}):
                        requested_fixture_id = self._resolve_fixture_id(
                            value,
                            requested_fixture_state=(
                                resolved_initial_state.get("fixtures", {}).get(value) or None
                            ),
                        )
                    if requested_fixture_id == parent_fixture_id:
                        resolved_args[arg_name] = None
                        continue
                resolved_args[arg_name] = self._resolve_support_site_reference(
                    value,
                    parent_fixture_id=parent_fixture_id,
                    symbolic_fixture_context=resolved_initial_state.get("fixtures", {}),
                )
            elif arg_name == "part_id":
                fixture_key = resolved_args.get("target_id")
                resolved_args[arg_name] = self._normalize_part_id(
                    value,
                    (
                        resolved_initial_state.get("fixtures", {})
                        .get(fixture_key, {})
                        .get("parts", {})
                        .get(value, {})
                    ),
                )
        if tool_name == "place_in_receptacle" and not isinstance(
            resolved_args.get("target_site_id"),
            str,
        ):
            receptacle_value = resolved_args.get("target_id") or resolved_args.get(
                "receptacle_id"
            )
            if isinstance(receptacle_value, str) and receptacle_value in self.scene.get(
                "fixtures",
                {},
            ):
                incoming_object_id = resolved_args.get("object_id")
                if (
                    hasattr(self.executor, "_fixture_requires_explicit_site")
                    and hasattr(
                        self.executor,
                        "_default_support_site_for_unspecified_fixture",
                    )
                    and self.executor._fixture_requires_explicit_site(  # noqa: SLF001
                        receptacle_value
                    )
                ):
                    default_site_id = self.executor._default_support_site_for_unspecified_fixture(  # noqa: SLF001
                        receptacle_value,
                        incoming_object_id=(
                            incoming_object_id
                            if isinstance(incoming_object_id, str)
                            else None
                        ),
                    )
                    if isinstance(default_site_id, str):
                        resolved_args["target_site_id"] = default_site_id
        return resolved_args

    def _resolve_output_path(
        self,
        image_path: str,
        output_dir: str | Path | None,
    ) -> Path:
        path = Path(image_path)
        if path.is_absolute() or output_dir is None:
            return path
        return Path(output_dir) / path

    def _resolve_image_output_path(
        self,
        image_path: str,
        output_dir: str | Path | None,
        *,
        view_name: str | None = None,
    ) -> Path:
        path = self._resolve_output_path(image_path, output_dir)
        normalized_view = str(view_name).strip().lower() if view_name is not None else None
        if normalized_view != "map" and path.suffix.lower() == ".png":
            return path.with_suffix(".jpg")
        return path

    def _resolve_dispenser_id(self, dispenser_id: str) -> str:
        if dispenser_id in self._dispenser_aliases:
            return self._dispenser_aliases[dispenser_id]
        if dispenser_id in self.scene.get("fixtures", {}):
            return dispenser_id
        raise ValueError(f"Unknown dispenser_id: {dispenser_id!r}")

    def _resolve_fixture_id(
        self,
        requested_id: str,
        requested_fixture_state: dict[str, Any] | None = None,
    ) -> str:
        if requested_id in self._fixture_aliases:
            return self._fixture_aliases[requested_id]
        if requested_id in self.scene.get("fixtures", {}):
            self._fixture_aliases[requested_id] = requested_id
            return requested_id

        fixture_type = None
        if requested_fixture_state is not None:
            fixture_type = requested_fixture_state.get("fixture_type")
        candidate_ids = self._fixture_candidates_for_type(fixture_type)
        resolved_id, method, confidence, reason = self._choose_candidate(
            requested_id=requested_id,
            candidate_ids=candidate_ids,
            entity_type="fixture",
            type_hint=fixture_type,
        )
        self._fixture_aliases[requested_id] = resolved_id
        self._resolution_log.append(
            ResolutionRecord(
                entity_type="fixture",
                requested_id=requested_id,
                resolved_id=resolved_id,
                method=method,
                confidence=confidence,
                reason=reason,
            )
        )
        return resolved_id

    def _resolve_object_id(
        self,
        requested_id: str,
        requested_object_state: dict[str, Any] | None = None,
        fixture_context: dict[str, Any] | None = None,
    ) -> str:
        if requested_id in self._object_aliases:
            return self._object_aliases[requested_id]
        if requested_id in self.scene.get("objects", {}):
            self._object_aliases[requested_id] = requested_id
            return requested_id

        object_type = None
        if requested_object_state is not None:
            object_type = requested_object_state.get("object_type")
        candidate_ids = self._object_candidates_for_type(object_type)
        candidate_ids = self._filter_object_candidates_by_location_hint(
            candidate_ids,
            requested_object_state,
            fixture_context=fixture_context,
        )
        candidate_ids = self._filter_distinct_object_candidates(
            candidate_ids,
            requested_id=requested_id,
        )
        if not candidate_ids:
            all_location_fallback_candidates = self._object_candidates_for_location_hint(
                requested_object_state,
                fixture_context=fixture_context,
            )
            zero_based_ordinal = self._extract_ordinal(requested_id)
            if (
                zero_based_ordinal is not None
                and 0 <= zero_based_ordinal < len(all_location_fallback_candidates)
            ):
                fallback_id = all_location_fallback_candidates[zero_based_ordinal]
                reserved_ids = {
                    resolved_id
                    for symbol, resolved_id in self._object_aliases.items()
                    if symbol != requested_id
                }
                if fallback_id not in reserved_ids:
                    self._object_aliases[requested_id] = fallback_id
                    self._resolution_log.append(
                        ResolutionRecord(
                            entity_type="object",
                            requested_id=requested_id,
                            resolved_id=fallback_id,
                            method="location_ordinal_fallback",
                            confidence=0.4,
                            reason=(
                                f"Location fallback for {requested_id!r} at "
                                f"{requested_object_state.get('location')!r}"
                            ),
                        )
                    )
                    return fallback_id
            location_fallback_candidates = self._filter_distinct_object_candidates(
                all_location_fallback_candidates,
                requested_id=requested_id,
            )
            candidate_ids = location_fallback_candidates
        resolved_id, method, confidence, reason = self._choose_candidate(
            requested_id=requested_id,
            candidate_ids=candidate_ids,
            entity_type="object",
            type_hint=object_type,
        )
        self._object_aliases[requested_id] = resolved_id
        self._resolution_log.append(
            ResolutionRecord(
                entity_type="object",
                requested_id=requested_id,
                resolved_id=resolved_id,
                method=method,
                confidence=confidence,
                reason=reason,
            )
        )
        return resolved_id

    def _fixture_candidates_for_type(self, fixture_type: str | None) -> list[str]:
        fixtures = self.scene.get("fixtures", {})
        if fixture_type is None:
            return sorted(fixtures.keys())
        fixture_type = str(fixture_type).lower()
        return sorted(
            fixture_id
            for fixture_id, fixture_info in fixtures.items()
            if self._fixture_type_matches(
                fixture_type,
                str(fixture_info.get("fixture_type", "")).lower(),
            )
        )

    def _object_candidates_for_type(self, object_type: str | None) -> list[str]:
        objects = self.scene.get("objects", {})
        if object_type is None:
            return sorted(objects.keys())
        object_type = str(object_type).lower()
        matches = []
        for object_id, object_info in objects.items():
            actual_type = str(object_info.get("object_type", "")).lower()
            if self._object_type_matches(object_type, actual_type):
                matches.append(object_id)
        return sorted(matches)

    def _broader_object_fallback_candidates(
        self,
        requested_id: str,
        type_hint: str,
    ) -> list[str]:
        scene_objects = self.scene.get("objects", {})
        reserved_ids = set(self._object_aliases.values())
        requested_type = str(type_hint).lower()
        semantic_tokens = {
            token
            for token in self._base_token(requested_id).split("_")
            if token
        }
        semantic_tokens.update(
            token for token in self._base_token(requested_type).split("_") if token
        )
        semantic_tokens.update(OBJECT_TYPE_FAMILIES.get(requested_type, set()))
        semantic_tokens.update(OBJECT_TYPE_ALIASES.get(requested_type, ()))

        candidate_ids: list[str] = []
        for object_id, object_info in scene_objects.items():
            if object_id in reserved_ids:
                continue
            actual_type = str(object_info.get("object_type", "")).lower()
            actual_tokens = {
                token for token in self._base_token(object_id).split("_") if token
            }
            actual_tokens.update(
                token for token in self._base_token(actual_type).split("_") if token
            )
            if semantic_tokens.intersection(actual_tokens):
                candidate_ids.append(object_id)
        return sorted(candidate_ids)

    def _object_type_matches(
        self,
        requested_type: str,
        actual_type: str,
    ) -> bool:
        return object_type_matches(requested_type, actual_type)

    def _implicit_container_candidate_for_symbol(
        self,
        requested_id: str,
        *,
        requested_object_state: dict[str, Any] | None,
        object_context: dict[str, Any] | None,
    ) -> str | None:
        """Resolve symbolic containers created by RoboCasa try_to_place_in.

        Native tasks often define only a child object, e.g. ``obj`` with
        ``try_to_place_in="pan"``. RoboCasa exposes the pan as ``obj_container``.
        TaskSpecs should still be able to say high-level ``steak`` and ``pan``.
        """

        if not isinstance(requested_object_state, dict) or not isinstance(
            object_context,
            dict,
        ):
            return None
        requested_type = requested_object_state.get("object_type")
        if not isinstance(requested_type, str):
            return None

        scene_objects = self.scene.get("objects", {})
        assigned_ids = set(self._object_aliases.values())
        candidates: list[str] = []
        for child_symbol, child_state in object_context.items():
            if child_symbol == requested_id or not isinstance(child_state, dict):
                continue
            if child_state.get("location") != requested_id:
                continue
            resolved_child_id = self._object_aliases.get(child_symbol)
            if not isinstance(resolved_child_id, str):
                continue
            for container_id in (
                f"{resolved_child_id}_container",
                f"{resolved_child_id}_container_0",
            ):
                if container_id in assigned_ids:
                    continue
                container_info = scene_objects.get(container_id)
                if not isinstance(container_info, dict):
                    continue
                actual_type = str(container_info.get("object_type", "")).lower()
                if self._object_type_matches(str(requested_type).lower(), actual_type):
                    candidates.append(container_id)

        if len(candidates) == 1:
            return candidates[0]
        return None

    def _resolve_requested_location_hint(
        self,
        requested_location: Any,
    ) -> str | None:
        if not isinstance(requested_location, str):
            return None
        if requested_location in self._fixture_aliases:
            return self._fixture_aliases[requested_location]
        if requested_location in self._object_aliases:
            return self._object_aliases[requested_location]
        if requested_location in self.scene.get("fixtures", {}):
            return requested_location
        if requested_location in self.scene.get("objects", {}):
            return requested_location
        return self._resolve_support_site_parent(requested_location)

    def _resolve_support_site_parent(self, requested_site_id: str) -> str | None:
        if not hasattr(self.executor, "get_support_sites"):
            return None
        matches: list[str] = []
        for fixture_id in self.scene.get("fixtures", {}):
            try:
                support_sites = self.executor.get_support_sites(fixture_id)
            except Exception:
                continue
            if requested_site_id in set(support_sites or ()):
                matches.append(fixture_id)
        if len(matches) == 1:
            return matches[0]
        return None

    def _is_known_support_site(
        self,
        requested_location: str,
        fixture_context: dict[str, Any],
    ) -> bool:
        for fixture_state in fixture_context.values():
            if not isinstance(fixture_state, dict):
                continue
            support_sites = fixture_state.get("support_sites")
            if isinstance(support_sites, dict) and requested_location in support_sites:
                return True
            if isinstance(support_sites, list) and requested_location in support_sites:
                return True
        return False

    def _filter_distinct_object_candidates(
        self,
        candidate_ids: list[str],
        *,
        requested_id: str,
    ) -> list[str]:
        if not candidate_ids:
            return candidate_ids
        assigned_ids = {
            resolved_id
            for symbol, resolved_id in self._object_aliases.items()
            if symbol != requested_id
        }
        distinct_candidates = [
            candidate_id
            for candidate_id in candidate_ids
            if candidate_id not in assigned_ids
        ]
        if distinct_candidates:
            return distinct_candidates
        raise ValueError(
            f"Unable to resolve distinct object_id {requested_id!r}; "
            f"all candidates are already assigned: {candidate_ids}"
        )

    def _select_distinct_object_candidate(
        self,
        requested_id: str,
        *,
        requested_object_state: dict[str, Any],
        env_object_ids: set[str],
        fixture_context: dict[str, Any] | None = None,
    ) -> str | None:
        object_type = str(requested_object_state.get("object_type", ""))
        reserved_ids = set(self._object_aliases.values())

        if requested_id in env_object_ids and requested_id not in reserved_ids:
            return requested_id
        if object_type in env_object_ids and object_type not in reserved_ids:
            return object_type

        candidate_ids = self._object_candidates_for_type(object_type)
        candidate_ids = self._filter_object_candidates_by_location_hint(
            candidate_ids,
            requested_object_state,
            fixture_context=fixture_context,
        )
        candidate_ids = self._filter_distinct_object_candidates(
            candidate_ids,
            requested_id=requested_id,
        )
        if not candidate_ids:
            all_location_fallback_candidates = self._object_candidates_for_location_hint(
                requested_object_state,
                fixture_context=fixture_context,
            )
            zero_based_ordinal = self._extract_ordinal(requested_id)
            if (
                zero_based_ordinal is not None
                and 0 <= zero_based_ordinal < len(all_location_fallback_candidates)
            ):
                fallback_id = all_location_fallback_candidates[zero_based_ordinal]
                reserved_ids = {
                    resolved_id
                    for symbol, resolved_id in self._object_aliases.items()
                    if symbol != requested_id
                }
                if fallback_id not in reserved_ids:
                    log.warning(
                        "Falling back from missing object type hint %r for %r to %s via location %r",
                        object_type,
                        requested_id,
                        fallback_id,
                        requested_object_state.get("location"),
                    )
                    return fallback_id
            location_fallback_candidates = self._filter_distinct_object_candidates(
                all_location_fallback_candidates,
                requested_id=requested_id,
            )
            candidate_ids = location_fallback_candidates
        resolved_id, _method, _confidence, _reason = self._choose_candidate(
            requested_id=requested_id,
            candidate_ids=candidate_ids,
            entity_type="object",
            type_hint=object_type,
        )
        return resolved_id

    def _filter_object_candidates_by_location_hint(
        self,
        candidate_ids: list[str],
        requested_object_state: dict[str, Any] | None,
        fixture_context: dict[str, Any] | None = None,
    ) -> list[str]:
        if not candidate_ids or not isinstance(requested_object_state, dict):
            return candidate_ids

        requested_location = requested_object_state.get("location")
        resolved_location = self._resolve_requested_location_hint(requested_location)
        if isinstance(resolved_location, str):
            scene_objects = self.scene.get("objects", {})
            matching_candidates = [
                candidate_id
                for candidate_id in candidate_ids
                if (scene_objects.get(candidate_id, {}) or {}).get("location") == resolved_location
            ]
            if matching_candidates:
                return matching_candidates

        if not isinstance(requested_location, str) or not isinstance(fixture_context, dict):
            return candidate_ids

        requested_fixture_state = fixture_context.get(requested_location)
        if not isinstance(requested_fixture_state, dict):
            return candidate_ids
        requested_fixture_type = requested_fixture_state.get("fixture_type")
        if not isinstance(requested_fixture_type, str):
            return candidate_ids

        scene_objects = self.scene.get("objects", {})
        scene_fixtures = self.scene.get("fixtures", {})
        matching_candidates = []
        for candidate_id in candidate_ids:
            candidate_location = (scene_objects.get(candidate_id, {}) or {}).get("location")
            if not isinstance(candidate_location, str):
                continue
            candidate_fixture_type = (
                (scene_fixtures.get(candidate_location, {}) or {}).get("fixture_type")
            )
            if not isinstance(candidate_fixture_type, str):
                continue
            if self._fixture_type_matches(
                str(requested_fixture_type).lower(),
                str(candidate_fixture_type).lower(),
            ):
                matching_candidates.append(candidate_id)
        return matching_candidates or candidate_ids

    def _object_candidates_for_location_hint(
        self,
        requested_object_state: dict[str, Any] | None,
        *,
        fixture_context: dict[str, Any] | None = None,
    ) -> list[str]:
        if not isinstance(requested_object_state, dict):
            return []

        requested_location = requested_object_state.get("location")
        resolved_location = self._resolve_requested_location_hint(requested_location)
        scene_objects = self.scene.get("objects", {})
        if isinstance(resolved_location, str):
            return sorted(
                object_id
                for object_id, object_info in scene_objects.items()
                if (object_info or {}).get("location") == resolved_location
            )

        if not isinstance(requested_location, str) or not isinstance(fixture_context, dict):
            return []
        requested_fixture_state = fixture_context.get(requested_location)
        if not isinstance(requested_fixture_state, dict):
            return []
        requested_fixture_type = requested_fixture_state.get("fixture_type")
        if not isinstance(requested_fixture_type, str):
            return []

        scene_fixtures = self.scene.get("fixtures", {})
        matching_candidates: list[str] = []
        for object_id, object_info in scene_objects.items():
            candidate_location = (object_info or {}).get("location")
            if not isinstance(candidate_location, str):
                continue
            candidate_fixture_type = (
                (scene_fixtures.get(candidate_location, {}) or {}).get("fixture_type")
            )
            if not isinstance(candidate_fixture_type, str):
                continue
            if self._fixture_type_matches(
                str(requested_fixture_type).lower(),
                str(candidate_fixture_type).lower(),
            ):
                matching_candidates.append(object_id)
        return sorted(matching_candidates)

    def _choose_candidate(
        self,
        requested_id: str,
        candidate_ids: list[str],
        entity_type: str,
        type_hint: str | None,
    ) -> tuple[str, str, float, str]:
        if not candidate_ids:
            if entity_type == "object" and isinstance(type_hint, str):
                fallback_candidates = self._broader_object_fallback_candidates(
                    requested_id,
                    type_hint,
                )
                if len(fallback_candidates) == 1:
                    fallback_id = fallback_candidates[0]
                    log.warning(
                        "Falling back from missing object type hint %r for %r to %s",
                        type_hint,
                        requested_id,
                        fallback_id,
                    )
                    return (
                        fallback_id,
                        "broader_family_fallback",
                        0.45,
                        f"Broader family fallback for type hint {type_hint!r}",
                    )
            raise ValueError(
                f"Unable to resolve {entity_type}_id {requested_id!r} with type hint {type_hint!r}"
            )

        if len(candidate_ids) == 1:
            return (
                candidate_ids[0],
                "type_match",
                0.9,
                f"Only candidate matching type hint {type_hint!r}",
            )

        exact_like = [
            candidate_id
            for candidate_id in candidate_ids
            if self._base_token(candidate_id) == self._base_token(requested_id)
        ]
        if len(exact_like) == 1:
            return (
                exact_like[0],
                "token_match",
                0.8,
                f"Matched normalized token for {requested_id!r}",
            )

        ordinal = self._extract_ordinal(requested_id)
        if ordinal is not None:
            ordinal_idx = ordinal - 1
            ordered = exact_like or candidate_ids
            if 0 <= ordinal_idx < len(ordered):
                return (
                    ordered[ordinal_idx],
                    "ordinal_guess",
                    0.6,
                    f"Selected candidate #{ordinal} among {len(ordered)} candidates",
                )

        if not self.allow_approximate_ids:
            raise ValueError(
                f"Ambiguous {entity_type}_id {requested_id!r}; candidates={candidate_ids}"
            )

        return (
            candidate_ids[0],
            "fallback_first_candidate",
            0.3,
            f"Fell back to first sorted candidate among {len(candidate_ids)} matches",
        )

    def _normalize_part_id(
        self,
        part_id: str,
        part_state: dict[str, Any] | None = None,
    ) -> str:
        part_id = str(part_id)
        part_type = ""
        if part_state is not None:
            part_type = str(part_state.get("part_type", "")).lower()
        if "hinged" in part_type:
            return "hinged"
        if "sliding" in part_type:
            return "sliding"
        lowered = part_id.lower()
        if lowered in {"door", "lid"}:
            return "hinged"
        if any(token in lowered for token in ("drawer", "slide", "sliding")):
            return "sliding"
        return part_id

    def _base_token(self, token: str) -> str:
        token = str(token).lower().strip()
        parts = [part for part in token.split("_") if part and not part.isdigit()]
        return "_".join(parts)

    def _extract_ordinal(self, token: str) -> int | None:
        pieces = str(token).split("_")
        if not pieces:
            return None
        tail = pieces[-1]
        if tail.isdigit():
            return int(tail)
        return None

    def _fixture_type_matches(
        self,
        requested_type: str,
        actual_type: str,
    ) -> bool:
        if requested_type == actual_type:
            return True

        requested_family = self._FIXTURE_TYPE_FAMILIES.get(requested_type)
        if requested_family is not None and actual_type in requested_family:
            return True

        actual_family = self._FIXTURE_TYPE_FAMILIES.get(actual_type)
        if actual_family is not None and requested_type in actual_family:
            return True

        return requested_type in actual_type or actual_type in requested_type


def adapt_trajectory(
    executor,
    trajectory: dict[str, Any],
    output_dir: str | Path | None = None,
    allow_approximate_ids: bool = True,
) -> dict[str, Any]:
    """Convenience wrapper around :class:`TrajectoryAdapter`."""
    return TrajectoryAdapter(
        executor=executor,
        allow_approximate_ids=allow_approximate_ids,
    ).adapt(trajectory, output_dir=output_dir)


def execute_trajectory(
    executor,
    trajectory: dict[str, Any],
    output_dir: str | Path | None = None,
    allow_approximate_ids: bool = True,
    fps: int = 2,
    skip_videos: bool = False,
    save_debug_frames: bool = False,
) -> dict[str, Any]:
    """Adapt and execute one external trajectory."""
    return TrajectoryAdapter(
        executor=executor,
        allow_approximate_ids=allow_approximate_ids,
    ).execute(
        trajectory,
        output_dir=output_dir,
        fps=fps,
        skip_videos=skip_videos,
        save_debug_frames=save_debug_frames,
    )


__all__ = [
    "ResolutionRecord",
    "TrajectoryAdapter",
    "adapt_trajectory",
    "execute_trajectory",
]
