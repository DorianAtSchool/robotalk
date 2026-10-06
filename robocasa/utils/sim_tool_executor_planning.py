"""Planning helpers for ``SimToolExecutor`` demo plan construction."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np


class SimToolExecutorPlanningMixin:
    def _semantic_ref(self, resolver: str, **kwargs) -> dict[str, Any]:
        return {"$ref": resolver, **kwargs}

    def _build_hotdog_setup_demo_template(self) -> list[dict[str, Any]]:
        return [
            {
                "tool": "communicate",
                "robot_idx": 0,
                "args": {
                    "to": "agent_1",
                    "message": "I will stage the bun and condiment at the dining table.",
                },
            },
            {
                "tool": "communicate",
                "robot_idx": 1,
                "args": {
                    "to": "agent_0",
                    "message": "I will bring the sausage from the fridge once the plate is ready.",
                },
            },
            {
                "tool": "navigate_to_fixture",
                "robot_idx": 0,
                "args": {
                    "fixture_id": self._semantic_ref(
                        "source_fixture",
                        object_id="hotdog_bun",
                    )
                },
            },
            {
                "tool": "pick_up_object",
                "robot_idx": 0,
                "args": {
                    "object_id": "hotdog_bun",
                    "source_id": self._semantic_ref(
                        "source_fixture",
                        object_id="hotdog_bun",
                    ),
                },
            },
            {
                "tool": "navigate_to_fixture",
                "robot_idx": 1,
                "args": {
                    "fixture_id": self._semantic_ref(
                        "source_fixture",
                        object_id="sausage",
                        preferred_fixture_types=["fridge"],
                    )
                },
            },
            {
                "tool": "open_hinged_part",
                "robot_idx": 1,
                "args": {
                    "target_id": self._semantic_ref(
                        "source_fixture",
                        object_id="sausage",
                        preferred_fixture_types=["fridge"],
                    ),
                    "part_id": "hinged",
                },
            },
            {
                "tool": "navigate_to_fixture",
                "robot_idx": 0,
                "args": {
                    "fixture_id": self._semantic_ref(
                        "object_anchor_fixture",
                        object_id="plate",
                        preferred_fixture_types=[
                            "dining_counter",
                            "island",
                            "counter_non_dining",
                        ],
                        require_placeable=True,
                    )
                },
            },
            {
                "tool": "place_on_object",
                "robot_idx": 0,
                "args": {
                    "object_id": "hotdog_bun",
                    "support_object_id": "plate",
                    "anchor_fixture_id": self._semantic_ref(
                        "object_anchor_fixture",
                        object_id="plate",
                        preferred_fixture_types=[
                            "dining_counter",
                            "island",
                            "counter_non_dining",
                        ],
                        require_placeable=True,
                    ),
                },
            },
            {
                "tool": "communicate",
                "robot_idx": 0,
                "args": {
                    "to": "agent_1",
                    "message": "Plate is staged on the dining table. Bring the sausage now.",
                },
            },
            {
                "tool": "pick_up_object",
                "robot_idx": 1,
                "args": {
                    "object_id": "sausage",
                    "source_id": self._semantic_ref(
                        "source_fixture",
                        object_id="sausage",
                        preferred_fixture_types=["fridge"],
                    ),
                },
            },
            {
                "tool": "navigate_to_fixture",
                "robot_idx": 1,
                "args": {
                    "fixture_id": self._semantic_ref(
                        "object_anchor_fixture",
                        object_id="plate",
                        preferred_fixture_types=[
                            "dining_counter",
                            "island",
                            "counter_non_dining",
                        ],
                        require_placeable=True,
                    )
                },
            },
            {
                "tool": "place_on_object",
                "robot_idx": 1,
                "args": {
                    "object_id": "sausage",
                    "support_object_id": "plate",
                    "anchor_fixture_id": self._semantic_ref(
                        "object_anchor_fixture",
                        object_id="plate",
                        preferred_fixture_types=[
                            "dining_counter",
                            "island",
                            "counter_non_dining",
                        ],
                        require_placeable=True,
                    ),
                },
            },
            {
                "tool": "communicate",
                "robot_idx": 1,
                "args": {
                    "to": "agent_0",
                    "message": "Sausage is placed. The dining table is clear for the condiment.",
                },
            },
            {
                "tool": "navigate_to_fixture",
                "robot_idx": 0,
                "args": {
                    "fixture_id": self._semantic_ref(
                        "source_fixture",
                        object_id="condiment",
                    )
                },
            },
            {
                "tool": "pick_up_object",
                "robot_idx": 0,
                "args": {
                    "object_id": "condiment",
                    "source_id": self._semantic_ref(
                        "source_fixture",
                        object_id="condiment",
                    ),
                },
            },
            {
                "tool": "navigate_to_fixture",
                "robot_idx": 0,
                "args": {
                    "fixture_id": self._semantic_ref(
                        "object_anchor_fixture",
                        object_id="plate",
                        preferred_fixture_types=[
                            "dining_counter",
                            "island",
                            "counter_non_dining",
                        ],
                        require_placeable=True,
                    )
                },
            },
            {
                "tool": "place_on_surface",
                "robot_idx": 0,
                "args": {
                    "object_id": "condiment",
                    "support_id": self._semantic_ref(
                        "object_anchor_fixture",
                        object_id="plate",
                        preferred_fixture_types=[
                            "dining_counter",
                            "island",
                            "counter_non_dining",
                        ],
                        require_placeable=True,
                    ),
                },
            },
            {
                "tool": "communicate",
                "robot_idx": 0,
                "args": {
                    "to": "agent_1",
                    "message": "Hot dog setup complete.",
                },
            },
        ]

    def _build_sandwich_station_demo_template(self) -> list[dict[str, Any]]:
        """Demo plan for PrepareSandwichStation (cooperative 2-robot)."""
        fridge_ref = self._semantic_ref(
            "source_fixture",
            object_id="ingredient_bowl",
            preferred_fixture_types=["fridge"],
        )
        counter_ref = self._semantic_ref(
            "nearest_fixture",
            anchor_fixture_type="toaster_oven",
            preferred_fixture_types=["counter", "counter_non_dining"],
            require_placeable=True,
        )

        return [
            {
                "tool": "communicate",
                "robot_idx": 0,
                "args": {
                    "to": "agent_1",
                    "message": "I'll open the fridge and grab the ingredient bowl. "
                    "You grab the baguette after me.",
                },
            },
            {
                "tool": "communicate",
                "robot_idx": 1,
                "args": {
                    "to": "agent_0",
                    "message": "Got it. I'll grab the baguette and close the fridge.",
                },
            },
            {
                "tool": "navigate_to_fixture",
                "robot_idx": 0,
                "args": {"fixture_id": fridge_ref},
            },
            {
                "tool": "open_hinged_part",
                "robot_idx": 0,
                "args": {"target_id": fridge_ref, "part_id": "hinged"},
            },
            {
                "tool": "pick_up_object",
                "robot_idx": 0,
                "args": {
                    "object_id": "ingredient_bowl",
                    "source_id": fridge_ref,
                },
            },
            {
                "tool": "give_space",
                "robot_idx": 0,
                "args": {"fixture_id": fridge_ref},
            },
            {
                "tool": "pick_up_object",
                "robot_idx": 1,
                "args": {
                    "object_id": "baguette",
                    "source_id": self._semantic_ref(
                        "source_fixture",
                        object_id="baguette",
                        preferred_fixture_types=["fridge"],
                    ),
                },
            },
            {
                "tool": "close_hinged_part",
                "robot_idx": 1,
                "args": {"target_id": fridge_ref, "part_id": "hinged"},
            },
            {
                "tool": "navigate_to_fixture",
                "robot_idx": 0,
                "args": {"fixture_id": counter_ref},
            },
            {
                "tool": "place_on_surface",
                "robot_idx": 0,
                "args": {
                    "object_id": "ingredient_bowl",
                    "support_id": counter_ref,
                },
            },
            {
                "tool": "communicate",
                "robot_idx": 0,
                "args": {
                    "to": "agent_1",
                    "message": "Bowl is placed. Your turn to place the baguette.",
                },
            },
            {
                "tool": "navigate_to_fixture",
                "robot_idx": 1,
                "args": {"fixture_id": counter_ref},
            },
            {
                "tool": "place_on_surface",
                "robot_idx": 1,
                "args": {
                    "object_id": "baguette",
                    "support_id": counter_ref,
                },
            },
            {
                "tool": "communicate",
                "robot_idx": 1,
                "args": {
                    "to": "agent_0",
                    "message": "Sandwich station is ready.",
                },
            },
        ]

    def _resolve_nearest_fixture(
        self,
        anchor_fixture_type: str,
        preferred_fixture_types: list[str] | set[str] | tuple[str, ...] | None = None,
        require_placeable: bool = False,
    ) -> str:
        """Find the nearest matching fixture to the anchor fixture type."""
        scene = self.get_scene_description()
        fixtures = scene.get("fixtures", {})

        anchor_id = None
        for fixture_id, fixture_info in fixtures.items():
            if fixture_info.get("fixture_type") == anchor_fixture_type:
                anchor_id = fixture_id
                break
        if anchor_id is None:
            raise ValueError(f"No fixture of type {anchor_fixture_type!r} found")
        anchor_pos = np.asarray(fixtures[anchor_id]["position"][:2], dtype=float)
        preferred = self._normalize_preferred_fixture_types(preferred_fixture_types)

        best_id = None
        best_dist = float("inf")
        for fixture_id, fixture_info in fixtures.items():
            if fixture_id == anchor_id:
                continue
            fixture_type = fixture_info.get("fixture_type")
            if preferred is not None and fixture_type not in preferred:
                continue
            if require_placeable and not fixture_info.get("can_place_objects", False):
                continue
            distance = float(
                np.linalg.norm(
                    np.asarray(fixture_info["position"][:2], dtype=float) - anchor_pos
                )
            )
            if distance < best_dist:
                best_dist = distance
                best_id = fixture_id

        if best_id is None:
            raise ValueError(
                f"No fixture of types {preferred} found near {anchor_fixture_type!r}"
            )
        return best_id

    def _resolve_semantic_ref(self, ref: dict[str, Any]) -> Any:
        resolver = ref.get("$ref")
        if resolver == "source_fixture":
            preferred_fixture_types = ref.get("preferred_fixture_types")
            return self._infer_source_fixture(
                ref["object_id"],
                preferred_fixture_types=preferred_fixture_types,
            )
        if resolver == "object_anchor_fixture":
            preferred_fixture_types = ref.get("preferred_fixture_types")
            return self._resolve_object_anchor_fixture(
                ref["object_id"],
                preferred_fixture_types=preferred_fixture_types,
                require_placeable=bool(ref.get("require_placeable", False)),
            )
        if resolver == "nearest_fixture":
            preferred_fixture_types = ref.get("preferred_fixture_types")
            return self._resolve_nearest_fixture(
                ref["anchor_fixture_type"],
                preferred_fixture_types=preferred_fixture_types,
                require_placeable=bool(ref.get("require_placeable", False)),
            )
        raise ValueError(f"Unknown semantic resolver {resolver!r}")

    def _ground_value(self, value: Any) -> Any:
        if isinstance(value, dict):
            if "$ref" in value:
                return self._resolve_semantic_ref(value)
            return {
                key: self._ground_value(subvalue) for key, subvalue in value.items()
            }
        if isinstance(value, list):
            return [self._ground_value(item) for item in value]
        return value

    def ground_plan_template(
        self, tool_calls: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        grounded_tool_calls = deepcopy(tool_calls)
        for tool_call in grounded_tool_calls:
            tool_call["args"] = self._ground_value(tool_call.get("args", {}))
        return grounded_tool_calls

    def build_demo_plan_template(self, demo_plan_name: str) -> list[dict[str, Any]]:
        normalized_name = demo_plan_name.strip().lower().replace("-", "_")

        if normalized_name == "cooperative_hotdog_setup":
            if self._get_task_class_name() != "HotDogSetup":
                raise ValueError(
                    "The cooperative_hotdog_setup demo plan requires --task HotDogSetup"
                )
            return self._build_hotdog_setup_demo_template()

        if normalized_name == "sandwich_station":
            if self._get_task_class_name() != "PrepareSandwichStation":
                raise ValueError(
                    "The sandwich_station demo plan requires --task PrepareSandwichStation"
                )
            return self._build_sandwich_station_demo_template()

        raise ValueError(
            f"Unknown demo plan {demo_plan_name!r}. "
            f"Available: {sorted(self._DEMO_TASK_BY_NAME)}"
        )

    def build_demo_plan(self, demo_plan_name: str) -> list[dict[str, Any]]:
        return self.ground_plan_template(self.build_demo_plan_template(demo_plan_name))
