from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock

import numpy as np

from robocasa.utils.sim_tool_executor import SimToolExecutor
from robocasa.utils.trajectory_adapter import TrajectoryAdapter


class FakeExecutor:
    def __init__(self):
        self.loaded_state = None
        self.executed_steps = []
        self.scene = {
            "fixtures": {
                "cab_main": {"fixture_type": "cabinet_double_door"},
                "counter_main": {"fixture_type": "counter_non_dining"},
                "coffee_machine_main": {"fixture_type": "coffee_machine"},
            },
            "objects": {
                "mug_main": {"object_type": "mug"},
            },
        }
        self.support_site_map = {}
        self.default_site_map = {}

    def get_scene_description(self):
        return self.scene

    def _parse_agent_idx(self, agent_id):
        if isinstance(agent_id, int):
            return agent_id
        return int(str(agent_id).replace("agent_", ""))

    def load_initial_state(self, initial_state):
        self.loaded_state = initial_state
        return {"loaded": True}

    def _resolve_fixture_site_id(self, fixture_id, requested_site_id):
        fixture_sites = self.support_site_map.get(fixture_id, {})
        if requested_site_id in fixture_sites:
            return fixture_sites[requested_site_id]
        for symbolic, concrete in fixture_sites.items():
            if concrete == requested_site_id:
                return concrete
        raise ValueError(f"Unknown site {requested_site_id!r} for fixture {fixture_id!r}")

    def _raw_support_site_to_external(self, raw_site_id):
        return str(raw_site_id)

    def get_support_sites(self, fixture_id):
        fixture_sites = self.support_site_map.get(fixture_id, {})
        return list(fixture_sites.values()) or list(fixture_sites.keys())

    def _normalize_target_site_id_for_placement(self, fixture_id, site_id):
        fixture_info = self.scene.get("fixtures", {}).get(fixture_id, {})
        if fixture_info.get("fixture_type") == "counter_non_dining" and str(site_id).startswith("geom_"):
            return None
        return site_id

    def _fixture_requires_explicit_site(self, fixture_id):
        return fixture_id in self.default_site_map

    def _default_support_site_for_unspecified_fixture(self, fixture_id, incoming_object_id=None):
        return self.default_site_map.get(fixture_id)

    def execute(self, tool_name, robot_idx=0, **kwargs):
        self.executed_steps.append((tool_name, robot_idx, kwargs))
        return SimpleNamespace(
            success=True, details={"tool_name": tool_name, "args": kwargs}
        )


class TestTrajectoryAdapter(unittest.TestCase):
    def test_explicit_object_index_maps_numbered_native_objects(self):
        executor = FakeExecutor()
        executor.scene["objects"] = {
            "obj_0": {"object_type": "soda"},
            "obj_1": {"object_type": "juice"},
            "obj_2": {"object_type": "water"},
        }
        adapter = TrajectoryAdapter(executor=executor)
        adapter._apply_sim_ground_truth(
            {
                "objects": {
                    "drink_0": {"object_type": "drink", "location": "counter"},
                    "drink_1": {"object_type": "drink", "location": "counter"},
                    "drink_2": {"object_type": "drink", "location": "counter"},
                },
                "fixtures": {"counter": {"fixture_type": "counter"}},
            },
            {
                f"drink_{index}": {
                    "resolver": "object_by_type", "object_type": "drink", "index": index
                }
                for index in range(3)
            },
        )

        self.assertEqual(
            adapter._object_aliases,
            {"drink_0": "obj_0", "drink_1": "obj_1", "drink_2": "obj_2"},
        )

    def test_place_on_object_target_id_resolves_as_object(self):
        executor = FakeExecutor()
        executor.scene["objects"]["plate_main"] = {"object_type": "plate"}
        adapter = TrajectoryAdapter(executor=executor)
        adapter._object_aliases["plate1"] = "plate_main"
        adapter._fixture_aliases["plate1"] = "cab_main"

        resolved = adapter._resolve_step_args(
            "place_on_object",
            {"object_id": "mug", "target_id": "plate1"},
            {
                "objects": {
                    "mug": {"object_type": "mug"},
                    "plate1": {"object_type": "plate"},
                },
                "fixtures": {},
            },
        )

        self.assertEqual(resolved["target_id"], "plate_main")

    def test_adapt_resolves_ids_and_rewrites_image_steps(self):
        executor = FakeExecutor()
        adapter = TrajectoryAdapter(executor=executor)
        trajectory = {
            "trajectory_id": "traj_1",
            "initial_state": {
                "agents": {
                    "agent_0": {"location": "staging_surface", "held_object": None},
                    "agent_1": {"location": "coffee_machine", "held_object": None},
                },
                "objects": {
                    "mug": {"object_type": "mug", "location": "mug_source_fixture"},
                },
                "fixtures": {
                    "mug_source_fixture": {"fixture_type": "cabinet"},
                    "staging_surface": {"fixture_type": "counter"},
                    "coffee_machine": {"fixture_type": "coffee_machine"},
                },
                "machine_state": {
                    "coffee_machine": {
                        "started": False,
                        "dispenser_id": "coffee_machine_dispenser",
                    }
                },
            },
            "steps": [
                {
                    "step": 0,
                    "agent": "agent_0",
                    "tool": "get_env_image",
                    "args": {"view": "top_view"},
                    "image_path": "images/traj_1/0_top.png",
                },
                {
                    "step": 1,
                    "agent": "agent_0",
                    "tool": "place_under_dispenser",
                    "args": {
                        "object_id": "mug",
                        "dispenser_id": "coffee_machine_dispenser",
                    },
                },
                {
                    "step": 2,
                    "agent": "agent_1",
                    "tool": "get_agent_image",
                    "args": {"view": "agentview_right"},
                    "image_path": "images/traj_1/2_right.png",
                },
            ],
        }

        adapted = adapter.adapt(trajectory, output_dir="tmp/output")

        self.assertEqual(
            adapted["initial_state"]["agents"]["agent_0"]["location"],
            "counter_main",
        )
        self.assertEqual(
            adapted["initial_state"]["objects"]["mug_main"]["location"],
            "cab_main",
        )
        self.assertEqual(adapted["tool_calls"][0]["tool"], "get_image")
        self.assertEqual(adapted["tool_calls"][0]["args"]["views"], ["top_view"])
        self.assertEqual(
            adapted["tool_calls"][0]["args"]["image_paths"],
            [str(Path("tmp/output") / "images/traj_1/0_top.jpg")],
        )
        self.assertEqual(adapted["tool_calls"][1]["tool"], "place_under")
        self.assertEqual(
            adapted["tool_calls"][1]["args"]["reference_fixture_id"],
            "coffee_machine_main",
        )
        self.assertEqual(adapted["tool_calls"][2]["tool"], "get_image")
        self.assertEqual(adapted["tool_calls"][2]["args"]["views"], ["agentview_right"])
        self.assertEqual(adapted["tool_calls"][2]["args"]["agent_id"], "agent_1")
        self.assertTrue(adapted["resolution_log"])

    def test_adapt_preserves_scalar_machine_flags(self):
        executor = FakeExecutor()
        adapter = TrajectoryAdapter(executor=executor)
        trajectory = {
            "trajectory_id": "traj_1",
            "initial_state": {
                "agents": {
                    "agent_0": {"location": "counter", "held_object": None},
                    "agent_1": {"location": "counter", "held_object": None},
                },
                "objects": {},
                "fixtures": {
                    "counter": {"fixture_type": "counter"},
                },
                "machine_state": {
                    "bowl1_placed_at_stool": False,
                    "counter": {"ready": True},
                },
            },
            "steps": [],
        }

        adapted = adapter.adapt(trajectory)

        self.assertIs(
            adapted["initial_state"]["machine_state"]["bowl1_placed_at_stool"],
            False,
        )
        self.assertEqual(
            adapted["initial_state"]["machine_state"]["counter_main"],
            {"ready": True},
        )

    def test_fixture_family_matching_handles_specific_scene_types(self):
        executor = FakeExecutor()
        adapter = TrajectoryAdapter(executor=executor)

        self.assertEqual(
            adapter._fixture_candidates_for_type("cabinet"),
            ["cab_main"],
        )
        self.assertEqual(
            adapter._fixture_candidates_for_type("counter"),
            ["counter_main"],
        )

    def test_object_family_matching_handles_specific_scene_types(self):
        executor = FakeExecutor()
        executor.scene = {
            "fixtures": {
                "counter_a": {"fixture_type": "counter_non_dining"},
            },
            "objects": {
                "juice_box": {"object_type": "juice", "location": "counter_a"},
                "party_candle": {"object_type": "candle", "location": "counter_a"},
                "skillet_main": {"object_type": "skillet", "location": "counter_a"},
            },
        }
        adapter = TrajectoryAdapter(executor=executor)

        self.assertEqual(adapter._object_candidates_for_type("drink"), ["juice_box"])
        self.assertEqual(
            adapter._object_candidates_for_type("decoration"),
            ["party_candle"],
        )
        self.assertEqual(adapter._object_candidates_for_type("pan"), ["skillet_main"])

    def test_object_family_matching_allows_role_alias_to_sim_type(self):
        executor = FakeExecutor()
        executor.scene = {
            "fixtures": {
                "counter_a": {"fixture_type": "counter_non_dining"},
            },
            "objects": {
                "wine_glass_main": {
                    "object_type": "wine_glass",
                    "location": "counter_a",
                },
            },
        }
        adapter = TrajectoryAdapter(executor=executor)

        self.assertEqual(
            adapter._object_candidates_for_type("wine_bottle"),
            ["wine_glass_main"],
        )

    def test_object_resolution_prefers_requested_location(self):
        executor = FakeExecutor()
        executor.scene = {
            "fixtures": {
                "counter_left": {"fixture_type": "counter_non_dining"},
                "counter_right": {"fixture_type": "counter_non_dining"},
            },
            "objects": {
                "bowl_left": {"object_type": "bowl", "location": "counter_left"},
                "bowl_right": {"object_type": "bowl", "location": "counter_right"},
            },
        }
        adapter = TrajectoryAdapter(executor=executor, allow_approximate_ids=False)

        resolved_id = adapter._resolve_object_id(
            "distractor",
            {"object_type": "bowl", "location": "counter_right"},
        )

        self.assertEqual(resolved_id, "bowl_right")

    def test_object_resolution_falls_back_to_location_order_when_type_missing(self):
        executor = FakeExecutor()
        executor.scene = {
            "fixtures": {
                "counter_main": {"fixture_type": "counter_non_dining"},
            },
            "objects": {
                "obj_0": {"object_type": "unknown", "location": "counter_main"},
                "obj_1": {"object_type": "unknown", "location": "counter_main"},
                "obj_2": {"object_type": "unknown", "location": "counter_main"},
            },
        }
        adapter = TrajectoryAdapter(executor=executor, allow_approximate_ids=False)

        self.assertEqual(
            adapter._resolve_object_id(
                "drink_0",
                {"object_type": "drink", "location": "counter_main"},
            ),
            "obj_0",
        )
        self.assertEqual(
            adapter._resolve_object_id(
                "drink_1",
                {"object_type": "drink", "location": "counter_main"},
            ),
            "obj_1",
        )
        self.assertEqual(
            adapter._resolve_object_id(
                "drink_2",
                {"object_type": "drink", "location": "counter_main"},
            ),
            "obj_2",
        )

    def test_object_resolution_requires_distinct_instances(self):
        executor = FakeExecutor()
        executor.scene = {
            "fixtures": {
                "counter_main": {"fixture_type": "counter_non_dining"},
            },
            "objects": {
                "candle_main": {"object_type": "candle", "location": "counter_main"},
            },
        }
        adapter = TrajectoryAdapter(executor=executor, allow_approximate_ids=False)

        first = adapter._resolve_object_id(
            "decoration_1",
            {"object_type": "decoration", "location": "counter_main"},
        )
        self.assertEqual(first, "candle_main")
        with self.assertRaises(ValueError):
            adapter._resolve_object_id(
                "decoration_2",
                {"object_type": "decoration", "location": "counter_main"},
            )

    def test_sim_ground_truth_uses_symbolic_fixture_type_for_object_location(self):
        executor = FakeExecutor()
        executor.scene = {
            "fixtures": {
                "counter_main": {"fixture_type": "counter_non_dining"},
                "dining_main": {"fixture_type": "dining_counter"},
            },
            "objects": {
                "plate_counter": {"object_type": "plate", "location": "counter_main"},
                "plate_dining": {"object_type": "plate", "location": "dining_main"},
            },
            "fixture_refs": {
                "counter": "counter_main",
            },
            "object_placements": {
                "plate_counter": "counter_main",
                "plate_dining": "dining_main",
            },
        }
        adapter = TrajectoryAdapter(executor=executor, allow_approximate_ids=False)

        adapted = adapter.adapt(
            {
                "trajectory_id": "traj_hotdog_like",
                "initial_state": {
                    "agents": {
                        "agent_0": {"location": "dining_table", "held_object": None},
                    },
                    "objects": {
                        "plate": {"object_type": "plate", "location": "dining_table"},
                    },
                    "fixtures": {
                        "dining_table": {"fixture_type": "dining_counter"},
                    },
                    "machine_state": {},
                },
                "steps": [],
            }
        )

        self.assertEqual(
            adapted["initial_state"]["agents"]["agent_0"]["location"],
            "dining_main",
        )
        self.assertIn("plate_dining", adapted["initial_state"]["objects"])
        self.assertEqual(
            adapted["initial_state"]["objects"]["plate_dining"]["location"],
            "dining_main",
        )

    def test_sim_ground_truth_prefers_exact_fixture_ref_over_family_match(self):
        executor = FakeExecutor()
        executor.scene = {
            "fixtures": {
                "counter_main": {"fixture_type": "counter_non_dining"},
                "dining_main": {"fixture_type": "dining_counter"},
            },
            "objects": {},
            "fixture_refs": {
                "counter": "counter_main",
                "dining_table": "dining_main",
            },
            "object_placements": {},
        }
        adapter = TrajectoryAdapter(executor=executor, allow_approximate_ids=False)

        adapted = adapter.adapt(
            {
                "trajectory_id": "traj_fixture_refs",
                "initial_state": {
                    "agents": {
                        "agent_0": {"location": "dining_table", "held_object": None},
                    },
                    "objects": {},
                    "fixtures": {
                        "dining_table": {"fixture_type": "dining_counter"},
                    },
                    "machine_state": {},
                },
                "steps": [],
            }
        )

        self.assertEqual(
            adapted["initial_state"]["agents"]["agent_0"]["location"],
            "dining_main",
        )

    def test_adapt_initial_state_preserves_support_site_locations(self):
        executor = FakeExecutor()
        executor.scene = {
            "fixtures": {
                "stove_main": {"fixture_type": "stove"},
            },
            "objects": {
                "pan_main": {"object_type": "pan"},
            },
        }
        adapter = TrajectoryAdapter(executor=executor)

        resolved = adapter._adapt_initial_state(
            {
                "agents": {},
                "objects": {
                    "pan": {"object_type": "pan", "location": "front_right_burner"},
                },
                "fixtures": {
                    "stove_fixture": {
                        "fixture_type": "stove",
                        "support_sites": {
                            "front_right_burner": {"site_type": "support"},
                        },
                    }
                },
                "machine_state": {},
            }
        )

        self.assertEqual(
            resolved["objects"]["pan_main"]["location"],
            "stove_main",
        )
        self.assertEqual(
            resolved["objects"]["pan_main"]["target_site_id"],
            "front_right_burner",
        )

    def test_adapt_initial_state_keeps_inferred_site_with_parent_fixture(self):
        executor = FakeExecutor()
        executor.scene = {
            "fixtures": {
                "cab_main": {"fixture_type": "cabinet_double_door"},
            },
            "objects": {
                "mug_main": {"object_type": "mug", "location": "cab_main"},
            },
        }
        executor.support_site_map = {"cab_main": {"shelf_0": "shelf_0"}}
        executor._infer_object_support_site = MagicMock(return_value="shelf_0")
        adapter = TrajectoryAdapter(executor=executor)

        resolved = adapter._adapt_initial_state(
            {
                "agents": {},
                "objects": {
                    "mug": {"object_type": "mug", "location": "cabinet"},
                },
                "fixtures": {
                    "cabinet": {"fixture_type": "cabinet"},
                },
                "machine_state": {},
            }
        )

        self.assertEqual(resolved["objects"]["mug_main"]["location"], "cab_main")
        self.assertEqual(
            resolved["objects"]["mug_main"]["target_site_id"],
            "shelf_0",
        )

    def test_adapt_initial_state_drops_generic_surface_geom_site(self):
        executor = FakeExecutor()
        executor.scene = {
            "fixtures": {
                "counter_main": {"fixture_type": "counter_non_dining"},
            },
            "objects": {
                "fork_main": {"object_type": "fork", "location": "counter_main"},
            },
        }
        executor.support_site_map = {"counter_main": {"geom_0": "geom_0"}}
        executor._infer_object_support_site = MagicMock(return_value="geom_0")
        adapter = TrajectoryAdapter(executor=executor)

        resolved = adapter._adapt_initial_state(
            {
                "agents": {},
                "objects": {
                    "fork": {"object_type": "fork", "location": "counter"},
                },
                "fixtures": {
                    "counter": {"fixture_type": "counter"},
                },
                "machine_state": {},
            }
        )

        self.assertEqual(resolved["objects"]["fork_main"]["location"], "counter_main")
        self.assertNotIn("target_site_id", resolved["objects"]["fork_main"])

    def test_adapt_initial_state_uses_default_site_for_partitioned_fixture(self):
        executor = FakeExecutor()
        executor.scene = {
            "fixtures": {
                "fridge_main": {"fixture_type": "fridge"},
            },
            "objects": {
                "baguette_main": {"object_type": "baguette", "location": "fridge_main"},
            },
        }
        executor.default_site_map = {"fridge_main": "shelf_1"}
        executor._infer_object_support_site = MagicMock(return_value=None)
        adapter = TrajectoryAdapter(executor=executor)

        resolved = adapter._adapt_initial_state(
            {
                "agents": {},
                "objects": {
                    "baguette": {"object_type": "baguette", "location": "fridge"},
                },
                "fixtures": {
                    "fridge": {"fixture_type": "fridge"},
                },
                "machine_state": {},
            }
        )

        self.assertEqual(resolved["objects"]["baguette_main"]["location"], "fridge_main")
        self.assertEqual(
            resolved["objects"]["baguette_main"]["target_site_id"],
            "shelf_1",
        )

    def test_adapt_resolves_symbolic_target_site_ids_against_concrete_fixture(self):
        executor = FakeExecutor()
        executor.scene = {
            "fixtures": {
                "sink_left_group": {
                    "fixture_type": "sink",
                    "support_sites": {"basin": {"site_type": "support"}},
                },
                "counter_main": {"fixture_type": "counter_non_dining"},
            },
            "objects": {
                "jug_main": {"object_type": "jug", "location": "counter_main"},
            },
        }
        executor.support_site_map = {"sink_left_group": {"sink_basin": "basin"}}
        adapter = TrajectoryAdapter(executor=executor)

        adapted = adapter.adapt(
            {
                "trajectory_id": "traj_sink",
                "initial_state": {
                    "agents": {"agent_0": {"location": "sink", "held_object": None}},
                    "objects": {"jug": {"object_type": "jug", "location": "counter"}},
                    "fixtures": {
                        "sink": {
                            "fixture_type": "sink",
                            "support_sites": {"sink_basin": {"site_type": "support"}},
                        },
                        "counter": {"fixture_type": "counter"},
                    },
                    "machine_state": {},
                },
                "steps": [
                    {
                        "step": 0,
                        "agent": "agent_0",
                        "tool": "place_under",
                        "args": {
                            "object_id": "jug",
                            "reference_fixture_id": "sink",
                            "target_site_id": "sink_basin",
                        },
                    }
                ],
            }
        )

        self.assertEqual(
            adapted["tool_calls"][0]["args"]["reference_fixture_id"],
            "sink_left_group",
        )
        self.assertEqual(adapted["tool_calls"][0]["args"]["target_site_id"], "basin")

    def test_adapt_drops_fixture_name_target_site_id(self):
        executor = FakeExecutor()
        executor.scene = {
            "fixtures": {
                "counter_main": {"fixture_type": "counter_non_dining"},
            },
            "objects": {
                "bowl_main": {"object_type": "bowl", "location": "counter_main"},
            },
        }
        adapter = TrajectoryAdapter(executor=executor)

        adapted = adapter.adapt(
            {
                "trajectory_id": "traj_counter_site",
                "initial_state": {
                    "agents": {
                        "agent_0": {"location": "dining_counter", "held_object": None},
                    },
                    "objects": {
                        "bowl": {"object_type": "bowl", "location": "dining_counter"},
                    },
                    "fixtures": {
                        "dining_counter": {"fixture_type": "counter"},
                    },
                    "machine_state": {},
                },
                "steps": [
                    {
                        "step": 0,
                        "agent": "agent_0",
                        "tool": "place_on_surface",
                        "args": {
                            "object_id": "bowl",
                            "target_site_id": "dining_counter",
                            "support_id": "dining_counter",
                        },
                    }
                ],
            }
        )

        self.assertIsNone(adapted["tool_calls"][0]["args"]["target_site_id"])

    def test_adapt_preserves_pose_for_reference_only_object(self):
        executor = FakeExecutor()
        executor.scene = {
            "fixtures": {
                "counter_main": {"fixture_type": "counter_non_dining"},
                "cabinet_main": {"fixture_type": "cabinet"},
            },
            "objects": {
                "steak_main": {"object_type": "steak", "location": "plate_main"},
                "shaker_main": {"object_type": "shaker", "location": "cabinet_main"},
            },
        }
        adapter = TrajectoryAdapter(executor=executor)

        adapted = adapter.adapt(
            {
                "trajectory_id": "traj_ref_only_anchor",
                "initial_state": {
                    "agents": {"agent_0": {"location": "cabinet", "held_object": None}},
                    "objects": {
                        "steak": {"object_type": "steak", "location": "counter"},
                        "shaker": {"object_type": "shaker", "location": "cabinet"},
                    },
                    "fixtures": {
                        "counter": {"fixture_type": "counter"},
                        "cabinet": {"fixture_type": "cabinet"},
                    },
                    "machine_state": {},
                },
                "steps": [
                    {
                        "step": 0,
                        "agent": "agent_0",
                        "tool": "pick_up_object",
                        "args": {"object_id": "shaker", "source_id": "cabinet"},
                    },
                    {
                        "step": 1,
                        "agent": "agent_0",
                        "tool": "place_next_to",
                        "args": {
                            "object_id": "shaker",
                            "reference_object_id": "steak",
                        },
                    },
                ],
            }
        )

        steak_state = adapted["initial_state"]["objects"]["steak_main"]
        self.assertTrue(steak_state.get("preserve_pose"))

    def test_adapt_preserves_support_site_receptacle_ids(self):
        executor = FakeExecutor()
        executor.scene = {
            "fixtures": {
                "blender_main": {
                    "fixture_type": "blender",
                    "support_sites": {"bowl": {"site_type": "support"}},
                }
            },
            "objects": {
                "tomato_main": {"object_type": "tomato", "location": "blender_main"},
            },
        }
        executor.support_site_map = {"blender_main": {"blender_bowl": "bowl"}}
        adapter = TrajectoryAdapter(executor=executor)

        adapted = adapter.adapt(
            {
                "trajectory_id": "traj_blender",
                "initial_state": {
                    "agents": {"agent_0": {"location": "blender", "held_object": None}},
                    "objects": {"tomato": {"object_type": "tomato", "location": "blender"}},
                    "fixtures": {
                        "blender": {
                            "fixture_type": "blender",
                            "support_sites": {"blender_bowl": {"site_type": "support"}},
                        }
                    },
                    "machine_state": {},
                },
                "steps": [
                    {
                        "step": 0,
                        "agent": "agent_0",
                        "tool": "place_in_receptacle",
                        "args": {
                            "object_id": "tomato",
                            "receptacle_id": "blender_bowl",
                        },
                    }
                ],
            }
        )

        self.assertEqual(adapted["tool_calls"][0]["args"]["receptacle_id"], "bowl")

    def test_adapt_adds_default_site_for_partitioned_fixture_receptacle(self):
        executor = FakeExecutor()
        executor.scene = {
            "fixtures": {
                "fridge_main": {
                    "fixture_type": "fridge",
                    "support_sites": {
                        "shelf_0": {"site_type": "support"},
                        "shelf_1": {"site_type": "support"},
                    },
                },
                "counter_main": {"fixture_type": "counter_non_dining"},
            },
            "objects": {
                "bowl_main": {"object_type": "bowl", "location": "counter_main"},
            },
        }
        executor.default_site_map = {"fridge_main": "shelf_1"}
        adapter = TrajectoryAdapter(executor=executor)

        adapted = adapter.adapt(
            {
                "trajectory_id": "traj_fridge_default_site",
                "initial_state": {
                    "agents": {"agent_0": {"location": "fridge", "held_object": None}},
                    "objects": {"bowl": {"object_type": "bowl", "location": "counter"}},
                    "fixtures": {
                        "fridge": {"fixture_type": "fridge"},
                        "counter": {"fixture_type": "counter"},
                    },
                    "machine_state": {},
                },
                "steps": [
                    {
                        "step": 0,
                        "agent": "agent_0",
                        "tool": "place_in_receptacle",
                        "args": {
                            "object_id": "bowl",
                            "receptacle_id": "fridge",
                        },
                    }
                ],
            }
        )

        self.assertEqual(adapted["tool_calls"][0]["args"]["receptacle_id"], "fridge_main")
        self.assertEqual(adapted["tool_calls"][0]["args"]["target_site_id"], "shelf_1")

    def test_adapt_rewrites_pickup_source_site_to_parent_fixture(self):
        executor = FakeExecutor()
        executor.scene = {
            "fixtures": {
                "toaster_oven_main": {
                    "fixture_type": "toaster_oven",
                    "support_sites": {"rack_0": {"site_type": "support"}},
                },
                "dishwasher_main": {
                    "fixture_type": "dishwasher",
                    "support_sites": {"rack_0": {"site_type": "support"}},
                },
            },
            "objects": {
                "bread_main": {
                    "object_type": "bread",
                    "location": "toaster_oven_main",
                },
            },
        }
        executor.support_site_map = {
            "toaster_oven_main": {"rack_0": "rack_0"},
            "dishwasher_main": {"rack_0": "rack_0"},
        }
        adapter = TrajectoryAdapter(executor=executor, allow_approximate_ids=False)

        adapted = adapter.adapt(
            {
                "trajectory_id": "traj_toaster_rack",
                "initial_state": {
                    "agents": {
                        "agent_0": {"location": "toaster_oven", "held_object": None}
                    },
                    "objects": {
                        "bread": {"object_type": "bread", "location": "rack_0"}
                    },
                    "fixtures": {
                        "toaster_oven": {
                            "fixture_type": "toaster_oven",
                            "support_sites": {"rack_0": {"site_type": "support"}},
                        }
                    },
                    "machine_state": {},
                },
                "steps": [
                    {
                        "step": 0,
                        "agent": "agent_0",
                        "tool": "pick_up_object",
                        "args": {
                            "object_id": "bread",
                            "source_id": "rack_0",
                        },
                    }
                ],
            }
        )

        self.assertEqual(
            adapted["tool_calls"][0]["args"]["source_id"],
            "toaster_oven_main",
        )
        self.assertEqual(
            adapted["tool_calls"][0]["args"]["source_site_id"],
            "rack_0",
        )

    def test_adapt_maps_symbolic_container_to_native_implicit_container(self):
        executor = FakeExecutor()
        executor.scene = {
            "fixtures": {
                "stove_main": {"fixture_type": "stove"},
                "dining_main": {"fixture_type": "dining_table"},
            },
            "objects": {
                "obj": {"object_type": "steak", "location": "stove_main"},
                "obj_container": {"object_type": "pan", "location": "stove_main"},
                "plate": {"object_type": "plate", "location": "dining_main"},
            },
            "fixture_refs": {
                "stove": "stove_main",
                "dining_table": "dining_main",
            },
        }
        adapter = TrajectoryAdapter(executor=executor)

        adapted = adapter.adapt(
            {
                "trajectory_id": "traj_serve_steak",
                "initial_state": {
                    "agents": {
                        "agent_0": {"location": "stove", "held_object": None}
                    },
                    "objects": {
                        "steak": {"object_type": "steak", "location": "pan"},
                        "pan": {"object_type": "pan", "location": "stove"},
                        "plate": {"object_type": "plate", "location": "dining_table"},
                    },
                    "fixtures": {
                        "stove": {"fixture_type": "stove"},
                        "dining_table": {"fixture_type": "dining_table"},
                    },
                    "machine_state": {},
                },
                "steps": [
                    {
                        "step": 0,
                        "agent": "agent_0",
                        "tool": "pick_up_object",
                        "args": {"object_id": "pan", "source_id": "stove"},
                    },
                    {
                        "step": 1,
                        "agent": "agent_0",
                        "tool": "pick_up_object",
                        "args": {"object_id": "steak", "source_id": "pan"},
                    }
                ],
            }
        )

        self.assertEqual(adapter._object_aliases["steak"], "obj")
        self.assertEqual(adapter._object_aliases["pan"], "obj_container")
        self.assertEqual(
            adapted["tool_calls"][0]["args"]["object_id"],
            "obj_container",
        )
        self.assertEqual(
            adapted["tool_calls"][1]["args"]["object_id"],
            "obj",
        )
        self.assertEqual(
            adapted["tool_calls"][1]["args"]["source_id"],
            "obj_container",
        )

    def test_sim_ground_truth_prefers_object_type_as_object_id(self):
        executor = FakeExecutor()
        executor.scene = {
            "fixtures": {
                "counter_main": {"fixture_type": "counter_non_dining"},
                "dining_main": {"fixture_type": "dining_table"},
            },
            "objects": {
                "hotdog_bun_container": {"object_type": "plate"},
                "plate": {"object_type": "plate"},
            },
            "fixture_refs": {
                "dining_table": "dining_main",
            },
            "object_placements": {
                "hotdog_bun_container": "counter_main",
                "plate": "dining_main",
            },
        }
        adapter = TrajectoryAdapter(executor=executor)

        adapter._apply_sim_ground_truth(
            {
                "objects": {
                    "serving_plate": {
                        "object_type": "plate",
                        "location": "serving_surface",
                    }
                },
                "fixtures": {
                    "serving_surface": {"fixture_type": "dining_table"},
                },
            }
        )

        self.assertEqual(adapter._object_aliases["serving_plate"], "plate")

    def test_normalize_part_id_maps_drawer_front_to_sliding(self):
        adapter = TrajectoryAdapter(executor=FakeExecutor())

        self.assertEqual(adapter._normalize_part_id("drawer_front"), "sliding")

    def test_anchor_resolves_counter_family_parent_fixture(self):
        executor = FakeExecutor()
        executor.scene = {
            "fixtures": {
                "toaster_oven_main": {
                    "fixture_type": "toaster_oven",
                    "parent_fixture": "prep_counter",
                    "position": [1.0, 1.0, 0.0],
                },
                "prep_counter": {
                    "fixture_type": "counter_non_dining",
                    "position": [1.0, 1.0, 0.0],
                },
                "dining_counter": {
                    "fixture_type": "dining_counter",
                    "position": [1.1, 1.0, 0.0],
                },
            },
            "objects": {
                "ingredient_bowl": {"object_type": "bowl", "location": "toaster_oven_main"},
            },
            "fixture_refs": {
                "toaster_oven": "toaster_oven_main",
            },
        }
        adapter = TrajectoryAdapter(executor=executor)
        trajectory = {
            "trajectory_id": "traj_anchor_counter",
            "initial_state": {
                "agents": {
                    "agent_0": {"location": "counter", "held_object": None},
                },
                "objects": {
                    "ingredient_bowl": {"object_type": "bowl", "location": "counter"},
                },
                "fixtures": {
                    "toaster_oven": {"fixture_type": "toaster_oven"},
                    "counter": {"fixture_type": "counter"},
                },
            },
            "steps": [
                {
                    "step": 0,
                    "agent": "agent_0",
                    "tool": "navigate_to_fixture",
                    "args": {"fixture_id": "counter"},
                }
            ],
            "grounding_map": {
                "symbols": {
                    "toaster_oven": {
                        "entity_type": "fixture",
                        "resolver": "unique_fixture_type",
                        "fixture_type": "toaster_oven",
                    },
                    "counter": {
                        "entity_type": "fixture",
                        "resolver": "unique_fixture_type",
                        "fixture_type": "counter",
                        "anchor_fixture_symbol": "toaster_oven",
                        "preferred_fixture_types": ["counter"],
                    },
                }
            },
        }

        adapted = adapter.adapt(trajectory)

        self.assertEqual(
            adapted["initial_state"]["agents"]["agent_0"]["location"],
            "prep_counter",
        )
        self.assertEqual(
            adapted["tool_calls"][0]["args"]["fixture_id"],
            "prep_counter",
        )

    def test_execute_loads_state_then_runs_steps(self):
        executor = FakeExecutor()
        adapter = TrajectoryAdapter(executor=executor)
        trajectory = {
            "initial_state": {
                "agents": {
                    "agent_0": {"location": "staging_surface", "held_object": None}
                },
                "objects": {
                    "mug": {"object_type": "mug", "location": "mug_source_fixture"}
                },
                "fixtures": {
                    "mug_source_fixture": {"fixture_type": "cabinet"},
                    "staging_surface": {"fixture_type": "counter"},
                },
            },
            "steps": [
                {
                    "step": 0,
                    "agent": "agent_0",
                    "tool": "communicate",
                    "args": {"to": "agent_1", "message": "hello"},
                }
            ],
        }

        metadata = adapter.execute(trajectory)

        self.assertTrue(metadata["load_initial_state"]["loaded"])
        self.assertIsNotNone(executor.loaded_state)
        self.assertEqual(len(executor.executed_steps), 1)
        self.assertEqual(executor.executed_steps[0][0], "communicate")


class TestSimToolExecutorObservationHelpers(unittest.TestCase):
    def _make_executor(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor.runner = SimpleNamespace(
            render_height=4,
            render_width=5,
            _render_room_view=lambda: np.full((4, 5, 3), 7, dtype=np.uint8),
            _render_top_view=lambda: np.full((4, 5, 3), 9, dtype=np.uint8),
        )
        executor.env = SimpleNamespace(
            sim=SimpleNamespace(
                render=lambda height, width, camera_name: np.full(
                    (height, width, 3), 13, dtype=np.uint8
                )
            )
        )
        return executor

    def test_get_image_saves_requested_env_view(self):
        executor = self._make_executor()
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "top.png"
            result = executor.get_image(
                views=["top_view"],
                image_paths=[str(image_path)],
            )

            self.assertTrue(Path(result.details["image_path"]).exists())
            self.assertEqual(result.details["camera_name"], "top_view")

    def test_get_image_supports_agentview_right(self):
        executor = self._make_executor()
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "right.png"
            result = executor.get_image(
                views=["agentview_right"],
                image_paths=[str(image_path)],
                agent_id="agent_1",
            )

            self.assertTrue(Path(result.details["image_path"]).exists())
            self.assertEqual(result.details["camera_name"], "robot1_agentview_right")

    def test_get_image_supports_multiple_views_and_map(self):
        executor = self._make_executor()
        with tempfile.TemporaryDirectory() as tmpdir:
            top_path = Path(tmpdir) / "top.png"
            map_path = Path(tmpdir) / "map.png"

            def _fake_save_map_image(requested_path, clean_labels=True):
                requested_path = Path(requested_path)
                requested_path.write_bytes(b"map")
                return requested_path

            executor._save_map_image = _fake_save_map_image
            result = executor.get_image(
                views=["top_view", "map"],
                image_paths=[str(top_path), str(map_path)],
            )

            self.assertTrue(Path(result.details["image_paths"][0]).exists())
            self.assertTrue(map_path.exists())
            self.assertEqual(result.details["views"], ["top_view", "map"])
            self.assertEqual(result.details["camera_names"], ["top_view", "map"])
            self.assertEqual(
                result.details["image_paths"],
                [str(top_path.with_suffix(".jpg")), str(map_path)],
            )

    def test_get_image_reuses_cached_shared_view_for_unchanged_state(self):
        top_render_count = 0

        def _render_top_view():
            nonlocal top_render_count
            top_render_count += 1
            return np.full((4, 5, 3), 9, dtype=np.uint8)

        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor.runner = SimpleNamespace(
            render_height=4,
            render_width=5,
            _render_room_view=lambda: np.full((4, 5, 3), 7, dtype=np.uint8),
            _render_top_view=_render_top_view,
        )
        executor.env = SimpleNamespace(
            sim=SimpleNamespace(
                render=lambda height, width, camera_name: np.full(
                    (height, width, 3), 13, dtype=np.uint8
                )
            )
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            executor.get_image(
                views=["top_view"],
                image_paths=[str(Path(tmpdir) / "top_a.png")],
            )
            executor.get_image(
                views=["top_view"],
                image_paths=[str(Path(tmpdir) / "top_b.png")],
            )

        self.assertEqual(top_render_count, 1)


    def test_run_tool_plan_skips_initial_render_when_videos_disabled(self):
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor.ground_plan_template = lambda tool_calls: tool_calls
        executor.get_scene_description = lambda: {
            "task": "demo_task",
            "cameras": ["top_view", "room_view"],
        }
        executor.runner = SimpleNamespace(
            _num_robots=0,
            camera_names=["top_view", "room_view"],
        )
        executor.render = MagicMock(side_effect=AssertionError("render should not be called"))

        with tempfile.TemporaryDirectory() as tmpdir:
            metadata = executor.run_tool_plan(
                tool_calls=[],
                output_dir=tmpdir,
                skip_videos=True,
            )

        self.assertEqual(metadata["cameras"], ["top_view", "room_view"])
        executor.render.assert_not_called()


class TestSimToolExecutorLoadInitialState(unittest.TestCase):
    def test_load_initial_state_repositions_agents_and_holds_objects(self):
        move_calls = []
        navigate_calls = []
        machine_calls = []
        location_updates = []
        synced = []

        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._held_objects = {}
        executor._support_parents = {}
        executor._robot_spawn = "trajectory"
        executor.runner = SimpleNamespace(
            _fixtures={"counter_main": object(), "cab_main": object()},
            _num_robots=2,
            get_scene_description=lambda: {"object_placements": {}, "objects": {}},
            move_object=lambda object_id, location: move_calls.append(
                (object_id, location)
            ),
            _set_object_location=lambda object_id, fixture_id: location_updates.append(
                (object_id, fixture_id)
            ),
            _move_robot_near_fixture=lambda robot_idx, fixture_id: (
                navigate_calls.append((robot_idx, fixture_id)) or True
            ),
            _rescue_robot_to_kitchen=lambda robot_idx: None,
        )
        executor.open_hinged_part = lambda target_id, part_id: None
        executor.close_hinged_part = lambda target_id, part_id: None
        executor.open_sliding_part = lambda target_id, part_id: None
        executor.close_sliding_part = lambda target_id, part_id: None
        executor.env = SimpleNamespace(
            sim=SimpleNamespace(forward=lambda: None),
            objects={},
            obj_body_id={},
        )
        executor._set_fixture_machine_state = (
            lambda fixture_id, started: machine_calls.append((fixture_id, started))
        )
        executor._require_object = lambda object_id: object_id
        executor._sync_held_object = lambda robot_idx: synced.append(robot_idx)
        executor._parse_agent_idx = lambda agent_id: int(
            str(agent_id).replace("agent_", "")
        )

        summary = executor.load_initial_state(
            {
                "agents": {
                    "agent_0": {"location": "counter_main", "held_object": "mug_main"},
                    "agent_1": {"location": "cab_main", "held_object": None},
                },
                "objects": {
                    "mug_main": {"location": "cab_main"},
                },
                "fixtures": {},
                "machine_state": {
                    "counter_main": {"started": False},
                },
            }
        )

        self.assertEqual(move_calls, [])
        self.assertEqual(navigate_calls, [(0, "counter_main"), (1, "cab_main")])
        self.assertEqual(machine_calls, [("counter_main", False)])
        self.assertEqual(executor._held_objects, {0: "mug_main"})
        self.assertEqual(synced, [0])
        self.assertEqual(location_updates, [("mug_main", "counter_main")])
        self.assertTrue(summary["loaded"])

    def test_load_initial_state_applies_fixture_part_states(self):
        opened = []
        closed = []

        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor._held_objects = {}
        executor._support_parents = {}
        executor._robot_spawn = "trajectory"
        fridge_fixture = SimpleNamespace(
            close_door=lambda env: closed.append(("fridge_main", "hinged", "closed"))
        )
        cab_fixture = SimpleNamespace(
            open_door=lambda env: opened.append(("cab_main", "hinged", "open"))
        )
        executor.runner = SimpleNamespace(
            _fixtures={"fridge_main": fridge_fixture, "cab_main": cab_fixture},
            _num_robots=2,
            get_scene_description=lambda: {"object_placements": {}, "objects": {}},
            move_object=lambda object_id, location, preferred_xy=None: None,
            _set_object_location=lambda object_id, fixture_id: None,
            _move_robot_near_fixture=lambda robot_idx, fixture_id: None,
            _rescue_robot_to_kitchen=lambda robot_idx: None,
        )
        executor.env = SimpleNamespace(
            sim=SimpleNamespace(forward=lambda: None),
            objects={},
            obj_body_id={},
        )
        executor._parse_agent_idx = lambda agent_id: 0
        executor._set_fixture_machine_state = lambda fixture_id, started: None
        executor._sync_held_object = lambda robot_idx: None
        executor._require_object = lambda object_id: object_id

        executor.load_initial_state(
            {
                "agents": {},
                "objects": {},
                "fixtures": {
                    "fridge_main": {"parts": {"hinged": {"state": "closed"}}},
                    "cab_main": {"parts": {"hinged": {"state": "open"}}},
                },
                "machine_state": {},
            }
        )

        self.assertEqual(opened, [("cab_main", "hinged", "open")])
        self.assertEqual(closed, [("fridge_main", "hinged", "closed")])


class TestSimToolExecutorFixtureStateSync(unittest.TestCase):
    def test_open_hinged_part_forwards_sim_state(self):
        fixture = SimpleNamespace(open_door=MagicMock())
        executor = SimToolExecutor.__new__(SimToolExecutor)
        executor.runner = SimpleNamespace(_fixtures={"fridge_main": fixture})
        executor.env = SimpleNamespace(sim=SimpleNamespace(forward=MagicMock()))
        executor._robot_near_fixture = lambda robot_idx, fixture_id: True
        executor._move_robot_near_fixture_with_retries = MagicMock()
        executor._sync_held_object = MagicMock()

        result = executor.open_hinged_part("fridge_main", "hinged", robot_idx=0)

        fixture.open_door.assert_called_once_with(env=executor.env)
        executor.env.sim.forward.assert_called()
        self.assertTrue(result.success)


if __name__ == "__main__":
    unittest.main()
