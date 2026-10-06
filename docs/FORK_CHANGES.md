# Changes to RoboCasa365

`robocasa/` is a modified copy of [RoboCasa365](https://github.com/robocasa/robocasa)
at upstream commit `1b19563` (2026-03-02). The RoboCasa code is MIT-licensed,
© the RoboCasa Team; see [`robocasa/LICENSE`](../robocasa/LICENSE). This file
lists our changes, as the license notices require.

## Modified upstream files

| File | Change |
|---|---|
| `environments/kitchen/kitchen.py` | Allows two robots (`robots=["PandaOmron", "PandaOmron"]`): per-robot initial placement and base poses, per-robot action defaults, and single-robot aliases kept for upstream code paths. Object configurations can be filtered and bound to a trajectory's symbolic objects, so a scene contains the objects a trajectory refers to. |
| `utils/env_utils.py` | `compute_robot_base_placement_pose`, `init_robot_base_pose` and `detect_robot_collision` take a robot index. Collision checks count contacts with the other robot. |
| `utils/camera_utils.py` | Creates each robot's cameras, not only robot 0's. |
| `environments/kitchen/composite/filling_serving_dishes/meat_skewer_assembly.py`, `garnishing_dishes/garnish_cake.py` | Placement regions constrain the object's center rather than its full footprint, so scenes that are otherwise valid initialize. |
| `environments/kitchen/composite/plating_food/plate_store_dinner.py` | Places the second steak in the first steak's container (`meat1_container`) instead of sampling its own spot on the stove. |
| `wrappers/gym_wrapper.py` | Default actions for every robot's controller parts. |
| `wrappers/enclosing_wall_render_wrapper.py`, `demos/demo_kitchen_scenes.py` | Run without a display (hotkeys and teleoperation devices are optional). The scene demo takes `--num_robots`. |
| `__init__.py` | Quieter optional-import warnings; robosuite version detection that tolerates editable installs. |

## Added files

| File | Purpose |
|---|---|
| `utils/sim_tool_executor*.py`, `utils/sim_tool_specs.py` | The skill executor: runs RoboTalk's symbolic tool calls (navigation, picking, placing, fixture parts, observations) in the simulator. |
| `utils/placement.py`, `utils/placement_map.py`, `utils/occupancy_grid.py` | Placement sampling, top-down maps and navigation occupancy for two robots. |
| `utils/trajectory_runner*.py`, `utils/trajectory_adapter.py`, `utils/trajectory_pruning.py` | Replay of symbolic trajectories in a scene, including rendering of requested camera views. |
| `LICENSE` | RoboCasa365's MIT license. |

## Assets

No RoboCasa365 asset is modified or redistributed here beyond the asset files
that upstream itself tracks. The kitchen assets (CC BY 4.0, © the RoboCasa
Team) are downloaded separately with
`python -m robocasa.scripts.download_kitchen_assets`.

## robosuite

robosuite (MIT) is a dependency, not part of this repository. It is installed
from [DorianAtSchool/robosuite](https://github.com/DorianAtSchool/robosuite) at
commit `94f9aa2`: upstream ARISE commit `aaa8b9b` plus one change in
`robosuite/models/robots/robot_model.py`. That change gives each robot's
`manipulator_mount` body its robot prefix, so two mobile manipulators can share
a scene.
