# Tool interface

Agents act only through tool calls. The same tool definitions are used to
generate demonstrations, train policies and evaluate them
(`robotalk/tools/subatomic_tool_specs.py`). Arguments are symbolic IDs taken
from the task's initial state: fixture IDs, object IDs, and the part and control
keys of a fixture.

| Family | Tools | Meaning |
|---|---|---|
| Navigation | `navigate_to_fixture(fixture_id)` | Move to a fixture's workspace |
| Picking | `pick_up_object(object_id, source_id)` | Pick an object from where it is, with an empty hand |
| Placing | `place_on_surface(object_id, support_id)` | On a fixture surface (counter, table, stove, …) |
| | `place_on_object(object_id, support_object_id)` | On a movable support (plate, tray, …) |
| | `place_in_receptacle(object_id, receptacle_id)` | Inside a receptacle (bowl, pot, cabinet, fridge, …) |
| | `place_next_to(object_id, reference_object_id=… \| reference_fixture_id=…)` | Beside a reference, on the supporting surface |
| | `place_under(object_id, reference_fixture_id)` | Under an appliance outlet, e.g. a mug under a coffee dispenser |
| Fixture parts | `open_hinged_part`, `close_hinged_part`, `open_sliding_part`, `close_sliding_part` `(target_id, part_id)` | Open or close a door or drawer the fixture exposes |
| Fixture controls | `press_button`, `press_lever` `(target_id, control_id)`; `set_rotary_control(target_id, control_id, goal)` | Operate a control the fixture exposes |
| Communication | `communicate(to, message, releases?, coordination_phase?)` | Send a message; optionally release a waiting partner or mark a protocol phase |
| Waiting | `wait_for_signal(from, about)` | Block until a matching release from that agent |
| Workspace handover | `give_space(fixture_id)` | Leave an exclusive fixture's workspace |
| Observation | `get_image(views)` | Return the requested camera views to the caller |

Camera views: `top_view`, `room_view`, `map`, `wrist`, `agentview_center`,
`agentview_left`, `agentview_right`.

## Symbolic state

```json
{
  "agents": {
    "agent_0": {"location": "fridge", "held_object": null},
    "agent_1": {"location": "counter", "held_object": null}
  },
  "objects": {
    "bowl": {"object_type": "bowl", "location": "counter"},
    "meat2": {"object_type": "steak", "location": "pan"}
  },
  "fixtures": {
    "counter": {"fixture_type": "counter"},
    "fridge": {"fixture_type": "fridge",
               "parts": {"door": {"part_type": "hinged_part", "state": "closed"}}}
  }
}
```

This is a fragment.
- `door` is the `part_id` that the part tools take.
- An agent holds at most one object.
- Containment is separate from support: a steak can be in a pan that is on a
  stove.

## Execution in the simulator

The validator checks every call against the task's preconditions and the
concurrency rules ([concurrency semantics](concurrency_semantics.md)). Valid
calls are executed in the RoboCasa365 scene by the skill executor
(`robocasa/utils/sim_tool_executor*.py`). It resolves symbols to simulator
entities, samples placements, and moves the robot bases and held objects.
Skills are scripted, so the policy decides *what* to do and the executor
decides *how*.
