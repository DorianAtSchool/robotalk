# Concurrency semantics

Two agents act at the same time. Each agent has a private context: its own
observations, its own tool-call history and the messages it receives. The task
goal, initial state, tool definitions and coordination rules are shared.

## Ticks

A tick holds one non-observation tool call for each runnable agent. An agent
blocked in `wait_for_signal` has no call in that tick.

Both agents' calls are checked against the symbolic state at the **beginning**
of the tick, together with checks on whether the two calls contend with each
other. The accepted calls are then executed and their effects committed. An
agent therefore cannot rely on its partner opening a door earlier in the same
tick. In the simulator the physical skills run one after the other, but they
are decided and validated concurrently.

Observation calls (`get_image`) don't take part in tick ordering. They return
images to the calling agent without advancing the schedule.

The validator is `robotalk.tasks.shared.concurrent_fsm.ConcurrentTaskValidator`.
The same scheduler semantics are used by offline validation (generation) and by
the live evaluator.

## Opening protocol (leader–follower)

One agent is the coordinator for the episode, and both agents are told which.

| Tick | Coordinator | Follower |
|---|---|---|
| 0 | `communicate(..., coordination_phase="propose")`: a concrete division of work | `communicate(..., coordination_phase="await_plan")` |
| 1 | `communicate(..., coordination_phase="await_confirmation")` | `communicate(..., coordination_phase="confirm")` |
| 2+ | Physical work may begin | Physical work may begin |

Messages are delivered after the tick in which they are sent. The follower's
tick-0 message therefore cannot depend on the proposal. The protocol fixes the
order of these phases; it doesn't verify that the plan in the message is good.

## Waiting and releasing

- `wait_for_signal(from, about)` blocks the caller. The wait is private: the
  partner sees only the message that precedes it, which should ask for the
  release and name the release keyword.
- A matching release is a later `communicate` call from that sender with
  `releases=<keyword>`. Other messages don't wake the waiter.
- The waiter stays blocked during the tick of the release and resumes on the
  next tick.
- Only an actual `wait_for_signal` call blocks an agent. A promise to wait
  doesn't.

Example: agent 1 is at the fridge, and agent 0 needs it.

| Tick | Agent 0 | Agent 1 |
|---|---|---|
| N | `communicate`: "Release `fridge` when you are done." | a fridge action |
| N+1 | `wait_for_signal(from=agent_1, about=fridge)` | `give_space(fridge)` |
| N+2 | (blocked) | `communicate(..., releases="fridge")` |
| N+3 | `navigate_to_fixture(fridge)` | … |

If both agents end up waiting on each other, the episode ends as a deadlock.

## Shared and exclusive resources

- **Objects.** An object being manipulated is exclusive. A supporting object
  such as a plate or tray can be shared while the agents handle different items
  on it, but moving it while the partner uses it is a conflict.
- **Shared workspaces.** Counters, islands and tables can be used by both
  agents. A cabinet belongs to its parent counter's workspace, but opening or
  closing a cabinet door conflicts with the partner using that cabinet in the
  same tick.
- **Exclusive fixtures.** Drawers, fridge, microwave, oven, dishwasher, stove,
  sink, toaster, toaster oven, coffee machine, blender, stand mixer and
  electric kettle admit one agent at a time.
  - An occupant leaves by calling `give_space(fixture)` or by navigating
    elsewhere.
  - The partner may enter on a later tick, not the same one.
  - Entering while the occupant is still there, or in the tick it leaves, is a
    conflict.

## Task completion

An episode succeeds when the symbolic state satisfies the task's goal
conditions; agents do not declare global completion. An agent that finishes its
part reports this to its partner and waits.
