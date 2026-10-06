"""A validator that replays a trajectory the way the executor actually runs it.

The old validator walks the flat step list top to bottom against one state. That
list is a *serialization* of two independent per-agent streams, so reading it in
order describes an interleaving that never happens: a step written at index 5 may
run long after one written at index 6. Every coordination defect measured on this
dataset traces back to that gap, and each was patched separately -- a lock-step
collision checker here, a tick-span contention rule there, a wait-necessity
heuristic in a third place. They were all approximations of one thing the
validator was not doing: running the plan.

This one runs it. Two agents, a clock, per-tool durations, and waits that block
until a release is *delivered*. Contention stops being something to infer -- if
two agents end up holding the same object or standing at the same exclusive
fixture at the same instant, the replay is simply in that state, and the check is
a state invariant rather than a guess about plan positions.

Two duration models, both faithful to a real executor regime:

    LOCK_STEP  every call costs 1.0  -- `live_sim_eval._UNIFORM_DURATIONS`
    EXECUTOR   the per-class floors  -- `live_sim_eval._tool_duration` default

LOCK_STEP is the "tick" regime: instants are integers and one instant is one
tick. EXECUTOR is what the live evaluation runs by default. A plan that is
correct only under one of them is correct only by luck, so `validate` checks
both by default; see `DURATION_MODELS`.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
import re
from typing import Any, Sequence

from .constants import (
    ACQUIRE_TOOL_NAMES,
    CLOSE_PART_TOOL_NAMES,
    DEPENDENCY_ARG_NAMES,
    EXCLUSIVE_FIXTURE_TYPES,
    FIXTURE_ARG_NAMES,
    GIVE_SPACE_TOOL_NAMES,
    NAVIGATION_TOOL_NAMES,
    OBSERVATION_TOOL_NAMES,
    OPEN_PART_TOOL_NAMES,
    RELEASE_TOOL_NAMES,
    SOCIAL_TOOL_NAMES,
    WAIT_TOOL_NAMES,
)
from .errors import (
    DeadlockSemanticValidationError,
    MissingInitialCommunicationSemanticValidationError,
    MissingTaskActionSemanticValidationError,
    PostGoalActionSemanticValidationError,
    ResourceConflictSemanticValidationError,
    TrajectoryValidationError,
    UnsatisfiedGoalSemanticValidationError,
    UnsupportedToolSemanticValidationError,
    WaitSignalSemanticValidationError,
)
from .scheduling import (
    ConcurrentScheduler,
    opening_protocol_error,
    release_keys,
    wait_key,
)
from .validation_contract import VALIDATOR_CONTRACT_VERSION
from .workspace_semantics import canonical_agent_workspace

LOCK_STEP = "lock_step"
EXECUTOR = "executor"
DURATION_MODELS = (LOCK_STEP, EXECUTOR)

# Mirrors live_sim_eval._tool_duration. The executor teleports, so measured
# sim-steps come out ~0 and these floors ARE the duration model there too.
_EXECUTOR_FLOORS = {"communicate": 0.25, "get_image": 0.25, "navigate_to_fixture": 4.0}
_EXECUTOR_DEFAULT_FLOOR = 2.0

# Float clocks accumulate error; instants are compared, not stored, so round.
_EPS = 1e-6

_WAIT_TOOL = "wait_for_signal"


def is_exclusive_fixture(
    fixture_id: str | None,
    initial_state: dict[str, Any],
) -> bool:
    """Return whether one fixture has the FSM's single-user workspace."""

    fixtures = initial_state.get("fixtures") or {}
    state = fixtures.get(fixture_id)
    if not isinstance(state, dict):
        return False
    return str(state.get("fixture_type", "")).lower() in EXCLUSIVE_FIXTURE_TYPES


def _cabinet_fixture_for_step(
    step: dict[str, Any], initial_state: dict[str, Any]
) -> tuple[str | None, bool]:
    """Return (cabinet id, changes-door-state) for one cabinet call."""

    from .workspace_semantics import is_cabinet

    tool = step.get("tool")
    args = step.get("args") or {}
    fixture_id = None
    if tool in OPEN_PART_TOOL_NAMES | CLOSE_PART_TOOL_NAMES:
        fixture_id = args.get("target_id")
    elif tool in ACQUIRE_TOOL_NAMES:
        fixture_id = args.get("source_id")
    elif tool in RELEASE_TOOL_NAMES:
        fixture_id = (
            args.get("target_id")
            or args.get("support_id")
            or args.get("receptacle_id")
        )
    if isinstance(fixture_id, str) and is_cabinet(initial_state, fixture_id):
        return fixture_id, tool in OPEN_PART_TOOL_NAMES | CLOSE_PART_TOOL_NAMES
    return None, False


def contention_resources(
    step: dict[str, Any],
    initial_state: dict[str, Any],
) -> frozenset[str]:
    """Return the resources claimed by one call under canonical FSM semantics.

    Roomy fixtures are deliberately absent. The object being manipulated is
    exclusive, while stationary source/support/receptacle/reference objects are
    shared reads. ``simultaneous_contentions`` separately catches the
    asymmetric case where one call moves such an object while another reads it.
    """

    tool = step.get("tool")
    if tool in SOCIAL_TOOL_NAMES or tool in OBSERVATION_TOOL_NAMES:
        return frozenset()
    if tool in GIVE_SPACE_TOOL_NAMES:
        return frozenset()
    objects = set(initial_state.get("objects") or {})
    args = step.get("args") or {}
    found: set[str] = set()
    object_id = args.get("object_id")
    if isinstance(object_id, str) and object_id in objects:
        found.add(object_id)
    for name in FIXTURE_ARG_NAMES:
        value = args.get(name)
        if isinstance(value, str) and is_exclusive_fixture(value, initial_state):
            found.add(value)
    return frozenset(found)


def simultaneous_contentions(
    left: dict[str, Any],
    right: dict[str, Any],
    initial_state: dict[str, Any],
) -> frozenset[str]:
    """Return resources that make two same-tick calls incompatible.

    This is the canonical pair classifier for both offline validation and live
    scheduling. In addition to ordinary shared claims, moving an object while
    the other call uses it as a source/support/reference is a conflict.
    """

    conflicts = set(contention_resources(left, initial_state)) & set(
        contention_resources(right, initial_state)
    )
    left_cabinet, left_transition = _cabinet_fixture_for_step(left, initial_state)
    right_cabinet, right_transition = _cabinet_fixture_for_step(right, initial_state)
    if (
        isinstance(left_cabinet, str)
        and left_cabinet == right_cabinet
        and (left_transition or right_transition)
    ):
        conflicts.add(f"cabinet_access:{left_cabinet}")
    left_args = left.get("args") or {}
    right_args = right.get("args") or {}
    left_object = left_args.get("object_id")
    right_object = right_args.get("object_id")
    left_secondary = {
        left_args.get(name)
        for name in DEPENDENCY_ARG_NAMES
        if name != "object_id" and isinstance(left_args.get(name), str)
    }
    right_secondary = {
        right_args.get(name)
        for name in DEPENDENCY_ARG_NAMES
        if name != "object_id" and isinstance(right_args.get(name), str)
    }
    if isinstance(left_object, str) and left_object in right_secondary:
        conflicts.add(left_object)
    if isinstance(right_object, str) and right_object in left_secondary:
        conflicts.add(right_object)
    return frozenset(conflicts)


def atomic_handover_conflicts(
    calls: Sequence[dict[str, Any]],
    runtime_state: Any,
    initial_state: dict[str, Any],
) -> dict[str, frozenset[str]]:
    """Return entrants that race an occupant's same-tick ``give_space``.

    A tick is validated against its beginning-of-tick state. Physically
    leaving an exclusive workspace is therefore visible to other agents only
    on the following tick, independent of call iteration order.
    """

    calls_by_agent = {
        str(call.get("agent")): call
        for call in calls
        if isinstance(call, dict) and isinstance(call.get("agent"), str)
    }
    vacated_by: dict[str, str] = {}
    for agent_id, call in calls_by_agent.items():
        if call.get("tool") not in GIVE_SPACE_TOOL_NAMES:
            continue
        fixture_id = (call.get("args") or {}).get("fixture_id")
        if not isinstance(fixture_id, str) or not is_exclusive_fixture(
            fixture_id, initial_state
        ):
            continue
        agent_state = getattr(runtime_state, "agents", {}).get(agent_id)
        if agent_state is not None and agent_state.location == fixture_id:
            vacated_by[fixture_id] = agent_id

    conflicts: dict[str, set[str]] = {}
    # Also reject entry/use while another agent remains at an exclusive
    # workspace. Offline replay catches the resulting co-location after a
    # commit, but live evaluation must reject the proposal *before* executing
    # it; otherwise the teleporting executor may silently displace the holder.
    for agent_id, call in calls_by_agent.items():
        claimed = contention_resources(call, initial_state)
        for fixture_id in claimed:
            if not is_exclusive_fixture(fixture_id, initial_state):
                continue
            for occupant_id, occupant_state in getattr(
                runtime_state, "agents", {}
            ).items():
                if occupant_id == agent_id:
                    continue
                occupant_workspace = canonical_agent_workspace(
                    initial_state, getattr(occupant_state, "location", None)
                )
                target_workspace = canonical_agent_workspace(
                    initial_state, fixture_id
                )
                if occupant_workspace == target_workspace:
                    conflicts.setdefault(agent_id, set()).add(fixture_id)
    for agent_id, call in calls_by_agent.items():
        for fixture_id, occupant in vacated_by.items():
            if agent_id == occupant:
                continue
            if fixture_id in contention_resources(call, initial_state):
                conflicts.setdefault(agent_id, set()).add(fixture_id)
    return {
        agent_id: frozenset(resources)
        for agent_id, resources in conflicts.items()
    }


def duration_of(tool_name: str, model: str) -> float:
    """Virtual-clock cost of one completed call under `model`.

    Observation is FREE. It is instrumentation, not authored work: the model
    never writes get_image in tick format -- every observation in the corpus
    was injected afterwards by the image pass, two per action. Charging them
    made a plan's coordination depend on how many actions each agent happened
    to take, because an agent's lock-step clock advances once per call IT
    makes. Agents with different action counts drifted apart, releases landed
    before their waiters blocked, and 47 of 1550 corpus trajectories that were
    valid as written became deadlocks purely by being photographed.

    `_apply` already treats these tools as having no symbolic effect; this is
    the scheduling half of the same statement. Nothing outside `replay()` reads
    this function -- live sim keeps its own `_tool_duration`, and neither the
    sim executor nor the training pipeline has a clock at all.
    """

    if tool_name in OBSERVATION_TOOL_NAMES:
        return 0.0
    if model == LOCK_STEP:
        return 1.0
    return _EXECUTOR_FLOORS.get(tool_name, _EXECUTOR_DEFAULT_FLOOR)


@dataclass
class Event:
    """One call, placed on the clock."""

    index: int  # position in the flat step list
    agent: str
    tool: str
    start: float
    end: float

    @property
    def tick(self) -> int:
        """Instant ordinal; under LOCK_STEP this is literally the tick."""

        return int(round(self.start))


@dataclass
class Replay:
    """What happened when the plan was run."""

    model: str
    events: list[Event] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    blocked: list[str] = field(default_factory=list)
    idle: dict[str, float] = field(default_factory=dict)
    makespan: float = 0.0
    goal_at: float | None = None
    post_goal: list[str] = field(default_factory=list)
    # Handover protocol violations. Unlike `conflicts` these need no clock: a
    # release that reports a departure that did not happen is wrong under every
    # duration model, whether or not the timing exposes it.
    protocol: list[str] = field(default_factory=list)
    first_action: bool = False
    # Exclusive fixtures the sampled initial state already put both agents at.
    initial_overlap: list[str] = field(default_factory=list)
    runtime_state: Any = None
    step_error: TrajectoryValidationError | None = None

    @property
    def deadlocked(self) -> bool:
        return bool(self.blocked)

    def ticks(self, total_steps: int) -> list[int | None]:
        """Per-step instant ordinals, indexed like the flat step list."""

        out: list[int | None] = [None] * total_steps
        for event in self.events:
            out[event.index] = event.tick
        return out


def _released_ids(step: dict[str, Any]) -> list[str]:
    value = (step.get("args") or {}).get("releases") or []
    values = [value] if isinstance(value, str) else list(value)
    return [str(item).strip() for item in values if str(item).strip()]


class ConcurrentTaskValidator:
    """Replays a candidate as concurrent per-agent streams on a virtual clock.

    Composes an existing `FiniteStateTaskValidator` (or any subclass, so the
    spec-driven task semantics come along unchanged) and reuses its per-step
    precondition and effect machinery. What is new here is only *when* each step
    is applied -- and that turns out to be where all the bugs were.
    """

    def __init__(
        self,
        validator: Any,
        *,
        models: Sequence[str] = DURATION_MODELS,
        require_initial_communication: bool = True,
    ) -> None:
        self.validator = validator
        self.composite_task = validator.composite_task
        self.agent_ids = tuple(validator.agent_ids)
        # Which duration regimes `validate` insists on by default. Tick
        # generation passes (LOCK_STEP,) alone. The model writes a grid of
        # equal instants; judging that grid under per-tool durations makes
        # correctness depend on wall-clock costs it was never told about. A
        # waiter whose own route is expensive arrives at its wait late, the
        # holder's cheap messages reach the release early, and a release that
        # fires with nobody blocked is discarded. 7 of the 9 deadlocks in job
        # 266947 were exactly this, and every one was clean under LOCK_STEP.
        self.models = tuple(models)
        self.require_initial_communication = bool(require_initial_communication)

    # -- replay -----------------------------------------------------------

    def replay(
        self,
        candidate: dict[str, Any],
        *,
        model: str = LOCK_STEP,
        stop_on_step_error: bool = True,
    ) -> Replay:
        """Runs the plan on a clock and reports what the run looked like."""

        validator = self.validator
        agents = validator._normalize_agents(candidate.get("agents"))
        steps = validator._normalize_steps(candidate.get("steps"))
        state = validator._build_runtime_state(agents)

        result = Replay(model=model, runtime_state=state)
        if not steps:
            return result

        streams: dict[str, list[int]] = {agent_id: [] for agent_id in self.agent_ids}
        for index, step in enumerate(steps):
            streams.setdefault(step["agent"], []).append(index)

        cursor = {agent_id: 0 for agent_id in streams}
        waiting_since: dict[str, float] = {}
        scheduler = ConcurrentScheduler(tuple(streams))
        # A release is an EVENT: it wakes only an agent already waiting when it
        # fires, exactly as AgentRuntime.deliver does -- a message sent before
        # the waiter blocked was never delivered to it and never will be.
        fired: dict[tuple[str, str], list[float]] = {}
        seen_conflicts: set[tuple[str, ...]] = set()
        # The tick whose co-location verdict is still open, and everyone who
        # acted during it. Flushed when the clock advances, and once at the end.
        open_clock: float | None = None
        open_acting: set[str] = set()
        # When each wait first blocked, and when each resource was actually
        # let go of, so the two orderings can be checked after the run.
        wait_started: dict[int, float] = {}
        departures: dict[tuple[str, str], list[float]] = {}
        # Last non-observation step each agent took, so a release can be checked
        # against what its sender did immediately before speaking.
        previous: dict[str, dict[str, Any] | None] = {a: None for a in streams}
        # Agents that START co-located are a property of the sampled initial
        # state, not of the plan, so the opening overlap is excused. The grace
        # expires the moment they separate: coming BACK to a fixture the other
        # never left is the plan's doing, and several trajectories do exactly
        # that under cover of having begun there.
        grace: set[tuple[str, ...]] = {
            ("at", location, *sorted(here))
            for location, here in self._occupancy(state).items()
            if len(here) > 1
        }
        result.initial_overlap = sorted(key[1] for key in grace)
        clock = 0.0

        while True:
            # An agent blocked on its LAST step still counts: it has nothing
            # left to do, but it is stuck rather than finished, and the
            # executor would sit there too.
            pending = [
                a for a in streams
                if cursor[a] < len(streams[a]) or scheduler.blocked(a)
            ]
            if not pending:
                break

            # A wait is CALLED once and then the agent sits blocked, exactly as
            # AgentRuntime does: `waiting_for` is set when the call is made and
            # cleared when a message is delivered. Deferring the call until it
            # could succeed put it on the clock at the instant it discharged,
            # which drew the agent as idle for the whole block and then calling
            # wait at the very moment it was already free.
            has_work = {a: cursor[a] < len(streams[a]) for a in streams}
            next_ready = scheduler.next_ready_time(has_work=has_work)
            if next_ready is not None:
                clock = max(clock, next_ready)
            runnable = scheduler.ready_agents(clock=clock, has_work=has_work)

            if not runnable and not scheduler.blocked_agents():
                break

            if not runnable:
                result.blocked = scheduler.blocked_agents()
                for agent_id in result.blocked:
                    index = streams[agent_id][cursor[agent_id] - 1]
                    step = steps[index]
                    args = step.get("args") or {}
                    waiting = scheduler.states[agent_id].waiting_for or {}
                    key = (str(waiting.get("from")), str(waiting.get("about")))
                    earlier = fired.get(key, ())
                    since = waiting_since[agent_id]
                    detail = (
                        f"the release fired at t={max(earlier):g}, before this "
                        f"agent began waiting at t={since:g}, so it was "
                        f"delivered to nobody -- the wait is too late to be "
                        f"woken by it"
                        if earlier
                        else "it is never released"
                    )
                    result.conflicts.append(
                        f"  {agent_id} is blocked forever from t={since:g}, at step "
                        f"{step['step']}: wait_for_signal(from={args.get('from')!r}, "
                        f"about={str(args.get('about'))!r}) -- {detail}."
                    )
                break

            acting = runnable

            # Co-location is judged once per TICK, not once per batch of calls.
            # A zero-duration call leaves its agent ready at the same clock, so
            # one tick can produce several batches -- and checking each batch
            # samples the world mid-tick. That caught seasoning_steak/traj_000001
            # handing the cabinet from agent_1 to agent_0 within tick 7: between
            # the two halves of the handoff both agents are momentarily recorded
            # there. A tick is atomic, so the only state worth judging is the one
            # it ends in. Deferred here rather than after the batch because this
            # is the last point at which `state` still holds the previous tick.
            if open_clock is not None and abs(clock - open_clock) > _EPS:
                self._check_co_location(
                    state, open_acting, open_clock, seen_conflicts, grace, result
                )
                open_acting = set()
            open_clock = clock
            open_acting.update(acting)

            instant = [(a, streams[a][cursor[a]]) for a in acting]
            instant_calls = [
                {**steps[index], "agent": agent_id}
                for agent_id, index in instant
            ]
            precondition_errors = self.validate_cycle_preconditions(
                instant_calls, state
            )
            if precondition_errors:
                failing_agent = next(
                    agent_id for agent_id, _ in instant
                    if agent_id in precondition_errors
                )
                error = precondition_errors[failing_agent]
                failing_index = next(
                    index for agent_id, index in instant
                    if agent_id == failing_agent
                )
                details = dict(error.details)
                details.setdefault(
                    "state_before_tick",
                    validator._build_final_state(deepcopy(state)),
                )
                details.setdefault("atomic_tick", clock)
                error = type(error)(
                    str(error), step=error.step, details=details
                )
                result.step_error = validator._validation_error_with_step(
                    error, steps[failing_index]["step"]
                )
                result.makespan = clock
                return result
            for entrant, resources in atomic_handover_conflicts(
                instant_calls, state, validator.initial_state
            ).items():
                for resource in resources:
                    key = ("atomic-handover", resource, entrant)
                    if key in seen_conflicts:
                        continue
                    seen_conflicts.add(key)
                    # The occupant is whoever stood at the resource's
                    # workspace at the start of the tick; it may be leaving
                    # with give_space, navigating away, or staying put.
                    target = canonical_agent_workspace(
                        validator.initial_state, resource
                    )
                    occupant = next(
                        (
                            agent_id
                            for agent_id, agent_state in state.agents.items()
                            if agent_id != entrant
                            and canonical_agent_workspace(
                                validator.initial_state,
                                getattr(agent_state, "location", None),
                            ) == target
                        ),
                        "another agent",
                    )
                    occupant_call = next(
                        (c for c in instant_calls if c["agent"] == occupant), {}
                    )
                    if occupant_call.get("tool") in GIVE_SPACE_TOOL_NAMES:
                        action = "calls give_space"
                    elif occupant_call.get("tool") in NAVIGATION_TOOL_NAMES:
                        action = "navigates away"
                    else:
                        action = "remains there"
                    result.conflicts.append(
                        f"  t={clock:g}: {entrant} enters or uses {resource!r} "
                        f"while its beginning-of-tick occupant {occupant} "
                        f"{action}. A departure takes effect after this atomic "
                        "tick; enter on the following tick."
                    )
            self._check_simultaneous_use(steps, instant, clock, seen_conflicts, result)

            for agent_id, index in instant:
                step = steps[index]
                tool = step["tool"]

                if tool in WAIT_TOOL_NAMES:
                    self._check_ask(step, agent_id, previous[agent_id], result)
                    args = step.get("args") or {}
                    key = (args.get("from"), str(args.get("about")))
                    scheduler.block(agent_id, step, clock=clock)
                    waiting_since[agent_id] = clock
                    wait_started[index] = clock
                    # A release fired at the very instant the wait is called is
                    # a race in the executor -- whether the message lands before
                    # or after `waiting_for` is set depends on the order the
                    # cycle happens to process proposals. Discharge stays
                    # permissive; the plan is flagged instead of gambled on.
                    if any(at <= clock + _EPS for at in fired.get(key, ())):
                        result.protocol.append(
                            f"  step {step['step']}: {agent_id} calls "
                            f"wait_for_signal on {str(args.get('about'))!r} at "
                            f"t={clock:g}, but {args.get('from')} released it at "
                            f"t={max(fired[key]):g}. The wait is inert -- block "
                            f"before the release, or drop it."
                        )
                else:
                    # Every call in this instant was already checked against
                    # the same frozen beginning-of-tick state. Commit effects
                    # only now; no call may gain a prerequisite from an
                    # earlier agent's commit in this tick.
                    self._commit(step, state)
                    if tool not in OBSERVATION_TOOL_NAMES | SOCIAL_TOOL_NAMES:
                        result.first_action = True

                if tool == "communicate":
                    for released in _released_ids(step):
                        self._check_release(
                            step, agent_id, released, state,
                            previous[agent_id], clock, result,
                        )
                        fired.setdefault((agent_id, released), []).append(clock)
                    scheduler.deliver(
                        step,
                        clock=clock,
                        resume_delay=duration_of(_WAIT_TOOL, model),
                    )

                for resource in self._let_go_of(step, previous[agent_id]):
                    departures.setdefault((agent_id, resource), []).append(clock)

                if tool not in OBSERVATION_TOOL_NAMES:
                    previous[agent_id] = step

                end = clock + duration_of(tool, model)
                result.events.append(
                    Event(index=index, agent=agent_id, tool=tool, start=clock, end=end)
                )
                scheduler.states[agent_id].ready_at = end
                cursor[agent_id] += 1

            if result.goal_at is None and validator.is_goal_state_satisfied(state):
                result.goal_at = clock
                # A waiter on its final call is not deadlocked when its partner
                # completes the global goal in this same atomic tick. Live
                # execution terminates the episode here, so no later release is
                # required and no agent is invoked again.
                if all(cursor[a] >= len(streams[a]) for a in streams):
                    scheduler.finish_cycle(goal_satisfied=True)
                    waiting_since.clear()
                    break
            elif result.goal_at is not None:
                for agent_id, index in instant:
                    step = steps[index]
                    if step["tool"] in OBSERVATION_TOOL_NAMES:
                        continue
                    result.post_goal.append(
                        f"  step {step['step']} ({step['tool']} by {agent_id}) starts "
                        f"at t={clock:g}, after the goal was reached at "
                        f"t={result.goal_at:g}."
                    )

        if open_clock is not None:
            self._check_co_location(
                state, open_acting, open_clock, seen_conflicts, grace, result
            )

        result.makespan = max((event.end for event in result.events), default=0.0)
        busy = {agent_id: 0.0 for agent_id in streams}
        for event in result.events:
            busy[event.agent] += event.end - event.start
        result.idle = {a: round(result.makespan - b, 3) for a, b in busy.items()}
        return result

    # -- per-step application ---------------------------------------------

    def validate_cycle_preconditions(
        self,
        calls: Sequence[dict[str, Any]],
        state: Any,
    ) -> dict[str, TrajectoryValidationError]:
        """Validate every call against one immutable beginning-of-tick state."""

        errors: dict[str, TrajectoryValidationError] = {}
        for call in calls:
            if call.get("tool") in WAIT_TOOL_NAMES:
                continue
            agent_id = str(call.get("agent"))
            try:
                self._validate_only(call, deepcopy(state))
            except TrajectoryValidationError as exc:
                errors[agent_id] = exc
        return errors

    def _validate_only(self, step: dict[str, Any], state: Any) -> None:
        """Validate one transition without committing any world effects."""

        validator = self.validator
        if step["tool"] in OBSERVATION_TOOL_NAMES:
            # Observation never reaches FsmMirror in the live evaluation either:
            # get_image is served from the renderer and has no symbolic effect.
            # Some specs never offer it, yet post-processing writes it into the
            # recorded trajectories, so gating on allowed_tool_specs here would
            # reject data the executor runs happily.
            return

        if step["tool"] == "communicate":
            if step["tool"] not in validator.allowed_tool_specs:
                raise UnsupportedToolSemanticValidationError(
                    f"Tool {step['tool']} is not allowed for {self.composite_task}.",
                    details={
                        "tool": step["tool"],
                        "composite_task": self.composite_task,
                    },
                )
            validator._validate_communicate_step(step)
            return

        if self.require_initial_communication and step["tool"] not in OBSERVATION_TOOL_NAMES:
            if state.communicated_agents != set(self.agent_ids):
                raise MissingInitialCommunicationSemanticValidationError(
                    "Both agents must coordinate via communication before the "
                    "first task action.",
                    details={"agent": step["agent"], "tool": step["tool"]},
                )

        if step["tool"] not in validator.allowed_tool_specs and not (
            validator._is_allowed_observation_tool(step["tool"])
        ):
            raise UnsupportedToolSemanticValidationError(
                f"Tool {step['tool']} is not allowed for {self.composite_task}.",
                details={"tool": step["tool"], "composite_task": self.composite_task},
            )

        validator._validate_task_local_symbolic_constraints(step)
        validator._validate_generic_transition(step, state)
        validator.validate_task_preconditions(step, state)

    def _commit(self, step: dict[str, Any], state: Any) -> None:
        """Commit one transition whose preconditions already passed."""

        validator = self.validator
        if step["tool"] in OBSERVATION_TOOL_NAMES:
            return
        if step["tool"] == "communicate":
            state.communicated_agents.add(step["agent"])
            validator.apply_task_effects(step, state)
            return
        validator._apply_generic_effects(step, state)
        validator.apply_task_effects(step, state)

    # -- the two contention invariants ------------------------------------

    def _exclusive(self, fixture_id: str | None) -> bool:
        return is_exclusive_fixture(fixture_id, self.validator.initial_state)

    def _named_resources(self, step: dict[str, Any]) -> list[str]:
        """Objects and exclusive fixtures this call reaches for."""

        return list(contention_resources(step, self.validator.initial_state))

    def _check_simultaneous_use(
        self,
        steps: Sequence[dict[str, Any]],
        instant: list[tuple[str, int]],
        clock: float,
        seen: set[tuple[str, ...]],
        result: Replay,
    ) -> None:
        """Invariant 1: two agents must not reach for one thing at one instant."""

        if len(instant) < 2:
            return
        by_resource: dict[str, list[tuple[str, int]]] = {}
        for agent_id, index in instant:
            for resource in self._named_resources(steps[index]):
                by_resource.setdefault(resource, []).append((agent_id, index))
        for resource, users in by_resource.items():
            if len(users) < 2:
                continue
            key = ("use", resource, *sorted(a for a, _ in users))
            if key in seen:
                continue
            seen.add(key)
            detail = ", ".join(
                f"{a} {steps[i]['tool']} (step {steps[i]['step']})" for a, i in users
            )
            result.conflicts.append(
                f"  t={clock:g}: {detail} -- both reach for {resource!r} at the "
                f"same instant. One of them must wait_for_signal on it and the "
                f"other must release it."
            )

        # Shared reads of a stationary plate/tray/support are safe, but moving
        # that support while the partner uses it is not. Catch that asymmetric
        # write/read case without turning two placements onto the support back
        # into a false collision.
        for left_pos, (left_agent, left_index) in enumerate(instant):
            left = steps[left_index]
            for right_agent, right_index in instant[left_pos + 1:]:
                right = steps[right_index]
                ordinary = set(self._named_resources(left)) & set(
                    self._named_resources(right)
                )
                asymmetric = simultaneous_contentions(
                    left, right, self.validator.initial_state
                ) - ordinary
                for resource in asymmetric:
                    key = (
                        "move-while-used",
                        resource,
                        *sorted((left_agent, right_agent)),
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    result.conflicts.append(
                        f"  t={clock:g}: {left_agent} {left['tool']} and "
                        f"{right_agent} {right['tool']} conflict because "
                        f"{resource!r} is being moved while the other call uses "
                        "it as a support, source, receptacle, or reference."
                    )

    def _let_go_of(
        self, step: dict[str, Any], previous: dict[str, Any] | None
    ) -> list[str]:
        """Resources this call actually hands back: a fixture left, a thing put down."""

        args = step.get("args") or {}
        if step["tool"] in GIVE_SPACE_TOOL_NAMES:
            return [str(args.get("fixture_id"))]
        if step["tool"] in RELEASE_TOOL_NAMES:
            objects = set(self.validator.initial_state.get("objects") or {})
            return [
                str(value)
                for value in args.values()
                if isinstance(value, str) and value in objects
            ]
        return []

    def _check_ask(
        self,
        step: dict[str, Any],
        agent_id: str,
        previous: dict[str, Any] | None,
        result: Replay,
    ) -> None:
        """A wait is announced before it is taken.

        The partner cannot see that an agent has stopped -- a blocked agent is
        simply absent from the plan. The ask is what tells it a handover is
        owed, so the wait goes IMMEDIATELY after it: any call in between is the
        agent doing other work while claiming it cannot proceed.
        """

        holder = (step.get("args") or {}).get("from")
        prefix = f"  step {step['step']}: {agent_id} waits on"
        about = str((step.get("args") or {}).get("about"))
        if not self._known_id(about):
            # Coordinating on a fiction. The model names the MOMENT it is
            # waiting for -- "mug_placed", "counter_free" -- instead of the
            # thing, and since a consistent fiction matches itself, a replay
            # that only pairs waits with releases would let it through.
            result.protocol.append(
                f"{prefix} {about!r}, which is not an object or fixture in "
                f"initial_state. `about` names a THING that exists, never an "
                f"event or a state."
            )
            return
        if previous is None or previous["tool"] != "communicate":
            was = "nothing" if previous is None else previous["tool"]
            result.protocol.append(
                f"{prefix} {about!r} but its previous call was {was}, not the "
                f"request. Ask {holder} for it on the tick before you block, "
                f"so the other agent knows a handover is owed."
            )
            return
        if (previous.get("args") or {}).get("to") != holder:
            result.protocol.append(
                f"{prefix} {about!r} on {holder}, but the message right before "
                f"it was addressed to "
                f"{(previous.get('args') or {}).get('to')!r}. Ask the agent you "
                f"are about to wait on."
            )
            return

        message = str((previous.get("args") or {}).get("message") or "")
        # The wait call is private, so its `about` value cannot teach the
        # holder which exact structural release will wake the waiter. Require
        # the visible request to carry that value explicitly. Keeping the
        # shape machine-readable also prevents semantically related but
        # mismatched releases such as about="fruit_plate" followed by
        # releases="dining_counter".
        exact_id = re.compile(
            rf"(?<![A-Za-z0-9_]){re.escape(about)}(?![A-Za-z0-9_])",
            flags=re.IGNORECASE,
        )
        release_language = re.compile(
            r"\breleas(?:e|es|ed|ing)\b",
            flags=re.IGNORECASE,
        )
        if exact_id.search(message) is None or release_language.search(message) is None:
            result.protocol.append(
                f"{prefix} {about!r}, but its request did not tell {holder} "
                f"the exact release value. The message immediately before the "
                f"wait must name {about!r} and identify it as the release "
                f"keyword, for example `When done, release \"{about}\"`."
            )

    # NOTE: an earlier rule here rejected a wait whose holder had already let
    # the resource go before the wait began. Measured against real tick output
    # it was wrong 6 times in 7: the holder vacates on the SAME instant the
    # waiter blocks, which is not a mistake but the tightest correct handover
    # there is -- the release still lands afterwards and still wakes the
    # waiter. The one genuinely-early case was equally harmless. What the rule
    # was reaching for is a wait that cannot be discharged, and that is already
    # caught exactly: by the inert check when the release lands on the wait's
    # own instant, and by the deadlock check when it lands before.

    def _check_release(
        self,
        step: dict[str, Any],
        agent_id: str,
        released: str,
        state: Any,
        previous: dict[str, Any] | None,
        clock: float,
        result: Replay,
    ) -> None:
        """A release is a report of a departure, so the departure must be real.

        `releases` is otherwise a pure speech act: nothing stops an agent
        handing over a fixture it is still standing at. The waiter then wakes
        and walks into an occupied space, and whether that shows up as a
        collision depends entirely on which duration model is running -- see
        `concurrent_fsm_demo.py` scenarios 8 and 9, the same defect with
        opposite verdicts. These three rules need no clock.
        """

        agent_state = state.agents.get(agent_id)
        if agent_state is None:
            return
        prefix = f"  step {step['step']}: {agent_id} releases {released!r}"

        if not self._known_id(released):
            result.protocol.append(
                f"{prefix}, which is not an object or fixture in initial_state. "
                f"`releases` names a THING that exists, never an event or a state."
            )
            return

        # 1. Still holding it.
        if agent_state.held_object == released:
            result.protocol.append(
                f"{prefix} while still holding it. Put it down first."
            )
            return

        # 2. Still standing at it.
        if self._exclusive(released) and agent_state.location == released:
            result.protocol.append(
                f"{prefix} while still standing at it. Call "
                f"give_space(fixture_id={released!r}) first -- announcing a "
                f"handover is not performing one."
            )
            return

        # Once the state proves the object is no longer held or the exclusive
        # fixture is no longer occupied, a later report is valid. Requiring the
        # release to be the immediately following call rejected trajectories
        # that merely kept the waiter blocked a little longer, even though the
        # same schedule executes correctly in live sim.

    def _known_id(self, symbol: str) -> bool:
        """True when this names something that actually exists in the scene."""

        initial = self.validator.initial_state
        return symbol in (initial.get("objects") or {}) or symbol in (
            initial.get("fixtures") or {}
        )

    def _occupancy(self, state: Any) -> dict[str, list[str]]:
        """Which agents are standing at each exclusive fixture right now."""

        occupants: dict[str, list[str]] = {}
        for agent_id, agent_state in state.agents.items():
            location = canonical_agent_workspace(
                self.validator.initial_state, agent_state.location
            )
            if location and self._exclusive(location):
                occupants.setdefault(location, []).append(agent_id)
        return occupants

    def _check_co_location(
        self,
        state: Any,
        acting: Sequence[str],
        clock: float,
        seen: set[tuple[str, ...]],
        grace: set[tuple[str, ...]],
        result: Replay,
    ) -> None:
        """Invariant 2: an exclusive fixture holds at most one agent.

        Occupancy persists between calls -- an agent that navigated to the
        cabinet is still standing there until it leaves. Checking only the
        calls, as every earlier pass did, cannot see two agents that arrived at
        different instants and never left.
        """

        occupancy = self._occupancy(state)
        live = {
            ("at", fixture_id, *sorted(here))
            for fixture_id, here in occupancy.items()
            if len(here) > 1
        }
        grace.intersection_update(live)  # separated at least once -> no excuse left

        for fixture_id, agents_here in occupancy.items():
            if len(agents_here) < 2 or not any(a in acting for a in agents_here):
                continue
            key = ("at", fixture_id, *sorted(agents_here))
            if key in seen or key in grace:
                continue
            seen.add(key)
            result.conflicts.append(
                f"  t={clock:g}: {' and '.join(sorted(agents_here))} are both at "
                f"{fixture_id!r}, which only one agent fits at. The one arriving "
                f"second must wait until the first calls give_space and releases it."
            )

    def _validate_tick_invocations(self, candidate: dict[str, Any]) -> None:
        """Reject tick grids that a live model scheduler cannot reproduce."""

        rows = candidate.get("tick_rows")
        if not isinstance(rows, list):
            rows = candidate.get("ticks")
        if not isinstance(rows, list):
            return

        self._validate_opening_handshake(rows)
        explicit_blocked = candidate.get("format") == "explicit_blocked_v1"

        scheduler = ConcurrentScheduler(self.agent_ids)
        for row_index, row in enumerate(rows):
            if not isinstance(row, dict):
                raise TrajectoryValidationError(
                    f"Tick row {row_index} must be an object."
                )
            tick = row.get("tick", row_index)
            if tick != row_index:
                raise TrajectoryValidationError(
                    f"Tick rows must be consecutive from 0; row {row_index} "
                    f"declares tick {tick!r}.",
                    details={"row": row_index, "tick": tick},
                )
            unknown = sorted(
                key for key in row if key != "tick" and key not in self.agent_ids
            )
            if unknown:
                raise TrajectoryValidationError(
                    f"Tick {tick} contains unknown agent field(s): "
                    + ", ".join(unknown),
                    details={"tick": tick, "unknown_agents": unknown},
                )
            represented = {agent for agent in self.agent_ids if agent in row}
            marker_agents = {
                agent
                for agent in represented
                if row.get(agent) == {"state": "blocked"}
            }
            present = represented - marker_agents
            blocked_agents = set(scheduler.blocked_agents())
            required = set(self.agent_ids) - blocked_agents
            missing = sorted(required - present)
            unexpected = sorted(present & blocked_agents)
            missing_markers = sorted(blocked_agents - marker_agents)
            false_markers = sorted(marker_agents - blocked_agents)
            missing_entries = sorted(set(self.agent_ids) - represented)
            if explicit_blocked and (missing_entries or missing_markers or false_markers):
                descriptions = []
                if missing_entries:
                    descriptions.append("agent field omitted: " + ", ".join(missing_entries))
                if missing_markers:
                    descriptions.append("blocked marker missing: " + ", ".join(missing_markers))
                if false_markers:
                    descriptions.append("unblocked marked blocked: " + ", ".join(false_markers))
                raise TrajectoryValidationError(
                    f"Tick {tick} violates explicit blocked-state format "
                    f"({'; '.join(descriptions)}).",
                    details={"tick": tick, "format": "explicit_blocked_v1"},
                )
            if missing or unexpected:
                descriptions = []
                if missing:
                    descriptions.append("unblocked omitted: " + ", ".join(missing))
                if unexpected:
                    descriptions.append(
                        "blocked invoked before release: " + ", ".join(unexpected)
                    )
                raise TrajectoryValidationError(
                    f"Tick {tick} is not executable by the live scheduler "
                    f"({'; '.join(descriptions)}). Every unblocked agent must "
                    "emit one call, and only a previously blocked agent may be absent.",
                    details={
                        "tick": tick,
                        "missing": missing,
                        "unexpected": unexpected,
                    },
                )

            new_waits: dict[str, tuple[str, str]] = {}
            releases: list[tuple[str, str]] = []
            for agent in present:
                call = row.get(agent) or {}
                key = wait_key({**call, "agent": agent})
                if key is not None:
                    new_waits[agent] = key
                releases.extend(release_keys({**call, "agent": agent}))
            same_tick = sorted(
                waiter
                for waiter, key in new_waits.items()
                if key in releases
            )
            if same_tick:
                wait_details = []
                for waiter in same_tick:
                    source, resource = new_waits[waiter]
                    releaser = next(
                        (
                            agent
                            for agent in present
                            if (source, resource) in release_keys(
                                {**(row.get(agent) or {}), "agent": agent}
                            )
                        ),
                        source,
                    )
                    wait_details.append(
                        {
                            "waiter": waiter,
                            "releaser": releaser,
                            "resource": resource,
                            "resource_requires_handover": (
                                resource in (self.validator.initial_state.get("objects") or {})
                                or self._exclusive(resource)
                            ),
                        }
                    )
                raise TrajectoryValidationError(
                    f"Tick {tick} waits and releases in the same atomic cycle "
                    f"for: {', '.join(same_tick)}. A waiter resumes only after "
                    "a release from a later tick.",
                    details={
                        "tick": tick,
                        "same_tick_wait_release": same_tick,
                        "wait_release_pairs": wait_details,
                    },
                )
            for waiter in new_waits:
                scheduler.block(waiter, row[waiter], clock=float(tick))
            for agent in present:
                call = {**(row.get(agent) or {}), "agent": agent}
                scheduler.deliver(call, clock=float(tick), resume_delay=1.0)

    def _validate_opening_handshake(self, rows: Sequence[dict[str, Any]]) -> None:
        """Require a causal leader proposal followed by follower confirmation."""

        coordinator = getattr(self.validator, "coordinator_id", None)
        if coordinator not in self.agent_ids:
            return  # no coordinator: a communication mode without the planning protocol
        follower = next(a for a in self.agent_ids if a != coordinator)
        # Rendering inserts observation-only rows. They are instrumentation,
        # not authored protocol turns, so locate the first two meaningful
        # cycles instead of assuming they remain physical rows 0 and 1.
        authored_rows = [
            (tick, row)
            for tick, row in enumerate(rows)
            if isinstance(row, dict)
            and any(
                isinstance(row.get(agent), dict)
                and row[agent].get("tool") not in OBSERVATION_TOOL_NAMES
                for agent in self.agent_ids
            )
        ]
        if len(authored_rows) < 2:
            raise TrajectoryValidationError(
                "Coordinator protocol requires two opening communication ticks."
            )
        opening_ticks = (authored_rows[0][0], authored_rows[1][0])
        expected = (
            (opening_ticks[0], coordinator, "propose"),
            (opening_ticks[0], follower, "await_plan"),
            (opening_ticks[1], coordinator, "await_confirmation"),
            (opening_ticks[1], follower, "confirm"),
        )
        for tick, agent, phase in expected:
            call = rows[tick].get(agent) if isinstance(rows[tick], dict) else None
            phase_index = 0 if phase in {"propose", "await_plan"} else 1
            error = opening_protocol_error(
                agent_id=agent,
                call=call or {},
                agent_ids=self.agent_ids,
                coordinator_id=coordinator,
                phase=phase_index,
                observation_tools=OBSERVATION_TOOL_NAMES,
            )
            if error is not None:
                raise TrajectoryValidationError(
                    f"Tick {tick}: {error}",
                    details={"tick": tick, "agent": agent, "expected_phase": phase},
                )

        premature = re.compile(
            r"\b(?:(?:the\s+)?task\s+(?:is\s+)?(?:now\s+)?"
            r"(?:complete|completed|done|finished)|we(?:'re|\s+are)\s+"
            r"(?:now\s+)?(?:done|finished))\b",
            re.IGNORECASE,
        )
        for tick, row in enumerate(rows):
            for agent in self.agent_ids:
                call = row.get(agent) if isinstance(row, dict) else None
                if (call or {}).get("tool") != "communicate":
                    continue
                message = str(((call or {}).get("args") or {}).get("message") or "")
                phase = ((call or {}).get("args") or {}).get("coordination_phase")
                scoped_portion = re.search(
                    r"\b(?:my|our)\s+(?:assigned\s+)?(?:portion|part|work)\b",
                    message,
                    re.IGNORECASE,
                )
                if premature.search(message) and not (
                    phase == "portion_complete" and scoped_portion
                ):
                    raise TrajectoryValidationError(
                        f"Tick {tick} {agent} makes a global completion claim. "
                        "Only the FSM terminates the global task; communicate "
                        "only that the agent's own assigned portion is finished.",
                        details={"tick": tick, "agent": agent, "message": message},
                    )
        self._validate_social_only_tails(rows)

    def _validate_social_only_tails(
        self, rows: Sequence[dict[str, Any]]
    ) -> None:
        """Turn a finished agent's filler tail into one status then a real wait."""

        inert = OBSERVATION_TOOL_NAMES | SOCIAL_TOOL_NAMES | WAIT_TOOL_NAMES
        active_ticks: dict[str, list[int]] = {a: [] for a in self.agent_ids}
        for tick, row in enumerate(rows):
            for agent in self.agent_ids:
                call = row.get(agent) if isinstance(row, dict) else None
                # An explicit {"state": "blocked"} invocation-grid marker is
                # not an action. Treating its missing tool (None) as a
                # non-inert tool made generation and persisted-record replay
                # disagree after canonicalization removed those markers.
                if (
                    isinstance(call, dict)
                    and call.get("tool")
                    and call.get("tool") not in inert
                ):
                    active_ticks[agent].append(tick)
        for agent in self.agent_ids:
            partner = next(a for a in self.agent_ids if a != agent)
            if not active_ticks[partner]:
                continue
            # If the accepted opening plan assigns this agent no physical
            # work, it has no "portion" to announce as complete.  Its correct
            # first post-handshake action is to block on the partner's real
            # outstanding resource.  Requiring a portion_complete message in
            # this case rejected exactly the efficient one-agent plans that
            # the canonical prompt permits.
            if not active_ticks[agent]:
                tail = [
                    (tick, rows[tick][agent])
                    for tick in range(2, len(rows))
                    if agent in rows[tick]
                    and isinstance(rows[tick].get(agent), dict)
                    and rows[tick][agent].get("tool")
                ]
                if tail and tail[0][1].get("tool") == _WAIT_TOOL:
                    continue
            own_last = max(active_ticks[agent], default=1)
            partner_last = max(active_ticks[partner])
            if own_last >= partner_last:
                continue
            tail_ticks = [
                tick for tick in range(max(2, own_last + 1), len(rows))
                if agent in rows[tick]
                and isinstance(rows[tick].get(agent), dict)
                and rows[tick][agent].get("tool")
            ]
            status_tick = next(
                (
                    tick
                    for tick in tail_ticks
                    if rows[tick][agent].get("tool") == "communicate"
                    and (rows[tick][agent].get("args") or {}).get(
                        "coordination_phase"
                    ) == "portion_complete"
                ),
                None,
            )
            if status_tick is None:
                if not tail_ticks:
                    continue  # invocation-grid validation reports the omission
                first_tick = tail_ticks[0]
                raise TrajectoryValidationError(
                    f"After finishing its physical work at tick {own_last}, "
                    f"{agent} must send one portion_complete message rather "
                    "than fill the remaining episode with status calls.",
                    details={
                        "agent": agent,
                        "partner": partner,
                        "own_last_physical_tick": own_last,
                        "tick": first_tick,
                    },
                )
            # Before portion_complete, permit only necessary release messages.
            for tick in tail_ticks:
                if tick >= status_tick:
                    break
                call = rows[tick][agent]
                args = call.get("args") or {}
                if call.get("tool") != "communicate" or not args.get("releases"):
                    raise TrajectoryValidationError(
                        f"After finishing its physical work at tick {own_last}, "
                        f"{agent} must send one portion_complete message rather "
                        "than fill the remaining episode with status calls.",
                        details={
                            "agent": agent,
                            "partner": partner,
                            "own_last_physical_tick": own_last,
                            "tick": tick,
                        },
                    )
            call = rows[status_tick][agent]
            args = call.get("args") or {}
            if partner_last <= status_tick:
                continue  # partner completes the global goal in this atomic tick
            wait_tick = status_tick + 1
            wait_call = rows[wait_tick].get(agent) if wait_tick < len(rows) else None
            if not isinstance(wait_call, dict) or wait_call.get("tool") != _WAIT_TOOL:
                raise TrajectoryValidationError(
                    f"{agent}'s portion_complete message at tick {status_tick} "
                    "must be followed on the next tick by wait_for_signal on "
                    "the partner's outstanding real object or fixture.",
                    details={
                        "agent": agent,
                        "partner": partner,
                        "own_last_physical_tick": own_last,
                        "tick": status_tick,
                    },
                )

    # -- drop-in validation ------------------------------------------------

    def canonicalize(self, candidate: dict[str, Any]) -> dict[str, Any]:
        """Return the canonical tick plan plus its derived flat step stream.

        Tick rows, not flattened steps, are the source of truth for live-policy
        certification. Flattening is deliberately owned here so no caller can
        erase an omitted-vs-blocked distinction before the FSM sees it.
        """

        rows = candidate.get("ticks")
        if not isinstance(rows, list):
            rows = candidate.get("tick_rows")
        if not isinstance(rows, list):
            raise TrajectoryValidationError(
                "Concurrent live-policy validation requires canonical ticks; "
                "flat steps alone cannot represent whether an absent agent was blocked."
            )
        from robotalk.tasks.shared.scheduling import (
            observation_transparent_tick_rows,
        )

        canonical = dict(candidate)
        # get_image is serialized as a supervised model call, but live sim
        # serves it inside the current scheduling cycle.  Project it out before
        # judging invocation, blocking, release, and completion semantics.
        rows = observation_transparent_tick_rows(
            rows,
            observation_tools=OBSERVATION_TOOL_NAMES,
        )
        canonical["tick_rows"] = deepcopy(rows)
        canonical.pop("ticks", None)
        self._validate_tick_invocations(canonical)

        clean_rows = []
        for row in rows:
            clean_row = {
                key: deepcopy(value)
                for key, value in row.items()
                if value != {"state": "blocked"}
            }
            clean_rows.append(clean_row)
        canonical["tick_rows"] = clean_rows
        canonical.pop("format", None)

        steps: list[dict[str, Any]] = []
        for row in clean_rows:
            if not isinstance(row, dict):
                continue
            for agent_id in self.agent_ids:
                action = row.get(agent_id)
                if not isinstance(action, dict) or not action.get("tool"):
                    continue
                step = deepcopy(action)
                step["agent"] = agent_id
                step["step"] = len(steps)
                steps.append(step)
        canonical["steps"] = steps
        return canonical

    def validate(
        self,
        candidate: dict[str, Any],
        *,
        models: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Validates under the configured duration models; FSM result shape.

        Where more than one regime is checked, all of them have to agree: a
        plan that survives only one survived by luck. Tick generation
        deliberately configures LOCK_STEP alone -- see `__init__`. The reported
        final state comes from LOCK_STEP, the regime with exactly one schedule.
        """

        candidate = self.canonicalize(candidate)
        return self._validate_canonical(candidate, models=models)

    def _validate_canonical(
        self,
        candidate: dict[str, Any],
        *,
        models: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        """Validate an already canonicalized plan."""

        models = tuple(models) if models is not None else self.models
        replays = {model: self.replay(candidate, model=model) for model in models}
        primary = replays.get(LOCK_STEP) or next(iter(replays.values()))

        for model in models:
            run = replays[model]
            if run.step_error is not None:
                raise run.step_error
            if run.blocked:
                raise DeadlockSemanticValidationError(
                    f"Under the {model} schedule the trajectory deadlocks:\n"
                    + "\n".join(run.conflicts),
                    details={"model": model, "blocked": run.blocked},
                )
            if run.conflicts:
                raise ResourceConflictSemanticValidationError(
                    f"Under the {model} schedule the agents collide:\n"
                    + "\n".join(run.conflicts),
                    details={"model": model, "conflicts": run.conflicts},
                )
            if run.protocol:
                raise WaitSignalSemanticValidationError(
                    "The handover protocol is not followed:\n"
                    + "\n".join(run.protocol),
                    details={"model": model, "protocol": run.protocol},
                )
            if run.post_goal:
                raise PostGoalActionSemanticValidationError(
                    f"Work is started after the {self.composite_task} goal is "
                    f"already satisfied ({model} schedule):\n"
                    + "\n".join(run.post_goal),
                    details={"model": model, "composite_task": self.composite_task},
                )
            if not run.first_action:
                raise MissingTaskActionSemanticValidationError(
                    "Trajectory did not contain any task action steps."
                )
            if run.goal_at is None:
                raise UnsatisfiedGoalSemanticValidationError(
                    f"Trajectory never satisfied the {self.composite_task} goal "
                    f"state under the {model} schedule.",
                    details={
                        "model": model,
                        **self.validator.goal_state_diagnostics(run.runtime_state),
                    },
                )

        validator = self.validator
        agents = validator._normalize_agents(candidate.get("agents"))
        steps = validator._normalize_steps(candidate.get("steps"))
        return {
            "is_valid": True,
            "validator_contract_version": VALIDATOR_CONTRACT_VERSION,
            "live_policy_certified": True,
            "checks": list(validator._all_checks) + [
                "concurrent_replay",
                "canonical_tick_invocations",
            ],
            "final_state": validator._build_final_state(primary.runtime_state),
            "normalized_candidate": {
                "agents": agents,
                "steps": steps,
                **(
                    {"tick_rows": deepcopy(candidate["tick_rows"])}
                    if isinstance(candidate.get("tick_rows"), list)
                    else {}
                ),
            },
            "signature": validator.trajectory_signature(
                {"agents": agents, "steps": steps}
            ),
            "schedule": {
                model: {
                    "makespan": round(run.makespan, 3),
                    "idle": run.idle,
                    "goal_at": run.goal_at,
                }
                for model, run in replays.items()
            },
        }

    # -- inspection --------------------------------------------------------

    def render(
        self,
        candidate: dict[str, Any],
        *,
        model: str = LOCK_STEP,
        width: int = 44,
    ) -> str:
        """A two-column view of the run: one row per instant, one column per agent."""

        run = self.replay(candidate, model=model, stop_on_step_error=False)
        steps = self.validator._normalize_steps(candidate.get("steps"))
        rows: dict[float, dict[str, str]] = {}
        for event in run.events:
            args = steps[event.index].get("args") or {}
            detail = ", ".join(
                f"{k}={v}" for k, v in args.items() if not isinstance(v, (dict, list))
            )
            rows.setdefault(event.start, {})[event.agent] = f"{event.tool}({detail})"

        header = f"{'t':>6}  " + "  ".join(a.ljust(width) for a in self.agent_ids)
        lines = [header, "-" * len(header)]
        for at in sorted(rows):
            cells = [rows[at].get(a, "")[:width].ljust(width) for a in self.agent_ids]
            lines.append(f"{at:>6g}  " + "  ".join(cells))
        return "\n".join(lines)
