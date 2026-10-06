"""Lock-step scheduling shared by the validator and the wait inserter.

Both need the same answer to "when does this step actually run", and when they
each had their own answer they drifted: the inserter reasoned about positions
in the file while the executor advanced two agents independently. Everything
downstream of that gap -- collisions, inert waits, deadlocks -- came from one
of them being wrong. There is one implementation here so that cannot recur.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import re
from typing import Any, Iterable, Mapping, MutableMapping, Sequence


OPENING_PHASES = (
    ("propose", "await_plan"),
    ("await_confirmation", "confirm"),
)


def observation_transparent_tick_rows(
    rows: Sequence[dict[str, Any]],
    *,
    observation_tools: Iterable[str] = ("get_image",),
) -> list[dict[str, Any]]:
    """Return the physical/coordination grid with observations made invisible.

    Live execution serves an agent-local ``get_image`` inside the current
    scheduling cycle: it changes that agent's context, but it cannot advance
    waits, releases, physical ordering, or the shared clock.  Image injection
    serializes those calls as extra rows so they remain SFT targets.  This
    projection recovers the canonical grid that the concurrent FSM must judge.
    """

    observations = set(observation_tools)
    projected: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        clean: dict[str, Any] = {}
        for key, value in row.items():
            if key == "tick":
                continue
            if isinstance(value, dict) and value.get("tool") in observations:
                continue
            clean[key] = deepcopy(value)
        if not clean:
            continue
        projected.append({"tick": len(projected), **clean})
    return projected


def symbolic_id_mentioned(text: str, symbol: str) -> bool:
    """Match an ID in communication text with underscores spoken as spaces.

    Tool arguments remain exact. This only lets a natural-language plan say
    ``glass cup`` for ``glass_cup`` without pretending the object was omitted.
    """

    pieces = [re.escape(piece) for piece in str(symbol).split("_") if piece]
    if not pieces:
        return False
    pattern = r"[_\s-]+".join(pieces)
    return re.search(
        rf"(?<![A-Za-z0-9_]){pattern}(?![A-Za-z0-9_])",
        str(text),
        re.IGNORECASE,
    ) is not None


def proposal_grounding_ids(
    assignment: Mapping[str, Sequence[str]],
    known_ids: Iterable[str],
) -> dict[str, tuple[str, ...]]:
    """Return ownership-distinguishing IDs required in the opening plan."""

    known = tuple(sorted(set(known_ids), key=lambda value: (-len(value), value)))
    required: dict[str, list[str]] = {}
    for agent_id, items in assignment.items():
        agent_required: list[str] = []
        manipulated: list[str] = []
        for item in items or ():
            mentions = [
                (match.start(), symbol)
                for symbol in known
                for match in [re.search(
                    rf"(?<![A-Za-z0-9_]){re.escape(symbol)}(?![A-Za-z0-9_])",
                    item,
                )]
                if match is not None
            ]
            if mentions:
                symbol = min(mentions)[1]
                if symbol not in agent_required:
                    agent_required.append(symbol)
                if re.match(r"^(?:pick_up_object|place_[A-Za-z0-9_]+)\b", item):
                    if symbol not in manipulated:
                        manipulated.append(symbol)
        # Manipulated objects distinguish ownership better than prerequisite
        # fixtures such as an opened cabinet. Keep fixture/control anchors only
        # for assignments with no manipulated object at all.
        required[str(agent_id)] = manipulated or agent_required
    return {agent: tuple(values) for agent, values in required.items()}


def opening_protocol_error(
    *,
    agent_id: str,
    call: Mapping[str, Any],
    agent_ids: Sequence[str],
    coordinator_id: str | None,
    phase: int,
    observation_tools: Iterable[str] = ("get_image",),
) -> str | None:
    """Shared offline/live verdict for one opening-protocol call."""

    ordered = tuple(agent_ids)
    if coordinator_id not in ordered or phase >= len(OPENING_PHASES):
        return None
    if call.get("tool") in set(observation_tools):
        return None
    role_index = 0 if agent_id == coordinator_id else 1
    expected = OPENING_PHASES[phase][role_index]
    actual = (call.get("args") or {}).get("coordination_phase")
    if call.get("tool") == "communicate" and actual == expected:
        return None
    return (
        f"opening protocol phase {phase} requires {agent_id} to call communicate "
        f"with coordination_phase={expected!r}; physical work starts only after "
        "both confirmation calls commit"
    )


def wait_key(call: dict[str, Any]) -> tuple[str, str] | None:
    """Return the exact sender/resource pair declared by a wait call."""

    if call.get("tool") != "wait_for_signal":
        return None
    args = call.get("args") or {}
    return str(args.get("from")), str(args.get("about"))


def release_keys(call: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    """Return release events emitted by one communication call."""

    if call.get("tool") != "communicate":
        return ()
    args = call.get("args") or {}
    values = args.get("releases") or []
    values = [values] if isinstance(values, str) else list(values)
    sender = str(call.get("agent"))
    return tuple((sender, str(value)) for value in values)


def message_releases_wait(
    call: dict[str, Any],
    waiting_for: dict[str, Any] | tuple[str, str] | None,
    *,
    lenient: bool = False,
) -> bool:
    """Shared FSM/live rule for whether a message discharges one wait."""

    if call.get("tool") != "communicate" or waiting_for is None:
        return False
    if lenient:
        return True
    if isinstance(waiting_for, tuple):
        expected = waiting_for
    else:
        expected = (
            str(waiting_for.get("from")), str(waiting_for.get("about"))
        )
    return expected in release_keys(call)


@dataclass
class SchedulerAgentState:
    """Simulator-independent scheduling state for one logical agent."""

    ready_at: float = 0.0
    waiting_for: dict[str, Any] | None = None


class ConcurrentScheduler:
    """Shared eligibility, blocking, release, and cycle-boundary semantics.

    Call generation and world transitions deliberately remain outside this
    class. The offline FSM supplies recorded calls and symbolic transitions;
    live evaluation supplies model calls and simulator transitions.
    """

    def __init__(
        self,
        agent_ids: Sequence[str],
        *,
        states: MutableMapping[str, Any] | None = None,
    ) -> None:
        self.agent_ids = tuple(agent_ids)
        self.states: MutableMapping[str, Any] = states or {
            agent_id: SchedulerAgentState() for agent_id in self.agent_ids
        }
        self.terminal = False

    def blocked(self, agent_id: str) -> bool:
        return self.states[agent_id].waiting_for is not None

    def blocked_agents(self) -> list[str]:
        return sorted(a for a in self.agent_ids if self.blocked(a))

    def block(self, agent_id: str, call: dict[str, Any], *, clock: float) -> None:
        key = wait_key({**call, "agent": agent_id})
        if key is None:
            raise ValueError("ConcurrentScheduler.block requires wait_for_signal")
        holder, about = key
        self.states[agent_id].waiting_for = {
            "from": holder,
            "about": about,
            "declared_at": clock,
        }

    def deliver(
        self,
        call: dict[str, Any],
        *,
        clock: float,
        resume_delay: float,
        lenient: bool = False,
    ) -> list[str]:
        """Deliver one message and return agents woken for a later instant."""

        woken: list[str] = []
        recipient = (call.get("args") or {}).get("to")
        for agent_id in self.agent_ids:
            if recipient is not None and agent_id != recipient:
                continue
            state = self.states[agent_id]
            if not message_releases_wait(
                call, state.waiting_for, lenient=lenient
            ):
                continue
            state.waiting_for = None
            state.ready_at = max(state.ready_at, clock + resume_delay)
            woken.append(agent_id)
        return woken

    def ready_agents(
        self,
        *,
        clock: float,
        has_work: Mapping[str, bool] | None = None,
    ) -> list[str]:
        """Agents eligible to be invoked at this instant."""

        return sorted(
            agent_id
            for agent_id in self.agent_ids
            if (has_work is None or has_work.get(agent_id, False))
            and not self.blocked(agent_id)
            and self.states[agent_id].ready_at <= clock
            and self.states[agent_id].ready_at != float("inf")
        )

    def next_ready_time(
        self, *, has_work: Mapping[str, bool] | None = None
    ) -> float | None:
        times = [
            self.states[a].ready_at
            for a in self.agent_ids
            if (has_work is None or has_work.get(a, False))
            and not self.blocked(a)
            and self.states[a].ready_at != float("inf")
        ]
        return min(times) if times else None

    def finish_cycle(self, *, goal_satisfied: bool) -> bool:
        """Atomically end a cycle without pretending termination is a release."""

        if goal_satisfied:
            self.terminal = True
        return self.terminal


def schedule(
    steps: Sequence[dict[str, Any]],
    *,
    permanent_releases: bool = False,
) -> tuple[list[int | None], bool]:
    """Assign each step the tick it runs at under lock-step.

    Every call costs one tick, so at each tick every agent not blocked on an
    undischarged wait acts -- all at once.

    A release is an EVENT, not a standing fact: it wakes only an agent already
    waiting when it fires. That matches the executor, which wakes a waiter by
    delivering a message -- one sent before the agent began waiting was never
    delivered to it and never will be. Treating releases as permanent made the
    schedule more permissive than the executor, so a plan could look clean here
    and hang there.

    Returns (ticks, deadlocked); a step never reached has tick None.
    """

    order: dict[str, list[int]] = {}
    for index, step in enumerate(steps):
        order.setdefault(step.get("agent"), []).append(index)

    pointer = {agent: 0 for agent in order}
    ticks: list[int | None] = [None] * len(steps)
    fired: dict[tuple[str, str], list[int]] = {}
    waiting_since: dict[str, int] = {}
    tick = 0

    while True:
        runnable: list[tuple[str, int]] = []
        for agent, indices in order.items():
            if pointer[agent] >= len(indices):
                continue
            index = indices[pointer[agent]]
            step = steps[index]
            if step.get("tool") == "wait_for_signal":
                args = step.get("args") or {}
                key = (args.get("from"), str(args.get("about")))
                since = waiting_since.setdefault(agent, tick)
                events = fired.get(key, ())
                if permanent_releases:
                    if not events:
                        continue
                elif not any(at >= since for at in events):
                    continue
                waiting_since.pop(agent, None)
            runnable.append((agent, index))

        if not runnable:
            break

        for agent, index in runnable:
            ticks[index] = tick
            step = steps[index]
            if step.get("tool") == "communicate":
                value = (step.get("args") or {}).get("releases") or []
                for released_id in ([value] if isinstance(value, str) else value):
                    fired.setdefault((agent, str(released_id)), []).append(tick)
            pointer[agent] += 1
        tick += 1

    return ticks, any(pointer[a] < len(order[a]) for a in order)


def usage_spans(
    steps: Sequence[dict[str, Any]],
    ticks: Sequence[int | None],
    resources_of,
) -> dict[str, dict[str, tuple[int, int]]]:
    """First and last tick at which each agent occupies each resource."""

    spans: dict[str, dict[str, list[int]]] = {}
    for index, step in enumerate(steps):
        if ticks[index] is None:
            continue
        agent = step.get("agent")
        for resource in resources_of(step):
            spans.setdefault(resource, {}).setdefault(agent, []).append(ticks[index])
    return {
        resource: {agent: (min(at), max(at)) for agent, at in by_agent.items()}
        for resource, by_agent in spans.items()
    }


def overlapping_users(
    spans: dict[str, dict[str, tuple[int, int]]],
) -> Iterable[tuple[str, str, str]]:
    """Yield (resource, first_user, later_user) where the tick spans overlap.

    Sequential use is not contention. The lock-step schedule is a function of
    the plan, so an agent that finishes with a bowl at tick 6 has finished
    before another that reaches it at tick 15, whether or not anything is
    said. Only an actual overlap needs ordering.
    """

    for resource, by_agent in spans.items():
        if len(by_agent) < 2:
            continue
        ordered = sorted(by_agent.items(), key=lambda kv: kv[1])
        for position, (agent, (start, end)) in enumerate(ordered):
            for other, (other_start, other_end) in ordered[position + 1 :]:
                if start <= other_end and other_start <= end:
                    yield resource, agent, other
