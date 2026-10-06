#!/usr/bin/env python
"""Phase 2: insert the sharing protocol into plans generated without it.

Phase 1 writes an ordinary sequential plan -- which the model does reliably --
and this computes where the two agents contend, with the whole plan in hand.
That is the part the model could not do while writing: four ordered obligations
spanning both agents and often ten steps. Four prompt-side approaches were tried
and all landed at 0-18%.

Contention is exactly two things:
  * the same OBJECT used by both agents;
  * the same EXCLUSIVE fixture (one opening, or too small for two robots).
A roomy counter shared by two agents on different objects is not contention.

For each one, three symbolically inert steps go in immediately before the
dependent action, so nothing downstream shifts in meaning:

    communicate(B -> A, "... <X> ...")     announcement: under partial
                                           observability the wait is invisible
                                           to A, so without this nobody knows
                                           to release it
    wait_for_signal(from=A, about=X)       B blocks
    communicate(A -> B, "... <X> ...")     release; A has already finished with
                                           X earlier in the recorded order

Nothing with world-state effects is inserted. `give_space` is never added: for a
fixture the releasing act is simply that A moved on, which the plan already has.

Deadlock is checked, not assumed: inserted waits can form a cycle (B waits on A
while A waits on B), which the FSM cannot see and which hangs live-sim. The
check replays the per-agent streams the way the scheduler does and fails loudly
if both agents end up blocked.

  python insert_waits.py --root <dir>           # report
  python insert_waits.py --root <dir> --out <dir> --apply
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
from collections import Counter

from robotalk.tasks.shared.constants import (
    DEPENDENCY_ARG_NAMES,
    EXCLUSIVE_FIXTURE_TYPES,
    FIXTURE_ARG_NAMES,
    GIVE_SPACE_TOOL_NAMES,
    NAVIGATION_TOOL_NAMES,
    OBSERVATION_TOOL_NAMES,
    SOCIAL_TOOL_NAMES,
)

# Varied phrasing on purpose. One fixed string would let a model resume on the
# template rather than on meaning, which is the degenerate policy this whole
# mechanism exists to measure.
ASKS = [
    "I need {x} next, so I am holding off until you are finished with it.",
    "Tell me when you are done with {x} and I will take it from there.",
    "I am waiting on {x} before I carry on with my part.",
    "Let me know once {x} is free; I cannot continue without it.",
]
FREES = [
    "I am finished with {x} now, it is all yours.",
    "Done with {x} -- go ahead whenever you are ready.",
    "{x} is free now, I have moved on to my next step.",
    "You can take {x}, I have finished what I needed there.",
]


def _contested(step, fixtures, objects):
    """Ids this step contends for: objects anywhere, exclusive fixtures only."""

    if step.get("tool") in SOCIAL_TOOL_NAMES or step.get("tool") in OBSERVATION_TOOL_NAMES:
        return []
    # Yielding a fixture is the opposite of claiming it. Counting give_space
    # here made the agent that stepped ASIDE the holder, so the next agent to
    # use the fixture was told to wait for someone who had already left --
    # a wait whose release had therefore always already fired.
    if step.get("tool") in GIVE_SPACE_TOOL_NAMES:
        return []
    args = step.get("args") or {}
    out = []
    for name in DEPENDENCY_ARG_NAMES:
        value = args.get(name)
        if isinstance(value, str) and value in objects:
            out.append(value)
    for name in FIXTURE_ARG_NAMES:
        value = args.get(name)
        if not isinstance(value, str) or value not in fixtures:
            continue
        kind = str((fixtures.get(value) or {}).get("fixture_type", "")).lower()
        if kind in EXCLUSIVE_FIXTURE_TYPES:
            out.append(value)
    return out


from robotalk.tasks.shared.scheduling import (  # noqa: E402
    schedule,
)


def _is_exclusive_fixture(resource, fixtures):
    kind = str((fixtures.get(resource) or {}).get("fixture_type", "")).lower()
    return kind in EXCLUSIVE_FIXTURE_TYPES


def _vacates(step, fixture_id):
    """True if this step takes the acting agent off `fixture_id`'s floor space."""

    args = step.get("args") or {}
    if step.get("tool") in GIVE_SPACE_TOOL_NAMES:
        return str(args.get("fixture_id")) == fixture_id
    if step.get("tool") in NAVIGATION_TOOL_NAMES:
        return str(args.get("fixture_id")) != fixture_id
    return False


def _already_waiting(steps, index, actor, resource, holder):
    """True if `actor` is standing on an undischarged wait for `resource`.

    The object-contention pass runs first and may already have guarded this very
    approach; a second wait on top of it would be redundant, and two waits with
    one release apiece read to the FSM as one that nobody answered. Scanning
    back stops at the actor's own last non-social step, since anything before
    that was already discharged.
    """

    for j in range(index - 1, -1, -1):
        step = steps[j]
        if step.get("agent") != actor:
            continue
        args = step.get("args") or {}
        if (step.get("tool") == "wait_for_signal"
                and str(args.get("about")) == resource
                and args.get("from") == holder):
            return True
        if step.get("tool") not in SOCIAL_TOOL_NAMES:
            return False
    return False


def _insert_occupancy_waits(steps, fixtures, locations, rng):
    """Guards exclusive fixtures whose approach a second robot is standing in.

    Distinct from object contention: nobody touches the same thing. An exclusive
    fixture mounted on a roomy one -- a toaster_oven sitting on a counter -- is
    reached from its parent's floor space, so a robot merely standing at that
    counter blocks the approach and MuJoCo refuses the navigation outright. It
    shows up on the first move, before any object is picked up, which is why
    tracking what has been *used* never sees it: the occupant is just standing
    where it started.

    The occupant's departure is read from the STARTING layout, not from where
    plan order has since put it. A plan that plausibly reads "agent_1 steps
    aside, then agent_0 moves in" carries no such guarantee: the two steps
    become ready at the same instant and the scheduler serves whichever agent it
    likes, so agent_0 can navigate before agent_1 has physically moved. Only a
    message orders two agents, so occupancy persists until a wait discharges it.
    """

    occupied: dict[str, set[str]] = {}
    for agent, where in locations.items():
        if isinstance(where, str):
            occupied.setdefault(where, set()).add(agent)

    standing = dict(locations)
    before: dict[int, list[dict]] = {}
    after: dict[int, list[dict]] = {}
    inserted = 0
    for index, step in enumerate(steps):
        actor = step.get("agent")
        # give_space is a departure. Only an arrival can be blocked.
        needs = () if step.get("tool") in GIVE_SPACE_TOOL_NAMES else _contested(
            step, fixtures, set()
        )
        for needed in needs:
            spec = fixtures.get(needed) or {}
            if str(spec.get("fixture_type", "")).lower() not in EXCLUSIVE_FIXTURE_TYPES:
                continue
            # Already at the fixture: the approach happened before anyone could
            # be in the way, and its parent's floor space is now irrelevant.
            if standing.get(actor) == needed:
                continue
            # the approach is blocked at the fixture itself or at its parent
            for blocking in (needed, spec.get("parent_fixture")):
                if not isinstance(blocking, str):
                    continue
                for occupant in sorted(occupied.get(blocking, set()) - {actor}):
                    leaves = next(
                        (j for j in range(len(steps))
                         if steps[j].get("agent") == occupant
                         and _vacates(steps[j], blocking)),
                        None,
                    )
                    if leaves is None:
                        continue  # occupant never moves; a wait would deadlock
                    if _already_waiting(steps, index, actor, needed, occupant):
                        occupied[blocking].discard(occupant)
                        continue
                    # The wait names the fixture being approached, not the
                    # ground it is approached over. A roomy parent gets used
                    # again by both agents later, and the FSM rightly rejects a
                    # release the sender goes on to contradict; what is really
                    # being handed over is the way in to `needed`.
                    before.setdefault(index, []).extend([
                        {"agent": actor, "tool": "communicate", "derived": True,
                         "args": {"to": occupant,
                                  "message": f"I cannot get to {needed} while "
                                             f"you are standing at {blocking}."},
                         "reasoning": f"{occupant} is blocking the approach to "
                                      f"{needed}."},
                        {"agent": actor, "tool": "wait_for_signal", "derived": True,
                         "args": {"from": occupant, "about": needed},
                         "reasoning": f"Waiting for the way to {needed}."},
                    ])
                    release = {
                        "agent": occupant, "tool": "communicate", "derived": True,
                        "args": {"to": actor,
                                 "message": f"I have moved off {blocking}, so "
                                            f"{needed} is clear for you now.",
                                 "releases": needed},
                        "reasoning": f"I am out of the way of {needed}.",
                    }
                    # If the departure is already behind us in plan order the
                    # release is simply true on arrival; otherwise it waits for
                    # the step that actually moves the occupant.
                    if leaves < index:
                        before[index].append(release)
                    else:
                        after.setdefault(leaves, []).append(release)
                    occupied[blocking].discard(occupant)
                    inserted += 1
        # Occupancy has to follow the agents, not just record where they began.
        # `occupied` was seeded from the starting layout and only ever had
        # entries removed, so an agent that walked somewhere new mid-plan was
        # never registered as occupying it -- the other agent then approached
        # an apparently empty fixture and no wait was inserted.
        args = step.get("args") or {}
        if step.get("tool") in NAVIGATION_TOOL_NAMES:
            arriving_at = args.get("fixture_id")
            previous = standing.get(actor)
            if isinstance(previous, str):
                occupied.get(previous, set()).discard(actor)
            if isinstance(arriving_at, str):
                occupied.setdefault(arriving_at, set()).add(actor)
            standing[actor] = arriving_at
        elif step.get("tool") in GIVE_SPACE_TOOL_NAMES:
            previous = standing.get(actor)
            if isinstance(previous, str):
                occupied.get(previous, set()).discard(actor)
            standing[actor] = None

    if not inserted:
        return steps, 0
    out: list[dict] = []
    for index, step in enumerate(steps):
        out.extend(before.get(index, []))
        out.append(step)
        out.extend(after.get(index, []))
    return out, inserted


def footprint(step, fixtures, objects):
    """What a step occupies: contested ids plus the floor space it stands on."""

    tool = step.get("tool")
    if tool in SOCIAL_TOOL_NAMES or tool in OBSERVATION_TOOL_NAMES:
        return set()
    # A departure is not a claim: one agent stepping off while the other steps
    # on is the handoff working.
    if tool in GIVE_SPACE_TOOL_NAMES:
        return set()
    out = set(_contested(step, fixtures, objects))
    args = step.get("args") or {}
    for name in FIXTURE_ARG_NAMES:
        value = args.get(name)
        if not isinstance(value, str):
            continue
        spec = fixtures.get(value) or {}
        if str(spec.get("fixture_type", "")).lower() in EXCLUSIVE_FIXTURE_TYPES:
            out.add(value)
            parent = spec.get("parent_fixture")
            if isinstance(parent, str):
                out.add(parent)
    return out


def tick_collisions(steps, fixtures, objects):
    """Pairs of steps sharing a tick whose footprints intersect."""

    ticks, _ = schedule(steps)
    by_tick: dict[int, list[int]] = {}
    for index, tick in enumerate(ticks):
        if tick is not None:
            by_tick.setdefault(tick, []).append(index)
    found = []
    for tick, indices in sorted(by_tick.items()):
        if len(indices) < 2:
            continue
        prints = {i: footprint(steps[i], fixtures, objects) for i in indices}
        for position, left in enumerate(indices):
            for right in indices[position + 1 :]:
                if steps[left].get("agent") == steps[right].get("agent"):
                    continue
                shared = prints[left] & prints[right]
                if shared:
                    found.append((tick, left, right, sorted(shared)))
    return found


def _insert_lockstep_waits(steps, fixtures, objects, rng, max_rounds=8):
    """Order agents that would otherwise act on the same tick.

    The occupancy pass asks "is anyone standing there when I reach this line",
    which is a question about plan order -- and plan order is not execution
    order. Written adjacency constrains nothing; only a wait and its release
    do. candle_cleanup/traj_000025 alternates at the cabinet perfectly on the
    page while agent_1 arrives seven ticks before agent_0 leaves.

    Inserting a wait shifts every later tick, so this runs to a fixpoint.
    """

    inserted = 0
    for _ in range(max_rounds):
        found = tick_collisions(steps, fixtures, objects)
        if not found:
            break
        _, left, right, shared = found[0]
        first, second = (left, right) if left < right else (right, left)
        holder = steps[first].get("agent")
        waiter = steps[second].get("agent")
        resource = shared[0]
        # KNOWN SOFT SPOT: in 2 of 1560 trajectories the waiter is already
        # waiting on this resource, and what clears the collision is the three
        # extra steps shifting the tick alignment rather than a new ordering
        # constraint. Guarding against the duplicate was tried and brings both
        # collisions back, so the duplicate stays -- two robots in one cabinet
        # is a worse defect than a redundant message. A principled fix would
        # re-place the existing wait instead of adding a second one.
        block = [
            {"agent": waiter, "tool": "communicate", "derived": True,
             "args": {"to": holder, "message": rng.choice(ASKS).format(x=resource)},
             "reasoning": f"I need {resource} and must wait."},
            {"agent": waiter, "tool": "wait_for_signal", "derived": True,
             "args": {"from": holder, "about": resource},
             "reasoning": f"Waiting for {resource}."},
            {"agent": holder, "tool": "communicate", "derived": True,
             "args": {"to": waiter,
                      "message": rng.choice(FREES).format(x=resource),
                      "releases": resource},
             "reasoning": f"I am done with {resource}."},
        ]
        steps = steps[:second] + block + steps[second:]
        inserted += 1
    return steps, inserted


def _resource_spans(steps, ticks, fixtures, objects):
    """Tick span over which each agent occupies each resource.

    KNOWN GAP: this is built from the calls that NAME a resource, which is not
    the same as occupying it -- an agent stands at a cabinet until it leaves,
    whether or not any later call mentions the cabinet. Extending the spans to
    cover the whole stay was tried and did not pay: it removed one collision of
    18 and introduced a deadlock, because the wait it then inserts has no use
    site to anchor the release to. The 15 co-occupancy collisions the concurrent
    validator still reports here need the insertion point derived from the stay
    itself, not from a call.
    """

    spans: dict[str, dict[str, list[int]]] = {}
    where: dict[tuple[str, str], list[int]] = {}
    for index, step in enumerate(steps):
        if ticks[index] is None:
            continue
        agent = step.get("agent")
        for resource in footprint(step, fixtures, objects):
            got = spans.setdefault(resource, {}).setdefault(agent, [])
            got.append(ticks[index])
            where.setdefault((resource, agent), []).append(index)
    return spans, where


def _first_overlap(steps, ticks, fixtures, objects):
    """The first resource two agents occupy over overlapping tick spans.

    Sequential use is safe on its own: the lock-step schedule is fully
    determined by the plan, so an agent that finishes at tick 6 has finished
    before another that arrives at tick 15, with or without a wait. Inserting
    one there produced a release that fired before the waiter existed --
    harmless in the old permissive schedule, a deadlock in the executor.

    An overlap is different: the holder is still using the resource when the
    waiter needs it. That is the case a wait exists for, and placing the
    release after the holder's last use then necessarily puts it after the
    waiter began waiting.
    """

    spans, where = _resource_spans(steps, ticks, fixtures, objects)
    best = None
    for resource, by_agent in spans.items():
        if len(by_agent) < 2:
            continue
        (a, a_ticks), (b, b_ticks) = sorted(by_agent.items())[:2]
        if min(a_ticks) > max(b_ticks) or min(b_ticks) > max(a_ticks):
            continue  # cleanly sequential: no wait needed
        # Whoever starts later is the one who must wait.
        holder, waiter = (a, b) if min(a_ticks) <= min(b_ticks) else (b, a)
        if _ordered_already(steps, waiter, holder, resource):
            continue
        holder_last = where[(resource, holder)][-1]
        waiter_first = where[(resource, waiter)][0]
        # Overlapping USE is not overlapping occupancy. If the holder already
        # walked away before the waiter got there, the handoff has happened and
        # a wait would block on a release that fired before it existed. Spans
        # cannot see this because give_space is deliberately not a use.
        if _departed_before(steps, ticks, holder, resource, ticks[waiter_first]):
            continue
        key = min(ticks[waiter_first], ticks[holder_last])
        if best is None or key < best[0]:
            best = (key, holder, waiter, resource, holder_last, waiter_first)
    return best[1:] if best else None


def _departed_before(steps, ticks, holder, resource, arrival_tick):
    """True when `holder` left `resource` no later than the waiter reached it."""

    if arrival_tick is None:
        return False
    return any(
        step.get("agent") == holder
        and ticks[index] is not None
        and ticks[index] <= arrival_tick
        and _vacates(step, resource)
        for index, step in enumerate(steps)
    )


def _ordered_already(steps, waiter, holder, resource):
    """True when the waiter already has a wait on this holder and resource."""

    return any(
        step.get("tool") == "wait_for_signal"
        and step.get("agent") == waiter
        and (step.get("args") or {}).get("from") == holder
        and str((step.get("args") or {}).get("about")) == resource
        for step in steps
    )


def _insert_temporal_waits(steps, fixtures, objects, rng, max_rounds=16):
    """Order agents only where the schedule shows them genuinely overlapping."""

    inserted = 0
    for _ in range(max_rounds):
        ticks, _ = schedule(steps)
        found = _first_overlap(steps, ticks, fixtures, objects)
        if found is None:
            break
        holder, waiter, resource, holder_last, waiter_first = found
        ask = {"agent": waiter, "tool": "communicate", "derived": True,
               "args": {"to": holder, "message": rng.choice(ASKS).format(x=resource)},
               "reasoning": f"I need {resource} and must wait."}
        wait = {"agent": waiter, "tool": "wait_for_signal", "derived": True,
                "args": {"from": holder, "about": resource},
                "reasoning": f"Waiting for {resource}."}
        release = {"agent": holder, "tool": "communicate", "derived": True,
                   "args": {"to": waiter,
                            "message": rng.choice(FREES).format(x=resource),
                            "releases": resource},
                   "reasoning": f"I am done with {resource}."}
        # A release REPORTS a departure, so the departure has to happen. The
        # holder's last USE is not one: it is still standing at the fixture,
        # and the waiter it wakes walks straight into it. Whether that showed
        # up as a collision used to depend on which duration model was running.
        # Emitting give_space first also pushes the release one tick later than
        # the wait, which is what stops it landing on the same instant and
        # discharging a wait that never blocked.
        vacate = None
        if _is_exclusive_fixture(resource, fixtures) and not _vacates(
            steps[holder_last], resource
        ):
            vacate = {"agent": holder, "tool": "give_space", "derived": True,
                      "args": {"fixture_id": resource},
                      "reasoning": f"Stepping away from {resource} for {waiter}."}
        rebuilt: list[dict] = []
        for index, step in enumerate(steps):
            if index == waiter_first:
                rebuilt.extend([ask, wait])
            rebuilt.append(step)
            if index == holder_last:
                if vacate is not None:
                    rebuilt.append(vacate)
                rebuilt.append(release)
        steps = rebuilt
        inserted += 1
    return steps, inserted


def _guarded(steps, upto, actor, resource, holder, since):
    return any(
        s.get("tool") == "wait_for_signal"
        and s.get("agent") == actor
        and str((s.get("args") or {}).get("about")) == resource
        and (s.get("args") or {}).get("from") == holder
        for s in steps[since + 1 : upto]
    )


def insert(steps, fixtures, objects, rng, locations=None):
    """Returns (new_steps, inserted_count).

    Coordination is decided on the SCHEDULE, not on positions in the file.
    The earlier passes asked "has anyone touched this before I reach this
    line", which inserted a wait wherever two agents shared a resource at all.
    Measured over 1560 trajectories that produced 357 waits that never blocked
    anyone -- and under live-sim message semantics, where a release wakes only
    an agent already waiting, 309 trajectories deadlocked outright because the
    release had fired before the waiter arrived.

    A wait is inserted only where the two agents' tick spans over a resource
    actually overlap. Sequential use needs none: the lock-step schedule is
    determined by the plan, so an agent finishing at tick 6 finishes before one
    arriving at tick 15 whether or not anything is said. `locations` is no
    longer consulted -- occupancy is read off the schedule.
    """

    del locations
    steps = [
        s for s in steps
        if s.get("tool") != "wait_for_signal" and not s.get("derived")
    ]
    steps, inserted = _insert_temporal_waits(steps, fixtures, objects, rng)
    for number, step in enumerate(steps):
        step["step"] = number
    return steps, inserted


def deadlocks(steps) -> bool:
    """Replays per-agent streams like the scheduler; True if both agents stick."""

    queues: dict[str, list[dict]] = {}
    for step in steps:
        queues.setdefault(step["agent"], []).append(step)
    waiting: dict[str, str | None] = {a: None for a in queues}
    delivered: dict[str, set[str]] = {a: set() for a in queues}
    while any(queues.values()):
        moved = False
        for agent, queue in queues.items():
            if not queue:
                continue
            if waiting[agent] is not None:
                if waiting[agent] in delivered[agent]:
                    waiting[agent] = None
                else:
                    continue
            step = queue[0]
            if step["tool"] == "wait_for_signal":
                about = str((step.get("args") or {}).get("about"))
                if about not in delivered[agent]:
                    waiting[agent] = about
                    continue
            queue.pop(0)
            moved = True
            if step["tool"] == "communicate":
                target = (step.get("args") or {}).get("to")
                message = str((step.get("args") or {}).get("message", ""))
                if target in delivered:
                    for token in message.replace(",", " ").split():
                        delivered[target].add(token.strip(".;:"))
        if not moved:
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    paths = sorted(glob.glob(os.path.join(args.root, "*", "trajectories", "*.json")))
    stats = Counter()
    for path in paths:
        payload = json.loads(open(path, encoding="utf-8").read())
        initial = payload.get("initial_state") or {}
        steps = payload.get("steps") or []
        new_steps, inserted = insert(
            steps,
            initial.get("fixtures") or {},
            set(initial.get("objects") or {}),
            rng,
            {a: (s or {}).get("location")
             for a, s in (initial.get("agents") or {}).items()},
        )
        stats["trajectories"] += 1
        stats["inserted"] += inserted
        stats["touched"] += bool(inserted)
        stats["grew"] += len(new_steps) - len(steps)
        if deadlocks(new_steps):
            stats["DEADLOCK"] += 1
            continue
        if args.apply and args.out:
            rel = os.path.relpath(path, args.root)
            dest = os.path.join(args.out, rel)
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            payload["steps"] = new_steps
            with open(dest, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False)

    n = max(stats["trajectories"], 1)
    print(f"trajectories        : {stats['trajectories']}")
    print(f"  needed a wait     : {stats['touched']}")
    print(f"  waits inserted    : {stats['inserted']}  ({stats['inserted']/n:.1f}/traj)")
    print(f"  steps added       : {stats['grew']}")
    print(f"  DEADLOCKED        : {stats['DEADLOCK']}")
    if not args.apply:
        print("\nreport only -- pass --out <dir> --apply to write")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
