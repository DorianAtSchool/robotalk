"""The validator runs the plan on a clock instead of reading it down the page."""

from __future__ import annotations

import unittest


from robotalk.tasks.shared import concurrent_fsm  # noqa: E402
from robotalk.tasks.shared.concurrent_fsm import (  # noqa: E402
    EXECUTOR,
    LOCK_STEP,
    VALIDATOR_CONTRACT_VERSION,
    ConcurrentTaskValidator,
    simultaneous_contentions,
)
from robotalk.tasks.shared.state import (  # noqa: E402
    AgentRuntimeState,
    TaskRuntimeState,
)
from robotalk.tasks.shared.errors import (  # noqa: E402
    TrajectoryValidationError,
)
from robotalk.tasks.shared.scheduling import (  # noqa: E402
    ConcurrentScheduler,
    symbolic_id_mentioned,
)

AGENTS = ("agent_0", "agent_1")
FIXTURES = {
    "counter": {"fixture_type": "counter", "parts": {}, "controls": {}},
    "cab": {"fixture_type": "cabinet", "parts": {}, "controls": {}},
    "sink": {"fixture_type": "sink", "parts": {}, "controls": {}},
}
OBJECTS = {"bowl": {"object_type": "bowl", "location": "counter"}}


def step(agent, tool, **args):
    return {"agent": agent, "tool": tool, "args": args, "reasoning": "r", "step": 0}


def plan(*steps):
    numbered = [dict(s, step=i) for i, s in enumerate(steps)]
    return {"agents": [{"agent": a} for a in AGENTS], "steps": numbered}


def tick_plan(*rows):
    return {
        "agents": [{"agent": a} for a in AGENTS],
        "ticks": [{"tick": i, **row} for i, row in enumerate(rows)],
    }


def action(tool, **args):
    return {"tool": tool, "args": args, "reasoning": "r"}


class FakeValidator:
    """A minimal stand-in exposing the surface ConcurrentTaskValidator uses.

    The real SpecDrivenTaskValidator needs a task spec and a sampled world; the
    scheduling behaviour under test is independent of both, so the seams are
    stubbed and only location/held effects are modelled.
    """

    composite_task = "Fake"
    agent_ids = AGENTS
    allowed_tool_specs = {
        "navigate_to_fixture": {}, "pick_up_object": {}, "give_space": {},
        "place_on_surface": {}, "place_on_object": {}, "communicate": {},
        "wait_for_signal": {},
    }
    _all_checks = ["fake"]

    def __init__(
        self,
        *,
        goal_after: int | None = None,
        coordinator_id: str | None = None,
    ) -> None:
        self.initial_state = {"fixtures": FIXTURES, "objects": OBJECTS,
                              "agents": {a: {"location": None} for a in AGENTS}}
        self._goal_after = goal_after
        self._applied = 0
        self.coordinator_id = coordinator_id

    # seams the concurrent validator calls
    def _normalize_agents(self, value):
        return [{"agent": a} for a in AGENTS]

    def _normalize_steps(self, value):
        return list(value or [])

    def _build_runtime_state(self, agents):
        self._applied = 0
        return TaskRuntimeState(
            agents={a: AgentRuntimeState(location=None, held_object=None)
                    for a in AGENTS},
            objects={}, fixtures={}, machine_state={},
        )

    def _validation_error_with_step(self, exc, step_index):
        return exc

    def _is_allowed_observation_tool(self, tool_name):
        return True

    def _validate_communicate_step(self, step):
        return None

    def _validate_task_local_symbolic_constraints(self, step):
        return None

    def _validate_generic_transition(self, step, state):
        return None

    def validate_task_preconditions(self, step, state):
        return None

    def apply_task_effects(self, step, state):
        return None

    def _apply_generic_effects(self, step, state):
        agent = state.agents[step["agent"]]
        args = step.get("args") or {}
        if step["tool"] == "navigate_to_fixture":
            agent.location = args.get("fixture_id")
        elif step["tool"] == "give_space":
            agent.location = None
        elif step["tool"] == "pick_up_object":
            agent.held_object = args.get("object_id")
        elif step["tool"] == "place_on_surface":
            agent.held_object = None
        self._applied += 1

    def is_goal_state_satisfied(self, state):
        if self._goal_after is None:
            return False
        return self._applied >= self._goal_after

    def _build_final_state(self, state):
        return {}

    def trajectory_signature(self, candidate):
        return "sig"


class AccessStateValidator(FakeValidator):
    """Tiny fixture-state model used to test transactional prerequisites."""

    allowed_tool_specs = {
        **FakeValidator.allowed_tool_specs,
        "open_hinged_part": {},
    }

    def _build_runtime_state(self, agents):
        state = super()._build_runtime_state(agents)
        state.fixtures = {
            "cab": {
                "fixture_type": "cabinet",
                "parts": {"door": {"state": "closed"}},
            }
        }
        return state

    def _validate_generic_transition(self, step, state):
        if (
            step["tool"] == "pick_up_object"
            and (step.get("args") or {}).get("source_id") == "cab"
            and state.fixtures["cab"]["parts"]["door"]["state"] != "open"
        ):
            raise TrajectoryValidationError(
                "cabinet must already be open at the beginning of the tick"
            )

    def _apply_generic_effects(self, step, state):
        if step["tool"] == "open_hinged_part":
            state.fixtures["cab"]["parts"]["door"]["state"] = "open"
            self._applied += 1
            return
        super()._apply_generic_effects(step, state)


def build(**kwargs):
    return ConcurrentTaskValidator(FakeValidator(**kwargs))


OPEN = (step("agent_0", "communicate", to="agent_1", message="a b c d"),
        step("agent_1", "communicate", to="agent_0", message="a b c d"))


class ScheduleTests(unittest.TestCase):
    def test_natural_plan_text_may_spell_symbolic_underscores_as_spaces(self):
        self.assertTrue(symbolic_id_mentioned("I will get the glass cup.", "glass_cup"))
        self.assertTrue(symbolic_id_mentioned("Handle lemon_wedge.", "lemon_wedge"))
        self.assertFalse(symbolic_id_mentioned("I will get the glass.", "glass_cup"))

    def test_goal_termination_does_not_unblock_a_waiter(self):
        scheduler = ConcurrentScheduler(AGENTS)
        scheduler.block(
            "agent_0",
            {"tool": "wait_for_signal", "args": {"from": "agent_1", "about": "bowl"}},
            clock=3.0,
        )
        self.assertTrue(scheduler.finish_cycle(goal_satisfied=True))
        self.assertTrue(scheduler.blocked("agent_0"))

    def test_same_tick_action_cannot_use_partner_created_prerequisite(self):
        validator = ConcurrentTaskValidator(AccessStateValidator())
        state = validator.validator._build_runtime_state(None)
        state.communicated_agents.update(AGENTS)
        calls = [
            step("agent_0", "open_hinged_part", target_id="cab", part_id="door"),
            step("agent_1", "pick_up_object", object_id="bowl", source_id="cab"),
        ]
        errors = validator.validate_cycle_preconditions(calls, state)
        self.assertEqual(set(errors), {"agent_1"})
        self.assertIn("beginning of the tick", str(errors["agent_1"]))
        reversed_errors = validator.validate_cycle_preconditions(
            list(reversed(calls)), state
        )
        self.assertEqual(set(reversed_errors), set(errors))
        self.assertEqual(
            str(reversed_errors["agent_1"]), str(errors["agent_1"])
        )

    def test_partner_created_prerequisite_is_available_next_tick(self):
        validator = ConcurrentTaskValidator(AccessStateValidator())
        state = validator.validator._build_runtime_state(None)
        state.communicated_agents.update(AGENTS)
        opening = step(
            "agent_0", "open_hinged_part", target_id="cab", part_id="door"
        )
        pickup = step(
            "agent_1", "pick_up_object", object_id="bowl", source_id="cab"
        )
        self.assertFalse(validator.validate_cycle_preconditions([opening], state))
        validator._commit(opening, state)
        self.assertFalse(validator.validate_cycle_preconditions([pickup], state))

    @staticmethod
    def coordinator_opening(*tail):
        return tick_plan(
            {
                "agent_0": action(
                    "communicate", to="agent_1",
                    message="I will handle bowl; you handle sink, cab, and counter.",
                    coordination_phase="propose",
                ),
                "agent_1": action(
                    "communicate", to="agent_0", message="Awaiting your plan.",
                    coordination_phase="await_plan",
                ),
            },
            {
                "agent_0": action(
                    "communicate", to="agent_1", message="Please confirm.",
                    coordination_phase="await_confirmation",
                ),
                "agent_1": action(
                    "communicate", to="agent_0", message="Confirmed.",
                    coordination_phase="confirm",
                ),
            },
            *tail,
        )

    def test_causal_coordinator_opening_is_accepted(self):
        candidate = self.coordinator_opening(
            {
                "agent_0": action("pick_up_object", object_id="bowl"),
                "agent_1": action("navigate_to_fixture", fixture_id="sink"),
            }
        )
        canonical = build(coordinator_id="agent_0").canonicalize(candidate)
        self.assertEqual(len(canonical["steps"]), 6)

    def test_rendered_observation_rows_do_not_shift_the_handshake(self):
        candidate = self.coordinator_opening()
        candidate["ticks"].insert(
            0,
            {
                "tick": 0,
                "agent_0": action("get_image", views=["top_view"]),
                "agent_1": action("get_image", views=["top_view"]),
            },
        )
        for tick, row in enumerate(candidate["ticks"]):
            row["tick"] = tick
        canonical = build(coordinator_id="agent_0").canonicalize(candidate)
        self.assertEqual(canonical["steps"][0]["tool"], "communicate")
        self.assertFalse(
            any(step["tool"] == "get_image" for step in canonical["steps"])
        )

    def test_follower_cannot_independently_propose_on_opening_tick(self):
        candidate = self.coordinator_opening()
        candidate["ticks"][0]["agent_1"]["args"]["coordination_phase"] = "propose"
        with self.assertRaisesRegex(Exception, "await_plan"):
            build(coordinator_id="agent_0").canonicalize(candidate)


    def test_global_completion_claim_is_rejected(self):
        candidate = self.coordinator_opening(
            {
                "agent_0": action(
                    "communicate", to="agent_1", message="Task complete!",
                ),
                "agent_1": action("navigate_to_fixture", fixture_id="sink"),
            }
        )
        with self.assertRaisesRegex(Exception, "global completion"):
            build(coordinator_id="agent_0").canonicalize(candidate)

    def test_scoped_portion_complete_is_not_global_completion(self):
        candidate = self.coordinator_opening(
            {
                "agent_0": action(
                    "communicate", to="agent_1",
                    message="My portion of the task is complete.",
                    coordination_phase="portion_complete",
                ),
                "agent_1": action("navigate_to_fixture", fixture_id="sink"),
            }
        )
        build(coordinator_id="agent_0").canonicalize(candidate)

    def test_required_release_may_precede_portion_complete(self):
        rows = [
            {
                "tick": 0,
                "agent_0": action("pick_up_object", object_id="bowl"),
                "agent_1": action("navigate_to_fixture", fixture_id="sink"),
            },
            {
                "tick": 1,
                "agent_0": action(
                    "communicate", to="agent_1", message="Bowl is free.",
                    releases="bowl",
                ),
                "agent_1": action("navigate_to_fixture", fixture_id="cab"),
            },
            {
                "tick": 2,
                "agent_0": action(
                    "communicate", to="agent_1", message="My portion is done.",
                    coordination_phase="portion_complete",
                ),
                "agent_1": action("navigate_to_fixture", fixture_id="counter"),
            },
        ]
        build()._validate_social_only_tails(rows)

    def test_agent_assigned_no_physical_work_waits_without_fake_completion(self):
        rows = [
            {"tick": 0},
            {"tick": 1},
            {
                "tick": 2,
                "agent_0": action(
                    "wait_for_signal", **{"from": "agent_1", "about": "bowl"}
                ),
                "agent_1": action("pick_up_object", object_id="bowl"),
            },
            {
                "tick": 3,
                "agent_1": action("navigate_to_fixture", fixture_id="sink"),
            },
        ]
        build()._validate_social_only_tails(rows)

    def test_blocked_markers_do_not_hide_invalid_social_tail(self):
        explicit_rows = [
            {"tick": 0},
            {"tick": 1},
            {
                "tick": 2,
                "agent_0": action("pick_up_object", object_id="bowl"),
                "agent_1": action("navigate_to_fixture", fixture_id="sink"),
            },
            {
                "tick": 3,
                "agent_0": action("communicate", to="agent_1", message="Status."),
                "agent_1": action("navigate_to_fixture", fixture_id="cab"),
            },
            {
                "tick": 4,
                "agent_0": {"state": "blocked"},
                "agent_1": action("navigate_to_fixture", fixture_id="counter"),
            },
        ]
        canonical_rows = [
            {key: value for key, value in row.items() if value != {"state": "blocked"}}
            for row in explicit_rows
        ]
        for rows in (explicit_rows, canonical_rows):
            with self.assertRaisesRegex(Exception, "portion_complete"):
                build()._validate_social_only_tails(rows)

    def test_explicit_blocked_marker_is_validated_then_removed(self):
        candidate = {
            "format": "explicit_blocked_v1",
            "ticks": [
                {
                    "tick": 0,
                    "agent_0": action(
                        "wait_for_signal", **{"from": "agent_1", "about": "bowl"}
                    ),
                    "agent_1": action("navigate_to_fixture", fixture_id="sink"),
                },
                {
                    "tick": 1,
                    "agent_0": {"state": "blocked"},
                    "agent_1": action(
                        "communicate", to="agent_0", message="Bowl is free.",
                        releases="bowl",
                    ),
                },
                {
                    "tick": 2,
                    "agent_0": action("navigate_to_fixture", fixture_id="counter"),
                    "agent_1": action("navigate_to_fixture", fixture_id="sink"),
                },
            ],
        }
        canonical = build().canonicalize(candidate)
        self.assertNotIn("format", canonical)
        self.assertNotIn("agent_0", canonical["tick_rows"][1])



    def test_finished_agent_reports_once_then_waits(self):
        candidate = self.coordinator_opening(
            {
                "agent_0": action("pick_up_object", object_id="bowl"),
                "agent_1": action("navigate_to_fixture", fixture_id="sink"),
            },
            {
                "agent_0": action(
                    "communicate", to="agent_1", message="My portion is finished.",
                    coordination_phase="portion_complete",
                ),
                "agent_1": action("navigate_to_fixture", fixture_id="cab"),
            },
            {
                "agent_0": action(
                    "wait_for_signal", **{"from": "agent_1", "about": "sink"}
                ),
                "agent_1": action("navigate_to_fixture", fixture_id="counter"),
            },
        )
        build(coordinator_id="agent_0").canonicalize(candidate)

    def test_live_certification_requires_ticks(self):
        with self.assertRaisesRegex(Exception, "requires canonical ticks"):
            build(goal_after=2).validate(plan(*OPEN), models=(LOCK_STEP,))

    def test_canonical_ticks_reject_an_unblocked_omission(self):
        candidate = tick_plan(
            {
                "agent_0": action("communicate", to="agent_1", message="a b c d"),
                "agent_1": action("communicate", to="agent_0", message="a b c d"),
            },
            {"agent_1": action("navigate_to_fixture", fixture_id="sink")},
        )
        with self.assertRaisesRegex(Exception, "unblocked omitted: agent_0"):
            build(goal_after=3).validate(candidate, models=(LOCK_STEP,))

    def test_terminal_wait_needs_no_release_when_same_tick_reaches_goal(self):
        candidate = tick_plan(
            {
                "agent_0": action("communicate", to="agent_1", message="a b c d"),
                "agent_1": action("communicate", to="agent_0", message="a b c d"),
            },
            {
                "agent_0": action(
                    "communicate", to="agent_1",
                    message='When done, send releases="bowl".',
                ),
                "agent_1": action("communicate", to="agent_0", message="a b c d"),
            },
            {
                "agent_0": action(
                    "wait_for_signal", **{"from": "agent_1", "about": "bowl"}
                ),
                "agent_1": action("navigate_to_fixture", fixture_id="sink"),
            },
        )
        validation = build(goal_after=1).validate(candidate, models=(LOCK_STEP,))
        self.assertTrue(validation["is_valid"])
        self.assertEqual(
            validation["validator_contract_version"],
            VALIDATOR_CONTRACT_VERSION,
        )
        self.assertEqual(validation["schedule"][LOCK_STEP]["goal_at"], 2.0)

    def test_terminal_wait_needs_no_release_when_partner_finishes_later(self):
        candidate = tick_plan(
            {
                "agent_0": action("communicate", to="agent_1", message="a b c d"),
                "agent_1": action("communicate", to="agent_0", message="a b c d"),
            },
            {
                "agent_0": action(
                    "communicate", to="agent_1",
                    message='When done, send releases="bowl".',
                ),
                "agent_1": action("communicate", to="agent_0", message="I will finish"),
            },
            {
                "agent_0": action("wait_for_signal", **{"from": "agent_1", "about": "bowl"}),
                "agent_1": action("navigate_to_fixture", fixture_id="sink"),
            },
            {
                "agent_0": {"state": "blocked"},
                "agent_1": action("navigate_to_fixture", fixture_id="counter"),
            },
        )
        candidate["format"] = "explicit_blocked_v1"
        validation = build(goal_after=2).validate(candidate, models=(LOCK_STEP,))
        self.assertTrue(validation["is_valid"])
        self.assertEqual(validation["schedule"][LOCK_STEP]["goal_at"], 3.0)

    def test_both_agents_advance_on_the_same_instant(self):
        run = build().replay(plan(*OPEN), model=LOCK_STEP)
        self.assertEqual([e.start for e in run.events], [0.0, 0.0])

    def test_lock_step_instants_are_integer_ticks(self):
        run = build().replay(
            plan(*OPEN, step("agent_0", "navigate_to_fixture", fixture_id="cab")),
            model=LOCK_STEP,
        )
        self.assertEqual([e.tick for e in run.events], [0, 0, 1])

    def test_executor_durations_desynchronise_the_two_streams(self):
        # navigate costs 4.0 and communicate 0.25, so the streams stop being
        # aligned -- which is the whole reason a tick model is an assumption.
        run = build().replay(
            plan(*OPEN,
                 step("agent_0", "navigate_to_fixture", fixture_id="cab"),
                 step("agent_1", "communicate", to="agent_0", message="a b c d"),
                 step("agent_1", "communicate", to="agent_0", message="e f g h")),
            model=EXECUTOR,
        )
        starts = {(e.agent, e.index): e.start for e in run.events}
        self.assertEqual(starts[("agent_0", 2)], 0.25)
        self.assertEqual(starts[("agent_1", 4)], 0.5)

    def test_position_in_the_file_does_not_order_the_agents(self):
        # agent_1's step is written last but runs first: it is that agent's
        # opening move, while agent_0 is three calls deep.
        run = build().replay(
            plan(*OPEN,
                 step("agent_0", "navigate_to_fixture", fixture_id="cab"),
                 step("agent_0", "give_space", fixture_id="cab"),
                 step("agent_1", "navigate_to_fixture", fixture_id="sink")),
            model=LOCK_STEP,
        )
        by_index = {e.index: e.tick for e in run.events}
        self.assertEqual(by_index[4], 1)
        self.assertEqual(by_index[3], 2)


class WaitTests(unittest.TestCase):
    def _waiting_plan(self, *, release: bool):
        release_args = {"releases": ["cab"]} if release else {}
        return plan(
            *OPEN,
            step("agent_0", "navigate_to_fixture", fixture_id="cab"),
            step("agent_1", "wait_for_signal", **{"from": "agent_0", "about": "cab"}),
            step("agent_1", "navigate_to_fixture", fixture_id="cab"),
            step("agent_0", "give_space", fixture_id="cab"),
            step("agent_0", "communicate", to="agent_1", message="cab is free now",
                 **release_args),
        )

    def test_a_wait_blocks_until_its_release_fires(self):
        # The wait is CALLED once, at the instant the agent reaches it, and the
        # agent then sits blocked -- it does not re-issue the call when woken.
        # Recording the call at its discharge instead drew the agent as idle
        # through the block and then calling wait once the way was already
        # clear, which is not what the executor does or what the plan says.
        run = build().replay(self._waiting_plan(release=True), model=LOCK_STEP)
        by_index = {e.index: e.tick for e in run.events}
        self.assertEqual(by_index[3], 1, "the wait is called at t=1")
        self.assertEqual(by_index[6], 3, "the release is sent at t=3")
        # The waiter resumes on the instant AFTER the release, never on it:
        # same-instant resumption is a delivery race, and costing the wake a
        # tick removes it.
        self.assertEqual(by_index[4], 4, "the waiter resumes after the release")
        self.assertEqual(run.idle["agent_1"], 2.0, "blocked from t=2 to t=4")

    def test_a_woken_agent_never_acts_on_the_release_instant(self):
        for model in (LOCK_STEP, EXECUTOR):
            with self.subTest(model=model):
                run = build().replay(self._waiting_plan(release=True), model=model)
                at = {e.index: e.start for e in run.events}
                self.assertGreater(at[4], at[6])

    def test_a_wait_nothing_releases_is_a_deadlock_not_a_pass(self):
        run = build().replay(self._waiting_plan(release=False), model=LOCK_STEP)
        self.assertTrue(run.deadlocked)
        self.assertEqual(run.blocked, ["agent_1"])

    def test_a_release_sent_before_the_wait_never_discharges_it(self):
        # The executor wakes a waiter by DELIVERING a message. One sent while
        # the agent was still busy was never delivered to it.
        run = build().replay(
            plan(*OPEN,
                 step("agent_0", "communicate", to="agent_1", message="cab is free",
                      releases=["cab"]),
                 step("agent_1", "navigate_to_fixture", fixture_id="sink"),
                 step("agent_1", "give_space", fixture_id="sink"),
                 step("agent_1", "wait_for_signal", **{"from": "agent_0",
                                                       "about": "cab"})),
            model=LOCK_STEP,
        )
        self.assertTrue(run.deadlocked)


class ContentionTests(unittest.TestCase):
    def test_entering_while_the_occupant_navigates_away_is_a_reported_conflict(self):
        # Regression: the occupant leaves by navigating (not give_space); the
        # conflict used to crash with StopIteration instead of being reported.
        validator = build()
        candidate = validator.canonicalize(tick_plan(
            {"agent_0": action("communicate", to="agent_1", message="a b c d"),
             "agent_1": action("communicate", to="agent_0", message="a b c d")},
            {"agent_0": action("navigate_to_fixture", fixture_id="counter"),
             "agent_1": action("navigate_to_fixture", fixture_id="sink")},
            {"agent_0": action("navigate_to_fixture", fixture_id="sink"),
             "agent_1": action("navigate_to_fixture", fixture_id="counter")},
        ))
        run = validator.replay(candidate, model=LOCK_STEP)
        self.assertEqual(len(run.conflicts), 1)
        self.assertIn("agent_1 navigates away", run.conflicts[0])

    def test_cabinet_door_transition_conflicts_with_content_access(self):
        opening = step(
            "agent_0", "open_hinged_part", target_id="cab", part_id="door"
        )
        pickup = step(
            "agent_1", "pick_up_object", object_id="bowl", source_id="cab"
        )
        self.assertEqual(
            simultaneous_contentions(opening, pickup, {"fixtures": FIXTURES, "objects": OBJECTS}),
            frozenset({"cabinet_access:cab"}),
        )

    def test_distinct_objects_may_be_picked_from_open_shared_cabinet(self):
        state = {
            "fixtures": FIXTURES,
            "objects": {
                "bowl_1": {"location": "cab"},
                "bowl_2": {"location": "cab"},
            },
        }
        left = step(
            "agent_0", "pick_up_object", object_id="bowl_1", source_id="cab"
        )
        right = step(
            "agent_1", "pick_up_object", object_id="bowl_2", source_id="cab"
        )
        self.assertEqual(simultaneous_contentions(left, right, state), frozenset())

    def test_distinct_objects_may_share_one_stationary_support(self):
        validator = build()
        validator.validator.initial_state["objects"].update({
            "cube_1": {"location": "counter"},
            "cube_2": {"location": "counter"},
            "plate": {"location": "counter"},
        })
        run = validator.replay(
            plan(
                *OPEN,
                step("agent_0", "place_on_object", object_id="cube_1", support_object_id="plate"),
                step("agent_1", "place_on_object", object_id="cube_2", support_object_id="plate"),
            ),
            model=LOCK_STEP,
        )
        self.assertEqual(run.conflicts, [])

    def test_moving_a_support_while_partner_uses_it_collides(self):
        validator = build()
        validator.validator.initial_state["objects"].update({
            "cube": {"location": "counter"},
            "plate": {"location": "counter"},
            "tray": {"location": "counter"},
        })
        run = validator.replay(
            plan(
                *OPEN,
                step("agent_0", "place_on_object", object_id="plate", support_object_id="tray"),
                step("agent_1", "place_on_object", object_id="cube", support_object_id="plate"),
            ),
            model=LOCK_STEP,
        )
        self.assertTrue(any("being moved" in conflict for conflict in run.conflicts))

    def test_two_agents_reaching_for_one_object_at_one_instant_collide(self):
        run = build().replay(
            plan(*OPEN,
                 step("agent_0", "pick_up_object", object_id="bowl", source_id="counter"),
                 step("agent_1", "pick_up_object", object_id="bowl", source_id="counter")),
            model=LOCK_STEP,
        )
        self.assertEqual(len(run.conflicts), 1)
        self.assertIn("bowl", run.conflicts[0])

    def test_sequential_use_of_one_object_is_not_contention(self):
        run = build().replay(
            plan(*OPEN,
                 step("agent_0", "pick_up_object", object_id="bowl", source_id="counter"),
                 step("agent_1", "navigate_to_fixture", fixture_id="sink"),
                 step("agent_1", "give_space", fixture_id="sink"),
                 step("agent_1", "pick_up_object", object_id="bowl", source_id="counter")),
            model=LOCK_STEP,
        )
        self.assertEqual(run.conflicts, [])

    def test_cabinet_parent_workspace_allows_co_occupancy(self):
        # The blind spot every earlier pass had: nobody ACTS at the same
        # instant, but agent_0 never left, so both are standing there.
        run = build().replay(
            plan(*OPEN,
                 step("agent_0", "navigate_to_fixture", fixture_id="cab"),
                 step("agent_1", "communicate", to="agent_0", message="a b c d"),
                 step("agent_1", "communicate", to="agent_0", message="e f g h"),
                 step("agent_1", "navigate_to_fixture", fixture_id="cab")),
            model=LOCK_STEP,
        )
        self.assertEqual(run.conflicts, [])

    def test_shared_cabinet_does_not_require_give_space_handoff(self):
        run = build().replay(
            plan(*OPEN,
                 step("agent_0", "navigate_to_fixture", fixture_id="cab"),
                 step("agent_0", "give_space", fixture_id="cab"),
                 step("agent_1", "communicate", to="agent_0", message="a b c d"),
                 step("agent_1", "navigate_to_fixture", fixture_id="cab")),
            model=LOCK_STEP,
        )
        self.assertEqual(run.conflicts, [])

    def test_handoff_entry_on_following_tick_is_not_a_collision(self):
        run = build().replay(
            plan(*OPEN,
                 step("agent_0", "navigate_to_fixture", fixture_id="cab"),
                 step("agent_0", "give_space", fixture_id="cab"),
                 step("agent_1", "communicate", to="agent_0", message="a b c d"),
                 step("agent_1", "communicate", to="agent_0", message="e f g h"),
                 step("agent_1", "navigate_to_fixture", fixture_id="cab")),
            model=LOCK_STEP,
        )
        self.assertEqual(run.conflicts, [])

    def test_a_shared_non_exclusive_fixture_is_fine(self):
        run = build().replay(
            plan(*OPEN,
                 step("agent_0", "navigate_to_fixture", fixture_id="counter"),
                 step("agent_1", "navigate_to_fixture", fixture_id="counter")),
            model=LOCK_STEP,
        )
        self.assertEqual(run.conflicts, [])

    def test_agents_starting_co_located_are_not_blamed_on_the_plan(self):
        validator = build()
        state = validator.validator._build_runtime_state(None)
        original = validator.validator._build_runtime_state

        def seeded(agents):
            fresh = original(agents)
            for agent in fresh.agents.values():
                agent.location = "cab"
            return fresh

        validator.validator._build_runtime_state = seeded
        run = validator.replay(
            plan(*OPEN, step("agent_0", "give_space", fixture_id="cab")),
            model=LOCK_STEP,
        )
        self.assertEqual(run.conflicts, [])
        del state

    def test_shared_cabinet_start_needs_no_overlap_excuse(self):
        # alcohol_serving_prep starts both agents at the cabinet. Excusing that
        # is right; excusing a RETURN to it after leaving is not, and a blanket
        # suppression hid exactly that case.
        validator = build()
        original = validator.validator._build_runtime_state

        def seeded(agents):
            fresh = original(agents)
            for agent in fresh.agents.values():
                agent.location = "cab"
            return fresh

        validator.validator._build_runtime_state = seeded
        run = validator.replay(
            plan(*OPEN,
                 step("agent_1", "give_space", fixture_id="cab"),
                 step("agent_1", "communicate", to="agent_0", message="a b c d"),
                 step("agent_1", "navigate_to_fixture", fixture_id="cab")),
            model=LOCK_STEP,
        )
        self.assertEqual(run.initial_overlap, [])
        self.assertEqual(run.conflicts, [])


class PromptedProtocolTests(unittest.TestCase):
    """The handover written into TICK_FORMAT_RULES must actually work.

    The rules show the model a four-part shape. If the implementation and the
    example ever disagree, the model is being taught to produce deadlocks --
    which is exactly what the first tick A/B measured, so pin it here.
    """

    def _handover(self):
        # ASK, then BLOCK on the very next tick; the holder keeps working for
        # as long as it likes, and only then LEAVES and REPORTS back to back.
        return plan(
            *OPEN,
            step("agent_1", "communicate", to="agent_0",
                 message='When done, send releases="cab".'),
            step("agent_1", "wait_for_signal", **{"from": "agent_0", "about": "cab"}),
            step("agent_0", "navigate_to_fixture", fixture_id="cab"),
            step("agent_0", "communicate", to="agent_1", message="still working here"),
            step("agent_0", "communicate", to="agent_1", message="nearly done"),
            step("agent_0", "give_space", fixture_id="cab"),
            step("agent_0", "communicate", to="agent_1", message="cab is yours now",
                 releases=["cab"]),
            step("agent_1", "navigate_to_fixture", fixture_id="cab"),
        )

    def test_the_prompted_handover_runs_clean(self):
        for model in (LOCK_STEP, EXECUTOR):
            with self.subTest(model=model):
                run = build().replay(self._handover(), model=model)
                self.assertFalse(run.deadlocked)
                self.assertEqual(run.conflicts, [])
                self.assertEqual(run.protocol, [])

    def test_the_wait_actually_blocks(self):
        # agent_1 reaches its wait at instant 2 and is not woken until the
        # release at 3, so it genuinely sits still for an instant. The wait
        # EVENT is recorded at the instant it discharges, which is why the
        # blocking shows up as idle time rather than as a tick difference.
        run = build().replay(self._handover(), model=LOCK_STEP)
        self.assertEqual(run.protocol, [], "not flagged inert")
        self.assertGreater(run.idle["agent_1"], 0.0)

    def test_the_rules_state_the_release_obligation(self):
        import robotalk.generation.tick_format as tick_format

        text = "\n".join(tick_format.TICK_FORMAT_RULES)
        self.assertIn("releases", text)
        self.assertIn("give_space", text, "release is coupled to departure")
        self.assertIn("on a LATER tick", text)
        self.assertIn("may be reported later", text)
        self.assertIn("must not reoccupy or reuse", text)
        self.assertIn("very next tick", text, "no gap between ask and wait")
        self.assertIn("tick AFTER the release", text, "no same-tick resumption")

    def test_the_wait_must_immediately_follow_the_ask(self):
        steps = list(self._handover()["steps"])
        steps.insert(3, step("agent_1", "get_image"))  # inert, must be allowed
        self.assertEqual(build().replay(plan(*steps), model=LOCK_STEP).protocol, [])

        steps[3] = step("agent_1", "communicate", to="agent_0",
                        message='When done, send releases="cab".')
        run = build().replay(plan(*steps), model=LOCK_STEP)
        self.assertEqual(run.protocol, [], "an extra ask is still an ask")

        steps[3] = step("agent_1", "navigate_to_fixture", fixture_id="sink")
        run = build().replay(plan(*steps), model=LOCK_STEP)
        self.assertTrue(any("not the request" in p for p in run.protocol))

    def test_the_ask_names_the_exact_structural_release(self):
        steps = list(self._handover()["steps"])
        steps[2] = step(
            "agent_1", "communicate", to="agent_0",
            message="Please tell me when the cabinet is free.",
        )
        run = build().replay(plan(*steps), model=LOCK_STEP)
        self.assertTrue(any("exact release value" in p for p in run.protocol))

        steps[2] = step(
            "agent_1", "communicate", to="agent_0",
            message='When done, send releases="counter".',
        )
        run = build().replay(plan(*steps), model=LOCK_STEP)
        self.assertTrue(any("exact release value" in p for p in run.protocol))

    def test_the_ask_may_describe_the_release_keyword_naturally(self):
        for message in (
            'When done, send releases="cab".',
            "Please release cab when finished.",
            "Please use the keyword cab as the release keyword.",
        ):
            with self.subTest(message=message):
                steps = list(self._handover()["steps"])
                steps[2] = step(
                    "agent_1", "communicate", to="agent_0", message=message,
                )
                run = build().replay(plan(*steps), model=LOCK_STEP)
                self.assertFalse(
                    any("exact release value" in p for p in run.protocol)
                )

    def test_a_holder_that_leaves_early_is_fine_if_it_reports_late(self):
        # The tightest correct handover has the holder vacating on the very
        # instant the waiter blocks, and leaving a tick earlier is no worse:
        # the release still lands afterwards and still wakes the waiter. An
        # earlier rule rejected both, and was wrong 6 times in 7 on real output.
        run = build().replay(
            plan(*OPEN,
                 step("agent_0", "navigate_to_fixture", fixture_id="cab"),
                 step("agent_0", "give_space", fixture_id="cab"),
                 step("agent_1", "communicate", to="agent_0",
                      message='When done, send releases="cab".'),
                 step("agent_1", "wait_for_signal", **{"from": "agent_0",
                                                       "about": "cab"}),
                 step("agent_0", "communicate", to="agent_1", message="yours",
                      releases=["cab"]),
                 step("agent_1", "navigate_to_fixture", fixture_id="cab")),
            model=LOCK_STEP,
        )
        self.assertEqual(run.protocol, [])
        self.assertFalse(run.deadlocked)

    def test_withdrawing_before_announcing_is_not_a_gap(self):
        # place the object, step clear of the fixture, THEN announce. The
        # give_space in between is part of the handover, not an interruption.
        run = build().replay(
            plan(*OPEN,
                 step("agent_0", "navigate_to_fixture", fixture_id="cab"),
                 step("agent_0", "pick_up_object", object_id="bowl",
                      source_id="counter"),
                 step("agent_0", "place_on_surface", object_id="bowl",
                      support_id="counter"),
                 step("agent_0", "give_space", fixture_id="cab"),
                 step("agent_0", "communicate", to="agent_1",
                      message="the bowl is yours", releases=["bowl"])),
            model=LOCK_STEP,
        )
        self.assertEqual(run.protocol, [])

    def test_dropping_the_release_from_the_prompted_shape_deadlocks(self):
        # The failure mode the rules exist to prevent, held fixed.
        steps = [s for s in self._handover()["steps"]
                 if not (s.get("args") or {}).get("releases")]
        run = build().replay(plan(*steps), model=LOCK_STEP)
        self.assertTrue(run.deadlocked)


class SymbolGroundingTests(unittest.TestCase):
    """Coordination has to be about things that exist.

    The tick A/B failed every trajectory this way: the model named the MOMENT
    it was waiting for -- "mug_placed", "counter_free" -- instead of the thing.
    A consistent fiction matches itself, so pairing waits with releases is not
    enough to catch it.
    """

    def test_waiting_on_an_invented_id_is_rejected(self):
        run = build().replay(
            plan(*OPEN,
                 step("agent_1", "communicate", to="agent_0", message="tell me when"),
                 step("agent_1", "wait_for_signal", **{"from": "agent_0",
                                                       "about": "mug_placed"}),
                 step("agent_0", "navigate_to_fixture", fixture_id="cab"),
                 step("agent_0", "give_space", fixture_id="cab"),
                 step("agent_0", "communicate", to="agent_1", message="done",
                      releases=["mug_placed"])),
            model=LOCK_STEP,
        )
        self.assertTrue(any("not an object or fixture" in p for p in run.protocol))

    def test_releasing_an_invented_id_is_rejected(self):
        run = build().replay(
            plan(*OPEN,
                 step("agent_0", "communicate", to="agent_1", message="your turn now",
                      releases=["your_turn"])),
            model=LOCK_STEP,
        )
        self.assertTrue(any("not an object or fixture" in p for p in run.protocol))

    def test_a_consistent_fiction_still_fails(self):
        # Both halves agree, so the replay pairs them and nothing hangs. The
        # plan is still coordinating on something that does not exist.
        run = build().replay(
            plan(*OPEN,
                 step("agent_1", "communicate", to="agent_0", message="tell me when"),
                 step("agent_1", "wait_for_signal", **{"from": "agent_0",
                                                       "about": "task_done"}),
                 step("agent_0", "navigate_to_fixture", fixture_id="cab"),
                 step("agent_0", "communicate", to="agent_1", message="done",
                      releases=["task_done"]),
                 step("agent_1", "navigate_to_fixture", fixture_id="sink")),
            model=LOCK_STEP,
        )
        self.assertTrue(
            run.deadlocked,
            "a same-tick release processed before the wait cannot wake it",
        )
        self.assertTrue(run.protocol, "but it is still rejected")

    def test_real_ids_pass(self):
        for about in ("cab", "bowl"):
            with self.subTest(about=about):
                self.assertTrue(build()._known_id(about))
        self.assertFalse(build()._known_id("mug_placed"))

    def test_the_prompt_forbids_invented_ids(self):
        import robotalk.generation.tick_format as tick_format

        text = "\n".join(tick_format.TICK_FORMAT_RULES)
        self.assertIn("mug_placed", text, "name the failure mode concretely")
        self.assertIn("initial_state", text)


class ReleaseProtocolTests(unittest.TestCase):
    """Three rules that need no clock, so no duration model can excuse them."""

    def test_legacy_shared_cabinet_release_is_not_an_occupancy_violation(self):
        run = build().replay(
            plan(*OPEN,
                 step("agent_0", "navigate_to_fixture", fixture_id="cab"),
                 step("agent_0", "communicate", to="agent_1",
                      message="you can take it", releases=["cab"])),
            model=LOCK_STEP,
        )
        self.assertFalse(any("still standing at it" in p for p in run.protocol))

    def test_releasing_an_object_you_are_still_holding_is_rejected(self):
        run = build().replay(
            plan(*OPEN,
                 step("agent_0", "pick_up_object", object_id="bowl",
                      source_id="counter"),
                 step("agent_0", "communicate", to="agent_1",
                      message="bowl is yours", releases=["bowl"])),
            model=LOCK_STEP,
        )
        self.assertTrue(any("still holding it" in p for p in run.protocol))

    def test_a_gap_between_leaving_and_reporting_is_allowed(self):
        # A delayed report is inefficient but live-valid: the waiter simply
        # remains blocked until the matching release arrives.
        run = build().replay(
            plan(*OPEN,
                 step("agent_0", "navigate_to_fixture", fixture_id="cab"),
                 step("agent_0", "give_space", fixture_id="cab"),
                 step("agent_0", "communicate", to="agent_1", message="just tidying"),
                 step("agent_0", "communicate", to="agent_1", message="all done",
                      releases=["cab"])),
            model=LOCK_STEP,
        )
        self.assertEqual(run.protocol, [])

    def test_observations_between_the_departure_and_the_report_are_fine(self):
        # get_image is inert and post-processing interleaves it everywhere.
        run = build().replay(
            plan(*OPEN,
                 step("agent_0", "navigate_to_fixture", fixture_id="cab"),
                 step("agent_0", "give_space", fixture_id="cab"),
                 step("agent_0", "get_image"),
                 step("agent_0", "communicate", to="agent_1", message="all done",
                      releases=["cab"])),
            model=LOCK_STEP,
        )
        self.assertEqual(run.protocol, [])

    def test_a_wait_released_on_its_own_instant_is_inert(self):
        run = build().replay(
            plan(*OPEN,
                 step("agent_0", "navigate_to_fixture", fixture_id="cab"),
                 step("agent_0", "give_space", fixture_id="cab"),
                 step("agent_0", "communicate", to="agent_1", message="yours",
                      releases=["cab"]),
                 step("agent_1", "communicate", to="agent_0", message="ok then"),
                 step("agent_1", "communicate", to="agent_0", message="still here"),
                 step("agent_1", "wait_for_signal", **{"from": "agent_0",
                                                       "about": "cab"})),
            model=LOCK_STEP,
        )
        self.assertTrue(any("inert" in p for p in run.protocol))

    def test_a_wait_that_arrives_after_the_release_says_so(self):
        run = build().replay(
            plan(*OPEN,
                 step("agent_0", "navigate_to_fixture", fixture_id="cab"),
                 step("agent_0", "give_space", fixture_id="cab"),
                 step("agent_0", "communicate", to="agent_1", message="yours",
                      releases=["cab"]),
                 step("agent_1", "communicate", to="agent_0", message="one"),
                 step("agent_1", "communicate", to="agent_0", message="two"),
                 step("agent_1", "communicate", to="agent_0", message="three"),
                 step("agent_1", "wait_for_signal", **{"from": "agent_0",
                                                       "about": "cab"})),
            model=LOCK_STEP,
        )
        self.assertTrue(run.deadlocked)
        self.assertTrue(any("too late" in c for c in run.conflicts))


class GoalTests(unittest.TestCase):
    def test_work_begun_after_the_goal_is_reported(self):
        # The goal lands on the two navigates at t=1; the give_space at t=2 is
        # started afterwards and has nothing left to contribute.
        run = build(goal_after=2).replay(
            plan(*OPEN,
                 step("agent_0", "navigate_to_fixture", fixture_id="cab"),
                 step("agent_1", "navigate_to_fixture", fixture_id="sink"),
                 step("agent_1", "give_space", fixture_id="sink")),
            model=LOCK_STEP,
        )
        self.assertIsNotNone(run.goal_at)
        self.assertTrue(run.post_goal)

    def test_work_already_in_flight_when_the_goal_lands_is_not_post_goal(self):
        # Both agents commit to their call on the same instant; one of them
        # happening to complete the goal does not make the other's call late.
        run = build(goal_after=3).replay(
            plan(*OPEN,
                 step("agent_0", "navigate_to_fixture", fixture_id="cab"),
                 step("agent_1", "navigate_to_fixture", fixture_id="sink")),
            model=LOCK_STEP,
        )
        self.assertEqual(run.post_goal, [])


class ReportingTests(unittest.TestCase):
    def test_idle_time_is_reported_per_agent(self):
        run = build().replay(
            plan(*OPEN,
                 step("agent_0", "navigate_to_fixture", fixture_id="cab"),
                 step("agent_0", "give_space", fixture_id="cab")),
            model=LOCK_STEP,
        )
        self.assertEqual(run.makespan, 3.0)
        self.assertEqual(run.idle["agent_1"], 2.0)

    def test_render_puts_one_agent_in_each_column(self):
        text = build().render(plan(*OPEN), model=LOCK_STEP)
        self.assertIn("agent_0", text.splitlines()[0])
        self.assertIn("agent_1", text.splitlines()[0])
        self.assertEqual(len(text.splitlines()), 3)

    def test_durations_match_the_live_evaluation(self):
        self.assertEqual(concurrent_fsm.duration_of("navigate_to_fixture", EXECUTOR), 4.0)
        self.assertEqual(concurrent_fsm.duration_of("communicate", EXECUTOR), 0.25)
        self.assertEqual(concurrent_fsm.duration_of("pick_up_object", EXECUTOR), 2.0)
        self.assertEqual(concurrent_fsm.duration_of("navigate_to_fixture", LOCK_STEP), 1.0)


if __name__ == "__main__":
    unittest.main()
