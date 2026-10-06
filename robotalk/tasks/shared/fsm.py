"""Apply the shared symbolic finite-state validator to task trajectories."""

from __future__ import annotations

from copy import deepcopy
from difflib import SequenceMatcher
import re
from typing import Any, Sequence

from robotalk.utils import stable_json_sha256

from .scheduling import overlapping_users, schedule, usage_spans
from .constants import (
    ACQUIRE_TOOL_NAMES,
    CLOSE_PART_TOOL_NAMES,
    GIVE_SPACE_TOOL_NAMES,
    INTERACTION_TOOL_NAMES,
    NAVIGATION_TOOL_NAMES,
    OBSERVATION_TOOL_NAMES,
    EXCLUSIVE_FIXTURE_TYPES,
    SOCIAL_TOOL_NAMES,
    WAIT_TOOL_NAMES,
    OPEN_PART_TOOL_NAMES,
    PLACE_LOCATION_ARG_NAMES,
    RELEASE_TOOL_NAMES,
)
from .errors import (
    CommunicationStepSemanticValidationError,
    HeldObjectSemanticValidationError,
    MissingInitialCommunicationSemanticValidationError,
    MissingTaskActionSemanticValidationError,
    NavigationSemanticValidationError,
    ObjectStateSemanticValidationError,
    ObservationSequenceSemanticValidationError,
    WaitSignalSemanticValidationError,
    PlacementDestinationSemanticValidationError,
    PostGoalActionSemanticValidationError,
    TaskPreconditionSemanticValidationError,
    ToolArgumentSemanticValidationError,
    TrajectoryStructureValidationError,
    TrajectoryValidationError,
    UnexpectedStepIndexSemanticValidationError,
    UnsupportedToolSemanticValidationError,
    UnsatisfiedGoalSemanticValidationError,
)
from .instances import build_canonical_agents
from .prompting import _format_agent_id_list
from .workspace_semantics import (
    accepted_agent_workspaces,
    canonical_agent_workspace,
)
from .schema import (
    _allowed_ids_key_for_arg_name,
    _normalize_mapping,
    _normalize_text,
    _resolve_tool_arg_schema_type,
)
from .state import AgentRuntimeState, TaskRuntimeState

_SUPPORT_SITE_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SUPPORT_SITE_STOPWORDS = frozenset(
    {"support", "site", "surface", "region", "placement", "place", "burner"}
)


def _normalize_support_site_signature(token: str) -> tuple[str, ...]:
    normalized_tokens: list[str] = []
    for raw_token in _SUPPORT_SITE_TOKEN_RE.findall(str(token).lower()):
        alias = "rear" if raw_token == "back" else raw_token
        if alias in _SUPPORT_SITE_STOPWORDS or not alias:
            continue
        if alias not in normalized_tokens:
            normalized_tokens.append(alias)
    if normalized_tokens:
        return tuple(normalized_tokens)
    lowered = str(token).strip().lower()
    return (lowered,) if lowered else ()


def _support_site_alias_matches(requested_id: str, candidate_id: str) -> bool:
    if requested_id == candidate_id:
        return True
    requested_signature = _normalize_support_site_signature(requested_id)
    candidate_signature = _normalize_support_site_signature(candidate_id)
    if requested_signature == candidate_signature:
        return True
    requested_text = "_".join(requested_signature)
    candidate_text = "_".join(candidate_signature)
    if not requested_text or not candidate_text:
        return False
    overlap = len(set(requested_signature) & set(candidate_signature))
    union = len(set(requested_signature) | set(candidate_signature))
    jaccard = overlap / union if union else 0.0
    ratio = SequenceMatcher(None, requested_text, candidate_text).ratio()
    return max(jaccard, ratio) >= 0.67


class FiniteStateTaskValidator:
    """Applies a legality-and-goal FSM over task-level tool calls."""

    def __init__(
        self,
        *,
        composite_task: str,
        agent_ids: Sequence[str],
        initial_state: dict[str, Any],
        allowed_tool_specs: dict[str, Any],
        checks: Sequence[str] | None = None,
        max_reasoning_chars: int = 200,
        initial_public_state: dict[str, Any] | None = None,
    ) -> None:
        """Stores task-local configuration for the FSM validator.

        Args:
            composite_task: Human-readable task name; used in validation errors.
            agent_ids: Ordered agent IDs the validator expects in the trajectory.
            initial_state: Symbolic initial world state.
            allowed_tool_specs: Task-local tool registry used to reject unsupported tools.
            checks: Validation check labels returned on successful validation.
            max_reasoning_chars: Maximum allowed character count for each reasoning string.
            initial_public_state: Extra task-specific summary fields seeded into final_state.
        """

        self.composite_task = composite_task
        self.agent_ids = tuple(agent_ids)
        self._agent_id_set = set(self.agent_ids)
        self.initial_state = deepcopy(initial_state)
        self.allowed_tool_specs = deepcopy(allowed_tool_specs)
        # Some tasks ask the model to emit observation steps directly, while
        # others synthesize them later during post-processing.
        self._requires_observation_steps = any(
            tool_name in OBSERVATION_TOOL_NAMES for tool_name in self.allowed_tool_specs
        )
        self.max_reasoning_chars = max_reasoning_chars
        self._all_checks = list(
            checks
            or (
                "initial_communication",
                "allowed_tools",
                "navigation_preconditions",
                "manipulation_preconditions",
                "effects",
                "final_success",
            )
        )
        self._initial_public_state = deepcopy(initial_public_state or {})

    def validate(self, candidate: dict[str, Any]) -> dict[str, Any]:
        """Validates a candidate trajectory by replaying legal steps to a goal."""

        agents = self._normalize_agents(candidate.get("agents"))
        steps = self._normalize_steps(candidate.get("steps"))
        self._validate_wait_protocol(steps)
        runtime_state = self._build_runtime_state(agents)
        first_action_seen = False
        goal_state_satisfied = self.is_goal_state_satisfied(runtime_state)
        # Step index at which the goal first became satisfied, or None if it
        # was already satisfied at the initial state (before any step).
        goal_satisfied_at_step: int | None = None

        for expected_index, step in enumerate(steps):
            state_before_step = self._build_final_state(runtime_state)
            try:
                if step["step"] != expected_index:
                    raise UnexpectedStepIndexSemanticValidationError(
                        f"step {step['step']} does not match expected index {expected_index}.",
                        details={
                            "actual_step": step["step"],
                            "expected_step": expected_index,
                        },
                    )

                # Preserve a final post-condition snapshot after the goal is reached,
                # but reject any later non-observation work.
                if goal_state_satisfied and step["tool"] not in OBSERVATION_TOOL_NAMES:
                    raise PostGoalActionSemanticValidationError(
                        f"No steps are allowed after the {self.composite_task} goal state is satisfied.",
                        details={
                            "tool": step["tool"],
                            "composite_task": self.composite_task,
                            "goal_satisfied_at_step": goal_satisfied_at_step,
                        },
                    )

                if step["tool"] in WAIT_TOOL_NAMES:
                    self._validate_wait_step(step, steps, expected_index)
                    continue

                if step["tool"] == "communicate":
                    self._validate_communicate_step(step)
                    runtime_state.communicated_agents.add(step["agent"])
                    self.apply_task_effects(step, runtime_state)
                    was_satisfied = goal_state_satisfied
                    goal_state_satisfied = self.is_goal_state_satisfied(runtime_state)
                    if goal_state_satisfied and not was_satisfied:
                        goal_satisfied_at_step = step["step"]
                    continue

                # Shared observation steps may appear before the opening
                # coordination block, but real task actions still require both
                # agents to communicate first.
                if (
                    step["tool"] not in OBSERVATION_TOOL_NAMES
                    and runtime_state.communicated_agents != self._agent_id_set
                ):
                    raise MissingInitialCommunicationSemanticValidationError(
                        "Both agents must coordinate via communication before the first task action.",
                        details={
                            "agent": step["agent"],
                            "tool": step["tool"],
                            "communicated_agents": sorted(
                                runtime_state.communicated_agents
                            ),
                            "required_agents": sorted(self._agent_id_set),
                        },
                    )

                if step[
                    "tool"
                ] not in self.allowed_tool_specs and not self._is_allowed_observation_tool(
                    step["tool"]
                ):
                    raise UnsupportedToolSemanticValidationError(
                        f"Tool {step['tool']} is not allowed for {self.composite_task}.",
                        details={
                            "tool": step["tool"],
                            "composite_task": self.composite_task,
                        },
                    )

                if self._requires_observation_steps:
                    self._validate_required_observation_sequence(steps, expected_index)
                self._validate_task_local_symbolic_constraints(step)
                self._validate_generic_transition(step, runtime_state)
                self.validate_task_preconditions(step, runtime_state)
                self._apply_generic_effects(step, runtime_state)
                self.apply_task_effects(step, runtime_state)
                first_action_seen = True
                was_satisfied = goal_state_satisfied
                goal_state_satisfied = self.is_goal_state_satisfied(runtime_state)
                if goal_state_satisfied and not was_satisfied:
                    goal_satisfied_at_step = step["step"]
            except TrajectoryValidationError as exc:
                details = dict(exc.details)
                details.setdefault("state_before_step", state_before_step)
                contextual_exc = type(exc)(
                    str(exc), step=exc.step, details=details
                )
                raise self._validation_error_with_step(
                    contextual_exc, step["step"]
                ) from exc

        if not first_action_seen:
            raise MissingTaskActionSemanticValidationError(
                "Trajectory did not contain any task action steps."
            )
        if not goal_state_satisfied:
            raise UnsatisfiedGoalSemanticValidationError(
                f"Trajectory never satisfied the {self.composite_task} goal state.",
                details=self.goal_state_diagnostics(runtime_state),
            )

        signature = self.trajectory_signature({"agents": agents, "steps": steps})
        return {
            "is_valid": True,
            "checks": list(self._all_checks),
            "final_state": self._build_final_state(runtime_state),
            # Reuse the normalized trajectory when persisting successful outputs.
            "normalized_candidate": {
                "agents": agents,
                "steps": steps,
            },
            "signature": signature,
        }

    def trajectory_signature(self, candidate: dict[str, Any]) -> str:
        """Builds a stable signature for duplicate-trajectory rejection."""

        normalized = {
            "initial_state": self.initial_state,
            "agents": sorted(candidate["agents"], key=lambda agent: agent["agent"]),
            "steps": sorted(candidate["steps"], key=lambda step: step["step"]),
        }
        return stable_json_sha256(normalized)

    def _build_runtime_state(
        self,
        agents: list[dict[str, str]],
    ) -> TaskRuntimeState:
        """Builds the mutable runtime state snapshot used by the FSM."""

        agent_states: dict[str, AgentRuntimeState] = {}
        for agent in agents:
            initial_agent_state = self.initial_state["agents"][agent["agent"]]
            agent_states[agent["agent"]] = AgentRuntimeState(
                location=initial_agent_state.get("location"),
                held_object=initial_agent_state.get("held_object"),
            )

        return TaskRuntimeState(
            agents=agent_states,
            objects=deepcopy(self.initial_state.get("objects", {})),
            fixtures=deepcopy(self.initial_state.get("fixtures", {})),
            machine_state=deepcopy(self.initial_state.get("machine_state", {})),
            public_state=deepcopy(self._initial_public_state),
        )

    def _build_final_state(self, runtime_state: TaskRuntimeState) -> dict[str, Any]:
        """Builds the shared validation payload from the final FSM state."""

        final_state = deepcopy(runtime_state.public_state)
        final_state["agents"] = {
            agent_id: {
                "location": agent_state.location,
                "held_object": agent_state.held_object,
            }
            for agent_id, agent_state in runtime_state.agents.items()
        }
        final_state["objects"] = deepcopy(runtime_state.objects)
        final_state["fixtures"] = deepcopy(runtime_state.fixtures)
        final_state["machine_state"] = deepcopy(runtime_state.machine_state)
        return final_state

    def goal_state_diagnostics(
        self, runtime_state: TaskRuntimeState
    ) -> dict[str, Any]:
        """Describe the terminal symbolic state without prescribing a repair."""

        return {"final_state": self._build_final_state(runtime_state)}

    def _normalize_agents(self, agents_value: Any) -> list[dict[str, str]]:
        """Normalizes the agent roster required by the task-level schema."""

        if agents_value is None:
            return build_canonical_agents(self.agent_ids)
        if not isinstance(agents_value, list) or len(agents_value) != len(
            self.agent_ids
        ):
            raise TrajectoryStructureValidationError(
                f"agents must be a list containing exactly {len(self.agent_ids)} agents."
            )

        normalized_agents: list[dict[str, str]] = []
        seen_agent_ids: set[str] = set()
        for agent in agents_value:
            if not isinstance(agent, dict):
                raise TrajectoryStructureValidationError(
                    "Each agent entry must be an object."
                )
            agent_id = _normalize_text(agent.get("agent"), "agent")
            if agent_id not in self._agent_id_set:
                raise TrajectoryStructureValidationError(
                    f"Unsupported agent {agent_id}."
                )
            if agent_id in seen_agent_ids:
                raise TrajectoryStructureValidationError(f"Duplicate agent {agent_id}.")
            seen_agent_ids.add(agent_id)
            normalized_agents.append({"agent": agent_id})

        if seen_agent_ids != self._agent_id_set:
            expected_agents = _format_agent_id_list(self.agent_ids)
            raise TrajectoryStructureValidationError(
                f"agents must contain exactly {expected_agents}."
            )
        return normalized_agents

    def _normalize_steps(self, steps_value: Any) -> list[dict[str, Any]]:
        """Normalizes step objects before semantic validation starts."""

        if not isinstance(steps_value, list) or not steps_value:
            raise TrajectoryStructureValidationError("steps must be a non-empty list.")

        normalized_steps: list[dict[str, Any]] = []
        for raw_step in steps_value:
            if not isinstance(raw_step, dict):
                raise TrajectoryStructureValidationError("Each step must be an object.")
            if not isinstance(raw_step.get("step"), int):
                raise TrajectoryStructureValidationError("step must be an integer.")

            step_index = raw_step["step"]
            try:
                agent_id = _normalize_text(
                    raw_step.get("agent"),
                    f"step[{step_index}].agent",
                )
                if agent_id not in self._agent_id_set:
                    raise TrajectoryStructureValidationError(
                        f"Unsupported step[{step_index}].agent {agent_id}."
                    )

                reasoning = _normalize_text(
                    raw_step.get("reasoning"),
                    f"step[{step_index}].reasoning",
                )
                if len(reasoning) > self.max_reasoning_chars:
                    raise TrajectoryStructureValidationError(
                        f"Reasoning for step {step_index} exceeds {self.max_reasoning_chars} characters."
                    )

                normalized_steps.append(
                    {
                        "step": step_index,
                        "agent": agent_id,
                        "tool": _normalize_text(
                            raw_step.get("tool"),
                            f"step[{step_index}].tool",
                        ),
                        "args": _normalize_mapping(
                            raw_step.get("args", {}),
                            f"step[{step_index}].args",
                        ),
                        "reasoning": reasoning,
                    }
                )
            except TrajectoryValidationError as exc:
                raise self._validation_error_with_step(exc, step_index) from exc
        return normalized_steps

    def _validation_error_with_step(
        self,
        exc: TrajectoryValidationError,
        step: int | None,
    ) -> TrajectoryValidationError:
        """Attaches the first failing step number to a validation exception."""

        if exc.step is not None:
            return exc
        return exc.with_step(step)

    def _validate_wait_protocol(self, steps: Sequence[dict[str, Any]]) -> None:
        """Reports every wait-protocol problem in one error.

        Missing waits and undischarged waits are opposite ends of one contract,
        and surfacing them one at a time makes the repair loop oscillate: the
        model adds the wait it was told about, which creates an unreleased
        wait, so the next attempt removes it again. Observed across all eight
        retries of a single trajectory. Reporting both together is what lets a
        single regeneration satisfy the whole contract.
        """

        problems = list(self._validate_dependency_waits(steps))
        for index, step in enumerate(steps):
            if step["tool"] not in WAIT_TOOL_NAMES:
                continue
            try:
                self._validate_wait_step(step, steps, index)
            except WaitSignalSemanticValidationError as exc:
                problems.append(f"  {exc.args[0]}")
        if not problems:
            return
        raise WaitSignalSemanticValidationError(
            f"{len(problems)} problem(s) with the wait protocol. Fix ALL of "
            f"them together in the next attempt -- fixing one at a time will "
            f"just break another:\n" + "\n".join(problems),
            details={"problem_count": len(problems)},
        )

    def _validate_dependency_waits(
        self,
        steps: Sequence[dict[str, Any]],
    ) -> list[str]:
        """Collects every unguarded contention between the agents.

        Validating only the waits that happen to be present leaves the real gap
        open: a plan can omit the wait and still pass, because a missing wait
        has no symbolic effect. Ordered replay hides that -- the recorded
        sequence supplies the ordering -- but the concurrent scheduler may
        re-interleave, and the dependent action then runs against a world the
        other agent is still changing. A demo that only works under a lucky
        interleaving is not a correct demo.

        Two things count as contention, and nothing else does:
          * the same OBJECT, wherever it sits; and
          * the same EXCLUSIVE fixture, whatever each agent is handling there.
        A roomy counter shared by two agents working on different objects is
        fine, which is why fixture type decides rather than fixture identity.
        """

        # Keep this temporal wait audit on the exact same resource definition
        # as concurrent replay and live evaluation. In particular, two agents
        # may read one stationary source plate while picking distinct children;
        # moving the plate while it is read is handled by the canonical pair
        # classifier in concurrent replay.
        from .concurrent_fsm import contention_resources

        objects = set(self.initial_state.get("objects") or {})

        def _resources(step: dict[str, Any]) -> list[str]:
            return list(contention_resources(step, self.initial_state))

        # Contention is a TEMPORAL fact, not a structural one. Requiring a wait
        # wherever two agents merely touch the same thing demanded one for
        # handoffs the schedule already separates, and a release for such a
        # wait fires before the waiter arrives -- which under the executor's
        # message semantics is a hang, not a redundancy. Measured over 1560
        # trajectories: 309 deadlocked that way. Only an overlap needs a wait.
        ticks, _ = schedule(steps)
        spans = usage_spans(steps, ticks, lambda step: _resources(step))
        overlapping = {
            (resource, first, later)
            for resource, first, later in overlapping_users(spans)
        }
        overlapping |= {(r, b, a) for r, a, b in overlapping}

        held: dict[str, tuple[str, int]] = {}
        violations: list[str] = []
        for index, step in enumerate(steps):
            tool = step["tool"]
            if tool in SOCIAL_TOOL_NAMES or tool in OBSERVATION_TOOL_NAMES:
                continue
            actor = step["agent"]
            args = step.get("args") or {}

            # give_space is the actor LEAVING. Recording it as a use would
            # renew the hold and reset the window, so the wait that already
            # guarded this handoff would stop counting.
            if tool in GIVE_SPACE_TOOL_NAMES:
                held.pop(str(args.get("fixture_id")), None)
                continue

            contested = _resources(step)

            for resource in contested:
                holder, held_at = held.get(resource, (None, -1))
                if holder is None or holder == actor:
                    continue
                if (resource, holder, actor) not in overlapping:
                    continue  # the schedule already separates them
                if any(
                    earlier["tool"] in WAIT_TOOL_NAMES
                    and earlier["agent"] == actor
                    and str((earlier.get("args") or {}).get("about")) == resource
                    and (earlier.get("args") or {}).get("from") == holder
                    for earlier in steps[held_at + 1 : index]
                ):
                    continue
                kind = "object" if resource in objects else "exclusive fixture"
                violations.append(
                    f"  step {step['step']} ({tool}): {actor} uses {resource!r}, "
                    f"the same {kind} {holder} used at step "
                    f"{steps[held_at]['step']}; {actor} must call "
                    f"wait_for_signal(from={holder!r}, about={resource!r}) "
                    f"before that step, and {holder} must announce and release it."
                )
            for resource in contested:
                held[resource] = (actor, index)

        return violations

    def _validate_wait_step(
        self,
        step: dict[str, Any],
        steps: Sequence[dict[str, Any]],
        step_index: int,
    ) -> None:
        """Checks one wait against two invariants.

        1. it names the other agent and a thing that exists;
        2. the partner later tells the waiter, by name, that it is done -- and
           really is done, meaning it never touches that thing again.

        Invariant 2 replaced a rule about when the "releasing action" had to
        occur relative to the wait. That rule was wrong twice: it rejected
        ordinary handoffs where the partner had finished beforehand, which is
        the normal case. "Never touches it again" says the same thing without
        needing to reason about ordering, and it still rejects a promise --
        "I will give you space" followed by the partner using it anyway.

        The announcement is deliberately NOT checked for the id. Its job is to
        make the wait visible, since under partial observability the partner
        cannot see it; identifying the thing is the release's job.

        A release is `communicate` carrying `releases: <id>`. Matching the id
        out of the message text still works, for trajectories written before
        that field existed, but new data should use the field: a wait that any
        sufficiently-worded sentence can discharge teaches the model to write
        the sentence rather than to finish the work.
        """

        tool_args = step["args"]
        from_agent = tool_args.get("from")
        about = tool_args.get("about")
        if from_agent not in self._agent_id_set or from_agent == step["agent"]:
            raise WaitSignalSemanticValidationError(
                "wait_for_signal requires from to reference the other agent.",
                details={"agent": step["agent"], "from": from_agent},
            )
        if not isinstance(about, str) or not about.strip():
            raise WaitSignalSemanticValidationError(
                "wait_for_signal requires a non-empty about in args.",
                details={"agent": step["agent"], "from": from_agent},
            )
        if set(tool_args) != {"from", "about"}:
            raise WaitSignalSemanticValidationError(
                "wait_for_signal args may only contain from and about.",
                details={"arg_names": sorted(tool_args)},
            )

        about = about.strip()
        step["args"] = {"from": from_agent, "about": about}
        known = set(self.initial_state.get("objects") or {}) | set(
            self.initial_state.get("fixtures") or {}
        )
        if known and about not in known:
            raise WaitSignalSemanticValidationError(
                f"wait_for_signal about={about!r} is not a symbolic id in this "
                f"task; it must name a declared object or fixture.",
                details={"agent": step["agent"], "about": about,
                         "known_ids": sorted(known), "step": step["step"]},
            )

        needle = about.casefold()
        # "After" means after in TIME, not after in the file. The two agents
        # advance independently, so a release written earlier in the list can
        # still fire once the waiter is already blocked -- and placing it where
        # it fires correctly is what stops it arriving before the waiter exists.
        wait_ticks, _ = schedule(steps)
        wait_tick = wait_ticks[step_index]
        for offset, later in enumerate(steps):
            later_tick = wait_ticks[offset]
            if wait_tick is not None and (later_tick is None or later_tick < wait_tick):
                continue
            if wait_tick is None and offset <= step_index:
                continue
            if later["agent"] != from_agent or later["tool"] != "communicate":
                continue
            later_args = later.get("args") or {}
            if later_args.get("to") != step["agent"]:
                continue
            # A structural release is the real signal. The textual rule below
            # it is a fallback for trajectories generated before `releases`
            # existed; it stays because those remain valid, not because
            # phrasing should be able to discharge a wait.
            released = later_args.get("releases") or []
            released = [released] if isinstance(released, str) else list(released)
            if about not in released:
                message = str(later_args.get("message", ""))
                if needle not in message.casefold():
                    continue
                # A message that is nothing but the id discharges no obligation;
                # generated data contained a release whose whole text was "bowl".
                if not message.casefold().replace(needle, "").strip(" .,;:!"):
                    continue
            # The partner must leave it alone until the waiter has had its
            # turn -- not forever. Two agents may legitimately use one fixture
            # in alternation, each handing it back when done.
            handed_back = next(
                (
                    later_index
                    for later_index, s2 in enumerate(steps[offset + 1 :], offset + 1)
                    if s2["agent"] == step["agent"]
                    and s2["tool"] not in SOCIAL_TOOL_NAMES
                    and s2["tool"] not in OBSERVATION_TOOL_NAMES
                    and any(str(v) == about for v in (s2.get("args") or {}).values())
                ),
                len(steps),
            )
            if self._uses_again(steps[offset + 1 : handed_back], from_agent, about):
                continue
            return

        raise WaitSignalSemanticValidationError(
            f"wait_for_signal at step {step['step']} is never released: "
            f"{from_agent} never tells {step['agent']} it is done with "
            f"{about!r} and then leaves it alone.",
            details={"agent": step["agent"], "from": from_agent,
                     "about": about, "step": step["step"]},
        )

    def _uses_again(
        self,
        remaining: Sequence[dict[str, Any]],
        agent: str,
        resource: str,
    ) -> bool:
        """True if `agent` touches `resource` again -- so it was not done."""

        for step in remaining:
            if step["agent"] != agent:
                continue
            if step["tool"] in SOCIAL_TOOL_NAMES or step["tool"] in OBSERVATION_TOOL_NAMES:
                continue
            # Stepping off a fixture is not using it. Counting give_space here
            # made a correct release look like a broken promise, because the
            # holder yields the space right after handing it over.
            if step["tool"] in GIVE_SPACE_TOOL_NAMES:
                continue
            if any(str(value) == resource for value in (step.get("args") or {}).values()):
                return True
        return False

    def _validate_communicate_step(self, step: dict[str, Any]) -> None:
        """Validates the shared synthetic communication tool."""

        tool_args = step["args"]
        to_agent = tool_args.get("to")
        message = tool_args.get("message")
        if to_agent not in self._agent_id_set or to_agent == step["agent"]:
            raise CommunicationStepSemanticValidationError(
                "communicate requires to to reference the other agent.",
                details={"agent": step["agent"], "to": to_agent},
            )
        if not isinstance(message, str) or not " ".join(message.strip().split()):
            raise CommunicationStepSemanticValidationError(
                "communicate requires a non-empty message in args.",
                details={"agent": step["agent"], "to": to_agent},
            )

        if not set(tool_args) <= {"to", "message", "releases", "coordination_phase"}:
            raise CommunicationStepSemanticValidationError(
                "communicate args may only contain to, message, releases and coordination_phase.",
                details={"arg_names": sorted(tool_args)},
            )
        normalized_message = " ".join(message.strip().split())
        step["args"] = {
            "to": to_agent,
            "message": normalized_message,
        }
        phase = tool_args.get("coordination_phase")
        allowed_phases = {
            "propose", "await_plan", "await_confirmation", "confirm",
            "portion_complete",
        }
        if phase is not None:
            if phase not in allowed_phases:
                raise CommunicationStepSemanticValidationError(
                    f"Unsupported coordination_phase {phase!r}.",
                    details={"coordination_phase": phase},
                )
            step["args"]["coordination_phase"] = phase
        released = self._normalize_released_ids(step, tool_args.get("releases"))
        if released:
            step["args"]["releases"] = released

    def _normalize_released_ids(
        self,
        step: dict[str, Any],
        raw: Any,
    ) -> list[str]:
        """Normalizes communicate.releases into a list of known symbolic ids.

        `releases` is what actually discharges a wait. Keeping it structural,
        rather than reading the id out of the message text, means a release
        cannot be produced by phrasing alone -- the model would otherwise learn
        one canonical sentence and emit it whether or not it had finished.
        """

        if raw is None:
            return []
        values = [raw] if isinstance(raw, str) else raw
        if not isinstance(values, (list, tuple)) or not all(
            isinstance(value, str) for value in values
        ):
            raise CommunicationStepSemanticValidationError(
                "communicate releases must be a symbolic id or a list of them.",
                details={"agent": step["agent"], "releases": raw},
            )
        known = set(self.initial_state.get("objects") or {}) | set(
            self.initial_state.get("fixtures") or {}
        )
        released: list[str] = []
        for value in values:
            value = value.strip()
            if not value:
                continue
            if known and value not in known:
                raise CommunicationStepSemanticValidationError(
                    f"communicate releases={value!r} is not a symbolic id in "
                    f"this task; it must name a declared object or fixture.",
                    details={"agent": step["agent"], "releases": value,
                             "known_ids": sorted(known)},
                )
            if value not in released:
                released.append(value)
        return released

    def _validate_required_observation_sequence(
        self,
        steps: Sequence[dict[str, Any]],
        step_index: int,
    ) -> None:
        """Requires each non-communication action to be bracketed by observation steps."""

        step = steps[step_index]
        if step["tool"] in SOCIAL_TOOL_NAMES or step["tool"] in OBSERVATION_TOOL_NAMES:
            return

        if (
            step_index == 0
            or steps[step_index - 1]["tool"] not in OBSERVATION_TOOL_NAMES
        ):
            raise ObservationSequenceSemanticValidationError(
                f"{step['tool']} at step {step['step']} must be immediately preceded by an observation step.",
                details={
                    "tool": step["tool"],
                    "step": step["step"],
                    "position": "before",
                },
            )
        if (
            step_index + 1 >= len(steps)
            or steps[step_index + 1]["tool"] not in OBSERVATION_TOOL_NAMES
        ):
            raise ObservationSequenceSemanticValidationError(
                f"{step['tool']} at step {step['step']} must be immediately followed by an observation step.",
                details={
                    "tool": step["tool"],
                    "step": step["step"],
                    "position": "after",
                },
            )

    def _validate_generic_transition(
        self,
        step: dict[str, Any],
        runtime_state: TaskRuntimeState,
    ) -> None:
        """Enforces the shared manipulation and navigation rules for every task."""

        agent_state = runtime_state.agents[step["agent"]]
        tool_name = step["tool"]
        tool_args = step["args"]
        required_fixture = self.resolve_required_fixture(step, runtime_state)

        if tool_name in ACQUIRE_TOOL_NAMES:
            if agent_state.held_object is not None:
                raise HeldObjectSemanticValidationError(
                    f"{step['agent']} cannot pick up a second object while already holding {agent_state.held_object}.",
                    details={
                        "agent": step["agent"],
                        "held_object": agent_state.held_object,
                        "tool": tool_name,
                    },
                )
            if required_fixture is not None:
                self._require_agent_location(
                    step=step,
                    current_location=agent_state.location,
                    expected_location=required_fixture,
                )
            object_id = tool_args["object_id"]
            source_id = tool_args["source_id"]
            other_holders = [
                other_agent_id
                for other_agent_id, other_agent_state in runtime_state.agents.items()
                if other_agent_id != step["agent"]
                and other_agent_state.held_object == object_id
            ]
            if other_holders:
                raise HeldObjectSemanticValidationError(
                    f"{step['agent']} cannot pick up {object_id}; it is already "
                    f"held by {other_holders[0]}.",
                    details={
                        "agent": step["agent"],
                        "object_id": object_id,
                        "held_by": other_holders[0],
                        "tool": tool_name,
                    },
                )
            object_location = runtime_state.objects.get(object_id, {}).get("location")
            if not self._object_location_matches_source(
                object_location=object_location,
                source_id=source_id,
                runtime_state=runtime_state,
            ):
                raise ObjectStateSemanticValidationError(
                    f"pick_up_object requires {object_id} to start at {source_id}.",
                    details={
                        "object_id": object_id,
                        "expected_location": source_id,
                        "actual_location": object_location,
                    },
                )
            self._require_referenced_objects_not_held(step, runtime_state)
            return

        # Once an agent is holding something, the only shared safe actions are
        # moving to the next fixture or placing that same object down.
        if (
            agent_state.held_object is not None
            and tool_name not in NAVIGATION_TOOL_NAMES
            and tool_name not in RELEASE_TOOL_NAMES
            and tool_name not in OBSERVATION_TOOL_NAMES
            and tool_name not in GIVE_SPACE_TOOL_NAMES
        ):
            raise HeldObjectSemanticValidationError(
                f"{step['agent']} must place {agent_state.held_object} before using {tool_name}.",
                details={
                    "agent": step["agent"],
                    "held_object": agent_state.held_object,
                    "tool": tool_name,
                },
            )

        if tool_name in OBSERVATION_TOOL_NAMES:
            return

        if tool_name in GIVE_SPACE_TOOL_NAMES:
            if required_fixture is not None:
                self._require_agent_location(
                    step=step,
                    current_location=agent_state.location,
                    expected_location=required_fixture,
                )
                self._require_other_agent_at_fixture(
                    step=step,
                    runtime_state=runtime_state,
                    fixture_id=required_fixture,
                )
            return

        if tool_name in RELEASE_TOOL_NAMES:
            # Report a held movable receptacle as the real problem before
            # attempting to resolve its supporting fixture. Otherwise a bowl
            # held by the partner degraded into the impossible instruction
            # navigate_to_fixture(bowl).
            self._require_referenced_objects_not_held(step, runtime_state)
            required_fixture = self.resolve_required_fixture(step, runtime_state)
            if required_fixture is not None:
                alternative_locations: tuple[str, ...] = ()
                if tool_name == "place_next_to":
                    reference_fixture_id = tool_args.get("reference_fixture_id")
                    if isinstance(reference_fixture_id, str):
                        adjacent_location_id = self._resolve_reference_location(
                            reference_id=reference_fixture_id,
                            runtime_state=runtime_state,
                        )
                        if (
                            isinstance(adjacent_location_id, str)
                            and required_fixture
                            == self._resolve_fixture_for_location(
                                location_id=adjacent_location_id,
                                runtime_state=runtime_state,
                            )
                        ):
                            alternative_locations = (reference_fixture_id,)
                self._require_agent_location(
                    step=step,
                    current_location=agent_state.location,
                    expected_location=required_fixture,
                    alternative_locations=alternative_locations,
                )
            self._require_held_object(
                step["agent"],
                agent_state,
                tool_args["object_id"],
            )
            # Destination resolution is a precondition, not an effect. Running
            # it only while committing let malformed but globally exposed
            # placement calls pass the frozen tick check and then crash the
            # harness. Resolve without mutation here so live evaluation reports
            # an ordinary model rejection and commits only known-safe effects.
            self._resolve_release_location(step, runtime_state)
            return

        if tool_name in INTERACTION_TOOL_NAMES:
            if required_fixture is not None:
                self._require_agent_location(
                    step=step,
                    current_location=agent_state.location,
                    expected_location=required_fixture,
                )
            return

        if tool_name not in NAVIGATION_TOOL_NAMES and required_fixture is not None:
            self._require_agent_location(
                step=step,
                current_location=agent_state.location,
                expected_location=required_fixture,
            )

    def _is_allowed_observation_tool(self, tool_name: str) -> bool:
        """Accepts post-processed observation tools when a task supports observation steps."""

        return self._requires_observation_steps and tool_name in OBSERVATION_TOOL_NAMES

    def _validate_task_local_symbolic_constraints(self, step: dict[str, Any]) -> None:
        """Rejects symbolic IDs that violate the task-local allowed_* tool overrides."""

        if self._is_allowed_observation_tool(step["tool"]) and (
            step["tool"] not in self.allowed_tool_specs
        ):
            return

        tool_spec = self.allowed_tool_specs[step["tool"]]
        tool_args = step["args"]

        for arg_name in tool_spec.get("tool_args", ()):
            self._validate_task_local_tool_arg(
                step=step,
                tool_spec=tool_spec,
                arg_name=arg_name,
            )

        for arg_name in tool_spec.get("optional_tool_args", ()):
            if tool_args.get(arg_name) is None:
                continue
            self._validate_task_local_tool_arg(
                step=step,
                tool_spec=tool_spec,
                arg_name=arg_name,
            )

        for arg_group in tool_spec.get("tool_arg_any_of", ()):
            if not isinstance(arg_group, (list, tuple)) or len(arg_group) < 2:
                raise ValueError(
                    f"{self.composite_task} configured {step['tool']}.tool_arg_any_of "
                    "with an invalid alternative-arg group."
                )
            normalized_group = tuple(
                arg_name for arg_name in arg_group if isinstance(arg_name, str)
            )
            present_args = [
                arg_name
                for arg_name in normalized_group
                if tool_args.get(arg_name) is not None
            ]
            if len(present_args) != 1:
                raise ToolArgumentSemanticValidationError(
                    f"{step['tool']} requires exactly one of {list(normalized_group)}.",
                    details={
                        "tool": step["tool"],
                        "arg_group": list(normalized_group),
                        "present_args": present_args,
                    },
                )
            self._validate_task_local_tool_arg(
                step=step,
                tool_spec=tool_spec,
                arg_name=present_args[0],
            )

    def _validate_task_local_tool_arg(
        self,
        *,
        step: dict[str, Any],
        tool_spec: dict[str, Any],
        arg_name: str,
    ) -> None:
        """Validate one concrete step arg against shared tool metadata."""

        tool_args = step["args"]
        arg_value = tool_args.get(arg_name)
        arg_schema_type = _resolve_tool_arg_schema_type(arg_name, tool_spec)
        if arg_schema_type == "STRING":
            if not isinstance(arg_value, str) or not " ".join(arg_value.strip().split()):
                raise ToolArgumentSemanticValidationError(
                    f"{step['tool']} requires {arg_name} to be a non-empty string.",
                    details={
                        "tool": step["tool"],
                        "arg_name": arg_name,
                        "arg_value": arg_value,
                    },
                )
            tool_args[arg_name] = " ".join(arg_value.strip().split())
        elif arg_schema_type == "INTEGER":
            if not isinstance(arg_value, int) or isinstance(arg_value, bool):
                raise ToolArgumentSemanticValidationError(
                    f"{step['tool']} requires {arg_name} to be an integer.",
                    details={
                        "tool": step["tool"],
                        "arg_name": arg_name,
                        "arg_value": arg_value,
                    },
                )
        elif arg_schema_type == "STRING_ARRAY":
            if not isinstance(arg_value, list) or not arg_value:
                raise ToolArgumentSemanticValidationError(
                    f"{step['tool']} requires {arg_name} to be a non-empty string list.",
                    details={
                        "tool": step["tool"],
                        "arg_name": arg_name,
                        "arg_value": arg_value,
                    },
                )

            normalized_values: list[str] = []
            for list_value in arg_value:
                if not isinstance(list_value, str) or not " ".join(
                    list_value.strip().split()
                ):
                    raise ToolArgumentSemanticValidationError(
                        f"{step['tool']} requires {arg_name} to contain only non-empty strings.",
                        details={
                            "tool": step["tool"],
                            "arg_name": arg_name,
                            "arg_value": arg_value,
                        },
                    )
                normalized_values.append(" ".join(list_value.strip().split()))
            tool_args[arg_name] = normalized_values

        allowed_ids_key = _allowed_ids_key_for_arg_name(arg_name)
        if allowed_ids_key is None or allowed_ids_key not in tool_spec:
            return

        allowed_ids = tool_spec[allowed_ids_key]
        if not isinstance(allowed_ids, list) or not all(
            isinstance(allowed_id, str) for allowed_id in allowed_ids
        ):
            raise ValueError(
                f"{self.composite_task} configured {step['tool']}.{allowed_ids_key} "
                "with a non-string list."
            )

        if tool_args[arg_name] not in allowed_ids:
            raise ToolArgumentSemanticValidationError(
                f"{step['tool']} requires {arg_name} to be one of "
                f"{allowed_ids}, got {tool_args[arg_name]!r}.",
                details={
                    "tool": step["tool"],
                    "arg_name": arg_name,
                    "arg_value": tool_args[arg_name],
                    "allowed_values": list(allowed_ids),
                },
            )

    def _apply_generic_effects(
        self,
        step: dict[str, Any],
        runtime_state: TaskRuntimeState,
    ) -> None:
        """Applies the shared symbolic state changes for the current step."""

        agent_state = runtime_state.agents[step["agent"]]
        tool_name = step["tool"]
        tool_args = step["args"]

        if tool_name in NAVIGATION_TOOL_NAMES:
            agent_state.location = canonical_agent_workspace(
                self.initial_state, tool_args["fixture_id"]
            )
            return

        if tool_name in OBSERVATION_TOOL_NAMES:
            return

        if tool_name in GIVE_SPACE_TOOL_NAMES:
            agent_state.location = None
            return

        if tool_name in OPEN_PART_TOOL_NAMES:
            self._set_part_state(
                runtime_state=runtime_state,
                target_id=tool_args["target_id"],
                part_id=tool_args["part_id"],
                part_state=self._resolve_part_state_transition(
                    runtime_state=runtime_state,
                    target_id=tool_args["target_id"],
                    part_id=tool_args["part_id"],
                    opening=True,
                ),
            )
            return

        if tool_name in CLOSE_PART_TOOL_NAMES:
            self._set_part_state(
                runtime_state=runtime_state,
                target_id=tool_args["target_id"],
                part_id=tool_args["part_id"],
                part_state=self._resolve_part_state_transition(
                    runtime_state=runtime_state,
                    target_id=tool_args["target_id"],
                    part_id=tool_args["part_id"],
                    opening=False,
                ),
            )
            return

        if tool_name in INTERACTION_TOOL_NAMES:
            control_state: str
            if tool_name == "set_rotary_control":
                control_state = tool_args["goal"]
            elif tool_name == "press_button":
                current_state = (
                    runtime_state.fixtures.get(tool_args["target_id"], {})
                    .get("controls", {})
                    .get(tool_args["control_id"], {})
                    .get("state")
                )
                control_state = "on" if current_state == "off" else "pressed"
            else:
                current_state = (
                    runtime_state.fixtures.get(tool_args["target_id"], {})
                    .get("controls", {})
                    .get(tool_args["control_id"], {})
                    .get("state")
                )
                control_state = "down" if current_state == "up" else "pressed"
            self._set_control_state(
                runtime_state=runtime_state,
                target_id=tool_args["target_id"],
                control_id=tool_args["control_id"],
                control_state=control_state,
            )
            return

        if tool_name in ACQUIRE_TOOL_NAMES:
            object_id = tool_args["object_id"]
            agent_state.held_object = object_id
            runtime_state.objects.setdefault(object_id, {})[
                "location"
            ] = f"held_by_{step['agent']}"
            return

        if tool_name in RELEASE_TOOL_NAMES:
            object_id = tool_args["object_id"]
            agent_state.held_object = None
            runtime_state.objects.setdefault(object_id, {})["location"] = (
                self._resolve_release_location(step, runtime_state)
            )

    def _resolve_release_location(
        self,
        step: dict[str, Any],
        runtime_state: TaskRuntimeState,
    ) -> str:
        """Resolves where a released object should live after placement."""

        tool_args = step["args"]
        target_site_id = tool_args.get("target_site_id")
        if (
            step["tool"] in {"place_in_receptacle", "place_on_surface", "place_under"}
            and isinstance(target_site_id, str)
        ):
            return target_site_id

        for arg_name in PLACE_LOCATION_ARG_NAMES:
            location_id = tool_args.get(arg_name)
            if isinstance(location_id, str):
                return location_id

        reference_id = tool_args.get("reference_id")
        if isinstance(reference_id, str):
            reference_location = self._resolve_reference_location(
                reference_id=reference_id,
                runtime_state=runtime_state,
            )
            if isinstance(reference_location, str):
                return reference_location
            if reference_id in runtime_state.fixtures:
                return reference_id

        reference_object_id = tool_args.get("reference_object_id")
        if isinstance(reference_object_id, str):
            reference_location = self._resolve_reference_location(
                reference_id=reference_object_id,
                runtime_state=runtime_state,
            )
            if isinstance(reference_location, str):
                return reference_location
            raise PlacementDestinationSemanticValidationError(
                f"{step['tool']} requires {reference_object_id} to have a known symbolic location.",
                details={
                    "tool": step["tool"],
                    "reference_object_id": reference_object_id,
                    "reference_location": reference_location,
                },
            )

        reference_fixture_id = tool_args.get("reference_fixture_id")
        if isinstance(reference_fixture_id, str):
            if step["tool"] == "place_next_to":
                adjacent_location_id = self._resolve_reference_location(
                    reference_id=reference_fixture_id,
                    runtime_state=runtime_state,
                )
                if isinstance(adjacent_location_id, str):
                    return adjacent_location_id
                raise PlacementDestinationSemanticValidationError(
                    f"{step['tool']} requires {reference_fixture_id} to expose an adjacent symbolic support location.",
                    details={
                        "tool": step["tool"],
                        "reference_fixture_id": reference_fixture_id,
                        "reference_location": adjacent_location_id,
                    },
                )
            inferred_site_id = self._infer_fixture_release_site(
                fixture_id=reference_fixture_id,
                runtime_state=runtime_state,
                tool_name=step["tool"],
            )
            if isinstance(inferred_site_id, str):
                return inferred_site_id
            return reference_fixture_id

        raise PlacementDestinationSemanticValidationError(
            "Placement tools must include a symbolic destination."
        )

    def _infer_fixture_release_site(
        self,
        *,
        fixture_id: str,
        runtime_state: TaskRuntimeState,
        tool_name: str,
    ) -> str | None:
        """Infer a symbolic support site when the placement tool implies one."""

        fixture_machine_state = runtime_state.machine_state.get(fixture_id, {})
        if isinstance(fixture_machine_state, dict):
            dispenser_id = fixture_machine_state.get("dispenser_id")
            if isinstance(dispenser_id, str):
                return dispenser_id

        fixture_state = runtime_state.fixtures.get(fixture_id, {})
        if not isinstance(fixture_state, dict):
            return None
        support_sites = fixture_state.get("support_sites", {})
        if isinstance(support_sites, dict):
            support_site_ids = [
                support_site_id
                for support_site_id in support_sites
                if isinstance(support_site_id, str)
            ]
        elif isinstance(support_sites, list):
            support_site_ids = [
                support_site_id
                for support_site_id in support_sites
                if isinstance(support_site_id, str)
            ]
        else:
            support_site_ids = []

        preferred_tokens_by_tool = {
            "place_under": ("dispenser", "basin"),
        }
        preferred_tokens = preferred_tokens_by_tool.get(tool_name, ())
        for support_site_id in support_site_ids:
            lowered_support_site_id = support_site_id.lower()
            if any(token in lowered_support_site_id for token in preferred_tokens):
                return support_site_id

        if len(support_site_ids) == 1:
            return support_site_ids[0]
        return None

    def _resolve_reference_location(
        self,
        *,
        reference_id: str,
        runtime_state: TaskRuntimeState,
    ) -> str | None:
        """Resolves the symbolic placement location for an object or fixture reference."""

        current_reference_id = reference_id
        seen_reference_ids: set[str] = set()

        # Some placements are specified relative to an object that sits on top
        # of another object (for example cake -> plate -> counter). Walk up the
        # object location chain until we reach a non-object symbolic location.
        while current_reference_id in runtime_state.objects:
            if current_reference_id in seen_reference_ids:
                return None
            seen_reference_ids.add(current_reference_id)

            reference_location = runtime_state.objects.get(current_reference_id, {}).get(
                "location"
            )
            if not isinstance(reference_location, str):
                return None
            if reference_location in runtime_state.objects:
                current_reference_id = reference_location
                continue
            return reference_location

        fixture_machine_state = runtime_state.machine_state.get(current_reference_id, {})
        if not isinstance(fixture_machine_state, dict):
            return None

        # Some tasks anchor fixture-relative placements to a named nearby surface.
        adjacent_location_id = fixture_machine_state.get("adjacent_location_id")
        if isinstance(adjacent_location_id, str):
            return adjacent_location_id
        return None

    def _set_part_state(
        self,
        *,
        runtime_state: TaskRuntimeState,
        target_id: str,
        part_id: str,
        part_state: str,
    ) -> None:
        """Updates the symbolic open or closed state for a fixture part."""

        target_state = runtime_state.fixtures.setdefault(target_id, {})
        parts_state = target_state.setdefault("parts", {})
        part_entry = parts_state.setdefault(part_id, {})
        part_entry["state"] = part_state

    def _set_control_state(
        self,
        *,
        runtime_state: TaskRuntimeState,
        target_id: str,
        control_id: str,
        control_state: str,
    ) -> None:
        """Updates the symbolic state for a fixture control."""

        target_state = runtime_state.fixtures.setdefault(target_id, {})
        controls_state = target_state.setdefault("controls", {})
        control_entry = controls_state.setdefault(control_id, {})
        control_entry["state"] = control_state

    def _resolve_part_state_transition(
        self,
        *,
        runtime_state: TaskRuntimeState,
        target_id: str,
        part_id: str,
        opening: bool,
    ) -> str:
        """Map open or close actions onto the symbolic state vocabulary for that part."""

        return "open" if opening else "closed"

    def _require_agent_location(
        self,
        *,
        step: dict[str, Any],
        current_location: str | None,
        expected_location: str,
        alternative_locations: tuple[str, ...] = (),
    ) -> None:
        """Checks that an agent navigated to the fixture before interacting there."""

        accepted_locations = tuple(dict.fromkeys(
            location
            for candidate in (expected_location, *alternative_locations)
            for location in accepted_agent_workspaces(
                self.initial_state,
                candidate,
                exclusive_fixture_types=EXCLUSIVE_FIXTURE_TYPES,
            )
        ))
        if current_location not in accepted_locations:
            step_number = step.get("step")
            step_prefix = (
                f"Step {step_number} ({step['tool']}): "
                if isinstance(step_number, int)
                else ""
            )
            current_location_label = current_location or "unknown location"
            destination_label = (
                expected_location
                if len(accepted_locations) == 1
                else "one of " + ", ".join(accepted_locations)
            )
            raise NavigationSemanticValidationError(
                f"{step_prefix}{step['agent']} is at {current_location_label} and must "
                f"use navigate_to_fixture to reach {destination_label} before using "
                f"{step['tool']}.",
                step=step_number if isinstance(step_number, int) else None,
                details={
                    "agent": step["agent"],
                    "tool": step["tool"],
                    "actual_ids": sorted(
                        str(value)
                        for value in (step.get("args") or {}).values()
                        if isinstance(value, str)
                    ),
                    "current_location": current_location,
                    "expected_location": expected_location,
                    "accepted_locations": list(accepted_locations),
                    "initial_location": (
                        (self.initial_state.get("agents") or {})
                        .get(step["agent"], {})
                        .get("location")
                    ),
                },
            )

    def _require_other_agent_at_fixture(
        self,
        *,
        step: dict[str, Any],
        runtime_state: TaskRuntimeState,
        fixture_id: str,
    ) -> None:
        """Checks that give_space is only used for an occupied or coordinated shared fixture."""

        other_agents_at_fixture = sorted(
            agent_id
            for agent_id, agent_state in runtime_state.agents.items()
            if agent_id != step["agent"]
            and canonical_agent_workspace(self.initial_state, agent_state.location)
            == canonical_agent_workspace(self.initial_state, fixture_id)
        )
        if other_agents_at_fixture:
            return

        # Allow the yielding agent to clear a fixture before the incoming agent arrives
        # once both agents have already coordinated through communicate.
        if runtime_state.communicated_agents == self._agent_id_set:
            return

        step_number = step.get("step")
        step_prefix = (
            f"Step {step_number} ({step['tool']}): "
            if isinstance(step_number, int)
            else ""
        )
        raise TaskPreconditionSemanticValidationError(
            f"{step_prefix}{step['agent']} can use give_space at {fixture_id} only "
            "when the agents have already coordinated and the fixture needs to be cleared.",
            step=step_number if isinstance(step_number, int) else None,
            details={
                "agent": step["agent"],
                "fixture_id": fixture_id,
                "other_agent_locations": {
                    agent_id: agent_state.location
                    for agent_id, agent_state in runtime_state.agents.items()
                    if agent_id != step["agent"]
                },
            },
        )

    def _require_held_object(
        self,
        agent_id: str,
        agent_state: AgentRuntimeState,
        object_id: str,
    ) -> None:
        """Checks that the acting agent is holding the required object."""

        if agent_state.held_object != object_id:
            raise HeldObjectSemanticValidationError(
                f"{agent_id} must be holding {object_id} before placing it.",
                details={
                    "agent": agent_id,
                    "expected_object": object_id,
                    "held_object": agent_state.held_object,
                },
            )

    def _iter_referenced_object_args(
        self,
        step: dict[str, Any],
        runtime_state: TaskRuntimeState,
    ) -> list[tuple[str, str]]:
        """Collect object-valued anchor args used by the current step."""

        referenced_args: list[tuple[str, str]] = []
        tool_args = step["args"]
        for arg_name in (
            "source_id",
            "target_id",
            "support_object_id",
            "receptacle_id",
            "reference_object_id",
            "reference_id",
        ):
            arg_value = tool_args.get(arg_name)
            if isinstance(arg_value, str) and arg_value in runtime_state.objects:
                referenced_args.append((arg_name, arg_value))
        return referenced_args

    def _require_referenced_objects_not_held(
        self,
        step: dict[str, Any],
        runtime_state: TaskRuntimeState,
    ) -> None:
        """Reject object-relative actions that target a held movable object."""

        for arg_name, object_id in self._iter_referenced_object_args(
            step,
            runtime_state,
        ):
            object_location = runtime_state.objects.get(object_id, {}).get("location")
            if (
                not isinstance(object_location, str)
                or not object_location.startswith("held_by_")
            ):
                continue
            holder_agent_id = object_location.removeprefix("held_by_")
            raise TaskPreconditionSemanticValidationError(
                f"{step['tool']} cannot use {arg_name}={object_id} because "
                f"{object_id} is currently held by {holder_agent_id}.",
                details={
                    "agent": step["agent"],
                    "tool": step["tool"],
                    "arg_name": arg_name,
                    "arg_value": object_id,
                    "holder_agent": holder_agent_id,
                },
            )

    def resolve_required_fixture(
        self,
        step: dict[str, Any],
        runtime_state: TaskRuntimeState,
    ) -> str | None:
        """Resolves which fixture an agent must already be at for a step."""

        tool_args = step["args"]
        for arg_name in (
            "fixture_id",
            "target_id",
            "source_id",
            "reference_id",
            "support_id",
            "receptacle_id",
            "reference_fixture_id",
            "support_object_id",
            "reference_object_id",
        ):
            value = tool_args.get(arg_name)
            if not isinstance(value, str):
                continue
            if arg_name == "fixture_id":
                return value
            if arg_name == "reference_fixture_id":
                if step["tool"] == "place_next_to":
                    reference_location = self._resolve_reference_location(
                        reference_id=value,
                        runtime_state=runtime_state,
                    )
                    resolved_fixture_id = self._resolve_fixture_for_location(
                        location_id=reference_location,
                        runtime_state=runtime_state,
                    )
                    if isinstance(resolved_fixture_id, str):
                        return resolved_fixture_id
                return value
            if value in runtime_state.fixtures and arg_name in {
                "target_id",
                "source_id",
                "support_id",
                "receptacle_id",
            }:
                return value
            # If the value is an object, resolve to the fixture it sits on.
            resolved = self._resolve_reference_location(
                reference_id=value,
                runtime_state=runtime_state,
            )
            resolved_fixture_id = self._resolve_fixture_for_location(
                location_id=resolved,
                runtime_state=runtime_state,
            )
            if isinstance(resolved_fixture_id, str):
                return resolved_fixture_id
            enclosing_fixture_id = self._resolve_fixture_for_location(
                location_id=value,
                runtime_state=runtime_state,
            )
            if isinstance(enclosing_fixture_id, str):
                return enclosing_fixture_id
            # Otherwise treat it as a fixture ID directly.
            return value
        return None

    def _resolve_fixture_for_location(
        self,
        *,
        location_id: str | None,
        runtime_state: TaskRuntimeState,
    ) -> str | None:
        """Map an object location token back to the fixture an agent must stand at."""

        if not isinstance(location_id, str):
            return None
        if location_id.startswith("held_by_"):
            holder_agent_id = location_id.removeprefix("held_by_")
            holder_state = runtime_state.agents.get(holder_agent_id)
            holder_location = (
                holder_state.location
                if holder_state is not None
                else None
            )
            if isinstance(holder_location, str):
                return holder_location
            return None
        if location_id in runtime_state.fixtures:
            return location_id
        enclosing_fixture_id = self._resolve_enclosing_fixture_id(
            reference_id=location_id,
            runtime_state=runtime_state,
        )
        if isinstance(enclosing_fixture_id, str):
            return enclosing_fixture_id
        return location_id

    def _object_location_matches_source(
        self,
        *,
        object_location: Any,
        source_id: str,
        runtime_state: TaskRuntimeState,
    ) -> bool:
        if object_location == source_id:
            return True
        if not isinstance(object_location, str):
            return False
        # A held object is not present at the fixture where its holder happens
        # to stand.  Treating held_by_agent_1 as that agent's counter location
        # allowed another agent to acquire the same object simultaneously.
        if object_location.startswith("held_by_"):
            return False

        location_fixture_id = self._resolve_fixture_for_location(
            location_id=object_location,
            runtime_state=runtime_state,
        )
        source_fixture_id = self._resolve_fixture_for_location(
            location_id=source_id,
            runtime_state=runtime_state,
        )
        if location_fixture_id == source_id:
            return True
        if source_fixture_id == object_location:
            return True
        return (
            isinstance(location_fixture_id, str)
            and isinstance(source_fixture_id, str)
            and location_fixture_id == source_fixture_id
        )

    def _resolve_enclosing_fixture_id(
        self,
        *,
        reference_id: str,
        runtime_state: TaskRuntimeState,
    ) -> str | None:
        """Resolve a sub-entity like a drawer part back to its containing fixture."""

        matching_fixture_ids: list[str] = []
        for fixture_id, fixture_state in runtime_state.fixtures.items():
            if not isinstance(fixture_state, dict):
                continue
            fixture_parts = fixture_state.get("parts", {})
            if isinstance(fixture_parts, dict) and reference_id in fixture_parts:
                matching_fixture_ids.append(fixture_id)
            fixture_controls = fixture_state.get("controls", {})
            if isinstance(fixture_controls, dict) and reference_id in fixture_controls:
                matching_fixture_ids.append(fixture_id)
            fixture_support_sites = fixture_state.get("support_sites", {})
            if (
                isinstance(fixture_support_sites, dict)
                and reference_id in fixture_support_sites
            ) or (
                isinstance(fixture_support_sites, list)
                and reference_id in fixture_support_sites
            ):
                matching_fixture_ids.append(fixture_id)
            if isinstance(fixture_support_sites, dict):
                if any(
                    isinstance(support_site_id, str)
                    and _support_site_alias_matches(reference_id, support_site_id)
                    for support_site_id in fixture_support_sites
                ):
                    matching_fixture_ids.append(fixture_id)
            elif isinstance(fixture_support_sites, list):
                if any(
                    isinstance(support_site_id, str)
                    and _support_site_alias_matches(reference_id, support_site_id)
                    for support_site_id in fixture_support_sites
                ):
                    matching_fixture_ids.append(fixture_id)

        for fixture_id, fixture_machine_state in runtime_state.machine_state.items():
            if not isinstance(fixture_machine_state, dict):
                continue
            if fixture_machine_state.get("dispenser_id") == reference_id:
                matching_fixture_ids.append(fixture_id)
            if fixture_machine_state.get("adjacent_location_id") == reference_id:
                matching_fixture_ids.append(fixture_id)

        deduped_fixture_ids = tuple(dict.fromkeys(matching_fixture_ids))
        if len(deduped_fixture_ids) == 1:
            return deduped_fixture_ids[0]
        return None

    def validate_task_preconditions(
        self,
        step: dict[str, Any],
        runtime_state: TaskRuntimeState,
    ) -> None:
        """Allows task validators to add small task-local preconditions."""

        _ = (step, runtime_state)

    def apply_task_effects(
        self,
        step: dict[str, Any],
        runtime_state: TaskRuntimeState,
    ) -> None:
        """Allows task validators to update task-local symbolic state."""

        _ = (step, runtime_state)

    def is_goal_state_satisfied(self, runtime_state: TaskRuntimeState) -> bool:
        """Checks whether the replayed symbolic state satisfies the task goal."""

        raise NotImplementedError(
            "Task validators must implement is_goal_state_satisfied()."
        )
