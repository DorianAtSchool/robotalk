"""Closed-loop (live-sim) evaluation of tool-calling VLMs.

The model controls both agents live in the MuJoCo sim. Success is judged by
the task env's own ``_check_success`` plus the symbolic FSM goal checker, and
visuals are rendered from current sim state.

Backends:
  oracle      replay the expert trajectory's own steps (verification gate:
              expect ~100% success; no model involved)
  vllm        a localhost vLLM OpenAI-compatible server
  gemini      Vertex Gemini with native function calling

Most runs go through ``robotalk.evaluation.live_sim_parallel_eval`` (or
``scripts/reproduce.py``). Example (oracle smoke on two trajectories):
  python -m robotalk.evaluation.live_sim_eval --backend oracle \
    --manifest configs/eval/fixed_live_sim_cohort.json \
    --cohort-split heldout_task_types \
    --dataset-root data/robotalk_rendered \
    --output-dir outputs/eval/oracle_smoke \
    --max-trajectories 2
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
import shutil
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeoutError
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import random
import re
from typing import Any
from urllib import error as urllib_error, request as urllib_request

from robotalk.training.prompting import (
    PROMPT_CONTRACT_VERSION,
    build_messages,
    build_partial_user_prompt,
)
from robotalk.training.communication_profiles import (
    COMMUNICATION_MODES,
    apply_communication_profile,
    get_communication_profile,
)
from robotalk.training.schema_utils import (
    augment_tool_specs_with_get_image,
    apply_eval_fixture_aliases,
)
from robotalk.training.task_registry import AGENT_IDS, get_task_metadata
from robotalk.training.tool_calling import (
    build_tool_schemas,
    parse_first_qwen_tool_call,
    tool_call_to_single_step_payload,
)
from robotalk.tools.subatomic_tool_specs import build_model_tool_specs
from robotalk.tasks.shared.concurrent_fsm import (
    contention_resources,
    is_exclusive_fixture,
    simultaneous_contentions,
)
from robotalk.tasks.shared.workspace_semantics import (
    canonical_agent_workspace,
)

# Expert view-selection rule, measured over 162 trajectories (~3,300 calls):
# views deterministically match the NEXT action (scout set before navigate,
# wrist set before manipulation, overhead at the episode opening). The next
# action is unknown before generation, so the first pass renders the overhead
# set and the second pass re-renders the canonical set for the predicted
# action (see --two-pass-views).
OVERHEAD_VIEWS = ("top_view", "room_view", "map")
SCOUT_VIEWS = ("agentview_center", "agentview_left", "agentview_right")
WRIST_VIEWS = ("wrist", "agentview_center")
# After the two required opening communications, the routing pass must not use
# a view family that was paired exclusively with one action family during SFT.
# Every overhead-only exp52 example targets ``communicate``. A fixed mixed
# bundle gives the router global and agent-centric evidence without conditioning
# its prediction on the eventual tool choice. Canonical refinement remains
# in-distribution for the selected tool.
ROUTER_VIEWS = ("top_view", "wrist", "agentview_center")
OPENING_COMMUNICATION_STEP_INDICES = frozenset({2, 3})
NAVIGATE_TOOLS = {"navigate_to_fixture"}
_NUMBERED_OBJECT_RE = re.compile(r"^obj_(\d+)$")


def _check_pruned_organize_condiments_success(
    env,
    *,
    object_utils=None,
) -> bool:
    """Evaluate OrganizeCondiments when trajectory pruning omitted distractors.

    The upstream task's native checker unconditionally indexes ``distractor``.
    Generated trajectories intentionally omit that non-goal object, so the
    trajectory-compatible native condition retains the three condiment and
    gripper-clearance requirements while omitting only the impossible
    distractor-on-counter and distractor-clearance clauses.
    """

    required_objects = ("condiment1", "condiment2", "condiment3")
    objects = getattr(env, "objects", {})
    missing = [name for name in required_objects if name not in objects]
    if missing:
        raise KeyError(
            "missing goal-relevant OrganizeCondiments object(s): "
            + ", ".join(missing)
        )
    if object_utils is None:
        import robocasa.utils.object_utils as object_utils

    return all(
        object_utils.obj_inside_of(env, name, env.cab)
        for name in required_objects
    ) and all(
        object_utils.gripper_obj_far(env, name)
        for name in required_objects
    )


# ---------------------------------------------------------------------------
# Partial-observability runtime: per-agent private state + concurrent scheduler
# ---------------------------------------------------------------------------

WAIT_TOOL_NAME = "wait_for_signal"
REJECTION_MODE_REPORT_FAILED = "report-failed"
REJECTION_MODE_TERMINATE_FIRST = "terminate-first-rejection"


def _prioritize_incumbent_fixture_users(
    agent_ids: list[str],
    proposal_by_agent: dict[str, dict[str, Any]],
    runtime_state: Any,
    initial_state: dict[str, Any],
) -> list[str]:
    """Put valid users already at an exclusive fixture before new entrants.

    Pairwise contention arbitration is order-sensitive by design: the first
    compatible proposal wins.  Alphabetical ordering can therefore reject an
    incumbent's manipulation when a lower-numbered agent simultaneously tries
    to enter the incumbent's workspace.  The occupancy gate later rejects the
    entrant too, producing a false reject-both livelock.  Stable sorting keeps
    the configured tie-break order within each class while ensuring the
    beginning-of-tick occupant gets first claim on its existing workspace.
    """

    agents = getattr(runtime_state, "agents", {})

    def is_incumbent(agent_id: str) -> bool:
        agent_state = agents.get(agent_id)
        if agent_state is None:
            return False
        current_workspace = canonical_agent_workspace(
            initial_state, getattr(agent_state, "location", None)
        )
        for resource in contention_resources(
            proposal_by_agent[agent_id], initial_state
        ):
            if not is_exclusive_fixture(resource, initial_state):
                continue
            if current_workspace == canonical_agent_workspace(
                initial_state, resource
            ):
                return True
        return False

    return sorted(agent_ids, key=lambda agent_id: not is_incumbent(agent_id))


def _sim_clock(session: "SimSession") -> float | None:
    """MuJoCo simulation time in seconds, or None if unavailable.

    ToolResult carries no timing, so the only real measure of how long a tool
    took is the simulator's own clock across the call.
    """

    try:
        return float(session.executor.env.sim.data.time)
    except Exception:
        return None


# Schedule-robustness probe. Because the executor teleports, measured sim-steps
# are ~0 and the floors below ARE the duration model -- so --duration-multiplier
# cannot perturb a schedule, and a null result from varying it means nothing.
# Jitter scales each completed call instead, which is the only knob that changes
# who wins a race. Off by default; a run that sets it is asking whether the
# plans survive a different interleaving, not measuring their quality.
_DURATION_JITTER = 0.0
_DURATION_RNG: random.Random | None = None
# Lock-step. Every call costs one tick, so both agents advance together and the
# ONLY thing that can order them is a wait and its release -- no timing is left
# to rescue a plan that forgot one. This REMOVES the timing dimension rather
# than testing it: there is exactly one schedule, so a plan that works is
# correct rather than lucky, and wait necessity becomes decidable by ablation.
# Two costs, both deliberate: a communicate costs the same tick as a navigate,
# so makespan degenerates to step count; and intra-tick order still exists (two
# sequential executor calls), so the contention model stays load-bearing.
_UNIFORM_DURATIONS = False


def _tool_duration(
    tool_name: str, *, sim_steps: float | None, multiplier: float
) -> float:
    """Virtual-clock duration for a completed call.

    Measured executor sim-steps scaled by `multiplier`, with per-class floors.
    The floors matter because the executor teleports: if measured costs come out
    uniform, equal durations from a common start would keep both agents in
    lockstep forever, which is precisely the artificial regime the virtual clock
    exists to avoid -- unless lock-step is what you asked for.
    """

    if _UNIFORM_DURATIONS:
        return 1.0
    if tool_name in ("communicate", "get_image"):
        floor = 0.25
    elif tool_name == "navigate_to_fixture":
        floor = 4.0
    else:
        floor = 2.0
    measured = 0.0 if not sim_steps else float(sim_steps) * multiplier
    duration = max(floor, measured)
    if _DURATION_JITTER and _DURATION_RNG is not None:
        duration *= _DURATION_RNG.uniform(
            max(0.05, 1.0 - _DURATION_JITTER), 1.0 + _DURATION_JITTER
        )
    return duration


def observation_ready_at(
    clock: float,
    *,
    consecutive_obs: int,
    max_free_observations: int,
    multiplier: float,
) -> tuple[float, int, bool]:
    """(ready_at, new consecutive count, capped) for one completed get_image.

    Observation is instrumentation, not work, so it must not move the agent's
    clock: an agent's clock advances per call IT makes, so charging cameras made
    the agent that acts more also drift ahead, and releases then landed before
    their waiters blocked. That rejected 47 of 1550 corpus trajectories in the
    concurrent validator, where the charge was removed -- the executor has to
    agree with the gate that certified the plan.

    Staying at `clock` re-admits the agent at this instant so it can emit its
    real action. The cap exists because a model that emits nothing but get_image
    would otherwise freeze the clock forever and starve its partner; past the
    cap the observation is charged normally.
    """

    if max_free_observations <= 0:
        return (
            clock + _tool_duration("get_image", sim_steps=None, multiplier=multiplier),
            0,
            False,
        )
    consecutive_obs += 1
    if consecutive_obs > max_free_observations:
        charged = clock + _tool_duration(
            "get_image", sim_steps=None, multiplier=multiplier
        )
        return charged, 0, True
    return clock, consecutive_obs, False


_LENIENT_WAIT_DISCHARGE = False


class AgentRuntime:
    """Private state for one logical agent under partial observability."""

    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self.private_history: list[dict[str, Any]] = []
        self.pending_obs: tuple[list[str], list[str]] | None = None
        self.ready_at: float = 0.0
        self.local_turn = 0
        self.silent_retries = 0
        self.active_s = 0.0
        self.last_proposal_key: str | None = None
        self.repeated_proposals = 0
        # wait_for_signal state. `waiting_for` records the DECLARED intent
        # ({"from","about"}). A wait is discharged by a message that names the
        # same id in `releases` -- not by any message at all. The old rule woke
        # a waiter on anything the partner said, which is how traj_000004 of
        # arrange_bread_bowl resumed on a message about something else and then
        # collided over the bowl. `--lenient-wait-discharge` restores it for
        # evaluating checkpoints trained before `releases` existed.
        self.waiting_for: dict[str, Any] | None = None
        self.consecutive_waits = 0
        # Observations are free on the virtual clock (see the get_image branch
        # in the concurrent loop), so an agent that only ever photographs would
        # sit at one instant forever. This counts observations taken without an
        # intervening non-observation call; the cap charges the clock instead.
        self.consecutive_obs = 0

    def deliver(self, step: dict[str, Any], clock: float | None = None) -> None:
        """Appends to this agent's private history (own act or delivered msg)."""

        self.private_history.append(deepcopy(step))

    def take_observation(self) -> tuple[list[str], list[str]]:
        """Consume-once: hand over the pending observation and clear it."""

        if self.pending_obs is None:
            return [], []
        image_paths, view_names = self.pending_obs
        self.pending_obs = None
        return list(image_paths), list(view_names)


def _proposal_key(step: dict[str, Any]) -> str:
    return json.dumps(
        {"tool": step.get("tool"), "args": step.get("args") or {}}, sort_keys=True
    )


def _budget_reference_action_count(trajectory: dict[str, Any]) -> int:
    """Return a nonzero action-length reference without requiring an expert carrier."""

    expert_count = sum(
        step.get("tool") != "get_image" for step in trajectory.get("steps", [])
    )
    if expert_count:
        return expert_count
    configured = trajectory.get("evaluation_reference_action_count")
    if isinstance(configured, int) and configured > 0:
        return configured
    return 16


# ---------------------------------------------------------------------------
# FSM mirror: incremental legality + goal checking over symbolic state
# ---------------------------------------------------------------------------


def _canonical_symbolic_initial_state(*, task_spec, trajectory: dict[str, Any]):
    """Normalize legacy trajectory symbols without introducing simulator IDs."""

    initial_state = deepcopy(trajectory.get("initial_state") or task_spec.initial_state)
    source_grounding = (
        (trajectory.get("grounding_map") or {}).get("symbols") or {}
    )
    target_grounding = (task_spec.grounding or {}).get("symbols") or {}
    legacy_aliases = (task_spec.grounding or {}).get("legacy_symbol_aliases") or {}
    symbol_map: dict[str, str] = {}

    def type_values(
        *,
        state_entry: dict[str, Any],
        grounding_entry: dict[str, Any],
        entity_type: str,
    ) -> set[str]:
        type_key = "object_type" if entity_type == "object" else "fixture_type"
        values = {
            str(value)
            for value in (state_entry.get(type_key), grounding_entry.get(type_key))
            if value
        }
        if entity_type == "fixture":
            values.update(
                str(value)
                for value in grounding_entry.get("preferred_fixture_types", ())
                if value
            )
        return values

    for state_key, entity_type in (("objects", "object"), ("fixtures", "fixture")):
        source_entities = initial_state.get(state_key) or {}
        target_entities = task_spec.initial_state.get(state_key) or {}
        used_targets: set[str] = set()

        for source_symbol in source_entities:
            target_symbol = source_symbol
            if target_symbol not in target_entities:
                target_symbol = legacy_aliases.get(source_symbol)
            if target_symbol in target_entities:
                symbol_map[source_symbol] = target_symbol
                used_targets.add(target_symbol)

        for source_symbol, source_entry in source_entities.items():
            if source_symbol in symbol_map:
                continue
            source_descriptor = source_grounding.get(source_symbol) or {}
            source_types = type_values(
                state_entry=source_entry,
                grounding_entry=source_descriptor,
                entity_type=entity_type,
            )
            candidates = []
            for target_symbol, target_entry in target_entities.items():
                if target_symbol in used_targets:
                    continue
                target_descriptor = target_grounding.get(target_symbol) or {}
                target_types = type_values(
                    state_entry=target_entry,
                    grounding_entry=target_descriptor,
                    entity_type=entity_type,
                )
                if source_types & target_types:
                    candidates.append(target_symbol)

            source_index = source_descriptor.get("index")
            if len(candidates) > 1 and source_index is not None:
                indexed = [
                    candidate
                    for candidate in candidates
                    if (target_grounding.get(candidate) or {}).get("index")
                    == source_index
                ]
                if len(indexed) == 1:
                    candidates = indexed
            if len(candidates) == 1:
                symbol_map[source_symbol] = candidates[0]
                used_targets.add(candidates[0])

    def rewrite(value):
        if isinstance(value, dict):
            rewritten = {
                symbol_map.get(key, key): rewrite(item)
                for key, item in value.items()
            }
            if len(rewritten) != len(value):
                raise ValueError("symbol normalization produced duplicate IDs")
            return rewritten
        if isinstance(value, list):
            return [rewrite(item) for item in value]
        if isinstance(value, str):
            return symbol_map.get(value, value)
        return value

    # The SAME symbol map must be applied to the trajectory's steps, not only
    # to initial_state. original_trajectory.json (what both training and
    # live-sim read) may name fixtures by role -- hot_dog_setup uses
    # `sausage_source_fixture`, `serving_surface`, `bun_source_fixture` -- and
    # leaving those unrewritten made every step referencing them illegal, since
    # the task's allowlist is expressed in canonical names.
    # Apply the SAME symbol map to the trajectory's steps, not only to
    # initial_state. original_trajectory.json (read by BOTH training and
    # live-sim) may name entities in a vocabulary the task allowlist does not
    # use -- hot_dog_setup says `bun` and `bun_source_fixture` where the spec
    # says `hotdog_bun` and `counter` -- and leaving steps unrewritten made
    # every such step illegal.
    #
    # Caveat worth knowing: this assumes the spec name is also resolvable in
    # the simulator. Where it is not, the call becomes legal-but-unexecutable
    # rather than rejected. prepare_cheese_station is the one known case
    # (spec `lettuce_bowl` vs scene `salad_bowl`); that is a spec/data naming
    # defect to reconcile, not a reason to skip the rewrite.
    trajectory["steps"] = rewrite(trajectory.get("steps") or [])
    # The TrajectoryAdapter is built from the ORIGINAL trajectory, so its alias
    # map keys on the pre-canonical names (bun_source_fixture). Rewriting steps
    # in place afterwards leaves the adapter unable to resolve the new names,
    # which is why a pickup worked only when a navigate happened to bind the
    # fixture first (6/6 with a prior navigate passed, 4/4 without failed).
    # Stash the map so callers can register the canonical aliases too.
    trajectory["_canonical_symbol_map"] = dict(symbol_map)
    return rewrite(initial_state)


def _register_canonical_fixture_aliases(adapter, trajectory: dict[str, Any]) -> None:
    """Teaches the adapter the canonical names the steps were rewritten to.

    Without this, `counter` (canonical) has no alias while `bun_source_fixture`
    (original) does, and the executor raises
    `Unknown support fixture/site 'counter'` unless some earlier navigate
    happened to bind it.
    """

    symbol_map = trajectory.get("_canonical_symbol_map") or {}
    aliases = getattr(adapter, "_fixture_aliases", None)
    if not symbol_map or aliases is None:
        return
    for original, canonical in symbol_map.items():
        if canonical in aliases:
            continue
        concrete = aliases.get(original)
        if concrete:
            aliases[canonical] = concrete


class FsmMirror:
    """Replays the FiniteStateTaskValidator loop body one step at a time.

    Mirrors fsm.py::FiniteStateTaskValidator.validate's per-step behavior
    (communicate gate, allowed tools, preconditions, effects, goal check)
    against the TRAJECTORY's sampled initial state instead of the spec's
    template. Legality failures are returned, never raised.
    """

    def __init__(
        self,
        *,
        composite_task: str,
        trajectory: dict[str, Any],
        communication_mode: str = "full",
    ) -> None:
        from robotalk.tasks.specs import load_task_spec
        from robotalk.tasks.specs.runtime import (
            SpecDrivenTaskValidator,
        )

        spec = load_task_spec(composite_task)
        # The policy sees the shared global tool interface. Its calls must not
        # then be rejected merely because an older task spec omitted an
        # otherwise valid, state-compatible tool. Generic FSM preconditions and
        # task-specific preconditions still apply, and the task goal is
        # unchanged.
        profile = get_communication_profile(communication_mode)
        model_tool_specs = apply_communication_profile(
            build_model_tool_specs(include_get_image=True), communication_mode
        )
        self.validator = SpecDrivenTaskValidator(
            spec,
            allowed_tool_specs_override=model_tool_specs,
        )
        initial_state = _canonical_symbolic_initial_state(
            task_spec=spec,
            trajectory=trajectory,
        )
        if initial_state:
            self.validator.initial_state = initial_state
        # Per-step legality now comes from the concurrent validator rather than
        # the linear one. It encodes the semantics the executor actually has:
        # observation is inert and never gated, a wait touches nothing, and
        # coordination is judged on a clock rather than on written order.
        from robotalk.tasks.shared.concurrent_fsm import (
            ConcurrentTaskValidator,
        )

        self._concurrent = ConcurrentTaskValidator(
            self.validator,
            require_initial_communication=profile.require_initial_communication,
        )
        agents = self.validator._normalize_agents(trajectory.get("agents"))
        self.runtime_state = self.validator._build_runtime_state(agents)
        self.goal_satisfied = self.validator.is_goal_state_satisfied(
            self.runtime_state
        )

    def atomic_handover_conflicts(
        self, calls: list[dict[str, Any]]
    ) -> dict[str, frozenset[str]]:
        """Apply the shared beginning-of-tick exclusive-handover rule."""

        from robotalk.tasks.shared.concurrent_fsm import (
            atomic_handover_conflicts,
        )

        return atomic_handover_conflicts(
            calls, self.runtime_state, self.validator.initial_state
        )

    def validate_cycle_preconditions(
        self, calls: list[dict[str, Any]]
    ) -> dict[str, str]:
        """Validate a complete proposal batch against one frozen state."""

        errors = self._concurrent.validate_cycle_preconditions(
            calls, self.runtime_state
        )
        return {
            agent_id: f"{type(exc).__name__}: {exc}"
            for agent_id, exc in errors.items()
        }

    def validate_wait_call(self, step: dict[str, Any]) -> str | None:
        """Validate the immediately knowable wait contract in live rollout.

        Offline validation can additionally prove that a future matching
        release exists. Live rollout cannot inspect the future, but it must
        still reject malformed callers and non-symbolic release keys before
        handing the call to the scheduler.
        """

        if WAIT_TOOL_NAME not in self.validator.allowed_tool_specs:
            return f"UnsupportedToolSemanticValidationError: Tool {WAIT_TOOL_NAME} is not allowed."
        args = step.get("args") or {}
        from_agent = args.get("from")
        about = args.get("about")
        if set(args) != {"from", "about"}:
            return "WaitSignalSemanticValidationError: wait_for_signal args may only contain from and about."
        if from_agent not in AGENT_IDS or from_agent == step.get("agent"):
            return "WaitSignalSemanticValidationError: wait_for_signal requires from to reference the other agent."
        known = set(self.validator.initial_state.get("objects") or {}) | set(
            self.validator.initial_state.get("fixtures") or {}
        )
        if not isinstance(about, str) or not about.strip() or (
            known and about.strip() not in known
        ):
            return "WaitSignalSemanticValidationError: wait_for_signal about must be an exact symbolic object or fixture ID."
        step["args"] = {"from": from_agent, "about": about.strip()}
        return None

    def commit(self, step: dict[str, Any]) -> None:
        """Commit a step whose preconditions passed the cycle-level check."""

        self._concurrent._commit(step, self.runtime_state)
        self.goal_satisfied = self.validator.is_goal_state_satisfied(
            self.runtime_state
        )

    def partial_goal_fraction(self) -> float:
        """Fraction of top-level spec goal conditions currently satisfied."""

        spec = self.validator._task_spec
        conditions = spec.goal_conditions
        if not conditions:
            return float(self.goal_satisfied)
        satisfied = 0
        for condition in conditions:
            original = self.validator._task_spec
            try:
                self.validator._task_spec = replace(
                    original, goal_conditions=(condition,)
                )
                satisfied += int(
                    self.validator.is_goal_state_satisfied(self.runtime_state)
                )
            finally:
                self.validator._task_spec = original
        return satisfied / len(conditions)


# ---------------------------------------------------------------------------
# Sim session: one env per task, rewound per trajectory
# ---------------------------------------------------------------------------


class SimSession:
    def __init__(
        self,
        *,
        composite_task: str,
        sample_trajectory: dict[str, Any],
        layout: int,
        style: int,
        seed: int,
        gl_backend: str,
        render_size: int,
        map_dpi: int = 300,
        map_renderer: str = "legacy",
    ) -> None:
        from robocasa.utils.sim_tool_executor import SimToolExecutor
        from robocasa.utils.trajectory_pruning import (
            build_trajectory_pruning_config,
        )

        self.composite_task = composite_task
        pruning = build_trajectory_pruning_config(sample_trajectory, layout=layout)
        kwargs: dict[str, Any] = {}
        if any(pruning.get(key) for key in (
            "update_fxtr_cfg_dict",
            "trajectory_object_names",
            "trajectory_object_types",
            "trajectory_object_specs",
        )):
            kwargs.update(
                update_fxtr_cfg_dict=pruning.get("update_fxtr_cfg_dict"),
                trajectory_object_names=pruning.get("trajectory_object_names"),
                trajectory_object_types=pruning.get("trajectory_object_types"),
                trajectory_object_specs=pruning.get("trajectory_object_specs"),
            )
        self.executor = SimToolExecutor(
            task_name=composite_task,
            robots=2,
            layout=layout,
            style=style,
            seed=seed,
            gl_backend=gl_backend,
            render_width=render_size,
            render_height=render_size,
            map_dpi=map_dpi,
            map_renderer=map_renderer,
            robot_spawn="trajectory",
            **kwargs,
        )
        self._synchronize_pruned_task_cardinality(composite_task)

    def _synchronize_pruned_task_cardinality(self, composite_task: str) -> None:
        """Keep native task counters consistent with trajectory pruning.

        Some RoboCasa tasks sample a variable object count while fixtures are
        being initialized.  Trajectory pruning changes RNG consumption and can
        subsequently remove objects that are absent from the recorded initial
        state, leaving the task's cached count larger than ``env.objects``.
        BeverageOrganization's native checker indexes every object up to that
        cached count, so the mismatch otherwise produces ``KeyError: obj_N``.

        Keep this compatibility fix deliberately task-specific: similarly named
        counters in other tasks need not describe contiguous ``obj_N`` objects.
        """

        if composite_task != "BeverageOrganization":
            return
        indices = sorted(
            int(match.group(1))
            for name in self.executor.env.objects
            if (match := _NUMBERED_OBJECT_RE.fullmatch(str(name)))
        )
        if indices == list(range(len(indices))) and indices:
            self.executor.env.num_bev = len(indices)

    def start_trajectory(self, trajectory: dict[str, Any]):
        from robocasa.utils.trajectory_adapter import TrajectoryAdapter

        self.executor.restore_baseline_state()
        adapter = TrajectoryAdapter(
            executor=self.executor,
            allow_approximate_ids=True,
        )
        adapted = adapter.adapt(trajectory)
        self.executor.load_initial_state(adapted["initial_state"])
        # load_initial_state may prune a sampled variable-cardinality object
        # set, so counters derived during env construction must be repaired
        # after loading each trajectory rather than only once in __init__.
        self._synchronize_pruned_task_cardinality(self.composite_task)
        return adapter, adapted

    def native_success(self) -> tuple[bool | None, str | None]:
        """(success, error). Unwraps common env wrappers to find _check_success."""

        env = self.executor.env
        for candidate in (env, getattr(env, "env", None), getattr(env, "unwrapped", None)):
            if candidate is None or not hasattr(candidate, "_check_success"):
                continue
            try:
                if (
                    self.composite_task == "OrganizeCondiments"
                    and "distractor" not in getattr(candidate, "objects", {})
                ):
                    return (
                        bool(
                            _check_pruned_organize_condiments_success(
                                candidate
                            )
                        ),
                        None,
                    )
                return bool(candidate._check_success()), None
            except Exception as exc:
                return None, f"{type(exc).__name__}: {exc}"
        return None, f"no _check_success on {type(env).__name__}"

    def render_views(
        self,
        views: tuple[str, ...],
        *,
        agent_id: str,
        out_dir: Path,
        tag: str,
    ) -> tuple[list[str], list[str]]:
        """Renders the requested views to files; returns (paths, view_names)."""

        out_dir.mkdir(parents=True, exist_ok=True)
        image_paths = [
            str(out_dir / f"{tag}_{view}.jpg") for view in views
        ]
        result = self.executor.get_image(
            views=list(views),
            image_paths=image_paths,
            agent_id=agent_id,
        )
        produced = result.details.get("image_paths") or image_paths
        return [str(p) for p in produced], list(views)

    def close(self) -> None:
        try:
            self.executor.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------


class OraclePolicy:
    """Replays the expert trajectory's own action steps."""

    def __init__(self, raw_steps: list[dict[str, Any]]) -> None:
        self._steps = [s for s in raw_steps if s.get("tool") != "get_image"]
        self._cursor = 0

    def next_step(self, **_ignored) -> dict[str, Any] | None:
        if self._cursor >= len(self._steps):
            return None
        step = self._steps[self._cursor]
        self._cursor += 1
        return {
            "agent": step["agent"],
            "tool": step["tool"],
            "args": deepcopy(step.get("args", {})),
        }


class VllmPolicy:
    """OpenAI-compatible client for a localhost vLLM multimodal server."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.endpoint = (
            args.vllm_base_url.rstrip("/") + "/chat/completions"
        )
        self.model = args.vllm_model
        self.api_key = args.vllm_api_key
        self.request_timeout = args.vllm_request_timeout
        self.max_new_tokens = args.max_new_tokens
        self.seed = args.seed
        self.temperature = float(getattr(args, "vllm_temperature", 0.0))
        self.top_p = getattr(args, "vllm_top_p", None)
        self.top_k = getattr(args, "vllm_top_k", None)
        self.enable_thinking = bool(getattr(args, "enable_thinking", False))
        # Opt-in debugging for parser integration. Keep this disabled by
        # default because raw assistant messages can be large and ordinary
        # evaluations should not change their logging behavior.
        self.log_raw_responses = os.environ.get(
            "VLLM_LOG_RAW_RESPONSES", ""
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.raw_text_mode = bool(getattr(args, "vllm_raw_text_mode", False)) or (
            os.environ.get("VLLM_RAW_TEXT_MODE", "").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        self.recover_missing_open_tool_tag = bool(
            getattr(args, "recover_missing_open_tool_tag", True)
        )
        self.last_usage: dict[str, Any] = {}

    @staticmethod
    def _image_data_url(image_path: str) -> str:
        path = Path(image_path)
        mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    @classmethod
    def _request_messages(cls, feature: dict[str, Any]) -> list[dict[str, Any]]:
        image_paths = iter(feature["image_paths"])
        used_images = 0
        messages: list[dict[str, Any]] = []
        # Match VisionGenerationCollator: the final labeled assistant turn is
        # a training-shape placeholder and is removed for generation.
        for message in feature["messages"][:-1]:
            content = message.get("content", "")
            if not isinstance(content, list):
                messages.append(dict(message))
                continue
            converted: list[dict[str, Any]] = []
            for item in content:
                if item.get("type") != "image":
                    converted.append(dict(item))
                    continue
                try:
                    image_path = next(image_paths)
                except StopIteration as exc:
                    raise ValueError(
                        "Message contains more image slots than image_paths."
                    ) from exc
                converted.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": cls._image_data_url(image_path)},
                    }
                )
                used_images += 1
            messages.append({**message, "content": converted})
        if used_images != len(feature["image_paths"]):
            raise ValueError(
                "image_paths contains images without matching message slots."
            )
        return messages

    def generate(self, feature: dict[str, Any]) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._request_messages(feature),
            "tools": feature["tool_schemas"],
            "temperature": self.temperature,
            "max_tokens": self.max_new_tokens,
            "seed": self.seed,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": self.enable_thinking},
        }
        # vLLM defaults to "auto" whenever tools are present, so parser-free
        # diagnostics must explicitly request "none" rather than omitting the
        # field. The schemas remain available to the model's chat template.
        payload["tool_choice"] = "none" if self.raw_text_mode else "auto"
        if self.top_p is not None:
            payload["top_p"] = float(self.top_p)
        if self.top_k is not None:
            payload["top_k"] = int(self.top_k)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib_request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib_request.urlopen(
                request, timeout=self.request_timeout
            ) as response:
                response_payload = json.loads(response.read())
        except urllib_error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"vLLM HTTP {exc.code}: {body[:1000]}"
            ) from exc
        except urllib_error.URLError as exc:
            raise RuntimeError(f"vLLM request failed: {exc.reason}") from exc

        self.last_usage = dict(response_payload.get("usage") or {})
        try:
            message = response_payload["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(
                f"Malformed vLLM response: {response_payload!r}"
            ) from exc
        if self.log_raw_responses:
            print(
                "[vllm_raw_response] "
                + json.dumps(message, ensure_ascii=False, sort_keys=True),
                flush=True,
            )
        tool_calls = message.get("tool_calls") or []
        reasoning_text = ""
        reasoning = message.get("reasoning_content")
        if isinstance(reasoning, str):
            reasoning_text = reasoning.strip()
        elif tool_calls and isinstance(message.get("content"), str):
            # Rationale-supervised Instruct adapters may emit their learned
            # <think> content here even though the base model has no native
            # enable_thinking switch. Keep it only as private diagnostics.
            reasoning_text = message["content"].strip()
        if tool_calls:
            function = tool_calls[0].get("function") or {}
            arguments = function.get("arguments", {})
            if isinstance(arguments, str):
                arguments = json.loads(arguments)
            if not isinstance(arguments, dict):
                raise RuntimeError("vLLM tool-call arguments are not an object.")
            decoded = (
                "<tool_call>\n"
                + json.dumps(
                    {
                        "name": function.get("name"),
                        "arguments": arguments,
                    },
                    separators=(",", ":"),
                )
                + "\n</tool_call>"
            )
            return _GeneratedPolicyText(decoded, reasoning_text=reasoning_text)
        content = message.get("content")
        if isinstance(content, str):
            if self.recover_missing_open_tool_tag:
                candidate = content.strip()
                closing_tag = "</tool_call>"
                if (
                    candidate.endswith(closing_tag)
                    and "<tool_call>" not in candidate
                ):
                    body = candidate[: -len(closing_tag)].strip()
                    try:
                        recovered_payload = json.loads(body)
                    except json.JSONDecodeError:
                        recovered_payload = None
                    if (
                        isinstance(recovered_payload, dict)
                        and set(recovered_payload) == {"name", "arguments"}
                        and isinstance(recovered_payload.get("name"), str)
                        and bool(recovered_payload["name"].strip())
                        and isinstance(recovered_payload.get("arguments"), dict)
                    ):
                        recovered = (
                            "<tool_call>\n"
                            + json.dumps(recovered_payload, separators=(",", ":"))
                            + "\n</tool_call>"
                        )
                        return _GeneratedPolicyText(
                            recovered,
                            reasoning_text=reasoning_text,
                            parser_recovery="recovered_missing_open_tool_tag",
                        )
            return content
        if isinstance(content, list):
            return "".join(
                str(item.get("text", ""))
                for item in content
                if isinstance(item, dict)
            )
        raise RuntimeError(
            f"vLLM response contained no text or tool call: {message!r}"
        )


class _GeneratedPolicyText(str):
    """Decoded action plus private, non-contextual model diagnostics."""

    def __new__(
        cls,
        value: str,
        *,
        reasoning_text: str = "",
        parser_recovery: str = "",
    ):
        instance = super().__new__(cls, value)
        instance.reasoning_text = reasoning_text
        instance.parser_recovery = parser_recovery
        return instance


class GeminiPolicy:
    """Vertex Gemini native-function-calling policy shared across turns."""

    def __init__(self, args: argparse.Namespace) -> None:
        import os

        from robotalk.generation.runtime.client import (
            load_dotenv_file,
        )

        load_dotenv_file()
        location = args.location or os.environ.get("GOOGLE_CLOUD_LOCATION") or "global"
        # Kept so the client can be rebuilt after a wedged connection: httpx
        # pools connections, so once one hangs every retry on that pool hangs
        # too, and a timeout alone just re-wedges at the next call.
        self._client_args = (args.project, location, args.request_timeout)
        self.client = self._build_client()
        self.model = args.model
        self.temperature = args.temperature
        self.max_output_tokens = args.max_output_tokens
        self.thinking_budget = args.thinking_budget
        self.max_retries = args.max_retries
        # Hard ceiling per SDK call. Healthy calls return in seconds -- whole
        # trajectories finish in 23-72s -- so 90s is ~20x headroom while
        # capping a wedged connection at 90s instead of the 360s a 2x-of-
        # request-timeout default cost. Wedges here are intermittent and hit
        # arbitrary tasks, so the cost per occurrence is what matters.
        self.hard_timeout = float(
            getattr(args, "generate_hard_timeout", 0) or 90.0
        )

    def _build_client(self):
        from robotalk.generation.runtime.client import (
            build_raw_google_genai_client,
        )

        project, location, timeout_sec = self._client_args
        return build_raw_google_genai_client(
            project, location, timeout_sec=timeout_sec
        )

    def _call_with_deadline(self, fn):
        """Runs one SDK call under a hard wall-clock bound.

        The SDK's own timeout is advisory here: it retries internally, so a
        stuck connection never surfaces as an error. This turns that into a
        TimeoutError the retry loop above can act on.
        """

        deadline = float(getattr(self, "hard_timeout", 0) or 0)
        if deadline <= 0:
            return fn()
        pool = ThreadPoolExecutor(max_workers=1)
        try:
            return pool.submit(fn).result(timeout=deadline)
        finally:
            # Do not join a thread that may still be blocked in ssl.read.
            pool.shutdown(wait=False)

    def generate(self, feature: dict[str, Any]) -> str:
        from google.genai import types

        from robotalk.evaluation.gemini_function_calling import (
            GEMINI_MIME_BY_SUFFIX,
            build_vertex_function_declarations,
            message_text,
        )

        messages = feature["messages"]
        parts: list[Any] = []
        for image_path in feature["image_paths"]:
            path = Path(image_path)
            parts.append(types.Part.from_bytes(
                data=path.read_bytes(),
                mime_type=GEMINI_MIME_BY_SUFFIX.get(path.suffix.lower(), "image/jpeg"),
            ))
        # build_messages() appends a trailing assistant turn to preserve the
        # training-example shape, and live-sim builds it with target_text="",
        # so messages[-1] is an EMPTY assistant turn -- not the prompt. Taking
        # it sent Gemini nothing but tool declarations, and because the tool
        # config forces a call (mode=ANY), the model still emitted plausible
        # calls with zero task context instead of failing loudly. Always read
        # the last user turn.
        user_message = next(
            (m for m in reversed(messages) if m.get("role") == "user"), None
        )
        if user_message is None:
            raise ValueError("No user message to send to Gemini.")
        prompt_text = message_text(user_message)
        if not prompt_text.strip():
            raise ValueError("Refusing to query Gemini with an empty prompt.")
        parts.append(types.Part.from_text(text=prompt_text))
        declarations = [
            types.FunctionDeclaration(**item)
            for item in build_vertex_function_declarations(
                feature["allowed_tool_specs"],
                # Under the partial contract the scheduler fixes the caller, so
                # the tools take no "agent" argument and the parsed call is not
                # stripped of one. Advertising the parameter anyway made Gemini
                # emit it on every call, which then failed validation -- the
                # agent relaunched the same rejected proposal until its
                # rejection budget drained, scoring 0 for harness reasons.
            )
        ]
        config_kwargs: dict[str, Any] = {
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "system_instruction": message_text(messages[0]),
            "tools": [types.Tool(function_declarations=declarations)],
            "tool_config": types.ToolConfig(function_calling_config=
                types.FunctionCallingConfig(mode="ANY")),
        }
        if self.thinking_budget >= 0:
            config_kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=self.thinking_budget
            )
        import os as _os

        if _os.environ.get("LIVESIM_DEBUG_PAYLOAD"):
            img_bytes = sum(
                len(getattr(getattr(p, "inline_data", None), "data", b"") or b"")
                for p in parts
            )
            print(
                f"[payload] images={len(feature['image_paths'])} "
                f"image_bytes={img_bytes/1e6:.2f}MB "
                f"prompt_chars={len(prompt_text)} "
                f"tools={len(declarations)}",
                flush=True,
            )
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                # --request-timeout reaches httpx, but google-genai wraps each
                # request in its own tenacity retry, so a wedged connection
                # retries instead of failing: one request was observed blocked
                # in ssl.read for 78 minutes under a 180s timeout, silently
                # eating the job's walltime and truncating the cell. Bound it
                # ourselves. The orphaned thread is leaked deliberately -- it
                # is bounded by max_retries and cannot be cancelled.
                response = self._call_with_deadline(
                    lambda: self.client.models.generate_content(
                        model=self.model,
                        contents=parts,
                        config=types.GenerateContentConfig(**config_kwargs),
                    )
                )
                for candidate in getattr(response, "candidates", None) or []:
                    content = getattr(candidate, "content", None)
                    for part in getattr(content, "parts", None) or []:
                        call = getattr(part, "function_call", None)
                        if call is not None and getattr(call, "name", None):
                            return (
                                "<tool_call>\n"
                                + json.dumps({"name": call.name, "arguments": dict(call.args or {})})
                                + "\n</tool_call>"
                            )
                raise ValueError("Gemini response contained no function call")
            except Exception as exc:
                last_error = exc
                if isinstance(exc, FuturesTimeoutError):
                    # The abandoned request still owns its pooled connection,
                    # so retrying on the same client wedges again immediately.
                    # Observed: a job sat 37min at 0 trajectories, every retry
                    # timing out. A fresh client gets a fresh pool.
                    try:
                        self.client = self._build_client()
                    except Exception:  # noqa: BLE001 -- keep the original error
                        pass
                if attempt < self.max_retries:
                    time.sleep(min(2 ** (attempt - 1), 8))
        assert last_error is not None
        raise last_error


def _uses_model_generation(policy: Any) -> bool:
    """Return whether a policy implements the shared model generate contract."""

    return callable(getattr(policy, "generate", None))


# ---------------------------------------------------------------------------
# The live loop for one trajectory
# ---------------------------------------------------------------------------


def _opening_protocol_error(
    *,
    agent_id: str,
    proposal: dict[str, Any],
    coordinator_id: str | None,
    phase: int,
) -> str | None:
    """Return why a live proposal violates the certified opening handshake."""

    if "error" in proposal:
        return None
    from robotalk.tasks.shared.scheduling import (
        opening_protocol_error,
    )

    return opening_protocol_error(
        agent_id=agent_id,
        call=proposal,
        agent_ids=AGENT_IDS,
        coordinator_id=coordinator_id,
        phase=phase,
    )


def run_trajectory_partial(
    *,
    session: SimSession,
    policy,
    args: argparse.Namespace,
    task_name: str,
    composite_task: str,
    trajectory: dict[str, Any],
    frames_dir: Path | None,
) -> dict[str, Any]:
    """Partial-observability episode: per-agent private state, virtual clock.

    Each agent becomes ready when its own tool completes, so the two agents
    desynchronize naturally and simultaneous proposals are the exception rather
    than every cycle. Neither proposal sees the other's result before it is
    generated. Execution stays serial (the executor runs a tool to completion);
    only readiness ordering is event-timed.
    """

    task_metadata = get_task_metadata(task_name)
    # Partial v3 supervises get_image with a FIXED caller, so the tool set gains
    # get_image but never task_complete or the agent argument.
    communication_mode = getattr(args, "communication_mode", "full")
    communication_profile = get_communication_profile(communication_mode)
    tool_specs = apply_communication_profile(
        augment_tool_specs_with_get_image(task_metadata.allowed_tool_specs),
        communication_mode,
    )
    model_tool_specs = apply_communication_profile(
        build_model_tool_specs(include_get_image=True), communication_mode
    )
    tool_schemas = build_tool_schemas(
        agent_ids=AGENT_IDS,
        allowed_tool_specs=model_tool_specs,
    )
    adapter, adapted = session.start_trajectory(trajectory)
    mirror = FsmMirror(
        composite_task=composite_task,
        trajectory=trajectory,
        communication_mode=communication_mode,
    )
    _register_canonical_fixture_aliases(adapter, trajectory)
    if frames_dir is not None:
        session.executor.save_scene_frames(str(frames_dir), prefix="step_-001")

    expert_action_steps = [
        s for s in trajectory["steps"] if s.get("tool") != "get_image"
    ]
    budget_reference_steps = _budget_reference_action_count(trajectory)
    continuation_step_budget = max(
        4, int(budget_reference_steps * args.step_budget_factor * 2)
    )
    replay_allowance = int(getattr(policy, "replay_call_count", 0))
    step_budget = continuation_step_budget + replay_allowance
    lock_priority = getattr(args, "conflict_priority", "agent-order")
    rejection_mode = getattr(args, "rejection_mode", REJECTION_MODE_REPORT_FAILED)
    multiplier = float(getattr(args, "duration_multiplier", 1.0))
    global _DURATION_JITTER, _DURATION_RNG, _UNIFORM_DURATIONS
    global _LENIENT_WAIT_DISCHARGE
    # Defaults True here too: programmatic callers that build their own args
    # namespace must land in the same regime the corpus was validated in.
    _UNIFORM_DURATIONS = bool(getattr(args, "uniform_durations", True))
    _LENIENT_WAIT_DISCHARGE = bool(getattr(args, "lenient_wait_discharge", False))
    _DURATION_JITTER = float(getattr(args, "duration_jitter", 0.0))
    # Per trajectory, so the same trajectory gets the same perturbed schedule
    # regardless of which shard or order it ran in.
    # sha256, not hash(): str hashing is salted per process, so hash() would
    # make a "reproducible" seed differ between runs.
    _DURATION_RNG = (
        random.Random(
            hashlib.sha256(
                f"{int(getattr(args, 'duration_jitter_seed', 0))}|"
                f"{composite_task}|{trajectory.get('trajectory_id', '')}".encode()
            ).hexdigest()
        )
        if _DURATION_JITTER else None
    )

    agents = {aid: AgentRuntime(aid) for aid in AGENT_IDS}
    from robotalk.tasks.shared.scheduling import ConcurrentScheduler

    scheduler = ConcurrentScheduler(AGENT_IDS, states=agents)
    is_model_policy = _uses_model_generation(policy)
    # The oracle replays each agent's recorded calls as its own queue, through
    # the same concurrent scheduler that models are evaluated with.
    expert_sequence: list[dict[str, Any]] = []
    if not is_model_policy:
        for raw in trajectory["steps"]:
            if raw.get("tool") == "get_image":
                continue
            if raw.get("agent") in agents:
                expert_sequence.append(
                    {"agent": raw["agent"], "tool": raw["tool"],
                     "args": deepcopy(raw.get("args", {}))}
                )
    records: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []
    clock = 0.0
    turn_index = 0          # productive (executed) steps -- what step_budget caps
    proposal_index = 0      # every model call, for logging/ids
    rejected_total = 0
    consecutive_rejections = 0
    max_consec = int(getattr(args, "max_consecutive_rejections", 8))
    # A rejected call is not productive work, so it must not consume the budget
    # meant for it -- but it cannot be free either, or a livelock runs forever.
    # Hence a separate allowance, generous relative to the step budget.
    max_rejected = int(step_budget * float(getattr(args, "rejection_budget_ratio", 1.0)))
    termination = "budget_exhausted"
    simultaneous_cycles = 0
    mutual_wait_deadlocks = 0
    waits_issued = 0
    wait_caps = 0
    max_consecutive_waits = int(getattr(args, 'max_consecutive_waits', 3))
    max_free_observations = int(getattr(args, "max_free_observations", 4))
    observation_caps = 0
    coordinator_id = (
        trajectory.get("coordinator_id")
        if communication_profile.require_opening_protocol
        else None
    )
    opening_phase = 0

    expert_queues: dict[str, list[dict[str, Any]]] = {}
    for entry in expert_sequence:
        expert_queues.setdefault(entry["agent"], []).append(entry)

    def _propose(agent: AgentRuntime) -> tuple[dict[str, Any], tuple[str, ...] | None]:
        """One decision for one agent, using only that agent's private state."""

        if not is_model_policy:
            queue = expert_queues.get(agent.agent_id) or []
            if not queue:
                return {"exhausted": True}, None
            return queue.pop(0), None
        image_paths, view_names = agent.take_observation()
        render_dir = frames_dir or Path(args.output_dir) / "_tmp_views"
        proposal = _generate_once(
            session=session,
            policy=policy,
            args=args,
            task_metadata=task_metadata,
            tool_specs=tool_specs,
            tool_schemas=tool_schemas,
            trajectory=trajectory,
            history=agent.private_history,
            step_index=turn_index,
            views=tuple(view_names),
            render_dir=render_dir,
            agent_id=agent.agent_id,
            pre_rendered_image_paths=image_paths if image_paths else None,
        )
        return proposal, tuple(view_names) if view_names else None

    # Productive work is bounded by ``step_budget``. Model proposals also need
    # an independent hard bound: rejected calls, capped waits, and other
    # nonproductive proposals may leave ``turn_index`` unchanged. The counters
    # existed previously but were never updated or checked, allowing one
    # pathological trajectory to generate forever (jobs 284515/284516).
    max_proposals = step_budget + max_rejected
    while turn_index < step_budget and proposal_index < max_proposals:
        if not is_model_policy and not any(expert_queues.values()):
            termination = "policy_exhausted"
            break
        if all(a.ready_at == float("inf") for a in agents.values()):
            termination = "policy_exhausted"
            break
        # A waiting agent is not polled and only a matching release may
        # wake it.  If every agent is waiting, the episode has reached a
        # real mutual-wait deadlock.  Never silently clear the waits: that
        # would manufacture actions which the agents could not take under
        # the protocol and would corrupt both evaluation and DAgger logs.
        if agents and all(scheduler.blocked(a) for a in agents):
            mutual_wait_deadlocks += 1
            termination = "mutual_wait_deadlock"
            break
        # A blocked agent must not drive the clock. Its ready_at is stale
        # -- the moment its wait WOULD have expired -- so including it here
        # pulled the clock back every iteration and the loop never
        # advanced past a partner that was merely busy.
        next_ready = scheduler.next_ready_time()
        runnable_times = (
            [next_ready] if next_ready is not None
            else [a.ready_at for a in agents.values()]
        )
        # Monotonic, but only over agents that still have work. `inf` is the
        # sentinel for an EXHAUSTED agent, and clamping upward against it
        # pins the clock at infinity forever -- after which the catch-up
        # below hands every other agent an infinite ready_at and nothing is
        # ever schedulable again. Without work to order there is no time to
        # advance to, so leave the clock where it is and let the
        # nobody-is-ready branch decide what happens next.
        finite = [t for t in runnable_times if t != float("inf")]
        if finite:
            clock = max(clock, min(finite))
        # A wait is released by a message, so the agent resumes at the
        # moment of release. Leaving its stale ready_at -- the time the
        # wait would have expired -- let it act "in the past", inverting
        # sim_time against the real execution order.
        for a in agents.values():
            if a.waiting_for is None and a.ready_at < clock:
                a.ready_at = clock
        ready = [agents[a] for a in scheduler.ready_agents(clock=clock)]
        if not ready:
            # A partner that is merely BUSY must not cancel a wait. This
            # fired whenever nobody was runnable at the current instant,
            # so every wait expired as soon as the other agent was
            # mid-action -- waits behaved as a 2s pause rather than a
            # block, which is why inserting them changed nothing.
            busy = [
                a for a in agents.values()
                if a.waiting_for is None and a.ready_at != float("inf")
            ]
            if busy:
                clock = max(clock, min(a.ready_at for a in busy))
                continue
            waiting = [a for a in agents.values() if a.waiting_for is not None]
            if not waiting:
                break
            # No non-waiting agent remains that could send a release.
            # Waiting is permanent regardless of the communication
            # ablation; modes differ in guidance/tool exposure, not in
            # the runtime meaning of wait_for_signal.
            termination = "permanent_wait_no_runnable_agents"
            break
        if len(ready) > 1:
            simultaneous_cycles += 1

        # --- concurrent generation: neither sees the other's result ---
        started = time.perf_counter()
        if len(ready) == 1 or not _uses_model_generation(policy):
            proposals = [(a, *_propose(a)) for a in ready]
        else:
            with ThreadPoolExecutor(max_workers=len(ready)) as pool:
                futures = {pool.submit(_propose, a): a for a in ready}
                proposals = [(futures[f], *f.result()) for f in futures]
            proposals.sort(key=lambda item: item[0].agent_id)
        elapsed = round(time.perf_counter() - started, 3)

        private_reasoning_by_agent = {
            agent.agent_id: str(proposal.pop("_private_reasoning", "") or "")
            for agent, proposal, _ in proposals
        }
        parser_recovery_by_agent = {
            agent.agent_id: str(proposal.pop("_parser_recovery", "") or "")
            for agent, proposal, _ in proposals
        }

        # --- resource conflict resolution against the same pre-batch state ---
        symbolic_initial_state = mirror.validator.initial_state
        proposal_by_agent = {
            agent.agent_id: prop
            for agent, prop, _ in proposals
            if "error" not in prop
        }
        # Deterministic priority breaks the tie. Rejecting BOTH proposals is
        # symmetric, so two agents wanting the same fixture would each retry the
        # same call forever (observed: 6 conflicts, zero progress, livelock).
        # One agent proceeds; the loser gets an ordinary tool error and replans.
        conflicted: set[str] = set()
        ids = sorted(proposal_by_agent)
        if lock_priority == "seeded":
            rng = random.Random(f"{args.seed}:{turn_index}")
            rng.shuffle(ids)
        ids = _prioritize_incumbent_fixture_users(
            ids,
            proposal_by_agent,
            mirror.runtime_state,
            symbolic_initial_state,
        )
        accepted: dict[str, dict[str, Any]] = {}
        conflict_resources: dict[str, set[str]] = {}
        for a_id in ids:
            proposal = proposal_by_agent[a_id]
            pair_conflicts: set[str] = set()
            for accepted_proposal in accepted.values():
                pair_conflicts.update(
                    simultaneous_contentions(
                        proposal,
                        accepted_proposal,
                        symbolic_initial_state,
                    )
                )
            if pair_conflicts:
                conflicted.add(a_id)
                conflict_resources[a_id] = pair_conflicts
            else:
                accepted[a_id] = proposal

        atomic_calls = [
            {**proposal, "agent": agent_id}
            for agent_id, proposal in proposal_by_agent.items()
        ]
        cycle_precondition_errors = mirror.validate_cycle_preconditions(
            atomic_calls
        )
        for agent_id, resources in mirror.atomic_handover_conflicts(
            atomic_calls
        ).items():
            conflicted.add(agent_id)
            conflict_resources.setdefault(agent_id, set()).update(resources)

        cycle_record_start = len(records)
        opening_observation_cycle = (
            coordinator_id in AGENT_IDS
            and opening_phase < 2
            and any(p.get("tool") == "get_image" for _, p, _ in proposals)
        )
        for agent, proposal, views_used in proposals:
            record: dict[str, Any] = {
                "step_index": turn_index,
                "sim_time": round(clock, 3),
                "agent": agent.agent_id,
                "local_turn": agent.local_turn,
                "proposal": deepcopy(proposal),
                "views": list(views_used) if views_used else None,
                "proposal_elapsed_s": elapsed,
                "cycle_agents": [a.agent_id for a in ready],
                "private_reasoning": private_reasoning_by_agent[agent.agent_id] or None,
                "parser_recovery": parser_recovery_by_agent[agent.agent_id] or None,
            }
            agent.local_turn += 1
            proposal_index += 1

            key = _proposal_key(proposal)
            if key == agent.last_proposal_key:
                agent.repeated_proposals += 1
            agent.last_proposal_key = key

            # Enforce the same two-cycle opening contract used to certify the
            # training tick grid. Observation calls may precede it, but no
            # physical call or competing proposal may slip through merely
            # because the prompt was misunderstood.
            protocol_error = _opening_protocol_error(
                agent_id=agent.agent_id,
                proposal=proposal,
                coordinator_id=coordinator_id,
                phase=opening_phase,
            )
            if opening_observation_cycle and proposal.get("tool") != "get_image":
                protocol_error = (
                    "opening observations and handshake messages cannot be mixed "
                    "in one atomic cycle; observe now and send the required "
                    "handshake message on the next cycle"
                )
            if protocol_error is not None:
                record.update(legal=False, reason=protocol_error, executed=False)
                _handle_rejection(
                    agent=agent, agents=agents, step=proposal,
                    reason=protocol_error, clock=clock, mode=rejection_mode,
                    multiplier=multiplier, record=record,
                )
                records.append(record)
                continue

            # wait_for_signal is a scheduling primitive, not a sim action: it
            # blocks the agent instead of touching the world. The shared
            # scheduler wakes it only for a later communicate call from the
            # requested sender whose exact releases ID matches `about`.
            # The first wait is free; later consecutive waits consume
            # budget so a wait->irrelevant-message->wait livelock is bounded
            # and visible rather than an unbounded stall.
            if proposal.get("tool") == WAIT_TOOL_NAME and "error" not in proposal:
                wait_args = proposal.get("args") or {}
                wait_step = {
                    "step": turn_index,
                    "agent": agent.agent_id,
                    "tool": WAIT_TOOL_NAME,
                    "args": deepcopy(wait_args),
                }
                wait_error = mirror.validate_wait_call(wait_step)
                if wait_error is not None:
                    record.update(
                        legal=False,
                        executed=False,
                        reason=wait_error,
                    )
                    _handle_rejection(
                        agent=agent,
                        agents=agents,
                        step=wait_step,
                        reason=wait_error,
                        clock=clock,
                        mode=rejection_mode,
                        multiplier=multiplier,
                        record=record,
                    )
                    records.append(record)
                    continue
                wait_args = wait_step["args"]
                agent.consecutive_waits += 1
                scheduler.block(agent.agent_id, wait_step, clock=clock)
                agent.waiting_for["declared_at"] = turn_index
                record["legal"] = True
                record["executed"] = True
                record["waited_for"] = deepcopy(agent.waiting_for)
                record["consecutive_waits"] = agent.consecutive_waits
                waits_issued += 1
                if agent.consecutive_waits > max_consecutive_waits:
                    # Treat as unproductive: stop blocking and make it cost a turn.
                    agent.waiting_for = None
                    agent.consecutive_waits = 0
                    record["wait_capped"] = True
                    wait_caps += 1
                    turn_index += 1
                agent.deliver(
                    {
                        "step": turn_index,
                        "agent": agent.agent_id,
                        "tool": WAIT_TOOL_NAME,
                        "args": deepcopy(wait_args),
                    }
                )
                agent.ready_at = clock + _tool_duration(
                    WAIT_TOOL_NAME, sim_steps=None, multiplier=multiplier
                )
                records.append(record)
                continue
            if proposal.get("tool") != WAIT_TOOL_NAME:
                agent.consecutive_waits = 0
            if proposal.get("tool") != "get_image":
                # The cap counts observations taken back-to-back; any real call
                # in between means the agent is making progress, not looping.
                agent.consecutive_obs = 0

            if proposal.get("error") or proposal.get("_rejected"):
                pass  # counted below
            if proposal.get("exhausted"):
                # This agent has no expert steps left; retire it from scheduling.
                agent.ready_at = float("inf")
                agent.local_turn -= 1
                turn_index -= 1
                continue

            if "error" in proposal:
                record.update(legal=False, executed=False, reason=proposal["error"])
                agent.ready_at = clock + _tool_duration(
                    "communicate", sim_steps=None, multiplier=multiplier
                )
                records.append(record)
                continue

            # Resource conflict -> ordinary tool error handed back to the agent.
            if agent.agent_id in conflicted:
                reason = "resource conflict: another agent holds a required resource"
                record.update(
                    legal=False, executed=False, reason=reason, conflict=True
                )
                conflicts.append(
                    {"agent": agent.agent_id, "sim_time": round(clock, 3),
                     "tool": proposal.get("tool"),
                     "claims": sorted(conflict_resources.get(agent.agent_id, ())) }
                )
                _handle_rejection(
                    agent=agent,
                    agents=agents,
                    step={"agent": agent.agent_id, "tool": proposal["tool"],
                          "args": deepcopy(proposal["args"])},
                    reason=reason,
                    clock=clock,
                    mode=rejection_mode,
                    multiplier=multiplier,
                    record=record,
                )
                records.append(record)
                continue

            # --- get_image: render, deliver to the requester only ---
            if proposal["tool"] == "get_image":
                views = tuple(proposal.get("args", {}).get("views", ()))
                known = set(OVERHEAD_VIEWS + SCOUT_VIEWS + WRIST_VIEWS)
                if not views or any(v not in known for v in views):
                    record.update(
                        legal=False, executed=False, reason="invalid observation views"
                    )
                    agent.ready_at = clock + _tool_duration(
                        "get_image", sim_steps=None, multiplier=multiplier
                    )
                    records.append(record)
                    continue
                render_dir = frames_dir or Path(args.output_dir) / "_tmp_views"
                image_paths, view_names = session.render_views(
                    views,
                    agent_id=agent.agent_id,
                    out_dir=render_dir,
                    tag=f"t{turn_index:03d}_{agent.agent_id}",
                )
                # Consume-once: only the requester ever sees this. Preserve the
                # global index in raw private state; prompt normalization below
                # converts it to local or removes it exactly as training does.
                observation_step_index = turn_index
                turn_index += 1
                consecutive_rejections = 0
                agent.pending_obs = (image_paths, view_names)
                agent.deliver(
                    {
                        "step": observation_step_index,
                        "agent": agent.agent_id,
                        "tool": "get_image",
                        "args": {"views": list(views)},
                    }
                )
                record.update(legal=True, executed=True, reason=None)
                # Persist the exact observation files used by this turn.
                record["image_paths"] = list(image_paths)
                agent.ready_at, agent.consecutive_obs, capped = observation_ready_at(
                    clock,
                    consecutive_obs=agent.consecutive_obs,
                    max_free_observations=max_free_observations,
                    multiplier=multiplier,
                )
                if capped:
                    record["observation_capped"] = True
                    observation_caps += 1
                record["consecutive_obs"] = agent.consecutive_obs
                agent.silent_retries = 0
                records.append(record)
                continue

            symbolic_step = {
                "step": turn_index,
                "agent": agent.agent_id,
                "tool": proposal["tool"],
                "args": deepcopy(proposal["args"]),
            }
            state_before = deepcopy(mirror.runtime_state)
            goal_before = mirror.goal_satisfied
            reason = cycle_precondition_errors.get(agent.agent_id)
            legal = reason is None
            record.update(legal=legal, reason=reason)

            if not legal:
                record["executed"] = False
                _handle_rejection(
                    agent=agent, agents=agents, step=symbolic_step,
                    reason=reason or "illegal", clock=clock, mode=rejection_mode,
                    multiplier=multiplier, record=record,
                )
                records.append(record)
                continue

            mirror.commit(symbolic_step)

            try:
                tool_call = adapter._adapt_step(
                    symbolic_step,
                    resolved_initial_state=adapted["initial_state"],
                    output_dir=None,
                )
                sim_t0 = _sim_clock(session)
                result = session.executor.execute(
                    tool_call["tool"],
                    robot_idx=tool_call.get("robot_idx", 0),
                    **tool_call.get("args", {}),
                )
                record.update(executed=True, sim_success=bool(result.success))
                if not result.success:
                    mirror.runtime_state = state_before
                    mirror.goal_satisfied = goal_before
                    details = getattr(result, "details", None) or {}
                    sim_reason = str(
                        details.get("error")
                        or details.get("reason")
                        or "simulator reported an unsuccessful tool call"
                    )
                    record["sim_error"] = sim_reason
                    _handle_rejection(
                        agent=agent, agents=agents, step=symbolic_step,
                        reason=sim_reason, clock=clock, mode=rejection_mode,
                        multiplier=multiplier, record=record,
                    )
                    records.append(record)
                    continue
            except Exception as exc:
                mirror.runtime_state = state_before
                mirror.goal_satisfied = goal_before
                record.update(
                    executed=False, sim_success=False,
                    sim_error=f"{type(exc).__name__}: {exc}",
                )
                _handle_rejection(
                    agent=agent, agents=agents, step=symbolic_step,
                    reason=f"{type(exc).__name__}: {exc}", clock=clock,
                    mode=rejection_mode,
                    multiplier=multiplier, record=record,
                )
                records.append(record)
                continue

            # --- committed: private history, then delivery to the recipient ---
            agent.silent_retries = 0
            turn_index += 1
            consecutive_rejections = 0
            agent.deliver(symbolic_step)
            if symbolic_step["tool"] == "communicate":
                recipient = (symbolic_step.get("args") or {}).get("to")
                if recipient in agents and recipient != agent.agent_id:
                    scheduler.deliver(
                        symbolic_step,
                        clock=clock,
                        resume_delay=_tool_duration(
                            WAIT_TOOL_NAME, sim_steps=None, multiplier=multiplier
                        ),
                        lenient=_LENIENT_WAIT_DISCHARGE,
                    )
                    agents[recipient].deliver(symbolic_step, clock=clock)
                    record["delivered_to"] = recipient
            sim_t1 = _sim_clock(session)
            measured = (
                (sim_t1 - sim_t0)
                if (sim_t0 is not None and sim_t1 is not None and sim_t1 > sim_t0)
                else None
            )
            duration = _tool_duration(
                symbolic_step["tool"], sim_steps=measured, multiplier=multiplier
            )
            record["measured_sim_seconds"] = (
                round(measured, 3) if measured is not None else None
            )
            agent.ready_at = clock + duration
            agent.active_s += duration
            record["duration"] = round(duration, 3)

            if frames_dir is not None:
                try:
                    session.executor.save_scene_frames(
                        str(frames_dir), prefix=f"step_{turn_index:03d}"
                    )
                except Exception:
                    pass

            record["fsm_goal"] = mirror.goal_satisfied
            native_now, native_err = session.native_success()
            record["native_success"] = native_now
            if native_err:
                record["native_error"] = native_err
            records.append(record)

            # Goal termination is decided after the whole concurrent proposal
            # batch commits. Proposals in one cycle were generated from the
            # same pre-cycle context, so stopping midway makes a tick
            # non-atomic and disagrees with ConcurrentTaskValidator.
        cycle_records = records[cycle_record_start:]
        if coordinator_id in AGENT_IDS and opening_phase < 2:
            expected_agents = set(AGENT_IDS)
            committed = {
                record.get("agent")
                for record in cycle_records
                if record.get("executed")
                and (record.get("proposal") or {}).get("tool") == "communicate"
            }
            if committed == expected_agents:
                opening_phase += 1
        cycle_rejected = _count_rejected_partial_records(cycle_records)
        rejected_total += cycle_rejected
        if cycle_rejected:
            consecutive_rejections += cycle_rejected
        elif cycle_records:
            consecutive_rejections = 0

        criterion = getattr(args, "success_criterion", "fsm")
        native_after_cycle, _ = session.native_success()
        reached = (
            mirror.goal_satisfied
            if criterion == "fsm"
            else (mirror.goal_satisfied and native_after_cycle)
        )
        if reached:
            termination = "goal_satisfied"
            scheduler.finish_cycle(goal_satisfied=True)

        # Calls in a cycle were proposed against one frozen pre-cycle state.
        # Classify the whole cycle, but do not generate another learner call.
        if cycle_rejected and rejection_mode == REJECTION_MODE_TERMINATE_FIRST:
            termination = "first_rejection"

        if not reached:
            budget_termination = _partial_budget_termination(
                rejected_total=rejected_total,
                max_rejected=max_rejected,
                consecutive_rejections=consecutive_rejections,
                max_consecutive_rejections=max_consec,
                proposal_index=proposal_index,
                max_proposals=max_proposals,
                turn_index=turn_index,
                step_budget=step_budget,
            )
            if budget_termination is not None:
                termination = budget_termination

        if termination in (
            "goal_satisfied",
            "max_consecutive_rejections",
            "rejection_budget_exhausted",
            "proposal_budget_exhausted",
            "first_rejection",
        ):
            break

    native, native_error = session.native_success()
    finite = [a.ready_at for a in agents.values() if a.ready_at != float("inf")]
    makespan = max(finite, default=0.0)
    first_rejection = next(
        (
            record for record in records
            if record.get("legal") is False or record.get("sim_success") is False
        ),
        None,
    )
    had_rejection = first_rejection is not None
    first_rejection_category = None
    if first_rejection is not None:
        first_rejection_category = (
            "simulator_execution_after_fsm_acceptance"
            if first_rejection.get("legal") is not False
            and first_rejection.get("sim_success") is False
            else "model_call_or_symbolic_validation"
        )
    return {
        "task_name": task_name,
        "composite_task": composite_task,
        "trajectory_id": trajectory.get("trajectory_id"),
        "partial_history": True,
        "communication_mode": communication_mode,
        "contention_policy": "concurrent_fsm",
        "rejection_mode": rejection_mode,
        "scene": {"layout": args.layout, "style": args.style, "seed": args.seed},
        "expert_steps": len(expert_action_steps),
        "steps_used": len(records),
        "productive_steps": turn_index,
        "rejected_total": rejected_total,
        "executed_steps": sum(1 for r in records if r.get("executed")),
        "communication_calls": sum(
            1
            for r in records
            if r.get("executed")
            and (r.get("proposal") or {}).get("tool") == "communicate"
        ),
        "wait_calls": waits_issued,
        "release_messages": sum(
            1
            for r in records
            if r.get("executed")
            and (r.get("proposal") or {}).get("tool") == "communicate"
            and ((r.get("proposal") or {}).get("args") or {}).get("releases")
        ),
        "rejected_steps": sum(
            1 for r in records
            if r.get("legal") is False or r.get("sim_success") is False
        ),
        "termination": termination,
        "declared_complete": False,
        "native_success": native,
        "native_error": native_error,
        "fsm_goal_satisfied": mirror.goal_satisfied,
        "had_rejection": had_rejection,
        "first_rejection": deepcopy(first_rejection),
        "first_rejection_category": first_rejection_category,
        "fsm_success_if_terminate_on_first_rejection": bool(
            mirror.goal_satisfied and not had_rejection
        ),
        "native_success_if_terminate_on_first_rejection": bool(
            native and not had_rejection
        ),
        "partial_goal_fraction": mirror.partial_goal_fraction(),
        "makespan": round(makespan, 3),
        "simultaneous_cycles": simultaneous_cycles,
        "waits_issued": waits_issued,
        "wait_caps": wait_caps,
        "observation_caps": observation_caps,
        "mutual_wait_deadlocks": mutual_wait_deadlocks,
        "resource_conflicts": len(conflicts),
        "conflict_details": conflicts,
        "reported_failures": rejected_total,
        "per_agent": {
            aid: {
                "turns": a.local_turn,
                "active_sim_time": round(a.active_s, 3),
                "idle_sim_time": round(max(0.0, makespan - a.active_s), 3),
                "repeated_proposals": a.repeated_proposals,
                "private_history_len": len(a.private_history),
            }
            for aid, a in agents.items()
        },
        "agent_turns": [r["agent"] for r in records if r.get("executed")],
        "steps": records,
    }


def _count_rejected_partial_records(records: list[dict[str, Any]]) -> int:
    """Count rejected proposals emitted during one concurrent scheduler cycle."""

    return sum(
        1
        for record in records
        if record.get("legal") is False or record.get("sim_success") is False
    )


def _partial_budget_termination(
    *,
    rejected_total: int,
    max_rejected: int,
    consecutive_rejections: int,
    max_consecutive_rejections: int,
    proposal_index: int,
    max_proposals: int,
    turn_index: int,
    step_budget: int,
) -> str | None:
    """Return the bounded-loop termination, if a partial episode exhausted it."""

    if rejected_total >= max_rejected:
        return "rejection_budget_exhausted"
    if consecutive_rejections >= max_consecutive_rejections:
        return "max_consecutive_rejections"
    if proposal_index >= max_proposals and turn_index < step_budget:
        return "proposal_budget_exhausted"
    return None


def _handle_rejection(
    *,
    agent: AgentRuntime,
    agents: dict[str, AgentRuntime],
    step: dict[str, Any],
    reason: str,
    clock: float,
    mode: str,
    multiplier: float,
    record: dict[str, Any],
) -> None:
    """Record a rejection, optionally without exposing it for recovery."""

    if mode not in (REJECTION_MODE_REPORT_FAILED, REJECTION_MODE_TERMINATE_FIRST):
        raise ValueError(f"Unsupported rejection mode: {mode!r}")
    if mode == REJECTION_MODE_TERMINATE_FIRST:
        # Avoidance-target DAgger branches before this rejected call. No later
        # learner call should receive a failure message in this mode.
        record["failure_reported"] = False
        record["terminated_on_rejection"] = True
        return
    # This is deliberately visible even though FAILED lines are absent from
    # SFT. Recovery therefore measures a general model capability; the
    # counterfactual no-rejection metric separately scores learned execution.
    # Parsed proposals do not necessarily carry an ``agent`` field; the
    # caller identity lives on AgentRuntime. Every private-history row must be
    # a complete symbolic step so the next prompt can render it safely.
    agent.deliver(
        {
            **step,
            "agent": step.get("agent") or agent.agent_id,
            "tool": step.get("tool") or "invalid",
            "args": deepcopy(step.get("args") or {}),
            "error": reason[:120],
        }
    )
    agent.silent_retries = 0
    record["silent_retries"] = 0
    record["failure_reported"] = True
    agent.ready_at = clock + _tool_duration(
        step.get("tool", "communicate"), sim_steps=None, multiplier=multiplier
    )


def _generate_once(
    *,
    session: SimSession,
    policy: "VllmPolicy | GeminiPolicy",
    args: argparse.Namespace,
    task_metadata,
    tool_specs: dict[str, Any],
    tool_schemas: list[dict[str, Any]],
    trajectory: dict[str, Any],
    history: list[dict[str, Any]],
    step_index: int,
    views: tuple[str, ...],
    render_dir: Path,
    agent_id: str,
    pre_rendered_image_paths: list[str] | None = None,
) -> dict[str, Any]:
    """One model proposal for ``agent_id`` from that agent's private context."""

    communication_mode = getattr(args, "communication_mode", "full")
    communication_profile = get_communication_profile(communication_mode)
    model_tool_specs = apply_communication_profile(
        build_model_tool_specs(include_get_image=True),
        communication_mode,
    )
    if pre_rendered_image_paths is not None:
        image_paths = list(pre_rendered_image_paths)
        view_names = list(views)
    elif views:
        image_paths, view_names = session.render_views(
            views,
            agent_id=agent_id,
            out_dir=render_dir,
            tag=f"turn_{step_index:03d}_{len(views)}v",
        )
    else:
        image_paths, view_names = [], []
    # The caller is the actor: no agent prediction and no step index.
    user_prompt = build_partial_user_prompt(
        composite_task=task_metadata.composite_task,
        task_instruction=task_metadata.task_goal,
        agent_id=agent_id,
        observation_views=view_names,
        history_steps=history,
        allowed_tool_specs=model_tool_specs,
        coordinator_id=(
            trajectory.get("coordinator_id")
            if communication_profile.require_opening_protocol
            else None
        ),
        initial_state=trajectory.get("initial_state"),
        communication_mode=communication_mode,
    )
    feature = {
        "sample_id": f"live/{trajectory.get('trajectory_id')}/turn_{step_index}",
        "task_name": task_metadata.dataset_name,
        "trajectory_id": str(trajectory.get("trajectory_id") or ""),
        "step_index": step_index,
        "agent_id": agent_id,
        "target_payload": None,
        "target_tool_call": None,
        "target_text": "",
        "messages": build_messages(
            user_prompt=user_prompt,
            num_images=len(image_paths),
            # The generation collator strips the labeled assistant turn before
            # tokenization, so retain its expected training-example shape.
            target_text="",
        ),
        "image_paths": image_paths,
        "tool_schemas": tool_schemas,
        "allowed_tool_specs": model_tool_specs,
    }
    try:
        decoded = policy.generate(feature)
        private_reasoning = str(getattr(decoded, "reasoning_text", "") or "")
        parser_recovery = str(getattr(decoded, "parser_recovery", "") or "")
        parsed = parse_first_qwen_tool_call(decoded)
        # eval-only cab<->cabinet leniency; never makes a legal call illegal
        parsed = apply_eval_fixture_aliases(parsed, tool_specs)
        # Caller identity is fixed by the scheduler; the model never names an
        # acting agent.
        payload = tool_call_to_single_step_payload(
            parsed,
            step_index=step_index,
            agent_id=agent_id,
            agent_ids=AGENT_IDS,
            allowed_tool_specs=model_tool_specs,
        )
        step = payload["steps"][0]
        result = {
            "agent": step["agent"],
            "tool": step["tool"],
            "args": step["args"],
        }
        if private_reasoning:
            result["_private_reasoning"] = private_reasoning
        if parser_recovery:
            result["_parser_recovery"] = parser_recovery
        return result
    except Exception as exc:
        result = {"error": f"{type(exc).__name__}: {exc}"}
        private_reasoning = str(
            getattr(locals().get("decoded"), "reasoning_text", "") or ""
        )
        if private_reasoning:
            result["_private_reasoning"] = private_reasoning
        parser_recovery = str(
            getattr(locals().get("decoded"), "parser_recovery", "") or ""
        )
        if parser_recovery:
            result["_parser_recovery"] = parser_recovery
        return result


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _latest_trajectory_records(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return the latest append-only record for each trajectory key."""

    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        key = (
            record.get("task_name"),
            record.get("episode_id") or record.get("trajectory_id"),
        )
        latest[key] = record
    return list(latest.values())


def _completed_trajectory_keys(
    records: list[dict[str, Any]],
) -> set[tuple[str, str]]:
    """Return resumable completions, leaving latest harness errors pending."""

    return {
        (
            record["task_name"],
            record.get("episode_id") or record["trajectory_id"],
        )
        for record in _latest_trajectory_records(records)
        if record.get("termination") != "harness_error"
    }


def _finalize_first_recording(
    *,
    frames_dir: Path | None,
    output_dir: Path,
    task_name: str,
    success: bool,
    firsts: dict[tuple[str, bool], bool],
    record_firsts: bool,
    save_frames: bool,
    record_fps: int,
) -> None:
    """Classify a recorded first outcome when frames were actually created."""

    if frames_dir is None or not frames_dir.exists():
        return
    if not firsts.get((task_name, success)):
        firsts[(task_name, success)] = True
        if record_firsts and not save_frames:
            classified = (
                output_dir
                / "recordings"
                / task_name
                / ("success" if success else "failure")
            )
            if classified.exists():
                shutil.rmtree(classified)
            frames_dir.rename(classified)
            _assemble_recording_videos(classified, fps=record_fps)
    elif record_firsts and not save_frames:
        shutil.rmtree(frames_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=("oracle", "vllm", "gemini"), required=True
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Save per-camera stills for every evaluated trajectory.",
    )
    parser.add_argument("--record-fps", type=int, default=2)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--cohort-split",
        choices=("train_task_types", "heldout_task_types"),
        default=None,
        help="Required for a fixed_live_sim manifest; selects one named split.",
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layout", type=int, default=11)
    parser.add_argument("--style", type=int, default=34)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--gl-backend", default="egl", choices=("egl", "osmesa"),
        help="OpenGL backend: 'egl' (headless GPU rendering) or 'osmesa' (CPU).",
    )
    parser.add_argument("--render-size", type=int, default=512)
    parser.add_argument(
        "--map-dpi",
        type=int,
        default=300,
        help=(
            "Placement-map export DPI. The compatibility default is 300 "
            "(6000x4800); lower values trade source-map fidelity for speed."
        ),
    )
    parser.add_argument(
        "--map-renderer",
        choices=("legacy", "raster"),
        default="legacy",
        help="Placement-grid renderer. Legacy is compatible; raster is faster.",
    )
    parser.add_argument("--step-budget-factor", type=float, default=2.0)
    parser.add_argument("--max-consecutive-rejections", type=int, default=3)
    parser.add_argument("--max-trajectories", type=int, default=None)
    parser.add_argument(
        "--episodes-per-task",
        type=int,
        default=None,
        help="Sampled-mode episodes per task (default: 10 for configuration cohorts).",
    )
    parser.add_argument(
        "--cohort-mode", choices=("sampled", "full_config"), default="sampled",
        help="Uniform sampled episodes or exhaustive configuration coverage.",
    )
    parser.add_argument("--episodes-per-config", type=int, default=1)
    parser.add_argument("--max-configs-per-task", type=int, default=None)
    parser.add_argument("--evaluation-seed", type=int, default=None)
    parser.add_argument(
        "--scene-compatibility-cache", type=Path, default=None,
        help="Certified configuration-scene cache used for reproducible random scenes.",
    )
    parser.add_argument("--scene-sampling-seed", type=int, default=20260819)
    parser.add_argument("--tasks", type=str, default=None,
                        help="Comma-separated task filter.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--record-firsts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Record stills for the first success and first failure per task.",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Local LoRA directory or Hugging Face adapter repository ID.",
    )
    parser.add_argument(
        "--vllm-base-url",
        default="http://127.0.0.1:8000/v1",
        help="OpenAI-compatible vLLM API base URL.",
    )
    parser.add_argument(
        "--vllm-model",
        default=None,
        help="Served vLLM model or LoRA name used in requests.",
    )
    parser.add_argument(
        "--vllm-api-key",
        default=None,
        help="Optional API key for the vLLM server.",
    )
    parser.add_argument(
        "--vllm-request-timeout",
        type=float,
        default=180.0,
        help="Per-request timeout for the vLLM server.",
    )
    parser.add_argument("--vllm-temperature", type=float, default=0.0)
    parser.add_argument("--vllm-top-p", type=float, default=None)
    parser.add_argument("--vllm-top-k", type=int, default=None)
    parser.add_argument(
        "--vllm-raw-text-mode",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Diagnostic mode: omit OpenAI tool_choice so a parser-free vLLM "
            "server returns ordinary assistant text. Tool schemas are still "
            "included in the request."
        ),
    )
    parser.add_argument(
        "--recover-missing-open-tool-tag",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Accept exactly one JSON tool call followed by </tool_call> whose "
            "opening <tool_call> tag is missing (Qwen3-VL-Thinking without a "
            "reasoning parser emits these)."
        ),
    )
    parser.add_argument("--image-resolution", type=int, default=512)
    parser.add_argument("--max-length", type=int, default=16384)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument(
        "--enable-thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Enable private model reasoning for vLLM inference. Reasoning is "
            "recorded for diagnostics but never added to either agent's history."
        ),
    )
    parser.add_argument(
        "--communication-mode",
        choices=COMMUNICATION_MODES,
        default="full",
        help=(
            "Communication ablation. 'full' is the unchanged canonical SFT/eval "
            "contract; 'intermediate' retains its communication guidance but "
            "removes the coordinator handshake; other modes progressively reduce "
            "eval-time guidance and tool exposure."
        ),
    )
    parser.add_argument("--model", default="gemini-3.5-flash-preview")
    parser.add_argument("--project", default=None)
    parser.add_argument("--location", default=None)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument(
        "--max-consecutive-waits", type=int, default=3,
        help="Consecutive wait_for_signal calls before the wait is capped "
             "and charged a turn. Bounds a wait->irrelevant-message->wait livelock.",
    )
    parser.add_argument(
        "--max-free-observations", type=int, default=4,
        help="Back-to-back get_image calls that do NOT advance the agent's "
             "virtual clock, matching the concurrent validator where "
             "observations are free. Beyond this the observation is charged, "
             "so an agent that only photographs cannot freeze the clock. "
             "Set 0 to charge every observation (pre-2026-08 behaviour).",
    )
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument(
        "--generate-hard-timeout", type=float, default=0.0,
        help="Hard per-call wall-clock bound in seconds (0 = 90s). "
             "Guards against google-genai retrying a wedged connection forever.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--thinking-budget", type=int, default=0)
    partial = parser.add_argument_group("partial observability")
    partial.add_argument(
        "--conflict-priority",
        choices=("agent-order", "seeded"),
        default="agent-order",
        help="Tie-break when two agents claim the same resource. 'agent-order' "
        "lets the lower agent id proceed; 'seeded' shuffles per turn. Rejecting "
        "both is symmetric and livelocks under deterministic retry.",
    )
    partial.add_argument(
        "--duration-multiplier",
        type=float,
        default=1.0,
        help="Scales measured executor sim-steps into virtual-clock duration.",
    )
    partial.add_argument(
        "--lenient-wait-discharge",
        action="store_true",
        help="Wake a waiting agent on ANY message from its partner, as the "
             "harness did before communicate.releases existed. Needed to "
             "evaluate checkpoints trained on data without that field; "
             "otherwise their waits are only cleared by the deadlock breaker.",
    )
    parser.add_argument(
        "--uniform-durations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="DEFAULT ON. Lock-step: every call costs one tick, so both agents "
             "advance together and only a wait+release can order them. This is "
             "the regime the corpus was validated in -- under the variable "
             "floors only 1055/1550 trajectories survive the concurrent "
             "validator, so evaluating there scores plans against timings they "
             "were never checked under. Pass --no-uniform-durations to test the "
             "timing dimension deliberately.",
    )
    partial.add_argument(
        "--duration-jitter",
        type=float,
        default=0.0,
        help="Randomly scale each call's duration by 1+-J. The only knob that "
             "actually changes the interleaving, since the executor teleports "
             "and measured sim-steps never reach the floors.",
    )
    partial.add_argument(
        "--duration-jitter-seed",
        type=int,
        default=0,
        help="Seed for --duration-jitter, so a perturbed schedule is reproducible.",
    )
    partial.add_argument(
        "--rejection-mode",
        choices=(REJECTION_MODE_REPORT_FAILED, REJECTION_MODE_TERMINATE_FIRST),
        default=REJECTION_MODE_REPORT_FAILED,
        help="'report-failed' appends a FAILED history line and permits "
        "recovery. 'terminate-first-rejection' stops after classifying the "
        "current proposal cycle and is intended for avoidance-target DAgger.",
    )
    partial.add_argument(
        "--rejection-budget-ratio",
        type=float,
        default=1.0,
        help="Rejected proposals get their own allowance, sized as this "
        "multiple of the step budget, instead of consuming the budget meant "
        "for productive work.",
    )
    partial.add_argument(
        "--success-criterion",
        choices=("fsm", "native-and-fsm"),
        default="fsm",
        help="What ends an episode successfully. 'fsm' (default) stops when the "
        "verified task spec's symbolic goal is satisfied; native success is "
        "still recorded as a diagnostic. Native disagreement is a property of "
        "the teleporting executor's geometric predicates, not of the policy, "
        "and requiring both wastes budget after the task is already done.",
    )
    parser.add_argument(
        "--two-pass-views",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Re-render canonical views for the predicted tool and re-predict.",
    )
    return parser.parse_args()


def _manifest_episodes(
    manifest: dict[str, Any],
    *,
    cohort_split: str | None,
    episodes_per_task: int | None,
    cohort_mode: str = "sampled",
    episodes_per_config: int = 1,
    max_configs_per_task: int | None = None,
    evaluation_seed: int | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Normalize legacy and fixed manifests into task -> episode records."""

    if episodes_per_task is not None and episodes_per_task < 1:
        raise ValueError("--episodes-per-task must be positive")
    if manifest.get("manifest_type") == "fixed_live_sim_configuration_targets":
        from robotalk.evaluation.fixed_cohort_selection import (
            select_configuration_episodes,
        )

        if cohort_split is None:
            raise ValueError("configuration cohorts require --cohort-split")
        source = manifest.get("configurations", {}).get(cohort_split)
        if source is None:
            raise ValueError(f"manifest has no split {cohort_split!r}")
        evaluation_seed = (
            manifest.get("cohort_seed") if evaluation_seed is None else evaluation_seed
        )
        count = 10 if episodes_per_task is None else episodes_per_task
        normalized = {}
        for task, configurations in source.items():
            rows = select_configuration_episodes(
                configurations,
                task_name=task,
                split=cohort_split,
                mode=cohort_mode,
                evaluation_seed=evaluation_seed,
                episodes_per_task=count,
                episodes_per_config=episodes_per_config,
                max_configs_per_task=max_configs_per_task,
            )
            normalized[task] = rows
        return normalized
    if cohort_split is not None:
        raise ValueError("--cohort-split is only valid with fixed_live_sim manifests")
    normalized = {
        task: [
            {"task_name": task, "trajectory_id": trajectory_id}
            for trajectory_id in sorted(ids)
        ]
        for task, ids in manifest["trajectory_ids_by_task"].items()
    }
    if episodes_per_task is not None:
        normalized = {
            task: episodes[:episodes_per_task] for task, episodes in normalized.items()
        }
    return normalized


def main() -> None:
    run_started = time.perf_counter()
    args = parse_args()
    if args.map_dpi < 1:
        raise SystemExit("--map-dpi must be at least 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    scene_compatibility_cache = None
    if args.scene_compatibility_cache is not None:
        from robotalk.scene_sampling import load_compatibility_cache
        scene_compatibility_cache = load_compatibility_cache(
            args.scene_compatibility_cache
        )
    try:
        episodes_by_task = _manifest_episodes(
            manifest,
            cohort_split=args.cohort_split,
            episodes_per_task=args.episodes_per_task,
            cohort_mode=args.cohort_mode,
            episodes_per_config=args.episodes_per_config,
            max_configs_per_task=args.max_configs_per_task,
            evaluation_seed=args.evaluation_seed,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if manifest.get("manifest_type") == "fixed_live_sim_configuration_targets":
        # The evaluator runs one fixed contract (private history, no step
        # indices); refuse a cohort frozen for any other.
        contract = manifest.get("contract", {})
        if contract.get("partial_step_index_mode", "none") != "none" or (
            contract.get("partial_history") is False
        ):
            raise SystemExit(
                "fixed cohort was built for a different prompt contract: "
                f"{contract!r}"
            )
    if args.tasks:
        keep = {t.strip() for t in args.tasks.split(",")}
        available = ", ".join(sorted(episodes_by_task))
        episodes_by_task = {
            t: episodes for t, episodes in episodes_by_task.items() if t in keep
        }
        if not episodes_by_task:
            raise SystemExit(
                f"--tasks matched no manifest tasks. Available tasks: {available}"
            )

    task_names = sorted(episodes_by_task)
    from robotalk.tasks.specs import load_verified_task_specs
    verified_specs_by_composite = {
        spec.composite_task: spec for spec in load_verified_task_specs()
    }

    results_path = args.output_dir / "live_sim_trajectories.jsonl"
    done: set[tuple[str, str]] = set()
    if args.resume and results_path.exists():
        existing_records = []
        for line in results_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                existing_records.append(json.loads(line))
        done = _completed_trajectory_keys(existing_records)

    policy = None
    policy_started = time.perf_counter()
    if args.backend == "vllm":
        if not args.vllm_model:
            raise SystemExit("--backend vllm requires --vllm-model")
        if args.vllm_request_timeout <= 0:
            raise SystemExit("--vllm-request-timeout must be positive")
        policy = VllmPolicy(args)
    elif args.backend == "gemini":
        policy = GeminiPolicy(args)
    policy_load_s = round(time.perf_counter() - policy_started, 3)
    print(f"[timing] policy_load_s={policy_load_s}", flush=True)

    total_done = len(done)
    firsts: dict[tuple[str, bool], bool] = {}
    if args.record_firsts:
        for task_name in episodes_by_task:
            for success, label in ((True, "success"), (False, "failure")):
                firsts[(task_name, success)] = (
                    args.output_dir / "recordings" / task_name / label
                ).exists()
    with results_path.open("a", encoding="utf-8") as out:
        for task_name, episodes in episodes_by_task.items():
            task_metadata = get_task_metadata(task_name)
            composite = task_metadata.composite_task
            session = None
            session_scene = None
            for episode in episodes:
                trajectory_id = episode.get("trajectory_id") or episode.get("episode_id")
                completion_id = episode.get("episode_id") or trajectory_id
                if (task_name, completion_id) in done:
                    continue
                if (
                    args.max_trajectories is not None
                    and total_done >= args.max_trajectories
                ):
                    break
                if episode.get("trajectory_id"):
                    traj_path = (
                        args.dataset_root
                        / task_metadata.dataset_name
                        / trajectory_id
                        / "original_trajectory.json"
                    )
                    trajectory = json.loads(traj_path.read_text(encoding="utf-8"))
                else:
                    from robotalk.tasks.shared.instances import (
                        initial_state_from_configuration,
                    )
                    task_spec = verified_specs_by_composite[composite]
                    trajectory = {
                        "trajectory_id": trajectory_id,
                        "task": task_spec.task_goal,
                        "composite_task": composite,
                        "initial_state": initial_state_from_configuration(
                            task_spec.initial_state,
                            episode["configuration"],
                        ),
                        "grounding_map": deepcopy(task_spec.grounding),
                        "coordinator_id": episode["configuration"].get(
                            "coordinator_id"
                        ),
                        "steps": [],
                        "evaluation_reference_action_count": max(
                            1,
                            len((task_spec.example_trajectory or {}).get("steps", [])),
                        ),
                        "configuration_signature": episode.get(
                            "configuration_signature"
                        ),
                        "physical_configuration_signature": episode.get(
                            "physical_configuration_signature"
                        ),
                    }
                episode_coordinator = episode.get("coordinator_id") or (
                    episode.get("configuration") or {}
                ).get("coordinator_id")
                if episode_coordinator is not None:
                    trajectory["coordinator_id"] = episode_coordinator
                scene = episode.get("scene")
                if scene is None and scene_compatibility_cache is not None:
                    from robotalk.scene_sampling import (
                        sample_compatible_scene,
                    )
                    physical_signature = episode.get(
                        "physical_configuration_signature"
                    )
                    if not physical_signature:
                        raise SystemExit(
                            "certified scene sampling requires each episode to expose "
                            "physical_configuration_signature"
                        )
                    scene = sample_compatible_scene(
                        scene_compatibility_cache,
                        task=composite,
                        physical_configuration_signature=physical_signature,
                        sampling_seed=args.scene_sampling_seed,
                        sample_index=int(episode.get("episode_rank", 0)),
                    )
                if scene is None:
                    scene = {
                        "layout": args.layout,
                        "style": args.style,
                        "seed": args.seed,
                    }
                scene_key = (scene["layout"], scene["style"], scene["seed"])
                if session is not None and session_scene != scene_key:
                    session.close()
                    session = None
                if session is None:
                    print(f"[{task_name}] building sim session...", flush=True)
                    session_started = time.perf_counter()
                    try:
                        session = SimSession(
                            composite_task=composite,
                            sample_trajectory=trajectory,
                            layout=scene["layout"],
                            style=scene["style"],
                            seed=scene["seed"],
                            gl_backend=args.gl_backend,
                            render_size=args.render_size,
                            map_dpi=args.map_dpi,
                            map_renderer=args.map_renderer,
                        )
                        session_build_s = round(
                            time.perf_counter() - session_started, 3
                        )
                        print(
                            f"[{task_name}] session_build_s={session_build_s}",
                            flush=True,
                        )
                        session_scene = scene_key
                    except Exception as exc:
                        result = {
                            "task_name": task_name,
                            "composite_task": composite,
                            "trajectory_id": trajectory_id,
                            "scene": scene,
                            "episode_id": episode.get("episode_id"),
                            "episode_rank": episode.get("episode_rank"),
                            "cohort_split": args.cohort_split,
                            "coordinator_id": episode_coordinator,
                            "configuration_signature": episode.get(
                                "configuration_signature"
                            ),
                            "termination": "harness_error",
                            "native_success": None,
                            "fsm_goal_satisfied": None,
                            "error": f"{type(exc).__name__}: {exc}",
                            "traceback": traceback.format_exc()[-2000:],
                        }
                        out.write(json.dumps(result, ensure_ascii=True) + "\n")
                        out.flush()
                        total_done += 1
                        print(
                            f"[{task_name}/{trajectory_id}] session build failed: "
                            f"{result['error']}", flush=True
                        )
                        break
                frames_dir = None
                if args.save_frames:
                    frames_dir = args.output_dir / "frames" / task_name / trajectory_id
                elif args.record_firsts:
                    # Record until both a success and a failure exist for the
                    # task; the final classification renames the directory.
                    if not (
                        firsts.get((task_name, True))
                        and firsts.get((task_name, False))
                    ):
                        frames_dir = (
                            args.output_dir
                            / "recordings"
                            / task_name
                            / trajectory_id
                        )
                started = time.time()
                try:
                    if args.backend == "oracle":
                        trajectory_policy = OraclePolicy(trajectory["steps"])
                    else:
                        trajectory_policy = policy
                    result = run_trajectory_partial(
                        session=session,
                        policy=trajectory_policy,
                        args=args,
                        task_name=task_name,
                        composite_task=composite,
                        trajectory=trajectory,
                        frames_dir=frames_dir,
                    )
                except Exception as exc:
                    result = {
                        "task_name": task_name,
                        "composite_task": composite,
                        "trajectory_id": trajectory_id,
                        "termination": "harness_error",
                        "native_success": None,
                        "fsm_goal_satisfied": None,
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc()[-2000:],
                    }
                result["communication_mode"] = args.communication_mode
                result["scene"] = scene
                if episode.get("episode_id") is not None:
                    result["episode_id"] = episode["episode_id"]
                    result["episode_rank"] = episode["episode_rank"]
                    result["cohort_split"] = args.cohort_split
                    result["coordinator_id"] = episode_coordinator
                    result["configuration_signature"] = episode.get(
                        "configuration_signature"
                    )
                result["elapsed_s"] = round(time.time() - started, 1)
                if "session_build_s" in locals() and session_build_s is not None:
                    result["session_build_s"] = session_build_s
                    session_build_s = None
                out.write(json.dumps(result, ensure_ascii=True) + "\n")
                out.flush()
                total_done += 1
                success = bool(result.get("native_success"))
                _finalize_first_recording(
                    frames_dir=frames_dir,
                    output_dir=args.output_dir,
                    task_name=task_name,
                    success=success,
                    firsts=firsts,
                    record_firsts=args.record_firsts,
                    save_frames=args.save_frames,
                    record_fps=args.record_fps,
                )
                native_error_text = (
                    f" native_error={result['native_error']}"
                    if result.get("native_error")
                    else ""
                )
                print(
                    f"[{task_name}/{trajectory_id}] "
                    f"native={result.get('native_success')}"
                    f"{native_error_text} "
                    f"fsm={result.get('fsm_goal_satisfied')} "
                    f"term={result.get('termination')} "
                    f"({result['elapsed_s']}s)",
                    flush=True,
                )
            if session is not None:
                session.close()

    _write_metrics(
        results_path,
        args.output_dir,
        policy_load_s=policy_load_s,
        total_elapsed_s=round(time.perf_counter() - run_started, 3),
    )


def _write_metrics(
    results_path: Path,
    output_dir: Path,
    *,
    policy_load_s: float | None = None,
    total_elapsed_s: float | None = None,
) -> None:
    raw_records = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records = _latest_trajectory_records(raw_records)
    n = len(records)
    if not n:
        return
    native = sum(1 for r in records if r.get("native_success"))
    fsm = sum(1 for r in records if r.get("fsm_goal_satisfied"))
    agree = sum(
        1
        for r in records
        if r.get("native_success") is not None
        and bool(r.get("native_success")) == bool(r.get("fsm_goal_satisfied"))
    )
    comparable = sum(
        1
        for r in records
        if r.get("native_success") is not None
        and r.get("fsm_goal_satisfied") is not None
    )
    rejection_counts = [int(r.get("rejected_total", r.get("rejected_steps", 0))) for r in records]
    parser_recovery_counts = [
        sum(
            1
            for step in r.get("steps", [])
            if step.get("parser_recovery") == "recovered_missing_open_tool_tag"
        )
        for r in records
    ]
    task_action_counts = [_task_action_counts_by_agent(r) for r in records]
    single_task_action_agent = [
        sum(count > 0 for count in counts.values()) == 1
        for counts in task_action_counts
    ]
    dominant_task_action_agent = [
        bool(sum(counts.values()))
        and max(counts.values(), default=0) / sum(counts.values()) >= 0.8
        for counts in task_action_counts
    ]
    fsm_success = [bool(r.get("fsm_goal_satisfied")) for r in records]
    native_success = [bool(r.get("native_success")) for r in records]
    metrics = {
        "prompt_contract_version": PROMPT_CONTRACT_VERSION,
        "communication_modes": sorted(
            {str(r.get("communication_mode", "full")) for r in records}
        ),
        "num_trajectories": n,
        "num_jsonl_records": len(raw_records),
        "num_superseded_records": len(raw_records) - n,
        "policy_load_s": policy_load_s,
        "total_elapsed_s": total_elapsed_s,
        "native_success_rate": native / n,
        "fsm_goal_rate": fsm / n,
        "trajectory_rejection_rate": sum(
            1 for r in records if r.get("had_rejection", r.get("rejected_steps", 0) > 0)
        ) / n,
        "num_trajectories_with_rejection": sum(
            1 for r in records if r.get("had_rejection", r.get("rejected_steps", 0) > 0)
        ),
        "total_rejected_tool_calls": sum(rejection_counts),
        "total_recovered_missing_open_tool_tags": sum(parser_recovery_counts),
        "num_trajectories_with_recovered_missing_open_tool_tag": sum(
            count > 0 for count in parser_recovery_counts
        ),
        "mean_recovered_missing_open_tool_tags_per_trajectory": (
            sum(parser_recovery_counts) / n
        ),
        "mean_rejected_tool_calls_per_trajectory": sum(rejection_counts) / n,
        "num_error_free_trajectories": sum(count == 0 for count in rejection_counts),
        "error_free_trajectory_rate": sum(count == 0 for count in rejection_counts) / n,
        "num_fsm_error_free_successes": sum(
            success and count == 0
            for success, count in zip(fsm_success, rejection_counts, strict=True)
        ),
        "fsm_error_free_success_rate": sum(
            success and count == 0
            for success, count in zip(fsm_success, rejection_counts, strict=True)
        ) / n,
        "num_native_error_free_successes": sum(
            success and count == 0
            for success, count in zip(native_success, rejection_counts, strict=True)
        ),
        "native_error_free_success_rate": sum(
            success and count == 0
            for success, count in zip(native_success, rejection_counts, strict=True)
        ) / n,
        "num_fsm_successes_with_recovery": sum(
            success and count > 0
            for success, count in zip(fsm_success, rejection_counts, strict=True)
        ),
        "fsm_success_with_recovery_rate": sum(
            success and count > 0
            for success, count in zip(fsm_success, rejection_counts, strict=True)
        ) / n,
        "num_single_task_action_agent_trajectories": sum(single_task_action_agent),
        "single_task_action_agent_rate": sum(single_task_action_agent) / n,
        "num_dominant_task_action_agent_trajectories_ge_0_8": sum(
            dominant_task_action_agent
        ),
        "dominant_task_action_agent_rate_ge_0_8": sum(
            dominant_task_action_agent
        ) / n,
        "num_single_task_action_agent_fsm_successes": sum(
            single and success
            for single, success in zip(single_task_action_agent, fsm_success, strict=True)
        ),
        "fsm_single_task_action_agent_success_rate": sum(
            single and success
            for single, success in zip(single_task_action_agent, fsm_success, strict=True)
        ) / n,
        "num_dominant_task_action_agent_fsm_successes_ge_0_8": sum(
            dominant and success
            for dominant, success in zip(dominant_task_action_agent, fsm_success, strict=True)
        ),
        "fsm_dominant_task_action_agent_success_rate_ge_0_8": sum(
            dominant and success
            for dominant, success in zip(dominant_task_action_agent, fsm_success, strict=True)
        ) / n,
        "single_task_action_agent_rate_among_fsm_successes": (
            sum(
                single and success
                for single, success in zip(single_task_action_agent, fsm_success, strict=True)
            ) / fsm
            if fsm
            else None
        ),
        "fsm_success_rate_if_terminate_on_first_rejection": sum(
            1
            for r in records
            if r.get(
                "fsm_success_if_terminate_on_first_rejection",
                bool(r.get("fsm_goal_satisfied")) and not r.get("rejected_steps", 0),
            )
        ) / n,
        "native_success_rate_if_terminate_on_first_rejection": sum(
            1
            for r in records
            if r.get(
                "native_success_if_terminate_on_first_rejection",
                bool(r.get("native_success")) and not r.get("rejected_steps", 0),
            )
        ) / n,
        "num_fsm_unsuccessful_if_terminate_on_first_rejection": n - sum(
            1
            for r in records
            if r.get(
                "fsm_success_if_terminate_on_first_rejection",
                bool(r.get("fsm_goal_satisfied")) and not r.get("rejected_steps", 0),
            )
        ),
        "num_native_unsuccessful_if_terminate_on_first_rejection": n - sum(
            1
            for r in records
            if r.get(
                "native_success_if_terminate_on_first_rejection",
                bool(r.get("native_success")) and not r.get("rejected_steps", 0),
            )
        ),
        "judge_agreement_rate": agree / comparable if comparable else None,
        "declared_complete_rate": sum(
            1 for r in records if r.get("declared_complete")
        ) / n,
        "harness_error_rate": sum(
            1 for r in records if r.get("termination") == "harness_error"
        ) / n,
        "terminations": {},
        "mean_steps_used": sum(r.get("steps_used", 0) for r in records) / n,
        "mean_communication_calls": sum(
            r.get("communication_calls", 0) for r in records
        ) / n,
        "mean_wait_calls": sum(r.get("wait_calls", 0) for r in records) / n,
        "mean_release_messages": sum(
            r.get("release_messages", 0) for r in records
        ) / n,
        "mean_trajectory_elapsed_s": _mean_present(
            r.get("elapsed_s") for r in records
        ),
        "total_session_build_s": sum(
            r.get("session_build_s", 0.0) for r in records
        ),
        "mean_model_proposal_elapsed_s": _mean_present(
            step.get("proposal_elapsed_s")
            for r in records
            for step in r.get("steps", [])
        ),
        "mean_expert_steps": sum(r.get("expert_steps", 0) for r in records) / n,
        "mean_partial_goal_fraction": sum(
            r.get("partial_goal_fraction", 0.0) for r in records
        ) / n,
        "mean_step_efficiency_ratio": _mean_present(
            r.get("step_efficiency_ratio") for r in records
        ),
        "mean_success_step_efficiency_ratio": _mean_present(
            r.get("step_efficiency_ratio")
            for r in records
            if r.get("native_success")
        ),
        "rejection_rate": (
            sum(r.get("rejected_steps", 0) for r in records)
            / max(1, sum(r.get("steps_used", 0) for r in records))
        ),
        "agent_turn_counts": {},
        "same_agent_transition_rate": None,
        "same_agent_run_lengths": {},
    }
    turns = [turn for r in records for turn in r.get("agent_turns", [])]
    metrics["agent_turn_counts"] = {
        agent: turns.count(agent) for agent in AGENT_IDS
    }
    same = total_transitions = 0
    run_lengths: dict[str, int] = {}
    for record in records:
        trajectory_turns = record.get("agent_turns", [])
        total_transitions += max(0, len(trajectory_turns) - 1)
        same += sum(a == b for a, b in zip(trajectory_turns, trajectory_turns[1:]))
        for length in _run_lengths(trajectory_turns):
            key = str(length)
            run_lengths[key] = run_lengths.get(key, 0) + 1
    metrics["same_agent_transition_rate"] = (
        same / total_transitions if total_transitions else None
    )
    metrics["same_agent_run_lengths"] = run_lengths

    # Scheduler metrics of the decentralized (private-context) episodes.
    partial_records = [r for r in records if r.get("partial_history")]
    if partial_records:
        conflicts = sum(r.get("resource_conflicts", 0) for r in partial_records)
        rejections = sum(r.get("rejected_steps", 0) for r in partial_records)
        cycles = sum(r.get("steps_used", 0) for r in partial_records)
        per_agent: dict[str, dict[str, float]] = {}
        for agent_id in AGENT_IDS:
            active = [
                r["per_agent"][agent_id]["active_sim_time"]
                for r in partial_records
                if r.get("per_agent", {}).get(agent_id)
            ]
            idle = [
                r["per_agent"][agent_id]["idle_sim_time"]
                for r in partial_records
                if r.get("per_agent", {}).get(agent_id)
            ]
            repeats = sum(
                r.get("per_agent", {}).get(agent_id, {}).get("repeated_proposals", 0)
                for r in partial_records
            )
            per_agent[agent_id] = {
                "mean_active_sim_time": _mean_present(active),
                "mean_idle_sim_time": _mean_present(idle),
                "repeated_proposals": repeats,
            }
        metrics["partial"] = {
            "trajectories": len(partial_records),
            "mean_makespan": _mean_present(
                r.get("makespan") for r in partial_records
            ),
            "simultaneous_cycle_rate": (
                sum(r.get("simultaneous_cycles", 0) for r in partial_records)
                / max(1, cycles)
            ),
            "resource_conflicts": conflicts,
            "rejections_observed": rejections,
            "failures_reported_to_model": rejections,
            "conflict_metrics_interpretable": rejections >= 20,
            "per_agent": per_agent,
        }
    for r in records:
        term = r.get("termination", "unknown")
        metrics["terminations"][term] = metrics["terminations"].get(term, 0) + 1
    (output_dir / "live_sim_metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


def _mean_present(values) -> float | None:
    present = [float(value) for value in values if value is not None]
    return sum(present) / len(present) if present else None


_NON_TASK_ACTION_TOOLS = {
    "communicate",
    "get_image",
    "give_space",
    "navigate_to_fixture",
    "report_failed",
    "task_complete",
    "wait_for_signal",
}


def _task_action_counts_by_agent(record: dict) -> dict[str, int]:
    """Count accepted task-state-changing actions, excluding coordination and motion."""

    counts = {agent_id: 0 for agent_id in AGENT_IDS}
    for step in record.get("steps", []):
        proposal = step.get("proposal") or {}
        tool = proposal.get("tool")
        agent = step.get("agent") or proposal.get("agent")
        if (
            agent in counts
            and step.get("executed")
            and step.get("sim_success", True) is not False
            and tool not in _NON_TASK_ACTION_TOOLS
        ):
            counts[agent] += 1
    return counts


def _run_lengths(turns: list[str]) -> list[int]:
    if not turns:
        return []
    lengths: list[int] = []
    current = turns[0]
    length = 1
    for turn in turns[1:]:
        if turn == current:
            length += 1
        else:
            lengths.append(length)
            current, length = turn, 1
    lengths.append(length)
    return lengths


def _assemble_recording_videos(recording_dir: Path, *, fps: int) -> None:
    """Create one MP4 per camera from the saved step stills."""

    import imageio.v2 as imageio

    for camera_dir in sorted(path for path in recording_dir.iterdir() if path.is_dir()):
        frames = sorted(camera_dir.glob("*.jpg"))
        if not frames:
            continue
        writer = imageio.get_writer(recording_dir / f"{camera_dir.name}.mp4", fps=fps)
        try:
            for frame in frames:
                writer.append_data(imageio.imread(frame))
        finally:
            writer.close()


if __name__ == "__main__":
    main()
