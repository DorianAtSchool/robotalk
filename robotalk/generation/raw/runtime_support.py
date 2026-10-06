"""Provide shared runtime helpers for validation, sampling, and JSON parsing."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
import re
from typing import Any

from robotalk.generation.raw.config import (
    RuntimeConfig,
    TRAJECTORY_ID_DIGITS,
)
from robotalk.generation.runtime.client import (
    GenerationResult,
    GenerationUsage,
)
from robotalk.generation.sampling import get_sampling_strategy
from robotalk.tasks import (
    ResponseFormatValidationError,
    ToolArgumentSemanticValidationError,
    TaskValidator,
    TrajectoryValidationError,
)
from robotalk.tasks.shared.concurrent_fsm import (
    LOCK_STEP,
    ConcurrentTaskValidator,
)
from robotalk.utils import stable_json_sha256


def format_trajectory_variation_key(
    trajectory_index: int,
    attempt_index: int,
) -> str:
    """Formats the retry variation key used to diversify model attempts."""

    return (
        f"traj-{trajectory_index:0{TRAJECTORY_ID_DIGITS}d}-attempt-{attempt_index:02d}"
    )


def _candidate_signature(candidate: dict[str, Any]) -> str:
    return stable_json_sha256(candidate, default=str)


def _sampling_strategy_for_runtime(runtime_config: RuntimeConfig):
    """Returns the configured sampling strategy for the current runtime."""

    return get_sampling_strategy(getattr(runtime_config, "sampling", "structured_random"))


def _unwrap_generation_response(
    raw_response: Any,
) -> tuple[Any, GenerationUsage | None]:
    # Both SDK wrappers and test doubles feed through here, so normalize the
    # payload shape before sampling code looks at it.
    if isinstance(raw_response, GenerationResult):
        return raw_response.payload, raw_response.usage
    return raw_response, None


def _is_tick_format(candidate: dict[str, Any]) -> bool:
    """Tick output is identified by shape, so no flag has to be threaded here."""

    return isinstance(candidate, dict) and isinstance(candidate.get("ticks"), list)


def _derive_coordination(
    candidate: dict[str, Any],
    initial_state: dict[str, Any],
) -> dict[str, Any]:
    """Fill in the waits and releases before the trajectory is validated.

    The FSM requires a wait wherever two agents share a resource, but the model
    is not asked to produce one -- placement is decidable from the plan, so
    asking is pure waste. It costs tokens, and the FSM's own repair loop
    oscillates on it: the model adds the wait it was told about, that wait is
    then unreleased, and the next attempt removes it again.

    Deriving here, before validation, means a retry is only ever spent on the
    work itself. Measured over the 1560-trajectory subset, this converts 836
    wait-protocol failures into passes and breaks nothing.
    """

    import random

    from robotalk.generation.insert_waits import insert

    steps = candidate.get("steps") or []
    if not steps:
        return candidate
    # Seeded from the plan so phrasing varies between trajectories but stays
    # reproducible for any one of them.
    seed = hashlib.sha256(
        json.dumps(steps, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    derived, _ = insert(
        deepcopy(steps),
        initial_state.get("fixtures") or {},
        set(initial_state.get("objects") or {}),
        random.Random(int(seed[:16], 16)),
        {
            agent: (state or {}).get("location")
            for agent, state in (initial_state.get("agents") or {}).items()
        },
    )
    updated = dict(candidate)
    updated["steps"] = derived
    return updated


def _validate_candidate(
    candidate: dict[str, Any],
    validator: TaskValidator,
    *,
    enforce_validation: bool,
    enable_static_referential_validation: bool = True,
    initial_state: dict[str, Any] | None = None,
    allowed_tool_specs: dict[str, dict[str, Any]] | None = None,
    derive_coordination: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    wrote_ticks = _is_tick_format(candidate)
    if wrote_ticks:
        # The concurrent FSM owns the canonical tick representation and does
        # the only permitted flattening after invocation state is validated.
        # ...and judge it with the validator that actually RUNS the plan. The
        # linear one replays the flat list in written order, where two agents
        # never collide and no wait ever blocks, so it cannot see whether the
        # coordination the model placed works. Under it, tick output that
        # deadlocks on the first tick validated clean. Raising here is what
        # makes the generation loop retry, which is the only way the data ends
        # up correct while the model is still allowed to be wrong.
        # LOCK_STEP only: the model writes a grid of equal instants, so that
        # is the regime its plan means. See ConcurrentTaskValidator.__init__.
        validator = ConcurrentTaskValidator(validator, models=(LOCK_STEP,))
    elif derive_coordination and isinstance(initial_state, dict):
        candidate = _derive_coordination(candidate, initial_state)
    try:
        if (
            enforce_validation
            and enable_static_referential_validation
            and isinstance(initial_state, dict)
            and isinstance(allowed_tool_specs, dict)
        ):
            reference_candidate = (
                validator.canonicalize(candidate) if wrote_ticks else candidate
            )
            _validate_candidate_references_without_sim(
                reference_candidate,
                initial_state=initial_state,
                allowed_tool_specs=allowed_tool_specs,
            )
        validation = dict(validator.validate(candidate))
        normalized_candidate = validation.pop("normalized_candidate", candidate)
        if wrote_ticks and "tick_rows" not in normalized_candidate:
            # Normalization rebuilds the record from a fixed field list. The
            # rows have to survive it: they are the schedule, and the flat
            # steps cannot be turned back into them.
            normalized_candidate = dict(normalized_candidate)
            normalized_candidate["tick_rows"] = candidate["tick_rows"]
        if wrote_ticks:
            # The persisted form omits explicit blocked markers. Require that
            # exact canonical representation to pass the same validator too;
            # generation must never save a record postprocessing will reject.
            persisted_validation = dict(validator.validate(normalized_candidate))
            persisted_normalized = persisted_validation.pop(
                "normalized_candidate", normalized_candidate
            )
            validation = persisted_validation
            normalized_candidate = persisted_normalized
        return validation, normalized_candidate
    except TrajectoryValidationError as exc:
        if enforce_validation:
            raise
        # Preserve the invalid trace for inspection when validation is disabled.
        return (
            {
                "is_valid": False,
                "validation_disabled": True,
                "error_type": exc.error_type,
                "error_base_type": exc.error_base_type,
                "error": str(exc),
                "error_details": dict(exc.details) if exc.details else None,
                "step": exc.step if isinstance(exc.step, int) else None,
                "checks": [],
                "final_state": None,
                "signature": _candidate_signature(candidate),
            },
            candidate,
        )


def _declared_tool_arg_names(tool_spec: dict[str, Any]) -> set[str]:
    declared_arg_names = {
        arg_name
        for arg_name in tool_spec.get("tool_args", ())
        if isinstance(arg_name, str)
    }
    declared_arg_names.update(
        arg_name
        for arg_name in tool_spec.get("optional_tool_args", ())
        if isinstance(arg_name, str)
    )
    for arg_group in tool_spec.get("tool_arg_any_of", ()):
        if not isinstance(arg_group, (list, tuple)):
            continue
        declared_arg_names.update(
            arg_name for arg_name in arg_group if isinstance(arg_name, str)
        )
    return declared_arg_names


def _support_site_parent_by_id(initial_state: dict[str, Any]) -> dict[str, str]:
    fixtures_by_id = initial_state.get("fixtures") or {}
    if not isinstance(fixtures_by_id, dict):
        return {}

    parent_by_site: dict[str, str] = {}
    for fixture_id, fixture_state in fixtures_by_id.items():
        if not isinstance(fixture_id, str) or not isinstance(fixture_state, dict):
            continue
        support_sites = fixture_state.get("support_sites") or {}
        if isinstance(support_sites, dict):
            for support_site_id in support_sites:
                if isinstance(support_site_id, str):
                    parent_by_site[support_site_id] = fixture_id
        elif isinstance(support_sites, list):
            for support_site_id in support_sites:
                if isinstance(support_site_id, str):
                    parent_by_site[support_site_id] = fixture_id
    return parent_by_site


def _step_fixture_id(
    args: dict[str, Any],
    support_site_parent_by_id: dict[str, str],
) -> str | None:
    for fixture_arg_name in (
        "target_id",
        "fixture_id",
        "reference_fixture_id",
        "source_id",
        "support_id",
        "receptacle_id",
    ):
        fixture_value = args.get(fixture_arg_name)
        if not isinstance(fixture_value, str):
            continue
        return support_site_parent_by_id.get(fixture_value, fixture_value)
    return None


def _validate_candidate_references_without_sim(
    candidate: dict[str, Any],
    *,
    initial_state: dict[str, Any],
    allowed_tool_specs: dict[str, dict[str, Any]],
) -> None:
    """Validate task-local symbolic part/control/site references without sim init.

    This is intentionally stricter than plain FSM replay for simulator-facing id
    fields. It blocks obvious hallucinated ids early in raw generation runs so
    the same protection applies both inside and outside the pipeline.
    """

    steps = candidate.get("steps")
    if not isinstance(steps, list):
        return

    fixtures_by_id = initial_state.get("fixtures") or {}
    if not isinstance(fixtures_by_id, dict):
        fixtures_by_id = {}
    support_site_parent_map = _support_site_parent_by_id(initial_state)
    fixture_ids = {
        fixture_id for fixture_id in fixtures_by_id if isinstance(fixture_id, str)
    }

    for step in steps:
        if not isinstance(step, dict):
            continue
        tool_name = step.get("tool")
        args = step.get("args")
        if not isinstance(tool_name, str) or not isinstance(args, dict):
            continue
        tool_spec = allowed_tool_specs.get(tool_name)
        if not isinstance(tool_spec, dict):
            continue

        step_index = step.get("step")
        step_number = step_index if isinstance(step_index, int) else None
        declared_arg_names = _declared_tool_arg_names(tool_spec)
        if declared_arg_names:
            unexpected_arg_names = sorted(set(args) - declared_arg_names)
            if unexpected_arg_names:
                raise ToolArgumentSemanticValidationError(
                    f"{tool_name} does not declare args {unexpected_arg_names}.",
                    step=step_number,
                    details={
                        "tool": tool_name,
                        "unexpected_args": unexpected_arg_names,
                        "declared_args": sorted(declared_arg_names),
                    },
                )

        fixture_id = _step_fixture_id(args, support_site_parent_map)
        fixture_state = (
            fixtures_by_id.get(fixture_id) if isinstance(fixture_id, str) else None
        )
        fixture_parts = (
            fixture_state.get("parts", {})
            if isinstance(fixture_state, dict)
            and isinstance(fixture_state.get("parts"), dict)
            else {}
        )
        fixture_controls = (
            fixture_state.get("controls", {})
            if isinstance(fixture_state, dict)
            and isinstance(fixture_state.get("controls"), dict)
            else {}
        )
        fixture_support_sites = (
            fixture_state.get("support_sites", {})
            if isinstance(fixture_state, dict)
            else {}
        )
        fixture_support_site_ids = (
            set(fixture_support_sites)
            if isinstance(fixture_support_sites, dict)
            else {
                site_id for site_id in fixture_support_sites if isinstance(site_id, str)
            }
            if isinstance(fixture_support_sites, list)
            else set()
        )

        part_id = args.get("part_id")
        allowed_part_ids = tool_spec.get("allowed_part_ids")
        if (
            isinstance(part_id, str)
            and isinstance(allowed_part_ids, list)
            and all(isinstance(part, str) for part in allowed_part_ids)
            and part_id not in allowed_part_ids
        ):
            raise ToolArgumentSemanticValidationError(
                f"Unknown part/control {part_id!r} for fixture {fixture_id!r}.",
                step=step_number,
                details={
                    "tool": tool_name,
                    "part_id": part_id,
                    "fixture_id": fixture_id,
                },
            )
        if isinstance(part_id, str) and fixture_parts and part_id not in fixture_parts:
            raise ToolArgumentSemanticValidationError(
                f"Unknown part/control {part_id!r} for fixture {fixture_id!r}.",
                step=step_number,
                details={
                    "tool": tool_name,
                    "part_id": part_id,
                    "fixture_id": fixture_id,
                },
            )

        control_id = args.get("control_id")
        allowed_control_ids = tool_spec.get("allowed_control_ids")
        if (
            isinstance(control_id, str)
            and isinstance(allowed_control_ids, list)
            and all(isinstance(control, str) for control in allowed_control_ids)
            and control_id not in allowed_control_ids
        ):
            raise ToolArgumentSemanticValidationError(
                f"Unknown part/control {control_id!r} for fixture {fixture_id!r}.",
                step=step_number,
                details={
                    "tool": tool_name,
                    "control_id": control_id,
                    "fixture_id": fixture_id,
                },
            )
        if (
            isinstance(control_id, str)
            and fixture_controls
            and control_id not in fixture_controls
        ):
            raise ToolArgumentSemanticValidationError(
                f"Unknown part/control {control_id!r} for fixture {fixture_id!r}.",
                step=step_number,
                details={
                    "tool": tool_name,
                    "control_id": control_id,
                    "fixture_id": fixture_id,
                },
            )

        for site_arg_name in ("source_site_id", "target_site_id"):
            site_id = args.get(site_arg_name)
            if not isinstance(site_id, str):
                continue
            if site_id in fixture_ids:
                raise ToolArgumentSemanticValidationError(
                    f"{tool_name} uses fixture id {site_id!r} as {site_arg_name}; expected a support-site id.",
                    step=step_number,
                    details={
                        "tool": tool_name,
                        "fixture_id": fixture_id,
                        site_arg_name: site_id,
                    },
                )
            allowed_site_ids = tool_spec.get(f"allowed_{site_arg_name[:-3]}_ids")
            if (
                isinstance(allowed_site_ids, list)
                and all(isinstance(site, str) for site in allowed_site_ids)
                and site_id not in allowed_site_ids
            ):
                raise ToolArgumentSemanticValidationError(
                    f"{tool_name} uses unknown {site_arg_name} {site_id!r}.",
                    step=step_number,
                    details={
                        "tool": tool_name,
                        site_arg_name: site_id,
                        "allowed_site_ids": allowed_site_ids,
                    },
                )
            if fixture_support_site_ids and site_id not in fixture_support_site_ids:
                raise ToolArgumentSemanticValidationError(
                    f"{tool_name} uses unknown {site_arg_name} {site_id!r} for fixture {fixture_id!r}.",
                    step=step_number,
                    details={
                        "tool": tool_name,
                        "fixture_id": fixture_id,
                        site_arg_name: site_id,
                    },
                )


def extract_json_candidate(raw_response: Any) -> dict[str, Any]:
    if isinstance(raw_response, dict):
        return raw_response
    if not isinstance(raw_response, str):
        raise ResponseFormatValidationError(
            f"Unsupported model response type: {type(raw_response).__name__}"
        )

    # Some SDK/model paths wrap the JSON object in fences or surrounding text.
    stripped = raw_response.strip()
    fenced_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, re.DOTALL)
    if fenced_match:
        stripped = fenced_match.group(1)

    try:
        return json.loads(stripped)
    except json.JSONDecodeError as exc:
        json_match = re.search(r"(\{.*\})", stripped, re.DOTALL)
        if not json_match:
            raise ResponseFormatValidationError(
                "Model response did not contain JSON."
            ) from exc
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError as inner_exc:
            raise ResponseFormatValidationError(
                "Model response contained invalid JSON."
            ) from inner_exc
