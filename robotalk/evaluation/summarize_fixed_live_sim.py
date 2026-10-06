"""Summarize fixed live-sim runs with micro/macro rates and uncertainty."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any


_NON_TASK_ACTION_TOOLS = {
    "communicate",
    "get_image",
    "give_space",
    "navigate_to_fixture",
    "report_failed",
    "task_complete",
    "wait_for_signal",
}

ERROR_CATEGORIES = (
    "call_construction_grounding",
    "state_action_execution",
    "coordination_progress",
)

_COORDINATION_ERROR_TYPES = {
    "CommunicationStepSemanticValidationError",
    "DeadlockSemanticValidationError",
    "MissingInitialCommunicationSemanticValidationError",
    "ResourceConflictSemanticValidationError",
    "WaitSignalSemanticValidationError",
}
_CALL_ERROR_TYPES = {
    "PlacementDestinationSemanticValidationError",
    "ResponseFormatValidationError",
    "ToolArgumentSemanticValidationError",
    "TrajectoryStructureValidationError",
    "UnexpectedStepIndexSemanticValidationError",
    "UnsupportedToolSemanticValidationError",
}


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0:
        return [float("nan"), float("nan")]
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return [max(0.0, center - margin), min(1.0, center + margin)]


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def hierarchical_bootstrap(
    outcomes_by_task: dict[str, list[int]], *, draws: int, seed: int
) -> dict[str, list[float]]:
    """Resample tasks, then episodes inside each sampled task."""
    if not outcomes_by_task:
        return {"micro": [float("nan"), float("nan")], "macro": [float("nan"), float("nan")]}
    rng = random.Random(seed)
    task_names = sorted(outcomes_by_task)
    micros: list[float] = []
    macros: list[float] = []
    for _ in range(draws):
        sampled_tasks = [rng.choice(task_names) for _ in task_names]
        task_rates = []
        pooled = []
        for task in sampled_tasks:
            source = outcomes_by_task[task]
            sampled = [rng.choice(source) for _ in source]
            task_rates.append(sum(sampled) / len(sampled))
            pooled.extend(sampled)
        micros.append(sum(pooled) / len(pooled))
        macros.append(sum(task_rates) / len(task_rates))
    return {
        "micro": [_percentile(micros, 0.025), _percentile(micros, 0.975)],
        "macro": [_percentile(macros, 0.025), _percentile(macros, 0.975)],
    }


def paired_hierarchical_bootstrap(
    differences_by_task: dict[str, list[int]], *, draws: int, seed: int
) -> list[float]:
    """CI for a paired success-rate difference on identical episode IDs."""
    if not differences_by_task:
        return [float("nan"), float("nan")]
    rng = random.Random(seed)
    tasks = sorted(differences_by_task)
    estimates = []
    for _ in range(draws):
        sampled_tasks = [rng.choice(tasks) for _ in tasks]
        values = []
        for task in sampled_tasks:
            source = differences_by_task[task]
            values.extend(rng.choice(source) for _ in source)
        estimates.append(sum(values) / len(values))
    return [_percentile(estimates, 0.025), _percentile(estimates, 0.975)]


def paired_comparison(
    records: list[dict[str, Any]], baseline: list[dict[str, Any]], *, draws: int, seed: int
) -> dict[str, Any]:
    current_by_id = {row["episode_id"]: row for row in records}
    baseline_by_id = {row["episode_id"]: row for row in baseline}
    if set(current_by_id) != set(baseline_by_id):
        missing = len(set(baseline_by_id) - set(current_by_id))
        extra = len(set(current_by_id) - set(baseline_by_id))
        raise ValueError(f"paired runs differ in episode IDs (missing={missing}, extra={extra})")
    by_split: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    for episode_id, row in current_by_id.items():
        other = baseline_by_id[episode_id]
        difference = int(bool(row.get("fsm_goal_satisfied"))) - int(
            bool(other.get("fsm_goal_satisfied"))
        )
        by_split[row["cohort_split"]][row["task_name"]].append(difference)
    result = {}
    for split, by_task in by_split.items():
        values = [x for task_values in by_task.values() for x in task_values]
        result[split] = {
            "micro_success_rate_difference": sum(values) / len(values),
            "paired_hierarchical_bootstrap_95_ci": paired_hierarchical_bootstrap(
                dict(by_task), draws=draws, seed=seed
            ),
            "num_paired_episodes": len(values),
        }
    return result


def load_latest(paths: list[Path]) -> list[dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            key = row.get("episode_id")
            if not key:
                raise ValueError(f"{path} contains a row without episode_id")
            latest[key] = row
    return list(latest.values())


def _episode_outcomes(row: dict[str, Any]) -> dict[str, int]:
    """Return comparable Bernoulli outcomes for one live-sim episode."""

    success = int(bool(row.get("fsm_goal_satisfied")))
    rejected = int(row.get("rejected_total", row.get("rejected_steps", 0))) > 0
    counts = {"agent_0": 0, "agent_1": 0}
    for step in row.get("steps", []):
        proposal = step.get("proposal") or {}
        agent = step.get("agent") or proposal.get("agent")
        tool = proposal.get("tool")
        if (
            agent in counts
            and step.get("executed")
            and step.get("sim_success", True) is not False
            and tool not in _NON_TASK_ACTION_TOOLS
        ):
            counts[agent] += 1
    total_actions = sum(counts.values())
    active_agents = sum(count > 0 for count in counts.values())
    dominant_share = max(counts.values(), default=0) / total_actions if total_actions else 0.0
    return {
        "fsm_success": success,
        "fsm_error_free_success": int(success and not rejected),
        "fsm_single_agent_success": int(success and active_agents == 1),
        "fsm_dominant_agent_success_ge_0_8": int(success and dominant_share >= 0.8),
        "fsm_error_free_dominant_agent_success_ge_0_8": int(
            success and not rejected and dominant_share >= 0.8
        ),
    }


def _metric_summary(values: list[int]) -> dict[str, Any]:
    count = sum(values)
    total = len(values)
    return {
        "count": count,
        "rate": count / total,
        "wilson_95_ci": wilson_interval(count, total),
    }


def classify_error(reason: str) -> tuple[str, str]:
    """Map one raw evaluator error to an exhaustive, stable plot category."""

    normalized = str(reason or "unknown error").strip()
    error_type = normalized.split(":", 1)[0].strip()
    lower = normalized.lower()
    if (
        error_type in _COORDINATION_ERROR_TYPES
        or "opening protocol" in lower
        or "opening observations and handshake" in lower
        or "resource conflict" in lower
        or "mutual wait" in lower
        or "permanent wait" in lower
    ):
        category = "coordination_progress"
    elif (
        error_type in _CALL_ERROR_TYPES
        or "response did not contain" in lower
        or "no function call" in lower
        or "tool_call> body" in lower
        or "missing/invalid \"agent\"" in lower
        or "malformed vllm" in lower
        or "unknown object" in lower
        or "unknown fixture" in lower
        or "unknown site" in lower
        or "unknown support" in lower
        or "invalid observation views" in lower
    ):
        category = "call_construction_grounding"
    else:
        # Once a proposal parses and grounds, remaining FSM and executor
        # failures concern current state, location, prerequisites, or action
        # execution. This includes physical simulator failures, whose raw
        # subtype remains available for diagnostic drill-down.
        category = "state_action_execution"
    return category, error_type or "unknown"


def _episode_errors(row: dict[str, Any]) -> list[dict[str, str]]:
    events: list[dict[str, str]] = []
    for step in row.get("steps", []):
        proposal = step.get("proposal") or {}
        rejected = (
            step.get("legal") is False
            or step.get("sim_success") is False
            or "error" in proposal
        )
        if rejected:
            reason = str(
                step.get("reason")
                or step.get("sim_error")
                or proposal.get("error")
                or "unknown rejected call"
            )
            category, subtype = classify_error(reason)
            events.append({"category": category, "subtype": subtype, "reason": reason})
        if step.get("observation_capped"):
            events.append({
                "category": "coordination_progress",
                "subtype": "RepetitiveObservationCall",
                "reason": "consecutive get_image calls exceeded the free-observation cap",
            })
        if step.get("wait_capped"):
            events.append({
                "category": "coordination_progress",
                "subtype": "RepetitiveWaitCall",
                "reason": "consecutive wait_for_signal calls exceeded the wait cap",
            })
    for _ in range(int(row.get("mutual_wait_deadlocks", 0))):
        events.append({
            "category": "coordination_progress",
            "subtype": "MutualWaitDeadlock",
            "reason": "all remaining agents were mutually blocked",
        })
    if row.get("termination") == "permanent_wait_no_runnable_agents":
        events.append({
            "category": "coordination_progress",
            "subtype": "PermanentWaitNoRunnableAgent",
            "reason": "all runnable work ended while an agent remained permanently blocked",
        })
    return events


def _error_summary(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    by_category: dict[str, dict[str, Any]] = {}
    for category in ERROR_CATEGORIES:
        recovered = failed = 0
        affected = 0
        subtypes: dict[str, int] = defaultdict(int)
        for episode in episodes:
            matching = [
                event for event in episode["errors"] if event["category"] == category
            ]
            if matching:
                affected += 1
            for event in matching:
                subtypes[event["subtype"]] += 1
            if episode["outcomes"]["fsm_success"]:
                recovered += len(matching)
            else:
                failed += len(matching)
        total = recovered + failed
        by_category[category] = {
            "total_events": total,
            "recovered_success_events": recovered,
            "failed_trajectory_events": failed,
            "affected_episodes": affected,
            "events_per_episode": total / len(episodes),
            "affected_episode_rate": affected / len(episodes),
            "subtypes": dict(sorted(subtypes.items())),
        }
    return {
        "num_episodes": len(episodes),
        "total_events": sum(row["total_events"] for row in by_category.values()),
        "categories": by_category,
    }


def summarize_records(records: list[dict[str, Any]], *, draws: int, seed: int) -> dict[str, Any]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    harness_errors = 0
    rejections = 0
    for row in records:
        split = row.get("cohort_split")
        if split is None:
            raise ValueError(f"episode {row['episode_id']} has no cohort_split")
        grouped[split][row["task_name"]].append({
            "outcomes": _episode_outcomes(row),
            "errors": _episode_errors(row),
        })
        harness_errors += int(row.get("termination") == "harness_error")
        rejections += int(bool(row.get("had_rejection") or row.get("rejected_steps", 0)))
    splits = {}
    for split, by_task in sorted(grouped.items()):
        outcomes = [
            value["outcomes"]["fsm_success"]
            for task_values in by_task.values()
            for value in task_values
        ]
        per_task = {}
        for task, task_outcomes in sorted(by_task.items()):
            metrics = {
                key: _metric_summary([row["outcomes"][key] for row in task_outcomes])
                for key in task_outcomes[0]["outcomes"]
            }
            final = metrics["fsm_success"]
            per_task[task] = {
                "successes": final["count"],
                "num_episodes": len(task_outcomes),
                "success_rate": final["rate"],
                "wilson_95_ci": final["wilson_95_ci"],
                "metrics": metrics,
                "errors": _error_summary(task_outcomes),
            }
        success_by_task = {
            task: [row["outcomes"]["fsm_success"] for row in task_outcomes]
            for task, task_outcomes in by_task.items()
        }
        task_rates = {
            task: sum(values) / len(values)
            for task, values in sorted(success_by_task.items())
        }
        boot = hierarchical_bootstrap(success_by_task, draws=draws, seed=seed)
        successes = sum(outcomes)
        pooled = [row for task_outcomes in by_task.values() for row in task_outcomes]
        metrics = {
            key: _metric_summary([row["outcomes"][key] for row in pooled])
            for key in pooled[0]["outcomes"]
        }
        splits[split] = {
            "num_tasks": len(by_task),
            "num_episodes": len(outcomes),
            "successes": successes,
            "micro_success_rate": successes / len(outcomes),
            "macro_success_rate": sum(task_rates.values()) / len(task_rates),
            "hierarchical_bootstrap_95_ci": boot,
            "wilson_micro_95_ci": wilson_interval(successes, len(outcomes)),
            "per_task_success_rate": task_rates,
            "per_task": per_task,
            "metrics": metrics,
            "errors": _error_summary(pooled),
        }
    return {
        "schema_version": 4,
        "success_definition": "fsm_goal_satisfied",
        "ci_method": {
            "chart_bars": "95% Wilson binomial interval",
            "micro_macro_inference": "hierarchical bootstrap: resample tasks, then episodes within task",
        },
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
        "num_records": len(records),
        "harness_error_episodes": harness_errors,
        "episodes_with_rejection": rejections,
        "error_taxonomy": {
            "call_construction_grounding": "Malformed/schema-invalid calls, unknown symbols, invalid destinations, and invalid camera requests.",
            "state_action_execution": "State, inventory, navigation, prerequisite, and simulator-execution failures.",
            "coordination_progress": "Communication, contention, wait/release, deadlock, and repetitive observation/wait failures.",
        },
        "splits": splits,
    }


def _markdown(label: str, summary: dict[str, Any]) -> str:
    lines = [f"# Fixed live-sim evaluation: {label}", "", "| Split | Episodes | Micro success (95% CI) | Macro success (95% CI) |", "|---|---:|---:|---:|"]
    for split, row in summary["splits"].items():
        micro_ci = row["hierarchical_bootstrap_95_ci"]["micro"]
        macro_ci = row["hierarchical_bootstrap_95_ci"]["macro"]
        lines.append(
            f"| {split} | {row['num_episodes']} | {row['micro_success_rate']:.1%} "
            f"({micro_ci[0]:.1%}–{micro_ci[1]:.1%}) | {row['macro_success_rate']:.1%} "
            f"({macro_ci[0]:.1%}–{macro_ci[1]:.1%}) |"
        )
    lines.extend(["", f"Harness errors: {summary['harness_error_episodes']}; episodes with any rejection: {summary['episodes_with_rejection']}."])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, action="append", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--baseline-results", type=Path, action="append", default=[])
    parser.add_argument("--baseline-label")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260812)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_latest(args.results)
    summary = summarize_records(records, draws=args.bootstrap_draws, seed=args.seed)
    summary["label"] = args.label
    if args.baseline_results:
        if not args.baseline_label:
            raise SystemExit("--baseline-results requires --baseline-label")
        summary["paired_comparison"] = {
            "baseline_label": args.baseline_label,
            "splits": paired_comparison(
                records,
                load_latest(args.baseline_results),
                draws=args.bootstrap_draws,
                seed=args.seed,
            ),
        }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(_markdown(args.label, summary), encoding="utf-8")


if __name__ == "__main__":
    main()
