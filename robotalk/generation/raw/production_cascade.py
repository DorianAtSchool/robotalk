"""Generate one task resumably with the validated bounded retry cascade."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import re

from robotalk.generation.raw import cascade_canary
from robotalk.scene_sampling import physical_configuration_signature
from robotalk.tasks import get_task_definition
from robotalk.tasks.shared.workspace_semantics import (
    canonical_agent_workspace,
)


def _canonical_physical_configuration(instance) -> dict:
    """Match the physical-state contract used by manifests and scene audits."""

    initial_state = instance.initial_state
    return {
        "agent_locations": {
            agent_id: canonical_agent_workspace(initial_state, state.get("location"))
            for agent_id, state in sorted((initial_state.get("agents") or {}).items())
        },
        "access_states": {
            f"{fixture_id}.{part_id}": part.get("state")
            for fixture_id, fixture in sorted(
                (initial_state.get("fixtures") or {}).items()
            )
            for part_id, part in sorted((fixture.get("parts") or {}).items())
            if part.get("state") in {"open", "closed"}
        },
    }


def _dataset_task_name(composite_task: str) -> str:
    """Return the canonical rendered/training directory name."""

    return re.sub(r"(?<!^)(?=[A-Z])", "_", composite_task).lower()


def _combined_usage(attempts: list[dict]) -> dict:
    usages = [
        usage
        for attempt in attempts
        for key in ("usage", "critic_usage")
        if isinstance((usage := attempt.get(key)), dict)
    ]
    prompt = sum(int(u.get("prompt_tokens") or 0) for u in usages)
    output = sum(int(u.get("candidates_tokens") or u.get("output_tokens") or 0) for u in usages)
    reasoning = sum(int(u.get("thoughts_tokens") or 0) for u in usages)
    return {
        "retry_costs_included": True,
        "api_call_count": len(usages),
        "successful_attempt_number": len(attempts),
        "prompt_tokens": prompt,
        "cached_input_tokens": sum(int(u.get("cached_content_tokens") or 0) for u in usages),
        "output_tokens": output,
        "reasoning_tokens": reasoning,
        "total_tokens": sum(int(u.get("total_tokens") or 0) for u in usages),
        "usage_source": "cascade_api_usage_metadata",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True)
    parser.add_argument("--num-runs", type=int, default=150)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--location", default="global")
    parser.add_argument("--temperature", type=float, default=0.6)
    args = parser.parse_args()
    task_root = args.output_root / _dataset_task_name(args.task)
    trajectory_root = task_root / "trajectories"
    attempt_root = task_root / "cascade_attempts"
    trajectory_root.mkdir(parents=True, exist_ok=True)
    attempt_root.mkdir(parents=True, exist_ok=True)
    definition = get_task_definition(args.task)
    stages: Counter[str] = Counter()
    failures = []
    completed = []
    for run_index in range(args.num_runs):
        trajectory_id = f"traj_{run_index:06d}"
        destination = trajectory_root / f"{trajectory_id}.json"
        result_path = attempt_root / f"run_{run_index:06d}.json"
        if destination.exists():
            existing = json.loads(destination.read_text())
            resume_config = argparse.Namespace(
                task=args.task,
                num_runs=args.num_runs,
                run_index=run_index,
                location=args.location,
                temperature=args.temperature,
            )
            runtime_config = cascade_canary._config(
                resume_config,
                model=cascade_canary.FLASH_MODEL,
                thinking="low",
            )
            resume_instance = definition.build_task_instance(run_index, runtime_config)
            canonical_physical = _canonical_physical_configuration(resume_instance)
            expected_signature = physical_configuration_signature(canonical_physical)
            if existing.get("physical_configuration_signature") != expected_signature:
                existing["physical_configuration"] = canonical_physical
                existing["physical_configuration_signature"] = expected_signature
                destination.write_text(json.dumps(existing, indent=2) + "\n")
            stages[str(existing.get("cascade_accepted_stage"))] += 1
            completed.append(run_index)
            continue
        cascade_args = argparse.Namespace(
            task=args.task,
            run_index=run_index,
            num_runs=args.num_runs,
            output=str(result_path),
            location=args.location,
            temperature=args.temperature,
            low_direct_only=False,
        )
        result = cascade_canary.run(cascade_args)
        if result.get("is_valid") is not True:
            failures.append(
                {
                    "run_index": run_index,
                    "validation": result.get("final_validation"),
                }
            )
            continue
        config = cascade_canary._config(
            cascade_args,
            model=cascade_canary.FLASH_MODEL,
            thinking="low",
        )
        instance = definition.build_task_instance(run_index, config)
        candidate = deepcopy(result["accepted_candidate"])
        record = definition.build_trajectory_record(
            candidate=candidate,
            validation=deepcopy(result["final_validation"]),
            trajectory_id=trajectory_id,
            generation_usage=_combined_usage(result["attempts"]),
            task_instance=instance,
        )
        record["tick_rows"] = deepcopy(candidate["tick_rows"])
        record["task"] = args.task
        record["coordinator_id"] = instance.coordinator_id
        normalized_physical = _canonical_physical_configuration(instance)
        record["physical_configuration"] = normalized_physical
        record["physical_configuration_signature"] = physical_configuration_signature(
            normalized_physical
        )
        record["cascade_accepted_stage"] = result["accepted_stage"]
        record["cascade_attempt_count"] = len(result["attempts"])
        record["sampling_metadata"] = {
            "strategy": "structured_random",
            "run_index": run_index,
            "attempt_number": len(result["attempts"]),
        }
        destination.write_text(json.dumps(record, indent=2) + "\n")
        stages[str(result["accepted_stage"])] += 1
        completed.append(run_index)
    summary = {
        "task": args.task,
        "num_runs": args.num_runs,
        "completed_run_indices": sorted(completed),
        "num_trajectories": len(completed),
        # Keep the production output directly consumable by the shared image
        # insertion/revalidation and rendering pipeline.  Paths are relative to
        # this summary, matching the canonical task-level dataset contract.
        "trajectory_files": [
            {
                "trajectory_id": f"traj_{run_index:06d}",
                "path": f"trajectories/traj_{run_index:06d}.json",
            }
            for run_index in sorted(completed)
        ],
        "failures": failures,
        "accepted_stage_counts": dict(sorted(stages.items())),
    }
    (task_root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: summary[k] for k in ("task", "num_runs", "num_trajectories", "accepted_stage_counts")}, indent=2))
    return 0 if len(completed) == args.num_runs else 1


if __name__ == "__main__":
    raise SystemExit(main())
