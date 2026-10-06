"""Freeze configuration targets before generated carriers or model outcomes exist."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from robotalk.training.prompting import PROMPT_CONTRACT_VERSION
from robotalk.training.task_registry import get_task_metadata
from robotalk.tasks.specs import load_verified_task_specs
from robotalk.tasks.shared.workspace_semantics import (
    canonicalize_configuration,
)
from robotalk.scene_sampling import (
    SCENE_POLICY_VERSION,
    physical_configuration_signature,
)


def _hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-config-manifest", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cohort-seed", type=int, default=20260817)
    args = parser.parse_args()

    source = json.loads(args.initial_config_manifest.read_text())
    specs = {
        spec.composite_task: spec for spec in load_verified_task_specs()
    }
    selection = json.loads(args.selection.read_text())
    train = set(selection["train_tasks"])
    declared_heldout = set(selection["held_out_tasks"])
    configurations: dict[str, dict[str, list[dict[str, Any]]]] = {
        "train_task_types": {}, "heldout_task_types": {}
    }
    for task in source["tasks"]:
        composite = task["task"]
        dataset_name = get_task_metadata(composite).dataset_name
        split = "train_task_types" if dataset_name in train else "heldout_task_types"
        if dataset_name not in train | declared_heldout:
            split = "heldout_task_types"
        unique: dict[str, dict[str, Any]] = {}
        for run in task["runs"]:
            configuration = canonicalize_configuration(
                run["configuration"], specs[composite].initial_state
            )
            signature = _hash(configuration)
            unique.setdefault(signature, configuration)
        ordered = list(unique.items())
        random.Random(f"{args.cohort_seed}:{dataset_name}").shuffle(ordered)

        def configuration_row(signature: str, configuration: dict[str, Any], rank: int) -> dict[str, Any]:
            return {
                "configuration_rank": rank,
                "composite_task": composite,
                "configuration_signature": signature,
                "configuration": configuration,
                "physical_configuration_signature": physical_configuration_signature(
                    configuration
                ),
                "initialization": "configuration_native",
                "scene_binding": "certified_cache_at_evaluation",
            }

        rows = [configuration_row(sig, config, rank) for rank, (sig, config) in enumerate(ordered)]
        if len({row["configuration_signature"] for row in rows}) != len(rows):
            raise ValueError(f"duplicate canonical configurations for {composite}")
        configurations[split][dataset_name] = rows

    payload = {
        "schema_version": 1,
        "manifest_type": "fixed_live_sim_configuration_targets",
        "frozen_configuration_selection": True,
        "executable": True,
        "cohort_seed": args.cohort_seed,
        "default_evaluation": {"mode": "sampled", "episodes_per_task": 10},
        "scene_policy_version": SCENE_POLICY_VERSION,
        "evaluation_modes": {
            "sampled": "Uniform shuffled passes over all unique configurations.",
            "full_config": "Every configuration once by default, with optional repetitions or a configuration cap.",
        },
        "source_manifest": str(args.initial_config_manifest.resolve()),
        "source_manifest_sha256": hashlib.sha256(args.initial_config_manifest.read_bytes()).hexdigest(),
        "prompt_contract_version": PROMPT_CONTRACT_VERSION,
        "contract": {
            "partial_history": True,
            "partial_step_index_mode": "none",
            "partial_observation_mode": "consume-once",
            "uniform_durations": True,
            "success_criterion": "fsm",
            "rejection_mode": "report_failed",
            "contention_policy": "concurrent_fsm",
        },
        "configurations": configurations,
    }
    payload["content_hash"] = _hash(payload)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
