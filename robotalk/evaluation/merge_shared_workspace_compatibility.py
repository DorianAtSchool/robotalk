"""Apply task-scene shared-workspace results to an initialization cache."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path
from typing import Any

from robotalk.scene_sampling import (
    SCENE_POLICY_VERSION,
    candidate_scenes_v1,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initialization-cache", type=Path, required=True)
    parser.add_argument("--workspace-audit-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--require-workspace-scene-count",
        type=int,
        default=len(candidate_scenes_v1()),
    )
    parser.add_argument("--positive-control-task")
    parser.add_argument("--positive-control-layout", type=int)
    args = parser.parse_args()

    if (args.positive_control_task is None) != (
        args.positive_control_layout is None
    ):
        raise ValueError(
            "--positive-control-task and --positive-control-layout must be used together"
        )

    cache = json.loads(args.initialization_cache.read_text())
    if cache.get("scene_policy_version") != SCENE_POLICY_VERSION:
        raise ValueError("initialization cache uses the wrong scene policy")
    files = sorted(
        args.workspace_audit_dir.glob("layout_*_style_*_seed_*.json")
    )
    if len(files) != args.require_workspace_scene_count:
        raise ValueError(
            f"found {len(files)} workspace audits, expected "
            f"{args.require_workspace_scene_count}"
        )

    incompatible: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    audit_errors: dict[tuple[str, str], list[str]] = defaultdict(list)
    control_hits = []
    audited_scene_signatures = []
    for path in files:
        payload = json.loads(path.read_text())
        scene = payload["scene"]
        scene_sig = scene["scene_signature"]
        audited_scene_signatures.append(scene_sig)
        for row in payload.get("records") or []:
            if row["compatible"]:
                continue
            incompatible[(row["task"], scene_sig)].append(row)
            if (
                args.positive_control_task is not None
                and scene["layout"] == args.positive_control_layout
                and row["task"] == args.positive_control_task
            ):
                control_hits.append({"scene": scene, "workspace": row["workspace_fixture_id"]})
        for row in payload.get("task_errors") or []:
            audit_errors[(row["task"], scene_sig)].append(row["error"])
    if len(audited_scene_signatures) != len(set(audited_scene_signatures)):
        raise ValueError("workspace audit contains duplicate scene signatures")
    initialization_scene_signatures = {
        scene["scene_signature"] for scene in cache.get("candidate_scenes", [])
    }
    if set(audited_scene_signatures) != initialization_scene_signatures:
        missing = initialization_scene_signatures - set(audited_scene_signatures)
        unexpected = set(audited_scene_signatures) - initialization_scene_signatures
        raise ValueError(
            "workspace and initialization scene sets differ: "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )
    if args.positive_control_task is not None and not control_hits:
        raise ValueError(
            "requested positive control was not detected: "
            f"task={args.positive_control_task}, "
            f"layout={args.positive_control_layout}"
        )

    merged = deepcopy(cache)
    removed = []
    counts = []
    zero = []
    for task, task_row in merged["tasks"].items():
        for config_sig, config in task_row["configurations"].items():
            kept = []
            for scene in config["compatible_scenes"]:
                key = (task, scene["scene_signature"])
                if key not in incompatible and key not in audit_errors:
                    kept.append(scene)
                    continue
                reason = (
                    "shared_workspace_incompatible"
                    if key in incompatible
                    else "shared_workspace_task_initialization_error"
                )
                removed.append(
                    {
                        "task": task,
                        "physical_configuration_signature": config_sig,
                        "scene": scene,
                        "reason": reason,
                        "workspaces": [
                            row["workspace_fixture_id"] for row in incompatible.get(key, [])
                        ],
                        "errors": audit_errors.get(key, []),
                    }
                )
            config["compatible_scenes"] = kept
            counts.append(len(kept))
            if not kept:
                zero.append(
                    {
                        "task": task,
                        "physical_configuration_signature": config_sig,
                    }
                )
    if zero:
        raise ValueError(
            f"{len(zero)} physical configurations have no scene after workspace gating"
        )

    merged["schema_version"] = 2
    merged["workspace_compatibility_gate"] = {
        "source_files": [str(path.resolve()) for path in files],
        "audited_scene_count": len(files),
        "positive_control": {
            "required": args.positive_control_task is not None,
            "task": args.positive_control_task,
            "layout": args.positive_control_layout,
            "detections": control_hits,
        },
        "incompatible_task_scene_pairs": len(incompatible),
        "task_scene_initialization_error_pairs": len(audit_errors),
        "removed_configuration_scene_bindings": removed,
    }
    merged["summary"]["post_workspace_compatible_scene_count_histogram"] = dict(
        sorted(Counter(counts).items())
    )
    merged["summary"]["post_workspace_zero_compatible_configurations"] = zero
    merged["summary"]["workspace_incompatible_task_scene_pairs"] = len(incompatible)
    merged["summary"]["workspace_task_scene_initialization_error_pairs"] = len(audit_errors)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(merged, indent=2) + "\n")
    print(json.dumps(merged["summary"], indent=2))


if __name__ == "__main__":
    main()
