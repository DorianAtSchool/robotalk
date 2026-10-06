"""Freeze a validated configuration-native cohort with immutable provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--scene-cache", type=Path, required=True)
    parser.add_argument("--scene-contract", type=Path, required=True)
    parser.add_argument("--canary-verification", type=Path, required=True)
    parser.add_argument("--replay-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.targets.read_text())
    if payload.get("executable") is not True:
        raise ValueError("configuration cohort is not executable")
    scene_contract = json.loads(args.scene_contract.read_text())
    canary = json.loads(args.canary_verification.read_text())
    replay = json.loads(args.replay_metrics.read_text())
    if not scene_contract.get("valid") or not canary.get("gate_passed"):
        raise ValueError("scene or generation canary gate did not pass")
    if replay.get("num_trajectories") != 6 or replay.get("terminations") != {
        "goal_satisfied": 6
    }:
        raise ValueError("exact live replay smoke did not pass 6/6")
    payload.update(
        {
            "frozen": True,
            "freeze_contract": "configuration_native_fixed_live_sim_v1",
            "provenance": {
                "targets_sha256_before_freeze": _sha(args.targets),
                "scene_cache": str(args.scene_cache.resolve()),
                "scene_cache_sha256": _sha(args.scene_cache),
                "scene_contract": str(args.scene_contract.resolve()),
                "scene_contract_sha256": _sha(args.scene_contract),
                "physical_canary_verification": str(
                    args.canary_verification.resolve()
                ),
                "physical_canary_verification_sha256": _sha(
                    args.canary_verification
                ),
                "exact_replay_metrics": str(args.replay_metrics.resolve()),
                "exact_replay_metrics_sha256": _sha(args.replay_metrics),
            },
        }
    )
    payload.pop("content_hash", None)
    payload["content_hash"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()
