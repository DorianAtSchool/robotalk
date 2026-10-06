"""Aggregate live-sim evaluation outputs into the paper's Figs. 5-7 results.

Reads ``outputs/eval/<model_id>/<cohort_split>/aggregate/live_sim_trajectories.jsonl``
for every model in ``configs/experiments.yaml`` and writes
``outputs/figures/43_10/artifact.json``, the input of the figure exporters, plus
a paper-vs-reproduced comparison table.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import yaml

from robotalk.utils import output_root

ROOT = Path(__file__).resolve().parents[2]
SPLIT_LABELS = {
    "train_task_types": "Trained tasks (43)",
    "heldout_task_types": "Held-out tasks (10)",
}
MODE_LABELS = {
    "full": "Full",
    "intermediate": "Intermediate",
    "minimal": "Minimal",
    "none": "None",
    "unguided": "Unguided",
}

NON_TASK_ACTION_TOOLS = {
    "communicate", "get_image", "give_space", "navigate_to_fixture",
    "report_failed", "task_complete", "wait_for_signal",
}


def load_trajectory_rows(path: Path) -> list[dict]:
    latest: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = str(row.get("trajectory_id") or row.get("episode_id"))
        latest[key] = row
    return list(latest.values())


def single_physical_agent_fsm_success(row: dict, *, require_error_free: bool) -> bool:
    counts = {"agent_0": 0, "agent_1": 0}
    for step in row.get("steps", []):
        proposal = step.get("proposal") or {}
        agent = step.get("agent") or proposal.get("agent")
        if (
            agent in counts and step.get("executed")
            and step.get("sim_success", True) is not False
            and proposal.get("tool") not in NON_TASK_ACTION_TOOLS
        ):
            counts[agent] += 1
    rejected = int(row.get("rejected_total", row.get("rejected_steps", 0))) > 0
    return (
        bool(row.get("fsm_goal_satisfied"))
        and (not require_error_free or not rejected)
        and sum(value > 0 for value in counts.values()) == 1
    )


def error_free_single_agent(row: dict) -> bool:
    return single_physical_agent_fsm_success(row, require_error_free=True)


def wilson(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def metric_row(*, model: str, checkpoint: str, split: str, successes: int, episodes: int,
               source_kind: str, source_path: str, communication_mode: str = "Full",
               single_agent_successes: int = 0, final_successes: int | None = None,
               final_single_agent_successes: int = 0) -> dict:
    final_successes = successes if final_successes is None else final_successes
    if not 0 <= single_agent_successes <= successes <= episodes:
        raise ValueError(
            f"invalid nested counts: one-agent error-free={single_agent_successes}, "
            f"error-free={successes}, episodes={episodes}"
        )
    if not 0 <= final_single_agent_successes <= final_successes <= episodes:
        raise ValueError(
            f"invalid final nested counts: one-agent final={final_single_agent_successes}, "
            f"final={final_successes}, episodes={episodes}"
        )
    low, high = wilson(successes, episodes)
    single_low, single_high = wilson(single_agent_successes, episodes)
    final_low, final_high = wilson(final_successes, episodes)
    return {
        "model": model,
        "checkpoint": checkpoint,
        "model_checkpoint": f"{model} · {checkpoint}",
        "split": split,
        "communication_mode": communication_mode,
        "model_and_mode": f"{communication_mode} · {model}",
        "error_free_successes": successes,
        "episodes": episodes,
        "error_free_success_rate": successes / episodes if episodes else 0.0,
        "ci_low": low,
        "ci_high": high,
        "error_free_single_agent_successes": single_agent_successes,
        "error_free_single_agent_success_rate": single_agent_successes / episodes if episodes else 0.0,
        "single_agent_ci_low": single_low,
        "single_agent_ci_high": single_high,
        "fsm_successes": final_successes,
        "fsm_success_rate": final_successes / episodes if episodes else 0.0,
        "fsm_ci_low": final_low,
        "fsm_ci_high": final_high,
        "fsm_single_physical_agent_successes": final_single_agent_successes,
        "fsm_single_physical_agent_success_rate": final_single_agent_successes / episodes if episodes else 0.0,
        "source_kind": source_kind,
        "source_path": source_path,
    }



def _counts(rows: list[dict]) -> dict[str, int]:
    def rejected(row: dict) -> bool:
        return int(row.get("rejected_total", row.get("rejected_steps", 0))) > 0

    return {
        "episodes": len(rows),
        "successes": sum(bool(r.get("fsm_goal_satisfied")) and not rejected(r) for r in rows),
        "single_agent_successes": sum(error_free_single_agent(r) for r in rows),
        "final_successes": sum(bool(r.get("fsm_goal_satisfied")) for r in rows),
        "final_single_agent_successes": sum(
            single_physical_agent_fsm_success(r, require_error_free=False) for r in rows
        ),
    }


def collect(config: dict, eval_root: Path) -> tuple[dict[str, list[dict]], list[tuple]]:
    """Return the exporter datasets and (model_id, split, n, reproduced, paper) rows."""

    datasets: dict[str, list[dict]] = {
        "scale_results": [],
        "thirty_results": [],
        "communication_results": [],
    }
    comparison = []
    for model_id, model in config["models"].items():
        for split in config["evaluation"]["cohort_splits"]:
            path = eval_root / model_id / split / "aggregate" / "live_sim_trajectories.jsonl"
            if not path.is_file():
                continue
            counts = _counts(load_trajectory_rows(path))
            fig5 = "fig5" in model["figures"]
            row = metric_row(
                model=model["label"],
                checkpoint="out of box" if fig5 else "epoch 1.0",
                split=SPLIT_LABELS[split],
                successes=counts["successes"],
                episodes=counts["episodes"],
                source_kind="live_sim_trajectories",
                source_path=str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
                communication_mode=MODE_LABELS[model.get("communication_mode", "full")],
                single_agent_successes=counts["single_agent_successes"],
                final_successes=counts["final_successes"],
                final_single_agent_successes=counts["final_single_agent_successes"],
            )
            if fig5:
                datasets["communication_results"].append(row)
            if "fig6" in model["figures"]:
                datasets["scale_results"].append(row)
            if "fig7" in model["figures"]:
                datasets["thirty_results"].append(row)
            paper = config.get("paper_results", {}).get(model_id)
            paper_value = None
            if paper is not None:
                paper_value = paper[config["evaluation"]["cohort_splits"].index(split)]
            comparison.append((model_id, split, counts["episodes"], counts["successes"], paper_value))
    return datasets, comparison


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/experiments.yaml")
    parser.add_argument("--eval-root", type=Path, default=output_root() / "eval")
    parser.add_argument("--output-dir", type=Path, default=output_root() / "figures/43_10")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    datasets, comparison = collect(config, args.eval_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact = {"snapshot": {"datasets": datasets}}
    (args.output_dir / "artifact.json").write_text(
        json.dumps(artifact, indent=2) + "\n", encoding="utf-8"
    )
    print(f"{'model':28} {'split':20} {'N':>4} {'reproduced':>11} {'paper':>6}")
    for model_id, split, n, ours, paper in comparison:
        paper_text = "" if paper is None else str(paper)
        print(f"{model_id:28} {split:20} {n:>4} {ours:>11} {paper_text:>6}")
    print(args.output_dir / "artifact.json")


if __name__ == "__main__":
    main()
