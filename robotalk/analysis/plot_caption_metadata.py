"""Caption metadata for PNGs in the fixed live-sim 43/10 artifact."""

from __future__ import annotations

import csv
from pathlib import Path


PLOT_METADATA = {
    "communication_ablation_error_free.png": ("Communication guidance ablation", (
        "All 53 tasks, n=530; whiskers show 95% Wilson intervals."
    )),
    "communication_ablation_error_free_single_agent.png": ("Communication guidance ablation", (
        "All 53 tasks, n=530; whiskers show 95% Wilson intervals."
    )),
    "communication_ablation_error_free_by_split.png": ("Communication guidance ablation", (
        "10 episodes per task; whiskers show 95% Wilson intervals."
    )),
    "communication_ablation_error_free_by_split_single_agent.png": ("Communication guidance ablation", (
        "All rates use total episodes; whiskers show 95% Wilson intervals."
    )),
    "sft_scaling_instruct.png": ("Qwen3-VL-8B-Instruct SFT scaling", (
        "Solid: in-training tasks; hatched: held-out tasks. One epoch; "
        "95% Wilson intervals."
    )),
    "sft_scaling_thinking_rationale.png": ("Qwen3-VL-8B-Thinking + rationale SFT scaling", (
        "Solid: in-training tasks; hatched: held-out tasks. One epoch; "
        "95% Wilson intervals."
    )),
    "sft_scaling_combined.png": ("Qwen3-VL-8B SFT scaling", (
        "One epoch; 10 episodes per task; whiskers show 95% Wilson intervals."
    )),
    "thinking_rationale_30task_ablations.png": ("Qwen3-VL-8B-Instruct vs. Qwen3-VL-8B-Thinking SFT ablations", (
        "Solid: in-training tasks; hatched: held-out tasks. "
        "30 trajectories/task, one epoch."
    )),
    "task_phase_distribution_combined.png": ("Task phase distribution", (
        "All 53 verified tasks; phases are defined by the number of task stages."
    )),
    "task_phase_distribution_by_split.png": ("Task phase distribution by split", (
        "Solid: in-training tasks; hatched: held-out tasks. Seeded 43/10 assignment."
    )),
}


def write_caption_csv(artifact_dir: Path) -> Path:
    path = artifact_dir / "plot_captions.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("plot", "title", "caption"))
        writer.writerows(
            (plot, title, caption)
            for plot, (title, caption) in PLOT_METADATA.items()
        )
    return path
