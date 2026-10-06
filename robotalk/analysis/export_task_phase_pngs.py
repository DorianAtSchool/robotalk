"""Export compact task-phase figures matching the 43/10 evaluation plots."""

from __future__ import annotations

from collections import Counter
import json
import re
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from robotalk.analysis.export_sft_scale_pngs import (
    ARTIFACT_DIR,
    BLUE,
    INK,
    ORANGE,
    _style_axis,
)
from robotalk.analysis.plot_caption_metadata import write_caption_csv


SPLIT = ROOT / "configs/splits/43_train_10_heldout.json"
ATTRIBUTES = ROOT / "robotalk/tasks/task_attributes.json"
VERIFIED_SPECS = ROOT / "robotalk/tasks/specs/verified"
PHASES = ("Phase 1", "Phase 2", "Phase 3", "Phase 4")
PHASE_LABELS = ("Phase 1\n1 stage", "Phase 2\n2–3 stages", "Phase 3\n4–5 stages", "Phase 4\n6+ stages")


def _snake_case(name: str) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower()


def _phase(num_stages: int) -> str:
    if num_stages == 1:
        return "Phase 1"
    if num_stages <= 3:
        return "Phase 2"
    if num_stages <= 5:
        return "Phase 3"
    return "Phase 4"


def _load_counts() -> tuple[Counter, dict[str, Counter]]:
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    attributes = json.loads(ATTRIBUTES.read_text(encoding="utf-8"))["tasks"]
    stage_counts = {
        _snake_case(row["name"]): int(row["num_subtasks"])
        for row in attributes
    }
    verified = {
        _snake_case(json.loads(path.read_text(encoding="utf-8"))["composite_task"])
        for path in VERIFIED_SPECS.glob("*.json")
    }
    trained = set(split["train_tasks"])
    held_out = set(split["held_out_tasks"])
    if trained & held_out:
        raise ValueError(f"Tasks appear in both splits: {sorted(trained & held_out)}")
    if trained | held_out != verified:
        raise ValueError(
            f"Split does not cover verified inventory: missing={sorted(verified - trained - held_out)}, "
            f"extra={sorted((trained | held_out) - verified)}"
        )
    missing = verified - stage_counts.keys()
    if missing:
        raise ValueError(f"Tasks missing num_subtasks metadata: {sorted(missing)}")

    by_split = {
        "In-training tasks": Counter(_phase(stage_counts[task]) for task in trained),
        "Held-out tasks": Counter(_phase(stage_counts[task]) for task in held_out),
    }
    return by_split["In-training tasks"] + by_split["Held-out tasks"], by_split


def _save(fig, filename: str) -> None:
    from robotalk.analysis.plot_style_utils import style_paper_figure
    style_paper_figure(fig)
    fig.tight_layout()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    path = ARTIFACT_DIR / filename
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(path)


def _combined_plot(overall: Counter) -> None:
    fig, axis = plt.subplots(figsize=(3.5, 2.55 * 0.70))
    x = np.arange(len(PHASES))
    values = [overall[phase] for phase in PHASES]
    bars = axis.bar(x, values, 0.62, color=BLUE, edgecolor=INK, linewidth=0.45)
    axis.bar_label(bars, labels=[str(value) for value in values], padding=1.5, fontsize=6.5)
    _style_axis(axis, list(PHASE_LABELS), ylim=30)
    axis.set_ylabel("Number of tasks")
    axis.set_yticks(np.arange(0, 31, 5), [str(value) for value in range(0, 31, 5)])
    _save(fig, "task_phase_distribution_combined.png")


def _split_plot(by_split: dict[str, Counter]) -> None:
    fig, axis = plt.subplots(figsize=(3.5, 2.55 * 0.70))
    x = np.arange(len(PHASES))
    width = 0.34
    for split_index, (split_name, color) in enumerate(
        (("In-training tasks", BLUE), ("Held-out tasks", ORANGE))
    ):
        values = [by_split[split_name][phase] for phase in PHASES]
        bars = axis.bar(
            x + (split_index - 0.5) * width,
            values,
            width,
            color=color,
            edgecolor=INK,
            linewidth=0.45,
            hatch="///" if split_index else None,
            label=f"{split_name} ({43 if split_index == 0 else 10})",
        )
        axis.bar_label(bars, labels=[str(value) for value in values], padding=1.5, fontsize=6.2)
    _style_axis(axis, list(PHASE_LABELS), ylim=24)
    axis.set_ylabel("Number of tasks")
    axis.set_yticks(np.arange(0, 25, 4), [str(value) for value in range(0, 25, 4)])
    axis.legend(
        handles=[
            Patch(facecolor=BLUE, edgecolor=INK, label="In-training tasks (43)"),
            Patch(facecolor=ORANGE, edgecolor=INK, hatch="///", label="Held-out tasks (10)"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        frameon=False,
        fontsize=6.1,
        ncol=2,
        borderaxespad=0,
        columnspacing=0.8,
        handletextpad=0.4,
    )
    _save(fig, "task_phase_distribution_by_split.png")


def main() -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.3})
    overall, by_split = _load_counts()
    _combined_plot(overall)
    _split_plot(by_split)
    print(write_caption_csv(ARTIFACT_DIR))


if __name__ == "__main__":
    main()
