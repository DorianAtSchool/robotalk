"""Export compact, single-column SFT scaling figures from the 43/10 artifact."""

from __future__ import annotations

import json
import re

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from robotalk.analysis.plot_caption_metadata import write_caption_csv
from robotalk.utils import output_root
from robotalk.analysis.plot_style_utils import label_bars_above_whiskers, style_paper_figure


ARTIFACT_DIR = output_root() / "figures/43_10"
ARTIFACT = ARTIFACT_DIR / "artifact.json"
SPLITS = ("Trained tasks (43)", "Held-out tasks (10)")
SPLIT_LABELS = ("In-training tasks", "Held-out tasks")
SCALES = (30, 60, 90, 120, 150)
BLUE = "#3976d2"
ORANGE = "#e07a5f"
INK = "#263241"
GRID = "#e3e7ec"


def _style_axis(axis, labels: list[str], *, ylim: float = 1.0) -> None:
    axis.set_ylabel("Error-free FSM success")
    axis.set_xticks(np.arange(len(labels)), labels)
    axis.set_ylim(0, ylim)
    ticks = np.arange(0, 1.01, 0.2)
    axis.set_yticks(ticks, [f"{value:.0%}" for value in ticks])
    axis.grid(axis="y", color=GRID, linewidth=0.55)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#9aa4b2")


def _error(row: dict) -> np.ndarray:
    value = row["error_free_success_rate"]
    return np.array([
        [max(0.0, value - row["ci_low"])],
        [max(0.0, row["ci_high"] - value)],
    ])


def _save(fig, name: str) -> None:
    style_paper_figure(fig)
    fig.tight_layout()
    path = ARTIFACT_DIR / name
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(path)


def _index(rows: list[dict]) -> dict[tuple[str, str], dict]:
    return {(row["model"], row["split"]): row for row in rows}


def _split_scale_plot(indexed: dict, models: list[str], name: str, color: str) -> None:
    fig, axis = plt.subplots(figsize=(3.5, 2.55))
    x = np.arange(len(models))
    width = 0.34
    for split_index, split in enumerate(SPLITS):
        positions = x + (split_index - 0.5) * width
        selected = [indexed[(model, split)] for model in models]
        values = [row["error_free_success_rate"] for row in selected]
        yerr = np.concatenate([_error(row) for row in selected], axis=1)
        bars = axis.bar(
            positions, values, width, color=color, edgecolor=INK, linewidth=0.45,
            hatch="///" if split_index else None, yerr=yerr,
            error_kw={"elinewidth": 0.65, "capsize": 1.6, "capthick": 0.65},
            label=SPLIT_LABELS[split_index],
        )
        axis.bar_label(bars, labels=[f"{value:.0%}" for value in values], padding=1.5, fontsize=5.7)
    _style_axis(axis, [re.search(r"(\d+)/task", model).group(1) for model in models], ylim=1.10)
    axis.set_xlabel("Training trajectories per task", labelpad=2)
    axis.legend(
        handles=[
            Patch(facecolor="white", edgecolor=INK, label=SPLIT_LABELS[0]),
            Patch(facecolor="white", edgecolor=INK, hatch="///", label=SPLIT_LABELS[1]),
        ],
        loc="upper left", frameon=False, fontsize=6.1, ncol=2,
    )
    _save(fig, name)


def _combined_scale_plot(indexed: dict) -> None:
    fig, axis = plt.subplots(figsize=(3.5, 2.55))
    x = np.arange(len(SCALES))
    width = 0.19
    series = [
        ("SFT", split, BLUE) for split in SPLITS
    ] + [
        ("Thinking + rationale SFT", split, ORANGE) for split in SPLITS
    ]
    for offset, (family, split, color) in zip((-1.5, -0.5, 0.5, 1.5), series):
        models = [
            f"SFT {scale}/task" if family == "SFT"
            else f"Thinking + rationale SFT {scale}/task"
            for scale in SCALES
        ]
        selected = [indexed[(model, split)] for model in models]
        values = [row["error_free_success_rate"] for row in selected]
        yerr = np.concatenate([_error(row) for row in selected], axis=1)
        bars = axis.bar(
            x + offset * width, values, width, color=color, edgecolor=INK,
            linewidth=0.4, hatch="///" if split.startswith("Held-out") else None,
            yerr=yerr,
            error_kw={"elinewidth": 0.55, "capsize": 1.1, "capthick": 0.55},
        )
        label_bars_above_whiskers(
            axis, bars, values, yerr[1], fontsize=4.9,
        )
    _style_axis(axis, [str(scale) for scale in SCALES], ylim=1.10)
    axis.set_xlabel("Training trajectories per task", labelpad=2)
    axis.legend(
        handles=[
            Patch(facecolor=BLUE, edgecolor=INK, label="Qwen3-VL-8B-Instruct SFT"),
            Patch(facecolor=ORANGE, edgecolor=INK, label="Qwen3-VL-8B-Thinking + rationale SFT"),
            Patch(facecolor="white", edgecolor=INK, label="In-training tasks"),
            Patch(facecolor="white", edgecolor=INK, hatch="///", label="Held-out tasks"),
        ],
        loc="lower center", bbox_to_anchor=(0.5, 1.01), frameon=False,
        fontsize=5.0, ncol=2, columnspacing=0.8, handletextpad=0.4,
    )
    _save(fig, "sft_scaling_combined.png")


def _thinking_ablation_plot(indexed: dict) -> None:
    models = ABLATION_MODELS
    labels = ["Qwen-Inst", "Qwen-Inst\n+ rationale", "Qwen-Think", "Qwen-Think\n+ rationale"]
    colors = (BLUE, "#8e79b8", "#d6a03a", ORANGE)
    fig, axis = plt.subplots(figsize=(3.5, 2.55))
    x = np.arange(len(models))
    width = 0.34
    for split_index, split in enumerate(SPLITS):
        positions = x + (split_index - 0.5) * width
        selected = [indexed[(model, split)] for model in models]
        values = [row["error_free_success_rate"] for row in selected]
        bars = axis.bar(
            positions, values, width, color=colors, edgecolor=INK, linewidth=0.45,
            hatch="///" if split_index else None,
            yerr=np.concatenate([_error(row) for row in selected], axis=1),
            error_kw={"elinewidth": 0.65, "capsize": 1.6, "capthick": 0.65},
            label=SPLIT_LABELS[split_index],
        )
        axis.bar_label(bars, labels=[f"{value:.0%}" for value in values], padding=1.5, fontsize=5.7)
    _style_axis(axis, labels, ylim=1.10)
    axis.legend(
        handles=[
            Patch(facecolor="white", edgecolor=INK, label=SPLIT_LABELS[0]),
            Patch(facecolor="white", edgecolor=INK, hatch="///", label=SPLIT_LABELS[1]),
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
    _save(fig, "thinking_rationale_30task_ablations.png")


ABLATION_MODELS = [
    "SFT 30/task",
    "Instruct + rationale SFT 30/task",
    "Thinking + no rationale SFT 30/task",
    "Thinking + rationale SFT 30/task",
]


def _complete(indexed: dict, models: list[str]) -> bool:
    return all((model, split) in indexed for model in models for split in SPLITS)


def main() -> None:
    """Draw every Fig. 6 and Fig. 7 panel whose cells have all been evaluated."""

    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    datasets = artifact["snapshot"]["datasets"]
    scale = _index(datasets["scale_results"])
    thirty = _index(datasets["thirty_results"])
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.3})

    instruct = [f"SFT {value}/task" for value in SCALES]
    thinking = [f"Thinking + rationale SFT {value}/task" for value in SCALES]
    if _complete(scale, instruct):
        _split_scale_plot(scale, instruct, "sft_scaling_instruct.png", BLUE)
    if _complete(scale, thinking):
        _split_scale_plot(scale, thinking, "sft_scaling_thinking_rationale.png", ORANGE)
    if _complete(scale, instruct + thinking):
        _combined_scale_plot(scale)
    else:
        print("Fig. 6: skipped, not every scale has been evaluated")
    if _complete(thirty, ABLATION_MODELS):
        _thinking_ablation_plot(thirty)
    else:
        print("Fig. 7: skipped, not every 30/task cell has been evaluated")
    print(write_caption_csv(ARTIFACT_DIR))

if __name__ == "__main__":
    main()
