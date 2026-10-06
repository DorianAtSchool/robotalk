"""Export the current 43/10 communication ablation as a compact PNG."""

from __future__ import annotations


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

import json

from robotalk.analysis.plot_caption_metadata import write_caption_csv
from robotalk.analysis.plot_style_utils import label_bars_above_whiskers, style_paper_figure
from robotalk.utils import output_root


OUTPUT = output_root() / "figures/43_10/communication_ablation_error_free.png"
SPLIT_OUTPUT = OUTPUT.with_name("communication_ablation_error_free_by_split.png")
SPLIT_SUBSET_OUTPUT = OUTPUT.with_name(
    "communication_ablation_error_free_by_split_single_agent.png"
)
POOLED_SUBSET_OUTPUT = OUTPUT.with_name(
    "communication_ablation_error_free_single_agent.png"
)
ARTIFACT = OUTPUT.with_name("artifact.json")
MODES = ("None", "Unguided", "Minimal", "Intermediate", "Full")
MODELS = ("Base Qwen3-VL-8B", "Gemini 3 Flash")
MODEL_LABELS = {
    "Base Qwen3-VL-8B": "Qwen3-VL-8B-Instruct",
    "Gemini 3 Flash": "Gemini 3 Flash",
}
SPLITS = ("Trained tasks (43)", "Held-out tasks (10)")
COLORS = {"Base Qwen3-VL-8B": "#3976d2", "Gemini 3 Flash": "#3b9b70"}
DARK_COLORS = {"Base Qwen3-VL-8B": "#174b91", "Gemini 3 Flash": "#1b5e40"}


def _legend_above(axis, *, handles=None, fontsize: float, ncol: int) -> None:
    kwargs = dict(
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
        frameon=False,
        fontsize=fontsize,
        ncol=ncol,
        borderaxespad=0,
        columnspacing=0.8,
        handletextpad=0.4,
        labelspacing=0.25,
    )
    if handles is not None:
        kwargs["handles"] = handles
    axis.legend(**kwargs)


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    rate = successes / total
    denominator = 1 + z * z / total
    center = (rate + z * z / (2 * total)) / denominator
    margin = z * np.sqrt(rate * (1 - rate) / total + z * z / (4 * total * total)) / denominator
    return center - margin, center + margin


def main() -> None:
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    rows = artifact["snapshot"]["datasets"]["communication_results"]
    indexed = {(row["communication_mode"], row["model"], row["split"]): row for row in rows}
    available_modes = [
        mode
        for mode in MODES
        if any((mode, model, split) in indexed for model in MODELS for split in SPLITS)
    ]

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 7.5})
    fig, axis = plt.subplots(figsize=(3.5, 2.55))
    x = np.arange(len(available_modes))
    width = 0.36

    for model_index, model in enumerate(MODELS):
        positions = x + (model_index - 0.5) * width
        values, lower, upper = [], [], []
        for mode in available_modes:
            selected = [indexed[(mode, model, split)] for split in SPLITS]
            successes = sum(row["error_free_successes"] for row in selected)
            total = sum(row["episodes"] for row in selected)
            value = successes / total
            ci_low, ci_high = _wilson(successes, total)
            values.append(value)
            lower.append(value - ci_low)
            upper.append(ci_high - value)
        bars = axis.bar(
            positions, values, width, color=COLORS[model], edgecolor="#263241",
            linewidth=0.4, label=MODEL_LABELS[model], yerr=np.array([lower, upper]),
            error_kw={"elinewidth": 0.7, "capsize": 1.8, "capthick": 0.7},
        )
        axis.bar_label(
            bars, labels=[f"{value:.0%}" for value in values], padding=1.5, fontsize=6.5
        )

    axis.set_ylabel("Error-free FSM success")
    axis.set_xticks(x, available_modes, rotation=20, ha="right", rotation_mode="anchor")
    axis.set_ylim(0, 1.0)
    ticks = np.arange(0, 1.01, 0.2)
    axis.set_yticks(ticks, [f"{value:.0%}" for value in ticks])
    axis.grid(axis="y", color="#e3e7ec", linewidth=0.55)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#9aa4b2")
    _legend_above(axis, fontsize=6.4, ncol=2)
    style_paper_figure(fig)
    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=220, bbox_inches="tight", facecolor="white")
    print(OUTPUT)

    fig, axis = plt.subplots(figsize=(3.5, 2.55))
    for model_index, model in enumerate(MODELS):
        positions = x + (model_index - 0.5) * width
        values, subsets, lower, upper = [], [], [], []
        for mode in available_modes:
            selected = [indexed[(mode, model, split)] for split in SPLITS]
            successes = sum(row["error_free_successes"] for row in selected)
            subset_successes = sum(
                row["error_free_single_agent_successes"] for row in selected
            )
            total = sum(row["episodes"] for row in selected)
            value = successes / total
            ci_low, ci_high = _wilson(successes, total)
            values.append(value)
            subsets.append(subset_successes / total)
            lower.append(value - ci_low)
            upper.append(ci_high - value)
        bars = axis.bar(
            positions, values, width, color=COLORS[model], edgecolor="#263241",
            linewidth=0.4, yerr=np.array([lower, upper]),
            error_kw={"elinewidth": 0.7, "capsize": 1.8, "capthick": 0.7},
        )
        axis.bar(
            positions, subsets, width * 0.52, color=DARK_COLORS[model],
            edgecolor="#263241", linewidth=0.35, zorder=3,
        )
        axis.bar_label(
            bars, labels=[f"{value:.0%}" for value in values], padding=1.5, fontsize=6.5
        )
    axis.set_ylabel("Error-free FSM success")
    axis.set_xticks(x, available_modes, rotation=20, ha="right", rotation_mode="anchor")
    axis.set_ylim(0, 1.0)
    axis.set_yticks(ticks, [f"{value:.0%}" for value in ticks])
    axis.grid(axis="y", color="#e3e7ec", linewidth=0.55)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#9aa4b2")
    _legend_above(
        axis,
        handles=[
            Patch(facecolor=COLORS[MODELS[0]], edgecolor="#263241", label=MODEL_LABELS[MODELS[0]]),
            Patch(facecolor="#4d5560", edgecolor="#263241", label="Inset: exactly one physical-work agent"),
            Patch(facecolor=COLORS[MODELS[1]], edgecolor="#263241", label="Gemini 3 Flash"),
        ],
        fontsize=5.4,
        ncol=2,
    )
    style_paper_figure(fig)
    fig.tight_layout()
    fig.savefig(POOLED_SUBSET_OUTPUT, dpi=220, bbox_inches="tight", facecolor="white")
    print(POOLED_SUBSET_OUTPUT)

    # Single-axis split view for a paper column: color identifies the model;
    # hatch identifies whether tasks were trained or held out.
    fig, axis = plt.subplots(figsize=(3.5, 2.55))
    bar_width = 0.19
    offsets = (-1.5, -0.5, 0.5, 1.5)
    series = [
        (model, split)
        for model in MODELS
        for split in SPLITS
    ]
    for offset, (model, split) in zip(offsets, series):
        values, lower, upper = [], [], []
        for mode in available_modes:
            row = indexed[(mode, model, split)]
            value = row["error_free_success_rate"]
            values.append(value)
            lower.append(max(0.0, value - row["ci_low"]))
            upper.append(max(0.0, row["ci_high"] - value))
        bars = axis.bar(
            x + offset * bar_width,
            values,
            bar_width,
            color=COLORS[model],
            edgecolor="#263241",
            linewidth=0.45,
            hatch="///" if split.startswith("Held-out") else None,
            label=f"{MODEL_LABELS[model]} · {'held out' if split.startswith('Held-out') else 'in-training'}",
            yerr=np.array([lower, upper]),
            error_kw={"elinewidth": 0.6, "capsize": 1.3, "capthick": 0.6},
        )
        label_bars_above_whiskers(
            axis, bars, values, upper, fontsize=5.6, minimum_value=0.025,
        )
    axis.set_ylabel("Error-free FSM success")
    axis.set_xticks(x, available_modes, rotation=20, ha="right", rotation_mode="anchor")
    axis.set_ylim(0, 1.0)
    axis.set_yticks(ticks, [f"{value:.0%}" for value in ticks])
    axis.grid(axis="y", color="#e3e7ec", linewidth=0.55)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#9aa4b2")
    _legend_above(axis, fontsize=4.9, ncol=2)
    style_paper_figure(fig)
    fig.tight_layout()
    fig.savefig(SPLIT_OUTPUT, dpi=220, bbox_inches="tight", facecolor="white")
    print(SPLIT_OUTPUT)

    fig, axis = plt.subplots(figsize=(3.5, 2.55))
    for offset, (model, split) in zip(offsets, series):
        values, subsets, lower, upper = [], [], [], []
        for mode in available_modes:
            row = indexed[(mode, model, split)]
            value = row["error_free_success_rate"]
            values.append(value)
            subsets.append(row["error_free_single_agent_success_rate"])
            lower.append(max(0.0, value - row["ci_low"]))
            upper.append(max(0.0, row["ci_high"] - value))
        positions = x + offset * bar_width
        hatch = "///" if split.startswith("Held-out") else None
        bars = axis.bar(
            positions, values, bar_width, color=COLORS[model],
            edgecolor="#263241", linewidth=0.45, hatch=hatch,
            yerr=np.array([lower, upper]),
            error_kw={"elinewidth": 0.6, "capsize": 1.3, "capthick": 0.6},
        )
        axis.bar(
            positions, subsets, bar_width * 0.52, color=DARK_COLORS[model],
            edgecolor="#263241", linewidth=0.35, hatch=hatch, zorder=3,
        )
        label_bars_above_whiskers(
            axis, bars, values, upper, fontsize=5.6, minimum_value=0.025,
        )
    axis.set_ylabel("Error-free FSM success")
    axis.set_xticks(x, available_modes, rotation=20, ha="right", rotation_mode="anchor")
    axis.set_ylim(0, 1.0)
    axis.set_yticks(ticks, [f"{value:.0%}" for value in ticks])
    axis.grid(axis="y", color="#e3e7ec", linewidth=0.55)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#9aa4b2")
    _legend_above(
        axis,
        handles=[
            Patch(facecolor=COLORS[MODELS[0]], edgecolor="#263241", label=MODEL_LABELS[MODELS[0]]),
            Patch(facecolor="white", edgecolor="#263241", label="In-training tasks"),
            Patch(facecolor="#4d5560", edgecolor="#263241", label="Inset: exactly one physical-work agent"),
            Patch(facecolor=COLORS[MODELS[1]], edgecolor="#263241", label="Gemini 3 Flash"),
            Patch(facecolor="white", edgecolor="#263241", hatch="///", label="Held-out tasks"),
        ],
        fontsize=4.8,
        ncol=2,
    )
    style_paper_figure(fig)
    fig.tight_layout()
    fig.savefig(SPLIT_SUBSET_OUTPUT, dpi=220, bbox_inches="tight", facecolor="white")
    print(SPLIT_SUBSET_OUTPUT)
    print(write_caption_csv(OUTPUT.parent))


if __name__ == "__main__":
    main()
