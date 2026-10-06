"""`scripts/reproduce.py --dry-run` must resolve every paper cell offline."""

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = yaml.safe_load((REPO_ROOT / "configs/experiments.yaml").read_text())


def _dry_run(tmp_path, *args):
    env = {
        **os.environ,
        "ROBOTALK_OUTPUT_ROOT": str(tmp_path / "outputs"),
        # Missing on purpose, so the dataset download + rebuild path is resolved too.
        "ROBOTALK_DATA_ROOT": str(tmp_path / "data"),
        "HF_HUB_OFFLINE": "1",
    }
    result = subprocess.run(
        [sys.executable, "scripts/reproduce.py", *args, "--dry-run"],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=300,
    )
    assert result.returncode == 0, result.stderr[-2000:]
    return [
        shlex.split(line[2:]) for line in result.stdout.splitlines() if line.startswith("+ ")
    ]


@pytest.mark.parametrize(
    "args", [("fig3",), ("fig5",), ("fig6",), ("fig6", "--train"), ("fig7",), ("fig7", "--train")]
)
def test_dry_run_resolves_every_cell(tmp_path, args):
    commands = _dry_run(tmp_path, *args)
    assert commands
    # Every repository file a command refers to exists.
    for command in commands:
        for token in command:
            if token.startswith(("configs/", str(REPO_ROOT / "configs"))):
                assert (REPO_ROOT / token).exists(), token
    figure = args[0]
    if figure == "fig3":
        return
    text = "\n".join(shlex.join(command) for command in commands)
    for model_id, model in CONFIG["models"].items():
        if figure in model["figures"]:
            assert f"/eval/{model_id}/" in text, f"{model_id} is not evaluated"
    if "--train" in args:
        assert "robotalk.training.main" in text
        assert "robotalk.release.materialize_training_layout" in text


# Output parsers of the paper's vLLM servers, read from their startup logs.
PAPER_PARSERS = {
    "instruct": ["--tool-call-parser", "hermes"],
    "thinking": ["--tool-call-parser", "qwen3_xml", "--reasoning-parser", "qwen3"],
    "thinking_no_rationale": ["--tool-call-parser", "qwen3_xml"],
}


@pytest.mark.parametrize(
    "model_id",
    [m for m, cfg in CONFIG["models"].items() if cfg.get("backend", "vllm") == "vllm"],
)
def test_vllm_server_uses_the_paper_parsers(tmp_path, model_id):
    model = CONFIG["models"][model_id]
    commands = _dry_run(tmp_path, model["figures"][0], "--models", model_id)
    serve = next(c for c in commands if "serve" in c and "vllm" in c)
    parser_args = [
        token for i, token in enumerate(serve)
        if token in ("--tool-call-parser", "--reasoning-parser")
        or (i and serve[i - 1] in ("--tool-call-parser", "--reasoning-parser"))
    ]
    assert parser_args == PAPER_PARSERS[model["profile"]]


def test_a_task_subset_spanning_both_splits_is_divided_between_them(tmp_path):
    """Each split's evaluator only accepts that split's tasks."""

    commands = _dry_run(
        tmp_path, "fig5", "--models", "gemini_full",
        "--tasks", "add_lemon_to_fish,garnish_cake", "--episodes-per-task", "1",
    )
    by_split = {
        command[command.index("--cohort-split") + 1]: command[command.index("--tasks") + 1]
        for command in commands if "--cohort-split" in command
    }
    assert by_split == {"train_task_types": "add_lemon_to_fish", "heldout_task_types": "garnish_cake"}


@pytest.mark.parametrize("model_id", ["instruct_s30", "thinking_rationale_s30", "thinking_noreason_s30"])
def test_evaluate_on_a_released_adapter_matches_its_paper_cell(tmp_path, model_id):
    model = CONFIG["models"][model_id]
    cell = _dry_run(tmp_path / "cell", model["figures"][0], "--models", model_id)
    custom = _dry_run(
        tmp_path / "custom", "evaluate", "--adapter", model["adapter"], "--base", model["base"],
        "--profile", model["profile"], "--name", model_id,
    )

    def normalize(commands, root):
        return [[t.replace(str(root), "ROOT") for t in c] for c in commands if "serve" in c or "--cohort-split" in c]

    assert normalize(custom, tmp_path / "custom") == normalize(cell, tmp_path / "cell")


def test_record_videos_turns_on_episode_recordings(tmp_path):
    commands = _dry_run(tmp_path, "evaluate", "--adapter", "x/y", "--record-videos")
    evals = [c for c in commands if "--cohort-split" in c]
    assert evals and all("--record-firsts" in c and "--no-record-firsts" not in c for c in evals)


def test_a_custom_split_and_cohort_reach_training_and_evaluation(tmp_path):
    split = {"train_tasks": ["prepare_coffee"], "held_out_tasks": ["garnish_cake"]}
    (tmp_path / "split.json").write_text(json.dumps(split))
    (tmp_path / "cohort.json").write_text("{}")
    commands = _dry_run(
        tmp_path, "fig6", "--models", "instruct_s30", "--train",
        "--task-split", str(tmp_path / "split.json"), "--cohort", str(tmp_path / "cohort.json"),
    )
    train = next(c for c in commands if "robotalk.training.main" in c)
    assert train[train.index("--train-tasks") + 1] == "prepare_coffee"
    evals = [c for c in commands if "--cohort-split" in c]
    assert evals and all(c[c.index("--manifest") + 1] == str(tmp_path / "cohort.json") for c in evals)
