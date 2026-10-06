#!/usr/bin/env python
"""Reproduce the RoboTalk paper's figures with one command per figure.

    python scripts/reproduce.py fig3                     # task-phase distribution (CPU)
    python scripts/reproduce.py fig5                     # communication ablation
    python scripts/reproduce.py fig6 [--train]           # SFT scaling
    python scripts/reproduce.py fig7 [--train]           # rationale ablation
    python scripts/reproduce.py figures                  # rebuild plots from outputs/eval
    python scripts/reproduce.py evaluate --adapter DIR   # any model, paper cohort and settings

Evaluation cells run the live-sim evaluator on both cohort splits and write
``outputs/eval/<model>/<split>/``. Fine-tuned cells evaluate the released
LoRA adapters from the Hugging Face Hub by default; ``--train`` retrains them
first. Completed cells are skipped, so an interrupted run can be resumed by
re-running the same command. ``--dry-run`` prints every command instead.

Open-weight cells start a local vLLM server (``vllm`` on PATH, or set
``VLLM_LAUNCHER``, e.g. ``"apptainer exec --nv vllm-openai-v0.27.1.sif"``).
Gemini cells need ``GOOGLE_API_KEY`` or Vertex credentials in the environment.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
FIGURE_EXPORTERS = {
    "fig3": ["robotalk.analysis.export_task_phase_pngs"],
    "fig5": ["robotalk.analysis.export_communication_ablation_png"],
    "fig6": ["robotalk.analysis.export_sft_scale_pngs"],
    "fig7": ["robotalk.analysis.export_sft_scale_pngs"],
}


def _load_dotenv(path: Path = ROOT / ".env") -> None:
    """Export KEY=VALUE lines of .env (see .env.example) without overriding the shell."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip().removeprefix("export ").strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            os.environ.setdefault(key.strip(), value.strip().strip("'\""))


def _flags(mapping: dict) -> list[str]:
    args: list[str] = []
    for key, value in mapping.items():
        if value is True:
            args.append(key)
        elif value is not False and value is not None:
            args += [key, str(value)]
    return args


def _run(cmd: list[str], *, dry_run: bool, env: dict | None = None) -> None:
    print("+", shlex.join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, check=True, cwd=ROOT, env={**os.environ, **(env or {})})


class Reproducer:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.config = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
        # A different train/held-out split (see robotalk.evaluation.resplit_cohort).
        if args.task_split:
            self.config["splits"]["task_split"] = str(Path(args.task_split).resolve())
        if args.cohort:
            self.config["evaluation"]["cohort"] = str(Path(args.cohort).resolve())
        if args.record_videos:
            self.config["evaluation"]["common_args"].pop("--no-record-firsts", None)
            self.config["evaluation"]["common_args"]["--record-firsts"] = True
        self.out = Path(os.environ.get("ROBOTALK_OUTPUT_ROOT", ROOT / "outputs"))
        # Partial (smoke) runs never count as finished paper cells.
        self.smoke = bool(args.tasks or args.episodes_per_task)
        self.eval_root = self.out / ("eval_smoke" if self.smoke else "eval")
        self.python = sys.executable
        self.data_root = Path(
            os.environ.get("ROBOTALK_DATA_ROOT", ROOT / self.config["dataset"]["default_root"])
        )

    # -- selection ---------------------------------------------------------
    def models_for(self, figure: str, *, selected_only: bool = True) -> list[str]:
        """Model cells of a figure; with selected_only, restricted to --models."""
        models = [
            model_id for model_id, model in self.config["models"].items()
            if figure in model["figures"]
        ]
        if selected_only and self.args.models:
            wanted = set(self.args.models.split(","))
            models = [m for m in models if m in wanted]
        return models

    def train_tasks(self) -> str:
        split = json.loads((ROOT / self.config["splits"]["task_split"]).read_text())
        return ",".join(split["train_tasks"])

    # -- adapters ------------------------------------------------------------
    def adapter_for(self, model_id: str) -> Path | None:
        model = self.config["models"][model_id]
        if "local_adapter" in model:
            return model["local_adapter"]
        if "scale" not in model and "adapter" not in model:
            return None  # untuned model
        if self.args.train:
            return self.train(model_id)
        repo = model.get("adapter")
        if not repo:
            raise SystemExit(
                f"{model_id}: no released adapter is configured; rerun with --train"
            )
        target = self.out / "adapters" / model_id
        if not (target / "adapter_config.json").exists():
            print(f"Downloading {repo} -> {target}", flush=True)
            if not self.args.dry_run:
                from huggingface_hub import snapshot_download

                snapshot_download(repo, local_dir=target)
        return target

    def ensure_dataset(self) -> None:
        """Download the dataset and rebuild the training layout if needed."""

        if self.data_root.is_dir() and any(self.data_root.iterdir()):
            return
        dataset = self.config["dataset"]
        hf_dir = self.out / "hf_dataset"
        print(f"Downloading {dataset['hf_repo']}@{dataset['revision']} -> {hf_dir}", flush=True)
        if not self.args.dry_run:
            from huggingface_hub import snapshot_download

            snapshot_download(
                dataset["hf_repo"], repo_type="dataset",
                revision=dataset["revision"], local_dir=hf_dir,
            )
        _run([
            self.python, "-m", "robotalk.release.materialize_training_layout",
            "--hf-dir", str(hf_dir), "--output", str(self.data_root),
            "--workers", str(self.args.cpu_workers),
        ], dry_run=self.args.dry_run)

    def train(self, model_id: str) -> Path:
        model = self.config["models"][model_id]
        self.ensure_dataset()
        kind = "rationale" if model["train_reasoning"] else "tool_call"
        reasoning = ["--train-reasoning"] if model["train_reasoning"] else []
        master = self.out / "preprocessed" / f"master_{kind}"
        subset = self.out / "preprocessed" / f"{kind}_s{model['scale']}"
        output = self.out / "checkpoints" / model_id
        if (output / "adapter_config.json").exists():
            return output
        if not (master / "preprocess_config.json").exists():
            _run([
                self.python, "-m", "robotalk.training.preprocess",
                "--dataset-root", str(self.data_root), "--output-dir", str(master),
                "--train-tasks", self.train_tasks(), "--val-tasks", "",
                "--no-pretokenize", "--no-use-example-cache",
                "--example-build-workers", str(self.args.cpu_workers),
                "--validation-split-mode", "none",
                "--validation-trajectory-fraction", "0",
                "--validation-min-trajectories-per-task", "0", *reasoning,
            ], dry_run=self.args.dry_run)
        if not (subset / "preprocess_config.json").exists():
            _run([
                self.python, "-m", "robotalk.training.materialize_scale_artifacts",
                "--master", str(master),
                "--selection", str(ROOT / self.config["splits"]["selections"][model["scale"]]),
                "--output", str(subset),
            ], dry_run=self.args.dry_run)
        training = self.config["training"]
        _run([
            "accelerate", "launch", "--config_file", "configs/train/accelerate_multigpu.yaml",
            "--num_processes", str(training["num_processes"]),
            "-m", "robotalk.training.main",
            "--dataset-root", str(self.data_root),
            "--model-name-or-path", model["base"], "--processor-name-or-path", model["base"],
            "--train-tasks", self.train_tasks(), "--val-tasks", "",
            "--preprocessed-data-dir", str(subset), "--output-dir", str(output),
            "--wandb-mode", self.args.wandb_mode,
            *_flags(training["main_args"]), *reasoning,
        ], dry_run=self.args.dry_run)
        return output

    # -- evaluation ----------------------------------------------------------
    def eval_args(self, model_id: str, split: str) -> list[str]:
        model = self.config["models"][model_id]
        evaluation = self.config["evaluation"]
        common = dict(evaluation["common_args"])
        if self.args.episodes_per_task:
            common["--episodes-per-task"] = self.args.episodes_per_task
        args = [
            "--backend", model.get("backend", "vllm"),
            "--cohort-split", split,
            "--manifest", str(ROOT / evaluation["cohort"]),
            "--scene-compatibility-cache", str(ROOT / evaluation["scene_compatibility_cache"]),
            "--dataset-root", str(self.data_root),
            "--communication-mode", model.get("communication_mode", "full"),
            "--output-dir", str(self.eval_root / model_id / split),
            *_flags(common),
            *_flags(evaluation["profiles"][model["profile"]]),
        ]
        if model.get("backend") == "gemini":
            args += ["--model", model["base"]]
        if self.args.tasks:
            args += ["--tasks", ",".join(self.tasks_in_split(split))]
        return args

    def cell_done(self, model_id: str, split: str) -> bool:
        return (self.eval_root / model_id / split / "aggregate" / "live_sim_metrics.json").exists()

    def splits_to_run(self, model_id: str) -> list[str]:
        splits = [
            s for s in self.config["evaluation"]["cohort_splits"]
            if not self.cell_done(model_id, s)
        ]
        if self.args.tasks:
            # A task subset may fall entirely in one split; skip the other.
            splits = [s for s in splits if self.tasks_in_split(s)]
        return splits

    def tasks_in_split(self, split: str) -> list[str]:
        """The --tasks that belong to a cohort split, in the order given."""
        cohort = json.loads((ROOT / self.config["evaluation"]["cohort"]).read_text())
        members = set(cohort["configurations"][split])
        return [t for t in self.args.tasks.split(",") if t in members]

    def evaluate(self, model_id: str) -> None:
        splits = self.splits_to_run(model_id)
        if not splits:
            print(f"{model_id}: already evaluated", flush=True)
            return
        model = self.config["models"][model_id]
        adapter = self.adapter_for(model_id)
        server = None
        extra: list[str] = []
        if model.get("backend", "vllm") == "vllm":
            port = self.args.vllm_port
            served = "robocasa-sft" if adapter else "qwen3vl8b-base"
            cmd = [
                *shlex.split(os.environ.get("VLLM_LAUNCHER", "")), "vllm", "serve", model["base"],
                "--host", "127.0.0.1", "--port", str(port), "--served-model-name", served,
                *self.config["evaluation"]["vllm"]["serve_args"],
                *self.config["evaluation"]["vllm"]["profile_serve_args"][model["profile"]],
            ]
            if adapter:
                cmd += [
                    "--enable-lora", *self.config["evaluation"]["vllm"]["lora_serve_args"],
                    "--lora-modules", f"{served}={adapter}",
                ]
                extra += ["--adapter-path", str(adapter)]
            extra += ["--vllm-model", served, "--vllm-base-url", f"http://127.0.0.1:{port}/v1"]
            print("+", shlex.join(cmd), flush=True)
            if not self.args.dry_run:
                log = open(self.out / f"vllm_{model_id}.log", "w")
                server_env = {**os.environ, **self.config["evaluation"]["vllm"].get("env", {})}
                server = subprocess.Popen(
                    cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=server_env
                )
                self._wait_for_server(port, server)
        try:
            for split in splits:
                _run([
                    self.python, "-m", "robotalk.evaluation.live_sim_parallel_eval",
                    "--workers", str(self.config["evaluation"]["workers"]), "--",
                    *self.eval_args(model_id, split), *extra,
                ], dry_run=self.args.dry_run, env={"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"})
        finally:
            if server is not None:
                server.terminate()
                server.wait()

    @staticmethod
    def _wait_for_server(port: int, server: subprocess.Popen, timeout_s: int = 900) -> None:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if server.poll() is not None:
                raise SystemExit("vLLM server exited before becoming ready; see outputs/vllm_*.log")
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2):
                    return
            except OSError:
                time.sleep(5)
        raise SystemExit("vLLM server did not become ready in time")

    # -- figures ---------------------------------------------------------------
    def figures(self, figure: str | None) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        if self.smoke:
            print(f"Smoke run finished; outputs are under {self.eval_root} (not plotted).")
            return
        if figure != "fig3":
            _run([self.python, "-m", "robotalk.analysis.paper_results"], dry_run=self.args.dry_run)
        figures = [figure] if figure else list(FIGURE_EXPORTERS)
        done_modules: set[str] = set()
        for name in figures:
            missing = [] if name == "fig3" else [
                f"{model_id}/{split}"
                # A figure is plotted only once all of its cells exist, not just
                # the ones selected with --models.
                for model_id in self.models_for(name, selected_only=False)
                for split in self.config["evaluation"]["cohort_splits"]
                if not self.cell_done(model_id, split)
            ]
            if missing and not self.args.dry_run:
                print(f"{name}: skipping plot, missing evaluations: {', '.join(missing)}")
                continue
            for module in FIGURE_EXPORTERS[name]:
                if module not in done_modules:
                    _run([self.python, "-m", module], dry_run=self.args.dry_run)
                    done_modules.add(module)

    def register_model(self) -> str:
        """Add the model given by `evaluate --base/--adapter/--profile` as a cell."""
        args = self.args
        name = args.name or (Path(args.adapter).name if args.adapter else Path(args.base).name)
        model = {
            "label": name, "base": args.base, "profile": args.profile,
            "backend": "gemini" if args.profile == "gemini" else "vllm",
            "communication_mode": args.communication_mode, "figures": [],
        }
        if args.adapter and Path(args.adapter).is_dir():
            model["local_adapter"] = Path(args.adapter).resolve()
        elif args.adapter:
            model["adapter"] = args.adapter  # a Hugging Face model repo
        self.config["models"][name] = model
        return name

    def report(self, model_id: str) -> None:
        for split in self.config["evaluation"]["cohort_splits"]:
            metrics = self.eval_root / model_id / split / "aggregate" / "live_sim_metrics.json"
            if metrics.exists():
                m = json.loads(metrics.read_text())
                k, n = m["num_fsm_error_free_successes"], m["num_jsonl_records"]
                print(f"{model_id} {split}: {k}/{n} error-free FSM successes ({k / n:.1%})")

    def run(self) -> None:
        figure = self.args.figure
        self.out.mkdir(parents=True, exist_ok=True)
        if figure == "evaluate":
            model_id = self.register_model()
            self.evaluate(model_id)
            if not self.args.dry_run:
                self.report(model_id)
            return
        if figure in ("fig5", "fig6", "fig7"):
            for model_id in self.models_for(figure):
                self.evaluate(model_id)
        self.figures(None if figure == "figures" else figure)


def main() -> None:
    _load_dotenv()
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("figure", choices=("fig3", "fig5", "fig6", "fig7", "figures", "evaluate"))
    parser.add_argument("--train", action="store_true", help="retrain adapters instead of downloading them")
    parser.add_argument("--models", help="comma-separated model ids from configs/experiments.yaml")
    parser.add_argument("--tasks", help="comma-separated task subset (smoke runs)")
    parser.add_argument("--episodes-per-task", type=int, help="override the 10 episodes per task")
    parser.add_argument("--vllm-port", type=int, default=18000)
    parser.add_argument("--cpu-workers", type=int, default=8)
    parser.add_argument("--wandb-mode", default="disabled", choices=("online", "offline", "disabled"))
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--record-videos", action="store_true",
        help="save videos of the first success and first failure of each task",
    )
    parser.add_argument("--task-split", help="train/held-out task split JSON (default: the paper's 43/10)")
    parser.add_argument("--cohort", help="evaluation cohort JSON matching --task-split")
    new = parser.add_argument_group("evaluate: any model on the paper's cohort and settings")
    new.add_argument("--base", default="Qwen/Qwen3-VL-8B-Instruct", help="base model, or a Gemini model id")
    new.add_argument("--adapter", help="LoRA adapter: a local directory or a Hugging Face repo")
    new.add_argument(
        "--profile", default="instruct",
        choices=("instruct", "thinking", "thinking_no_rationale", "gemini"),
        help="decoding profile and vLLM parsers (configs/experiments.yaml)",
    )
    new.add_argument("--communication-mode", default="full",
                     choices=("none", "unguided", "minimal", "intermediate", "full"))
    new.add_argument("--name", help="output name under outputs/eval/ (default: adapter directory name)")
    Reproducer(parser.parse_args()).run()


if __name__ == "__main__":
    main()
