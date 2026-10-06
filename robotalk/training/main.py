"""Multi-GPU image-conditioned SFT entrypoint for task-level BC VLM training."""

from __future__ import annotations

import argparse
import inspect
import importlib.util
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any

import torch
from accelerate.state import PartialState
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoProcessor, Trainer, TrainerCallback, TrainingArguments, set_seed
from transformers.trainer_pt_utils import LengthGroupedSampler
from transformers.utils import is_torch_bf16_gpu_available

try:
    from transformers import AutoModelForImageTextToText as AutoVisionLanguageModel
except ImportError:  # pragma: no cover - compatibility with older transformers
    from transformers import AutoModelForVision2Seq as AutoVisionLanguageModel

from robotalk.training.dataset import (
    AgentTurnDataset,
    LazyVisionSFTCollator,
    build_example_cache_fingerprint,
    build_example_cache_path,
    build_agent_turn_examples,
    build_split_manifest,
    estimate_example_length,
    list_available_task_names,
    list_task_trajectory_ids,
    load_examples_from_cache,
    save_examples_to_cache,
    select_held_out_trajectory_ids,
)
from robotalk.generation.runtime.client import load_dotenv_file
from robotalk.training.preprocessed_data import (
    PreprocessedFeatureDataset,
    load_preprocessed_artifact_from_disk,
)
from robotalk.training.task_registry import resolve_task_name, supported_task_names
from robotalk.utils import output_root, paper_train_tasks

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_WANDB_PROJECT = "robotalk"


@dataclass(frozen=True)
class RunConfiguration:
    dataset_root: str
    preprocessed_data_dir: str | None
    model_name_or_path: str
    processor_name_or_path: str
    train_tasks: list[str]
    val_tasks: list[str]
    validation_trajectories_per_task: int
    validation_trajectory_fraction: float
    validation_split_seed: int
    train_reasoning: bool
    init_adapter_path: str | None
    use_example_cache: bool
    trust_example_cache: bool
    training_samples_cache_dir: str
    output_dir: str
    per_device_batch_size: int
    per_device_eval_batch_size: int | None
    grad_accum: int
    train_sampling_strategy: str
    num_epochs: float
    learning_rate: float
    weight_decay: float
    max_length: int | None
    max_tokens: int | None
    image_resolution: int | None
    max_images_per_sample: int | None
    supervise_last_assistant_turn_only: bool
    num_workers: int
    example_build_workers: int
    ddp_timeout_seconds: int
    bf16: bool
    fp16: bool
    gradient_checkpointing: bool
    ddp_find_unused_parameters: bool
    attn_implementation: str
    trust_remote_code: bool
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: list[str]
    save_steps: int
    checkpoints_per_epoch: int
    eval_steps: int
    logging_steps: int
    max_steps: int
    save_total_limit: int
    warmup_ratio: float
    lr_scheduler_type: str
    optim: str
    eval_max_samples: int | None
    report_to: list[str]
    wandb_project: str | None
    wandb_entity: str | None
    wandb_run_name: str | None
    wandb_tags: list[str]
    wandb_mode: str
    resume_from_checkpoint: str | None
    seed: int


def parse_args() -> argparse.Namespace:
    # Defaults are the paper's training settings (configs/experiments.yaml).
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(os.environ.get("ROBOTALK_DATA_ROOT", "data/robotalk_rendered")),
        help=(
            "Root directory of the rendered trajectory dataset (see "
            "robotalk.release.materialize_training_layout)."
        ),
    )
    parser.add_argument(
        "--preprocessed-data-dir",
        type=Path,
        default=None,
        help=(
            "Load train/validation examples produced by preprocess.py instead "
            "of rebuilding them from --dataset-root. The artifact contract is "
            "checked against the runtime SFT flags before model loading."
        ),
    )
    parser.add_argument(
        "--model-name-or-path",
        default="Qwen/Qwen3.5-0.8B",
        help=(
            "Multimodal checkpoint to fine-tune. Defaults to the post-trained "
            "chat model because this pipeline uses chat-formatted JSON supervision."
        ),
    )
    parser.add_argument(
        "--processor-name-or-path",
        default=None,
        help="Optional processor path override. Defaults to the model path.",
    )
    parser.add_argument(
        "--train-tasks",
        default=paper_train_tasks(),
        help=(
            "Comma-separated task names for the training split. Use 'all' for "
            "every task present in the dataset root."
        ),
    )
    parser.add_argument(
        "--val-tasks",
        default="",
        help=(
            "Comma-separated task names for the validation split. Pass an empty "
            "string to disable validation. Use 'all' for every task present in "
            "the dataset root."
        ),
    )
    parser.add_argument(
        "--validation-trajectories-per-task",
        type=int,
        default=0,
        help=(
            "If positive, evaluate on at most this many held-out trajectories "
            "per validation task. Overlapping train/validation tasks are allowed "
            "only in this mode, and selected validation trajectories are removed "
            "from training."
        ),
    )
    parser.add_argument(
        "--validation-trajectory-fraction",
        type=float,
        default=0.0,
        help=(
            "If positive, evaluate on this fraction of trajectories per "
            "validation task. Overlapping train/validation tasks are allowed "
            "only in held-out validation mode, and selected validation "
            "trajectories are removed from training. Mutually exclusive with "
            "--validation-trajectories-per-task."
        ),
    )
    parser.add_argument(
        "--validation-split-seed",
        type=int,
        default=None,
        help="Seed for selecting held-out validation trajectories. Defaults to --seed.",
    )
    parser.add_argument(
        "--train-reasoning",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Supervise a `<think>{reasoning}</think>` prefix before each "
            "assistant tool call, sourced from the trajectory's per-step "
            "rationale."
        ),
    )
    parser.add_argument(
        "--init-adapter-path",
        type=str,
        default=None,
        help=(
            "Initialize LoRA weights from an existing adapter (local dir or "
            "HF repo) and continue training it, instead of a fresh LoRA. "
            "Unlike --resume-from-checkpoint this does not restore optimizer "
            "state."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for checkpoints, manifests, and metrics.",
    )
    parser.add_argument(
        "--use-example-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse cached serialized training samples across training runs.",
    )
    parser.add_argument(
        "--trust-example-cache",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Load existing task example caches without restatting source "
            "trajectories. Use only when the dataset root and cache are known "
            "to match."
        ),
    )
    parser.add_argument(
        "--training-samples-cache-dir",
        dest="training_samples_cache_dir",
        type=Path,
        default=None,
        help=(
            "Directory for cached serialized training samples. Defaults to "
            "<repo>/.cache/robotalk/examples."
        ),
    )
    parser.add_argument(
        "--example-cache-dir",
        dest="training_samples_cache_dir",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--per-device-batch-size", type=int, default=8)
    parser.add_argument(
        "--per-device-eval-batch-size",
        type=int,
        default=None,
        help=(
            "Per-device batch size for base evaluation. Defaults to "
            "--per-device-batch-size when unset."
        ),
    )
    parser.add_argument("--grad-accum", type=int, default=2)
    parser.add_argument("--num-epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="AdamW decoupled weight decay applied to trainable parameters.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=8192,
        help=(
            "Maximum tokenized sequence length, also logged as max_tokens. "
            "Non-positive values disable token truncation."
        ),
    )
    parser.add_argument(
        "--image-resolution",
        type=int,
        default=512,
        help=(
            "Square pixel budget (N*N) applied to the processor's "
            "min/max_pixels for live (non-pretokenized) training, matching "
            "the pretokenization pipeline's --image-resolution. Unset keeps "
            "the processor's native resolution."
        ),
    )
    parser.add_argument(
        "--max-images-per-sample",
        type=int,
        default=4,
        help=(
            "Optional cap on recent image observations kept per multimodal SFT "
            "sample before tokenization. Non-positive values disable the cap."
        ),
    )
    parser.add_argument(
        "--supervise-last-assistant-turn-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "For multi-turn conversation samples, keep only the final "
            "user/assistant turn for loss computation. This avoids unstable "
            "multi-assistant label masks in processor chat templates."
        ),
    )
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument(
        "--train-sampling-strategy",
        choices=("random", "group_by_length"),
        default="random",
        help=(
            "Training sampler. group_by_length batches examples with similar "
            "estimated text-plus-image lengths and puts the longest batch first."
        ),
    )
    parser.add_argument(
        "--example-build-workers",
        type=int,
        default=1,
        help=(
            "Number of worker threads used while converting staged trajectories "
            "into in-memory SFT examples before Trainer dataloading starts."
        ),
    )
    parser.add_argument(
        "--ddp-timeout-seconds",
        type=int,
        default=600,
        help=(
            "Distributed process-group timeout in seconds. Increase this when "
            "rank-0-only structured generation evaluation can run longer than "
            "the default collective watchdog."
        ),
    )
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--ddp-find-unused-parameters",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "DDP find_unused_parameters. Off by default (as for the paper): "
            "turning it on breaks gradient checkpointing (double-marked "
            "params -> crash or silent hang). Only needed when some batches "
            "leave LoRA params without gradients."
        ),
    )
    parser.add_argument("--attn-implementation", default="flash_attention_2")
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated module names to target with LoRA.",
    )
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument(
        "--checkpoints-per-epoch",
        type=int,
        default=2,
        help=(
            "When positive, save whenever training crosses each 1/N epoch "
            "boundary. This supersedes --save-steps and supports half-epoch saves."
        ),
    )
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Maximum optimizer steps for short smoke/performance runs.",
    )
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lr-scheduler-type", default="cosine")
    parser.add_argument("--optim", default="adamw_torch")
    parser.add_argument(
        "--eval-max-samples",
        type=int,
        default=None,
        help="Optional cap for the evaluation dataset used by Trainer eval.",
    )
    parser.add_argument(
        "--report-to",
        default="wandb",
        help="Comma-separated Trainer report targets. Use 'none' to disable.",
    )
    parser.add_argument(
        "--wandb-project",
        default=os.environ.get("WANDB_PROJECT", _DEFAULT_WANDB_PROJECT),
    )
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--wandb-tags", default="robotalk,sft")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="online",
    )
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _split_csv(raw_value: str) -> list[str]:
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def _available_dataset_task_names(dataset_root: Path) -> list[str]:
    if not dataset_root.is_dir():
        return list(supported_task_names())

    return list_available_task_names(dataset_root)


def _resolve_task_list(
    raw_value: str,
    *,
    allow_empty: bool = False,
    dataset_root: Path | None = None,
) -> list[str]:
    raw_task_names = _split_csv(raw_value)
    if not raw_task_names:
        if allow_empty:
            return []
        supported = ", ".join(supported_task_names())
        raise ValueError(
            f"At least one task is required. Supported tasks: {supported}."
        )

    resolved: list[str] = []
    for task_name in raw_task_names:
        if task_name.strip().lower() in {"all", "*"}:
            if dataset_root is None:
                resolved.extend(supported_task_names())
            else:
                available_task_names = _available_dataset_task_names(dataset_root)
                if not available_task_names:
                    raise ValueError(
                        f"No supported task directories found in {dataset_root}."
                    )
                resolved.extend(available_task_names)
        else:
            resolved.append(resolve_task_name(task_name))

    if not resolved:
        if allow_empty:
            return []
        supported = ", ".join(supported_task_names())
        raise ValueError(
            f"At least one task is required. Supported tasks: {supported}."
        )
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"Duplicate task names are not allowed: {resolved}.")
    return resolved


def _resolve_output_dir(output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return output_root() / "train" / timestamp


def _resolve_training_samples_cache_dir(
    training_samples_cache_dir: Path | None,
) -> Path:
    if training_samples_cache_dir is not None:
        return training_samples_cache_dir
    return _REPO_ROOT / ".cache" / "robotalk" / "examples"


def _resolve_report_targets(raw_value: str) -> list[str]:
    targets = [item.strip() for item in raw_value.split(",") if item.strip()]
    if not targets or targets == ["none"]:
        return []
    return targets


def _ensure_optional_dependency(module_name: str) -> None:
    if importlib.util.find_spec(module_name) is None:
        raise ImportError(
            f"Missing optional dependency {module_name!r}. "
            "Install the training dependencies first: pip install -e '.[train]'."
        )


def _configure_wandb(
    args: argparse.Namespace, report_targets: list[str], output_dir: Path
) -> None:
    if "wandb" not in report_targets:
        return
    _ensure_optional_dependency("wandb")

    if args.wandb_project:
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    if args.wandb_entity:
        os.environ.setdefault("WANDB_ENTITY", args.wandb_entity)
    if args.wandb_run_name:
        os.environ.setdefault("WANDB_NAME", args.wandb_run_name)
    if args.wandb_tags:
        os.environ.setdefault("WANDB_TAGS", args.wandb_tags)
    if args.wandb_mode == "disabled":
        os.environ.setdefault("WANDB_DISABLED", "true")
    else:
        os.environ.setdefault("WANDB_MODE", args.wandb_mode)
    os.environ.setdefault("WANDB_DIR", str((output_dir / "wandb").resolve()))


def _initialize_wandb_run(
    *,
    args: argparse.Namespace,
    config: RunConfiguration,
    report_targets: list[str],
    output_dir: Path,
    state: PartialState,
) -> None:
    if "wandb" not in report_targets or args.wandb_mode == "disabled":
        return
    if not state.is_main_process:
        return

    import wandb

    if wandb.run is not None:
        return

    wandb_dir = (output_dir / "wandb").resolve()
    wandb_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_run_name,
        tags=config.wandb_tags,
        mode=args.wandb_mode,
        dir=str(wandb_dir),
        config=asdict(config),
    )
    run_url = getattr(run, "url", None)
    if run_url:
        _log_startup(f"Initialized W&B run: {run_url}", state=state)
    else:
        _log_startup("Initialized W&B run", state=state)


def _log_wandb_metrics(
    metrics: dict[str, float | int],
    *,
    state: PartialState | None = None,
) -> None:
    if state is not None and not state.is_main_process:
        return
    try:
        import wandb
    except ImportError:
        return
    if wandb.run is None:
        return
    wandb.log(metrics)


def _count_parameters(model) -> dict[str, int]:
    total_params = 0
    trainable_params = 0
    for parameter in model.parameters():
        total_params += parameter.numel()
        if parameter.requires_grad:
            trainable_params += parameter.numel()
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
    }


def _build_run_configuration(args: argparse.Namespace) -> RunConfiguration:
    if args.bf16 and args.fp16:
        raise ValueError("Enable at most one of --bf16 or --fp16.")
    processor_name_or_path = args.processor_name_or_path or args.model_name_or_path
    report_targets = _resolve_report_targets(args.report_to)
    if args.wandb_mode == "disabled":
        # transformers refuses report_to="wandb" together with WANDB_DISABLED.
        report_targets = [target for target in report_targets if target != "wandb"]
    dataset_root = args.dataset_root.resolve()
    validation_trajectories_per_task = max(args.validation_trajectories_per_task, 0)
    validation_trajectory_fraction = args.validation_trajectory_fraction
    if validation_trajectory_fraction < 0.0 or validation_trajectory_fraction > 1.0:
        raise ValueError("--validation-trajectory-fraction must be between 0 and 1.")
    if validation_trajectories_per_task > 0 and validation_trajectory_fraction > 0.0:
        raise ValueError(
            "Set only one of --validation-trajectories-per-task or "
            "--validation-trajectory-fraction."
        )
    uses_held_out_validation = (
        validation_trajectories_per_task > 0 or validation_trajectory_fraction > 0.0
    )
    train_tasks = _resolve_task_list(args.train_tasks, dataset_root=dataset_root)
    val_tasks = _resolve_task_list(
        args.val_tasks,
        allow_empty=True,
        dataset_root=dataset_root,
    )
    if uses_held_out_validation and not val_tasks:
        val_tasks = list(train_tasks)
    overlapping_tasks = sorted(set(train_tasks).intersection(val_tasks))
    if overlapping_tasks and not uses_held_out_validation:
        overlap_text = ", ".join(overlapping_tasks)
        raise ValueError(
            "Train and validation tasks must be disjoint unless "
            "--validation-trajectories-per-task or "
            "--validation-trajectory-fraction is positive. "
            f"Overlap: {overlap_text}."
        )
    max_tokens = args.max_length if args.max_length and args.max_length > 0 else None
    return RunConfiguration(
        dataset_root=str(dataset_root),
        preprocessed_data_dir=(
            str(args.preprocessed_data_dir.resolve())
            if args.preprocessed_data_dir is not None
            else None
        ),
        model_name_or_path=args.model_name_or_path,
        processor_name_or_path=processor_name_or_path,
        train_tasks=train_tasks,
        val_tasks=val_tasks,
        validation_trajectories_per_task=validation_trajectories_per_task,
        validation_trajectory_fraction=validation_trajectory_fraction,
        validation_split_seed=(
            args.validation_split_seed
            if args.validation_split_seed is not None
            else args.seed
        ),
        train_reasoning=args.train_reasoning,
        init_adapter_path=args.init_adapter_path,
        use_example_cache=args.use_example_cache,
        trust_example_cache=args.trust_example_cache,
        training_samples_cache_dir=str(
            _resolve_training_samples_cache_dir(
                args.training_samples_cache_dir
            ).resolve()
        ),
        output_dir=str(_resolve_output_dir(args.output_dir).resolve()),
        per_device_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=(
            args.per_device_eval_batch_size
            if args.per_device_eval_batch_size and args.per_device_eval_batch_size > 0
            else None
        ),
        grad_accum=args.grad_accum,
        train_sampling_strategy=args.train_sampling_strategy,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_length=max_tokens,
        max_tokens=max_tokens,
        image_resolution=(
            args.image_resolution
            if args.image_resolution and args.image_resolution > 0
            else None
        ),
        max_images_per_sample=(
            args.max_images_per_sample
            if args.max_images_per_sample and args.max_images_per_sample > 0
            else None
        ),
        supervise_last_assistant_turn_only=args.supervise_last_assistant_turn_only,
        num_workers=args.num_workers,
        example_build_workers=max(args.example_build_workers, 1),
        ddp_timeout_seconds=max(args.ddp_timeout_seconds, 1),
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.gradient_checkpointing,
        ddp_find_unused_parameters=args.ddp_find_unused_parameters,
        attn_implementation=args.attn_implementation,
        trust_remote_code=args.trust_remote_code,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=_split_csv(args.lora_target_modules),
        save_steps=args.save_steps,
        checkpoints_per_epoch=max(args.checkpoints_per_epoch, 0),
        eval_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        max_steps=args.max_steps,
        save_total_limit=args.save_total_limit,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        optim=args.optim,
        eval_max_samples=(
            args.eval_max_samples
            if args.eval_max_samples and args.eval_max_samples > 0
            else None
        ),
        report_to=report_targets,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_tags=_split_csv(args.wandb_tags),
        wandb_mode=args.wandb_mode,
        resume_from_checkpoint=args.resume_from_checkpoint,
        seed=args.seed,
    )


def _save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _load_processor(
    processor_name_or_path: str,
    *,
    trust_remote_code: bool,
):
    processor = AutoProcessor.from_pretrained(
        processor_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        tokenizer.padding_side = "right"
        tokenizer.truncation_side = "left"
    return processor


def _resolve_lora_target_modules(model, target_modules: list[str]) -> list[str]:
    named_modules = list(model.named_modules())
    requested_targets = tuple(target_modules)
    resolved_targets: list[str] = []
    for module_name, module in named_modules:
        if type(module) is not torch.nn.Linear:
            continue
        for target_module in requested_targets:
            if module_name == target_module or module_name.endswith(
                f".{target_module}"
            ):
                resolved_targets.append(module_name)
                break
            wrapped_linear_target = f"{target_module}.linear"
            if module_name == wrapped_linear_target or module_name.endswith(
                f".{wrapped_linear_target}"
            ):
                resolved_targets.append(module_name)
                break

    if not resolved_targets:
        return target_modules

    resolved_targets = sorted(set(resolved_targets))
    if resolved_targets != target_modules:
        _log_startup(
            "Resolved LoRA target modules to "
            f"{len(resolved_targets)} supported Linear modules"
        )
    return resolved_targets


def _load_model(config: RunConfiguration):
    torch_dtype = None
    if torch.cuda.is_available():
        if config.bf16:
            torch_dtype = torch.bfloat16
        elif config.fp16:
            torch_dtype = torch.float16

    model_kwargs: dict[str, Any] = {
        "trust_remote_code": config.trust_remote_code,
    }
    if torch_dtype is not None:
        model_kwargs["torch_dtype"] = torch_dtype
    if config.attn_implementation:
        model_kwargs["attn_implementation"] = config.attn_implementation

    model = AutoVisionLanguageModel.from_pretrained(
        config.model_name_or_path,
        **model_kwargs,
    )
    if config.gradient_checkpointing:
        # use_reentrant=False: reentrant checkpointing (the default) is
        # incompatible with DDP + find_unused_parameters when some batches
        # don't exercise every module (e.g. no-image communicate/give_space
        # steps skip the vision tower) -- it raises "Expected to mark a
        # variable ready only once".
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        # Model stores fewer intermediate activations during the forward pass and recomputes them during backprop. Lower VRAM use, but slower training.
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        # Inference KV-cache. Not useful for training, so disable it.
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    if config.init_adapter_path:
        # v2-continue: start from a published adapter's weights and keep
        # training them (fresh optimizer state, unlike --resume-from-checkpoint).
        _ensure_peft_tensor_parallel_import_compatibility(model)
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model,
            config.init_adapter_path,
            is_trainable=True,
        )
        return model

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=_resolve_lora_target_modules(
            model,
            config.lora_target_modules,
        ),
    )
    model = get_peft_model(model, lora_config)
    return model


def _ensure_peft_tensor_parallel_import_compatibility(model: Any) -> None:
    """Bridge PEFT 0.19.1 to Transformers 4.57 for non-TP adapter loading.

    PEFT imports ``EmbeddingParallel`` unconditionally while loading any
    adapter, but Transformers 4.57 does not export that newer class. The PEFT
    function only uses it for modules carrying an active HF tensor-parallel
    plan and device mesh. Our DDP LoRA training has neither, so a distinct
    placeholder restores the import without altering execution. Fail closed
    if a future caller actually enables tensor parallelism.
    """

    from transformers.integrations import tensor_parallel

    if hasattr(tensor_parallel, "EmbeddingParallel"):
        return
    active_tp_modules = [
        name
        for name, module in model.named_modules()
        if getattr(module, "_hf_tp_plan", None) is not None
        and getattr(module, "_hf_device_mesh", None) is not None
    ]
    if active_tp_modules:
        raise RuntimeError(
            "The installed PEFT/Transformers pair lacks EmbeddingParallel "
            "while the model uses tensor parallelism; use compatible package "
            f"versions instead. Example modules: {active_tp_modules[:3]}"
        )

    class EmbeddingParallel:  # intentionally not a Col/Row parallel subclass
        pass

    tensor_parallel.EmbeddingParallel = EmbeddingParallel


def _launcher_world_size() -> int:
    raw_value = os.environ.get("WORLD_SIZE", "1")
    try:
        return max(int(raw_value), 1)
    except (TypeError, ValueError):
        return 1


def _bind_distributed_cuda_device(state: PartialState) -> None:
    if not torch.cuda.is_available() or state.num_processes <= 1:
        return
    torch.cuda.set_device(getattr(state, "local_process_index", 0))


def _is_primary_process() -> bool:
    raw_value = os.environ.get("RANK", "0")
    try:
        return int(raw_value) == 0
    except (TypeError, ValueError):
        return True


def _validate_runtime_environment() -> None:
    if _launcher_world_size() <= 1 and os.environ.get("LOCAL_RANK") is None:
        return
    if torch.cuda.is_available():
        return
    cuda_build = getattr(torch.version, "cuda", None) or "unknown"
    raise RuntimeError(
        "This run was launched for distributed GPU training, but PyTorch cannot "
        "initialize CUDA in the current environment. "
        f"Installed torch={torch.__version__} "
        f"(CUDA build {cuda_build}). Update the NVIDIA driver on the host or "
        "install a torch build compatible with the host driver before retrying."
    )


def _resolve_precision_config(
    config: RunConfiguration,
    *,
    state: PartialState,
) -> RunConfiguration:
    if not config.bf16:
        return config
    if is_torch_bf16_gpu_available():
        return config
    if not torch.cuda.is_available():
        _log_startup(
            "bf16 requested but CUDA is unavailable; falling back to fp32",
            state=state,
        )
        return replace(config, bf16=False, fp16=False)
    _log_startup(
        "bf16 requested but not supported by the current CUDA runtime/GPU; "
        "falling back to fp16",
        state=state,
    )
    return replace(config, bf16=False, fp16=True)


def _selected_mixed_precision(config: RunConfiguration) -> str:
    if config.bf16:
        return "bf16"
    if config.fp16:
        return "fp16"
    return "no"


def _synchronize_accelerate_precision_env(config: RunConfiguration) -> None:
    os.environ["ACCELERATE_MIXED_PRECISION"] = _selected_mixed_precision(config)


def _log_startup(message: str, *, state: PartialState | None = None) -> None:
    if not _is_primary_process():
        return
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", file=sys.stderr, flush=True)


def _task_example_cache_path(
    *,
    cache_dir: Path,
    dataset_root: Path,
    task_name: str,
    trajectory_ids: set[str] | None,
) -> Path:
    cache_path = build_example_cache_path(
        cache_dir=cache_dir,
        dataset_root=dataset_root,
        task_name=task_name,
    )
    if trajectory_ids is None:
        return cache_path

    selection_key = sha256(
        "\n".join(sorted(trajectory_ids)).encode("utf-8")
    ).hexdigest()[:16]
    return cache_path.with_name(
        f"{cache_path.stem}.traj-{selection_key}{cache_path.suffix}"
    )


def _build_examples_for_tasks(
    *,
    dataset_root: Path,
    task_names: list[str],
    split_name: str,
    train_reasoning: bool = False,
    use_example_cache: bool,
    trust_example_cache: bool,
    training_samples_cache_dir: Path,
    trajectory_ids_by_task: dict[str, set[str]] | None = None,
    example_build_workers: int = 1,
    state: PartialState,
) -> list[Any]:
    examples: list[Any] = []
    total_tasks = len(task_names)
    for task_index, task_name in enumerate(task_names, start=1):
        task_start = time.perf_counter()
        task_label = (
            f"{split_name} examples for task {task_name} "
            f"({task_index}/{total_tasks})"
        )
        progress_description = f"{split_name} {task_name}"
        fingerprint: dict[str, Any] | None = None
        cache_path: Path | None = None
        task_examples: list[Any] | None = None
        task_source = "build"
        trajectory_count: int | None = None
        task_trajectory_ids: set[str] | None = None
        if trajectory_ids_by_task is not None:
            task_trajectory_ids = set(trajectory_ids_by_task.get(task_name, set()))
            trajectory_count = len(task_trajectory_ids)

        cache_allowed = use_example_cache
        if cache_allowed:
            cache_path = _task_example_cache_path(
                cache_dir=training_samples_cache_dir,
                dataset_root=dataset_root,
                task_name=task_name,
                trajectory_ids=task_trajectory_ids,
            )
            # Always fingerprint (and therefore validate) the raw source before
            # trusting a cache.  --trust-example-cache may skip signature
            # comparison, but it may not bypass the trajectory validity gate.
            fingerprint = build_example_cache_fingerprint(
                dataset_root=dataset_root,
                task_name=task_name,
                trajectory_ids=task_trajectory_ids,
                train_reasoning=train_reasoning,
            )
            if trust_example_cache:
                task_examples = load_examples_from_cache(
                    cache_path=cache_path,
                    expected_fingerprint=None,
                )
                if task_examples is not None:
                    task_source = "trusted-cache"
                    trajectory_count = len(
                        {example.trajectory_id for example in task_examples}
                    )
                    _log_startup(
                        f"Loaded {task_label} from trusted cache {cache_path}",
                        state=state,
                    )
            if task_examples is None:
                fingerprint = build_example_cache_fingerprint(
                    dataset_root=dataset_root,
                    task_name=task_name,
                    trajectory_ids=task_trajectory_ids,
                    train_reasoning=train_reasoning,
                )
                trajectory_count = len(fingerprint.get("trajectories", ()))
                task_examples = load_examples_from_cache(
                    cache_path=cache_path,
                    expected_fingerprint=fingerprint,
                )
                if task_examples is not None:
                    task_source = "cache"
                    _log_startup(
                        f"Loaded {task_label} from cache {cache_path}",
                        state=state,
                    )

        if task_examples is None:
            if task_trajectory_ids is not None and not task_trajectory_ids:
                task_examples = []
                task_source = "selected"
            else:
                _log_startup(f"Building {task_label}", state=state)
            if task_examples is not None:
                pass
            elif cache_allowed and not state.is_main_process:
                state.wait_for_everyone()
                if cache_path is not None and fingerprint is not None:
                    task_examples = load_examples_from_cache(
                        cache_path=cache_path,
                        expected_fingerprint=fingerprint,
                    )
                    if task_examples is not None:
                        task_source = "cache"
            else:
                task_examples = build_agent_turn_examples(
                    dataset_root=dataset_root,
                    task_names=[task_name],
                    show_progress=state.is_main_process,
                    progress_description=progress_description,
                    train_reasoning=train_reasoning,
                    trajectory_ids_by_task=(
                        None
                        if task_trajectory_ids is None
                        else {task_name: task_trajectory_ids}
                    ),
                    example_build_workers=example_build_workers,
                )
                if cache_allowed and cache_path is not None and fingerprint is not None:
                    try:
                        save_examples_to_cache(
                            cache_path=cache_path,
                            fingerprint=fingerprint,
                            examples=task_examples,
                        )
                    except OSError as exc:
                        _log_startup(
                            "Warning: failed to save training-samples cache "
                            f"{cache_path}: {exc}",
                            state=state,
                        )
                if cache_allowed:
                    state.wait_for_everyone()

            if task_examples is None:
                task_examples = build_agent_turn_examples(
                    dataset_root=dataset_root,
                    task_names=[task_name],
                    show_progress=False,
                    progress_description=progress_description,
                    train_reasoning=train_reasoning,
                    trajectory_ids_by_task=(
                        None
                        if task_trajectory_ids is None
                        else {task_name: task_trajectory_ids}
                    ),
                    example_build_workers=example_build_workers,
                )

        examples.extend(task_examples)
        elapsed_seconds = time.perf_counter() - task_start
        example_rate = len(task_examples) / elapsed_seconds if elapsed_seconds else 0.0
        log_message = (
            f"Finished {split_name} task {task_name}: {len(task_examples)} "
            f"examples ({len(examples)} cumulative) in {elapsed_seconds:.1f}s "
            f"from {task_source} ({example_rate:.2f} examples/s)"
        )
        metrics: dict[str, float | int] = {
            f"dataset/{split_name}/{task_name}/seconds": elapsed_seconds,
            f"dataset/{split_name}/{task_name}/examples": len(task_examples),
            f"dataset/{split_name}/{task_name}/examples_per_second": example_rate,
            f"dataset/{split_name}/{task_name}/example_build_workers": (
                example_build_workers
            ),
            f"dataset/{split_name}/{task_name}/cache_hit": (
                1 if task_source in {"cache", "trusted-cache"} else 0
            ),
            f"dataset/{split_name}/cumulative_examples": len(examples),
        }
        if trajectory_count is not None:
            trajectory_rate = (
                trajectory_count / elapsed_seconds if elapsed_seconds else 0.0
            )
            log_message += (
                f", {trajectory_count} trajectories "
                f"({trajectory_rate:.2f} trajectories/s)"
            )
            metrics[f"dataset/{split_name}/{task_name}/trajectories"] = trajectory_count
            metrics[
                f"dataset/{split_name}/{task_name}/trajectories_per_second"
            ] = trajectory_rate
        _log_startup(
            log_message,
            state=state,
        )
        _log_wandb_metrics(metrics, state=state)
    return examples


def _select_validation_trajectory_ids(
    *,
    dataset_root: Path,
    val_tasks: list[str],
    train_tasks: list[str],
    trajectories_per_task: int,
    trajectory_fraction: float,
    seed: int,
    state: PartialState,
) -> dict[str, set[str]]:
    selected_by_task = select_held_out_trajectory_ids(
        dataset_root=dataset_root,
        val_tasks=val_tasks,
        train_tasks=train_tasks,
        trajectories_per_task=trajectories_per_task,
        trajectory_fraction=trajectory_fraction,
        seed=seed,
    )
    if not selected_by_task and (
        trajectories_per_task <= 0 and trajectory_fraction <= 0.0
    ):
        return {}

    metrics: dict[str, float | int] = {}
    for task_name in sorted(val_tasks):
        selected_ids = selected_by_task.get(task_name)
        if not selected_ids:
            _log_startup(
                "Selected 0 validation trajectories for task "
                f"{task_name}; not enough trajectories to hold out.",
                state=state,
            )
            continue
        task_trajectory_ids = list_task_trajectory_ids(
            dataset_root=dataset_root,
            task_name=task_name,
        )
        _log_startup(
            "Selected "
            f"{len(selected_ids)}/{len(task_trajectory_ids)} validation "
            f"trajectories for task {task_name}",
            state=state,
        )
        metrics[f"dataset/validation_split/{task_name}/trajectories"] = len(
            selected_ids
        )
        metrics[f"dataset/validation_split/{task_name}/available_trajectories"] = len(
            task_trajectory_ids
        )

    if metrics:
        if trajectories_per_task > 0:
            metrics[
                "dataset/validation_split/trajectories_per_task"
            ] = trajectories_per_task
        if trajectory_fraction > 0.0:
            metrics[
                "dataset/validation_split/trajectory_fraction"
            ] = trajectory_fraction
        _log_wandb_metrics(metrics, state=state)
    return selected_by_task


def _select_train_trajectory_ids(
    *,
    dataset_root: Path,
    train_tasks: list[str],
    selected_validation_trajectory_ids: dict[str, set[str]],
) -> dict[str, set[str]]:
    train_ids_by_task: dict[str, set[str]] = {}
    for task_name in train_tasks:
        held_out_ids = selected_validation_trajectory_ids.get(task_name, set())
        train_ids_by_task[task_name] = {
            trajectory_id
            for trajectory_id in list_task_trajectory_ids(
                dataset_root=dataset_root,
                task_name=task_name,
            )
            if trajectory_id not in held_out_ids
        }
    return train_ids_by_task


def _filter_examples_by_trajectory_ids(
    examples: list[Any],
    trajectory_ids_by_task: dict[str, set[str]],
    *,
    include_selected: bool,
) -> list[Any]:
    if not trajectory_ids_by_task:
        return examples

    filtered_examples: list[Any] = []
    for example in examples:
        selected_ids = trajectory_ids_by_task.get(example.task_name)
        is_selected = selected_ids is not None and example.trajectory_id in selected_ids
        if is_selected == include_selected:
            filtered_examples.append(example)
    return filtered_examples


def _apply_validation_trajectory_split(
    *,
    train_examples: list[Any],
    val_examples: list[Any],
    selected_validation_trajectory_ids: dict[str, set[str]],
    config: RunConfiguration,
    state: PartialState,
) -> tuple[list[Any], list[Any]]:
    if (
        config.validation_trajectories_per_task <= 0
        and config.validation_trajectory_fraction <= 0.0
    ):
        return train_examples, val_examples

    if not selected_validation_trajectory_ids:
        return train_examples, []

    filtered_train_examples = _filter_examples_by_trajectory_ids(
        train_examples,
        selected_validation_trajectory_ids,
        include_selected=False,
    )
    removed_train_examples = len(train_examples) - len(filtered_train_examples)
    _log_startup(
        "Applied held-out validation split: "
        f"{len(filtered_train_examples)} training examples "
        f"({removed_train_examples} held out), "
        f"{len(val_examples)} validation examples",
        state=state,
    )
    _log_wandb_metrics(
        {
            "dataset/validation_split/train_examples": len(filtered_train_examples),
            "dataset/validation_split/validation_examples": len(val_examples),
            "dataset/validation_split/held_out_train_examples": (
                removed_train_examples
            ),
        },
        state=state,
    )
    return filtered_train_examples, val_examples


def _cap_eval_examples(
    *,
    val_examples: list[Any],
    config: RunConfiguration,
    state: PartialState,
) -> list[Any]:
    if config.eval_max_samples is None or len(val_examples) <= config.eval_max_samples:
        return val_examples

    rng = random.Random(config.seed)
    selected_indices = sorted(
        rng.sample(range(len(val_examples)), config.eval_max_samples)
    )
    capped_examples = [val_examples[index] for index in selected_indices]
    _log_startup(
        "Capped evaluation examples to "
        f"{len(capped_examples)}/{len(val_examples)} samples "
        f"with seed {config.seed}",
        state=state,
    )
    _log_wandb_metrics(
        {
            "dataset/eval_cap/original_examples": len(val_examples),
            "dataset/eval_cap/effective_examples": len(capped_examples),
            "config/eval_max_samples": config.eval_max_samples,
        },
        state=state,
    )
    return capped_examples


def _validate_preprocessed_contract(
    *, preprocess_config: dict[str, Any] | None, config: RunConfiguration
) -> None:
    if preprocess_config is None:
        raise ValueError(
            "Preprocessed artifact is missing preprocess_config.json; refusing "
            "to train because its prompt contract cannot be verified."
        )
    from robotalk.tasks.shared.validation_contract import (
        VALIDATOR_CONTRACT_VERSION,
    )
    from robotalk.training.prompting import PROMPT_CONTRACT_VERSION

    expected = {
        "train_example_format": "private_history_consume_once",
        "train_reasoning": config.train_reasoning,
        "trajectory_validator_contract_version": VALIDATOR_CONTRACT_VERSION,
        "prompt_contract_version": PROMPT_CONTRACT_VERSION,
    }
    mismatches: list[str] = []
    for key, runtime_value in expected.items():
        artifact_value = preprocess_config.get(key)
        if key == "train_reasoning" and artifact_value is None:
            # Artifacts created before reasoning became part of the persisted
            # preprocessing contract are unambiguously non-reasoning.
            artifact_value = False
        if artifact_value != runtime_value:
            mismatches.append(
                f"{key}: artifact={artifact_value!r}, runtime={runtime_value!r}"
            )
    if mismatches:
        raise ValueError(
            "Preprocessed artifact does not match the runtime prompt contract: "
            + "; ".join(mismatches)
        )


def _estimate_feature_length(
    feature: dict[str, Any], *, max_images_per_sample: int | None
) -> int:
    text_characters = len(
        json.dumps(feature["messages"], ensure_ascii=False, separators=(",", ":"))
    ) + len(
        json.dumps(feature["tool_schemas"], ensure_ascii=False, separators=(",", ":"))
    )
    image_count = len(feature.get("image_paths", ()))
    if max_images_per_sample is not None and max_images_per_sample > 0:
        image_count = min(image_count, max_images_per_sample)
    return max(1, math.ceil(text_characters / 4)) + image_count * 256


def _build_training_arguments(
    *,
    config: RunConfiguration,
    output_dir: Path,
    evaluation_strategy: str,
    has_validation: bool,
) -> TrainingArguments:
    training_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": config.per_device_batch_size,
        "per_device_eval_batch_size": (
            config.per_device_eval_batch_size or config.per_device_batch_size
        ),
        "gradient_accumulation_steps": config.grad_accum,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "num_train_epochs": config.num_epochs,
        "bf16": config.bf16,
        "fp16": config.fp16,
        "logging_steps": config.logging_steps,
        "save_steps": config.save_steps,
        "eval_steps": config.eval_steps,
        "max_steps": config.max_steps,
        "save_strategy": (
            "no" if config.checkpoints_per_epoch > 0 else "steps"
        ),
        "save_total_limit": config.save_total_limit,
        "remove_unused_columns": False,
        "report_to": config.report_to,
        "run_name": config.wandb_run_name or output_dir.name,
        "dataloader_num_workers": config.num_workers,
        "gradient_checkpointing": config.gradient_checkpointing,
        # True is required when some batches leave LoRA params without
        # gradients (variable image counts do exactly that), but it is also
        # what makes gradient checkpointing fail: checkpointing re-runs the
        # forward during backward, firing each param's hook a second time, so
        # DDP raises "marked as ready twice" -- or, seen on job 239493, just
        # hangs silently until the NCCL timeout. transformers' own default is
        # `not model.is_gradient_checkpointing` for this reason.
        # Kept True by default so existing runs are unchanged; pass
        # --no-ddp-find-unused-parameters to try checkpointing + batch 8.
        "ddp_find_unused_parameters": config.ddp_find_unused_parameters,
        "warmup_ratio": config.warmup_ratio,
        "lr_scheduler_type": config.lr_scheduler_type,
        "optim": config.optim,
        "seed": config.seed,
        "label_names": ["labels"],
        # Keep the final trained adapter active. Reloading the "best" PEFT
        # checkpoint is both misleading when eval/save happens only once
        # before the final step (it discards all later updates) and currently
        # crosses an incompatible PEFT/transformers tensor-parallel load path.
        # Step checkpoints are still written and remain available for explicit
        # resume/evaluation.
        "load_best_model_at_end": False,
    }
    signature = inspect.signature(TrainingArguments.__init__)
    if "evaluation_strategy" in signature.parameters:
        training_kwargs["evaluation_strategy"] = evaluation_strategy
    else:
        training_kwargs["eval_strategy"] = evaluation_strategy
    if "eval_on_start" in signature.parameters:
        training_kwargs["eval_on_start"] = has_validation
    if "ddp_timeout" in signature.parameters:
        training_kwargs["ddp_timeout"] = config.ddp_timeout_seconds
    return TrainingArguments(**training_kwargs)


def _is_loggable_metric_value(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _wandb_eval_section_metrics(
    metrics: dict[str, float],
    *,
    metric_key_prefix: str,
    section: str,
) -> dict[str, float]:
    prefix = f"{metric_key_prefix}_"
    # Hugging Face's W&B callback rewrites only eval_* and test_* keys into
    # top-level eval/ and test/ namespaces. A literal eval/... key is treated
    # as a train metric and becomes train/eval/... in W&B.
    grouped_prefix = f"{metric_key_prefix}_{section}"
    grouped_metrics: dict[str, float] = {}
    for key, value in metrics.items():
        if not _is_loggable_metric_value(value):
            continue
        if key.startswith(prefix):
            grouped_metrics[f"{grouped_prefix}/{key[len(prefix):]}"] = value
        elif key == "epoch":
            grouped_metrics[f"{grouped_prefix}/epoch"] = value
    return grouped_metrics


class FractionalEpochCheckpointCallback(TrainerCallback):
    """Request checkpoints when training crosses fixed fractional epochs."""

    def __init__(self, checkpoints_per_epoch: int) -> None:
        if checkpoints_per_epoch <= 0:
            raise ValueError("checkpoints_per_epoch must be positive")
        self.checkpoints_per_epoch = checkpoints_per_epoch
        self.next_boundary = 1.0 / checkpoints_per_epoch

    def on_step_end(self, args, state, control, **kwargs):
        epoch = float(state.epoch or 0.0)
        epsilon = 1e-9
        if epoch + epsilon >= self.next_boundary:
            control.should_save = True
            while epoch + epsilon >= self.next_boundary:
                self.next_boundary += 1.0 / self.checkpoints_per_epoch
        return control


class LengthGroupedTrainer(Trainer):
    """Trainer that can sample batches grouped by precomputed example length."""

    def __init__(
        self,
        *,
        train_example_lengths: list[int] | None = None,
        **kwargs: Any,
    ) -> None:
        self.train_example_lengths = train_example_lengths
        super().__init__(**kwargs)

    def _get_train_sampler(self, train_dataset: Any | None = None):
        if self.train_example_lengths is None:
            return super()._get_train_sampler(train_dataset)
        resolved_dataset = self.train_dataset if train_dataset is None else train_dataset
        if resolved_dataset is None:
            return None
        if len(resolved_dataset) != len(self.train_example_lengths):
            raise ValueError(
                "Length-grouped sampler metadata does not match the training dataset: "
                f"{len(self.train_example_lengths)} lengths for "
                f"{len(resolved_dataset)} examples."
            )
        return LengthGroupedSampler(
            self.args.train_batch_size * self.args.gradient_accumulation_steps,
            lengths=self.train_example_lengths,
        )

    def evaluate(
        self,
        eval_dataset: Any | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ) -> dict[str, float]:
        metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        base_section_metrics = _wandb_eval_section_metrics(
            metrics,
            metric_key_prefix=metric_key_prefix,
            section="base",
        )
        if base_section_metrics:
            self.log(base_section_metrics)
        return metrics


def main() -> None:
    load_dotenv_file()
    args = parse_args()
    config = _build_run_configuration(args)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _configure_wandb(args, config.report_to, output_dir)
    set_seed(config.seed)
    distributed_state = PartialState(
        timeout=timedelta(seconds=config.ddp_timeout_seconds)
    )
    _bind_distributed_cuda_device(distributed_state)
    _validate_runtime_environment()
    config = _resolve_precision_config(config, state=distributed_state)
    _synchronize_accelerate_precision_env(config)
    _initialize_wandb_run(
        args=args,
        config=config,
        report_targets=config.report_to,
        output_dir=output_dir,
        state=distributed_state,
    )
    dataset_root = Path(config.dataset_root)

    _log_startup(
        f"Starting task VLM training run in {output_dir}",
        state=distributed_state,
    )
    _log_startup(
        f"Using precision {_selected_mixed_precision(config)}",
        state=distributed_state,
    )
    _log_startup(
        "Using max_tokens "
        f"{config.max_tokens if config.max_tokens is not None else 'unbounded'}",
        state=distributed_state,
    )
    _log_startup(
        "Using example build workers "
        f"{config.example_build_workers} "
        f"(example cache={'on' if config.use_example_cache else 'off'}, "
        f"trust={'on' if config.trust_example_cache else 'off'})",
        state=distributed_state,
    )
    if config.max_tokens is not None:
        _log_wandb_metrics(
            {"config/max_tokens": config.max_tokens},
            state=distributed_state,
        )
    _log_wandb_metrics(
        {
            "config/example_build_workers": config.example_build_workers,
            "config/use_example_cache": int(config.use_example_cache),
            "config/trust_example_cache": int(config.trust_example_cache),
        },
        state=distributed_state,
    )

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    loaded_artifact = None
    if config.preprocessed_data_dir is not None:
        _log_startup(
            f"Loading preprocessed artifact from {config.preprocessed_data_dir}",
            state=distributed_state,
        )
        loaded_artifact = load_preprocessed_artifact_from_disk(
            Path(config.preprocessed_data_dir)
        )
        _validate_preprocessed_contract(
            preprocess_config=loaded_artifact.preprocess_config,
            config=config,
        )

    selected_validation_trajectory_ids: dict[str, set[str]] = {}
    uses_held_out_validation = (
        config.validation_trajectories_per_task > 0
        or config.validation_trajectory_fraction > 0.0
    )
    if loaded_artifact is None and uses_held_out_validation and config.val_tasks:
        selected_validation_trajectory_ids = _select_validation_trajectory_ids(
            dataset_root=dataset_root,
            val_tasks=config.val_tasks,
            train_tasks=config.train_tasks,
            trajectories_per_task=config.validation_trajectories_per_task,
            trajectory_fraction=config.validation_trajectory_fraction,
            seed=config.validation_split_seed,
            state=distributed_state,
        )

    train_trajectory_ids_by_task = (
        _select_train_trajectory_ids(
            dataset_root=dataset_root,
            train_tasks=config.train_tasks,
            selected_validation_trajectory_ids=selected_validation_trajectory_ids,
        )
        if uses_held_out_validation and selected_validation_trajectory_ids
        else None
    )
    train_examples = [] if loaded_artifact is not None else _build_examples_for_tasks(
        dataset_root=dataset_root,
        task_names=config.train_tasks,
        split_name="train",
        train_reasoning=config.train_reasoning,
        use_example_cache=config.use_example_cache,
        trust_example_cache=config.trust_example_cache,
        training_samples_cache_dir=Path(config.training_samples_cache_dir),
        trajectory_ids_by_task=train_trajectory_ids_by_task,
        example_build_workers=config.example_build_workers,
        state=distributed_state,
    )
    validation_trajectory_ids_by_task = (
        selected_validation_trajectory_ids if uses_held_out_validation else None
    )
    val_examples = [] if loaded_artifact is not None else _build_examples_for_tasks(
        dataset_root=dataset_root,
        task_names=config.val_tasks,
        split_name="validation",
        train_reasoning=config.train_reasoning,
        use_example_cache=config.use_example_cache,
        trust_example_cache=config.trust_example_cache,
        training_samples_cache_dir=Path(config.training_samples_cache_dir),
        trajectory_ids_by_task=validation_trajectory_ids_by_task,
        example_build_workers=config.example_build_workers,
        state=distributed_state,
    )
    if loaded_artifact is not None:
        train_split = loaded_artifact.train_split
        val_split = loaded_artifact.validation_split
        if config.eval_max_samples is not None and len(val_split) > config.eval_max_samples:
            rng = random.Random(config.seed)
            selected_indices = sorted(
                rng.sample(range(len(val_split)), config.eval_max_samples)
            )
            val_split = val_split.select(selected_indices)
        if len(train_split) == 0:
            raise ValueError("Preprocessed training split is empty.")
        train_dataset = PreprocessedFeatureDataset(
            train_split, artifact_root=loaded_artifact.artifact_root,
            include_pretokenized=False,
        )
        val_dataset = PreprocessedFeatureDataset(
            val_split, artifact_root=loaded_artifact.artifact_root,
            include_pretokenized=False,
        )
        split_manifest = loaded_artifact.split_manifest
        _log_startup(
            "Loaded preprocessed splits: "
            f"{len(train_dataset)} train / {len(val_dataset)} validation examples",
            state=distributed_state,
        )
    else:
        train_examples, val_examples = _apply_validation_trajectory_split(
            train_examples=train_examples,
            val_examples=val_examples,
            selected_validation_trajectory_ids=selected_validation_trajectory_ids,
            config=config,
            state=distributed_state,
        )
        val_examples = _cap_eval_examples(
            val_examples=val_examples,
            config=config,
            state=distributed_state,
        )
        if not train_examples:
            raise ValueError("Training split is empty after validation holdout filtering.")
        train_dataset = AgentTurnDataset(train_examples)
        val_dataset = AgentTurnDataset(val_examples)
        _log_startup("Building split manifest", state=distributed_state)
        split_manifest = build_split_manifest(
            dataset_root=dataset_root,
            train_examples=train_examples,
            val_examples=val_examples,
        )
    train_example_lengths: list[int] | None = None
    if config.train_sampling_strategy == "group_by_length":
        if loaded_artifact is not None:
            train_example_lengths = [
                _estimate_feature_length(
                    train_dataset[index],
                    max_images_per_sample=config.max_images_per_sample,
                )
                for index in range(len(train_dataset))
            ]
        else:
            train_example_lengths = [
                estimate_example_length(
                    example,
                    max_images_per_sample=config.max_images_per_sample,
                )
                for example in train_examples
            ]
        sorted_lengths = sorted(train_example_lengths)
        percentile = lambda fraction: sorted_lengths[  # noqa: E731
            min(len(sorted_lengths) - 1, int(fraction * (len(sorted_lengths) - 1)))
        ]
        _log_startup(
            "Length-grouped training enabled with estimated lengths: "
            f"min={sorted_lengths[0]}, median={percentile(0.5)}, "
            f"p95={percentile(0.95)}, max={sorted_lengths[-1]}. "
            "The longest grouped batch is scheduled first.",
            state=distributed_state,
        )

    _log_startup(
        f"Loading processor from {config.processor_name_or_path}",
        state=distributed_state,
    )
    processor = _load_processor(
        config.processor_name_or_path,
        trust_remote_code=config.trust_remote_code,
    )
    _log_startup(
        f"Loading model weights from {config.model_name_or_path}",
        state=distributed_state,
    )
    model = _load_model(config)
    parameter_counts = _count_parameters(model)
    _log_startup(
        "Model loaded "
        f"({parameter_counts['trainable_params']:,} trainable / "
        f"{parameter_counts['total_params']:,} total parameters)",
        state=distributed_state,
    )

    run_config_payload = asdict(config) | parameter_counts
    if distributed_state.is_main_process:
        _log_startup("Saving run metadata", state=distributed_state)
        _save_json(output_dir / "run_config.json", run_config_payload)
        _save_json(output_dir / "split_manifest.json", split_manifest)
        if distributed_state.num_processes == 1:
            processor.save_pretrained(output_dir / "processor")
        else:
            # The processor is restored from config.processor_name_or_path; avoid
            # the native save path during distributed startup.
            _log_startup(
                "Skipping processor snapshot during distributed startup",
                state=distributed_state,
            )

    data_collator = LazyVisionSFTCollator(
        processor_name_or_path=config.processor_name_or_path,
        max_length=config.max_length,
        max_images_per_sample=config.max_images_per_sample,
        supervise_last_assistant_turn_only=config.supervise_last_assistant_turn_only,
        trust_remote_code=config.trust_remote_code,
        image_resolution=config.image_resolution,
    )

    evaluation_strategy = "steps" if len(val_dataset) > 0 else "no"
    if (
        len(val_dataset) > 0
        and config.checkpoints_per_epoch == 0
        and config.save_steps % config.eval_steps != 0
    ):
        raise ValueError(
            "--save-steps must be a multiple of --eval-steps when validation is enabled."
        )
    training_args = _build_training_arguments(
        config=config,
        output_dir=output_dir,
        evaluation_strategy=evaluation_strategy,
        has_validation=len(val_dataset) > 0,
    )

    _log_startup("Initializing Trainer", state=distributed_state)
    callbacks = (
        [FractionalEpochCheckpointCallback(config.checkpoints_per_epoch)]
        if config.checkpoints_per_epoch > 0
        else None
    )
    trainer = LengthGroupedTrainer(
        train_example_lengths=train_example_lengths,
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=val_dataset if len(val_dataset) > 0 else None,
        callbacks=callbacks,
    )

    _log_startup(
        f"Starting trainer.train() with {len(train_dataset)} training samples "
        f"and {len(val_dataset)} validation samples",
        state=distributed_state,
    )
    if len(val_dataset) > 0 and not getattr(training_args, "eval_on_start", False):
        _log_startup(
            "Running initial evaluation before training",
            state=distributed_state,
        )
        initial_eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", initial_eval_metrics)
        trainer.save_metrics("eval", initial_eval_metrics)
    train_result = trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    trainer.save_model()
    trainer.save_state()
    trainer.log(train_result.metrics)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_metrics("train", train_result.metrics)

    eval_metrics: dict[str, float] = {}
    if len(val_dataset) > 0:
        eval_metrics = trainer.evaluate()
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero():
        final_metrics = train_result.metrics | eval_metrics
        _save_json(output_dir / "final_metrics.json", final_metrics)
    trainer.accelerator.wait_for_everyone()
    _log_startup("Run complete", state=distributed_state)


if __name__ == "__main__":
    main()
