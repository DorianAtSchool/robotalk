"""CPU-oriented task-VLM preprocessing and optional Hugging Face publishing."""

from __future__ import annotations

import argparse
import importlib
import os
import resource
import sys
import time
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional in lightweight test envs
    tqdm = None

from robotalk.generation.runtime.client import load_dotenv_file
from robotalk.training.dataset import (
    build_batched_pretokenized_tensors,
    build_agent_turn_examples,
    build_example_cache_fingerprint,
    build_example_cache_path,
    build_same_task_trajectory_split,
    build_split_manifest,
    load_examples_from_cache,
    save_examples_to_cache,
    serialize_pretokenized_tensors,
)
from robotalk.training.preprocessed_data import (
    _PREPROCESSED_FORMAT_VERSION,
    PretokenizedShardWriter,
    build_or_load_cached_pretokenization_metadata,
    build_preprocessed_dataset_dict,
    build_pretokenization_metadata,
    existing_artifact_image_relpaths_for_examples,
    push_preprocessed_artifact_to_hub,
    save_preprocessed_artifact,
    stage_artifact_images_for_examples,
)
from robotalk.training.task_registry import resolve_task_name, supported_task_names
from robotalk.training.prompting import (
    PROMPT_CONTRACT_VERSION,
)
from robotalk.tasks.shared.validation_contract import (
    VALIDATOR_CONTRACT_VERSION,
)
from robotalk.utils import paper_train_tasks

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATASET_ROOT = Path(os.environ.get("ROBOTALK_DATA_ROOT", "data/robotalk_rendered"))
_DEFAULT_IMAGE_RESOLUTION = 512
_PRETOKENIZATION_PROGRESS_INTERVAL = 1000
_EXAMPLE_CACHE_LOCK_TIMEOUT_SECONDS = 1800.0


def _split_csv(raw_value: str) -> list[str]:
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def _resolve_task_list(raw_value: str) -> list[str]:
    resolved = [resolve_task_name(task_name) for task_name in _split_csv(raw_value)]
    if not resolved:
        supported = ", ".join(supported_task_names())
        raise ValueError(
            f"At least one task is required. Supported tasks: {supported}."
        )
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"Duplicate task names are not allowed: {resolved}.")
    return resolved


def _resolve_validation_task_list(
    raw_value: str | None,
    *,
    train_tasks: list[str],
    validation_split_mode: str,
) -> list[str]:
    if validation_split_mode == "none":
        return []
    if raw_value is None or raw_value.strip() in {"", "same_as_train", "train"}:
        return list(train_tasks)
    return _resolve_task_list(raw_value)


def _resolve_training_samples_cache_dir(
    training_samples_cache_dir: Path | None,
) -> Path:
    if training_samples_cache_dir is not None:
        return training_samples_cache_dir
    return _REPO_ROOT / ".cache" / "robotalk" / "examples"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=_DEFAULT_DATASET_ROOT,
        help="Root directory of the rendered trajectory dataset.",
    )
    parser.add_argument(
        "--train-tasks",
        default=paper_train_tasks(),
        help="Comma-separated task names for the training split.",
    )
    parser.add_argument(
        "--val-tasks",
        default=None,
        help=(
            "Comma-separated task names for the validation split. Defaults to "
            "`same_as_train` in same-task mode."
        ),
    )
    parser.add_argument(
        "--train-reasoning",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Supervise each assistant tool call with the trajectory's "
            "<think>{reasoning}</think> prefix."
        ),
    )
    parser.add_argument(
        "--validation-split-mode",
        choices=("none", "same-task"),
        default="none",
        help=(
            "No validation set (as for the paper), or hold out a fraction of "
            "each training task's trajectories for validation."
        ),
    )
    parser.add_argument(
        "--validation-trajectory-fraction",
        type=float,
        default=0.0,
        help="Fraction of trajectories per validation task held out in same-task mode.",
    )
    parser.add_argument(
        "--validation-min-trajectories-per-task",
        type=int,
        default=0,
        help="Minimum validation trajectories per task in same-task mode.",
    )
    parser.add_argument(
        "--validation-split-seed",
        type=int,
        default=42,
        help="Seed used to shuffle each task's trajectory IDs before holdout.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Artifact output directory.",
    )
    parser.add_argument(
        "--use-example-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse cached serialized training samples across preprocessing runs.",
    )
    parser.add_argument(
        "--training-samples-cache-dir",
        dest="training_samples_cache_dir",
        type=Path,
        default=None,
        help="Directory for cached serialized training samples.",
    )
    parser.add_argument(
        "--example-cache-dir",
        dest="training_samples_cache_dir",
        type=Path,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--example-build-workers",
        type=int,
        default=1,
        help="Number of CPU worker processes to use when building examples.",
    )
    parser.add_argument(
        "--pretokenize",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Tokenize examples with the selected processor during preprocessing "
            "and store reusable tensor payloads in the artifact."
        ),
    )
    parser.add_argument(
        "--processor-name-or-path",
        default=None,
        help=(
            "Processor id or local path used for pretokenization. Required when "
            "--pretokenize is enabled."
        ),
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Optional sequence truncation cap applied during pretokenization.",
    )
    parser.add_argument(
        "--image-resolution",
        type=int,
        default=None,
        help=(
            "Square image resolution passed to the multimodal processor. "
            "Defaults to --artifact-image-size when set, otherwise 512. "
            "Internally this sets Qwen min_pixels and max_pixels to "
            "image_resolution ** 2."
        ),
    )
    parser.add_argument(
        "--artifact-image-size",
        type=int,
        default=None,
        help=(
            "Optional square pixel size for images stored in the artifact. "
            "For example, 256 writes 256x256 artifact images without modifying "
            "the source dataset."
        ),
    )
    parser.add_argument(
        "--resume-existing-artifact-images",
        nargs="?",
        const="validated",
        choices=("validated", "unchecked"),
        default="disabled",
        help=(
            "Resume a partial artifact whose output directory already contains "
            "the staged images/ directory. The default mode validates every "
            "expected staged image; pass `unchecked` to reconstruct deterministic "
            "artifact relpaths without touching each image file."
        ),
    )
    parser.add_argument(
        "--no-resume-existing-artifact-images",
        dest="resume_existing_artifact_images",
        action="store_const",
        const="disabled",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--skip-existing-artifact-image-validation",
        action="store_true",
        help=(
            "Alias for `--resume-existing-artifact-images unchecked` when "
            "resuming known-good staged artifact images."
        ),
    )
    parser.add_argument(
        "--pretokenize-batch-size",
        type=int,
        default=1,
        help="Number of examples sent through the processor per pretokenization call.",
    )
    parser.add_argument(
        "--pretokenize-flush-interval",
        type=int,
        default=400,
        help=(
            "Number of serialized examples to buffer before flushing a "
            "pretokenization shard chunk to disk."
        ),
    )
    parser.add_argument(
        "--pretokenize-shard-index",
        type=int,
        default=None,
        help="Zero-based pretokenization shard index for Slurm array jobs.",
    )
    parser.add_argument(
        "--pretokenize-num-shards",
        type=int,
        default=None,
        help="Total number of pretokenization shards.",
    )
    parser.add_argument(
        "--pretokenize-shard-output-dir",
        type=Path,
        default=None,
        help="Directory where this pretokenization shard writes Arrow outputs.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Pass trust_remote_code through to AutoProcessor.from_pretrained.",
    )
    parser.add_argument(
        "--push-to-hub",
        default=None,
        help="Optional Hugging Face dataset repo id to publish after preprocessing.",
    )
    parser.add_argument(
        "--hub-revision",
        default=None,
        help="Optional Hugging Face revision or branch name for publishing.",
    )
    parser.add_argument(
        "--hub-private",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Create or update the Hub dataset repo as private.",
    )
    parser.add_argument(
        "--hub-data-dir",
        default=None,
        help="Optional subdirectory inside the Hub repo for dataset files and sidecars.",
    )
    return parser.parse_args()


def _log(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    formatted_message = f"[{timestamp}] {message}"
    if tqdm is not None:
        tqdm.write(formatted_message, file=sys.stderr)
        return
    print(formatted_message, file=sys.stderr, flush=True)


def _format_memory_bytes(byte_count: int) -> str:
    value = float(byte_count)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TiB"


def _read_current_rss_bytes() -> int | None:
    status_path = Path("/proc/self/status")
    try:
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if not line.startswith("VmRSS:"):
                continue
            fields = line.split()
            if len(fields) >= 2:
                return int(fields[1]) * 1024
    except (OSError, ValueError):
        return None
    return None


def _read_peak_rss_bytes() -> int | None:
    try:
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (OSError, ValueError):
        return None
    if peak_rss < 0:
        return None
    if sys.platform == "darwin":
        return int(peak_rss)
    return int(peak_rss) * 1024


def _log_process_memory_usage(label: str) -> None:
    memory_fields = []
    current_rss_bytes = _read_current_rss_bytes()
    if current_rss_bytes is not None:
        memory_fields.append(f"current_rss={_format_memory_bytes(current_rss_bytes)}")
    peak_rss_bytes = _read_peak_rss_bytes()
    if peak_rss_bytes is not None:
        memory_fields.append(f"peak_rss={_format_memory_bytes(peak_rss_bytes)}")
    if memory_fields:
        _log(f"Memory usage {label}: {', '.join(memory_fields)}")


def _create_progress_bar(*, total: int, description: str, unit: str):
    if total <= 0 or tqdm is None:
        return None
    return tqdm(
        total=total,
        desc=description,
        dynamic_ncols=True,
        leave=False,
        unit=unit,
    )


def _load_processor(
    *,
    processor_name_or_path: str,
    trust_remote_code: bool,
):
    try:
        transformers_module = importlib.import_module("transformers")
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise ImportError(
            "Missing optional dependency 'transformers'. Install "
            "the training dependencies: pip install -e '.[train]'."
        ) from exc

    auto_processor_cls = getattr(transformers_module, "AutoProcessor", None)
    if auto_processor_cls is None:  # pragma: no cover - defensive
        raise ImportError(
            "transformers.AutoProcessor is unavailable in the active environment."
        )

    try:
        processor = auto_processor_cls.from_pretrained(
            processor_name_or_path,
            trust_remote_code=trust_remote_code,
        )
    except ImportError as exc:
        if "Torchvision" in str(exc):
            raise ImportError(
                "Pretokenization processor loading requires torchvision. Install "
                "the training dependencies: pip install -e '.[train]'."
            ) from exc
        raise
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None:
        tokenizer.padding_side = "right"
        tokenizer.truncation_side = "left"
    return processor


def _should_report_progress(index: int, total: int, *, interval: int) -> bool:
    if total <= 0:
        return False
    return index == total or index % max(1, interval) == 0


def stable_sample_rank(sample_id: str) -> int:
    return int(sha256(sample_id.encode("utf-8")).hexdigest(), 16)


def _select_pretokenization_shard_examples(
    *,
    examples: list[Any],
    shard_index: int,
    num_shards: int,
) -> list[Any]:
    return [
        example
        for example in examples
        if stable_sample_rank(example.sample_id) % num_shards == shard_index
    ]


def _build_pretokenization_shard_sample_id_filter(
    *,
    shard_index: int,
    num_shards: int,
) -> Callable[[str], bool]:
    return lambda sample_id: stable_sample_rank(sample_id) % num_shards == shard_index


def _pretokenize_examples(
    *,
    split_name: str,
    examples: list[Any],
    processor,
    max_length: int | None,
    image_resolution: int | None,
    batch_size: int = 1,
    artifact_root: Path | None = None,
    image_relpaths_by_source: dict[str, str] | None = None,
    flush_callback: Callable[[dict[str, bytes]], None] | None = None,
    flush_interval: int | None = None,
) -> dict[str, bytes]:
    if not examples:
        return {}
    if flush_callback is not None and (flush_interval is None or flush_interval < 1):
        raise ValueError("flush_interval must be at least 1 when flushing is enabled.")

    total_examples = len(examples)
    _log(f"Pretokenizing {split_name} examples ({total_examples} samples)")
    blobs_by_sample_id: dict[str, bytes] = {}
    pending_blobs_by_sample_id: dict[str, bytes] = {}
    progress_bar = _create_progress_bar(
        total=total_examples,
        description=f"pretokenize {split_name}",
        unit="example",
    )
    processed_count = 0

    def flush_pending(*, force: bool = False) -> None:
        if flush_callback is None:
            return
        if not pending_blobs_by_sample_id:
            return
        if not force and len(pending_blobs_by_sample_id) < int(flush_interval):
            return
        flush_callback(dict(pending_blobs_by_sample_id))
        pending_blobs_by_sample_id.clear()

    try:
        for start_index in range(0, total_examples, batch_size):
            batch_examples = examples[start_index : start_index + batch_size]
            features = []
            for example in batch_examples:
                feature = example.to_feature_dict()
                if image_relpaths_by_source is not None:
                    if artifact_root is None:
                        raise ValueError(
                            "artifact_root is required with image_relpaths_by_source."
                        )
                    feature["image_paths"] = [
                        str(
                            (
                                artifact_root / image_relpaths_by_source[image_path]
                            ).resolve()
                        )
                        for image_path in feature.get("image_paths", ())
                    ]
                features.append(feature)
            pretokenized_tensor_batch = build_batched_pretokenized_tensors(
                processor=processor,
                features=features,
                max_length=max_length,
                image_resolution=image_resolution,
            )
            for example, pretokenized_tensors in zip(
                batch_examples,
                pretokenized_tensor_batch,
                strict=True,
            ):
                serialized_tensors = serialize_pretokenized_tensors(
                    pretokenized_tensors
                )
                if flush_callback is None:
                    blobs_by_sample_id[example.sample_id] = serialized_tensors
                else:
                    pending_blobs_by_sample_id[example.sample_id] = serialized_tensors
                    flush_pending()
            processed_count += len(batch_examples)
            if progress_bar is not None:
                progress_bar.update(len(batch_examples))
            elif _should_report_progress(
                processed_count,
                total_examples,
                interval=_PRETOKENIZATION_PROGRESS_INTERVAL,
            ):
                _log(
                    "Pretokenized "
                    f"{split_name} examples: {processed_count}/"
                    f"{total_examples}"
                )
        flush_pending(force=True)
    finally:
        if progress_bar is not None:
            progress_bar.close()
    if progress_bar is not None:
        _log(
            f"Pretokenized {split_name} examples: " f"{total_examples}/{total_examples}"
        )
    return blobs_by_sample_id


def _build_examples_for_tasks(
    *,
    dataset_root: Path,
    task_names: list[str],
    split_name: str,
    trajectory_ids_by_task: dict[str, list[str]] | None,
    use_example_cache: bool,
    training_samples_cache_dir: Path,
    example_build_workers: int,
    train_reasoning: bool = False,
    example_filter: Callable[[str], bool] | None = None,
) -> list[Any]:
    examples: list[Any] = []
    total_tasks = len(task_names)
    for task_index, task_name in enumerate(task_names, start=1):
        task_label = (
            f"{split_name} examples for task "
            f"{task_name} ({task_index}/{total_tasks})"
        )
        progress_description = f"{split_name} {task_name}"
        task_examples: list[Any] | None = None
        trajectory_ids = (
            None
            if trajectory_ids_by_task is None
            else trajectory_ids_by_task.get(task_name, [])
        )

        cache_allowed = use_example_cache and trajectory_ids is None
        if cache_allowed:
            fingerprint = build_example_cache_fingerprint(
                dataset_root=dataset_root,
                task_name=task_name,
                trajectory_ids=trajectory_ids,
                train_reasoning=train_reasoning,
            )
            cache_path = build_example_cache_path(
                cache_dir=training_samples_cache_dir,
                dataset_root=dataset_root,
                task_name=task_name,
            )
            task_examples = load_examples_from_cache(
                cache_path=cache_path,
                expected_fingerprint=fingerprint,
                sample_id_filter=example_filter,
            )
            if task_examples is not None:
                _log(f"Loaded {task_label} from cache {cache_path}")
        else:
            fingerprint = None
            cache_path = None

        if task_examples is None:
            lock_dir = (
                None
                if not cache_allowed or cache_path is None
                else cache_path.with_name(f"{cache_path.name}.lock")
            )
            owns_lock = False
            if lock_dir is not None:
                deadline = time.monotonic() + _EXAMPLE_CACHE_LOCK_TIMEOUT_SECONDS
                while True:
                    try:
                        lock_dir.mkdir(parents=True)
                    except FileExistsError:
                        task_examples = load_examples_from_cache(
                            cache_path=cache_path,
                            expected_fingerprint=fingerprint,
                            sample_id_filter=example_filter,
                        )
                        if task_examples is not None:
                            _log(f"Loaded {task_label} from cache {cache_path}")
                            break
                        if time.monotonic() >= deadline:
                            raise TimeoutError(
                                "Timed out waiting for example cache lock: "
                                f"{lock_dir}"
                            )
                        time.sleep(1.0)
                    else:
                        owns_lock = True
                        task_examples = load_examples_from_cache(
                            cache_path=cache_path,
                            expected_fingerprint=fingerprint,
                            sample_id_filter=example_filter,
                        )
                        if task_examples is not None:
                            _log(f"Loaded {task_label} from cache {cache_path}")
                        break
            try:
                if task_examples is None:
                    _log(f"Building {task_label}")
                    task_examples = build_agent_turn_examples(
                        dataset_root=dataset_root,
                        task_names=[task_name],
                        trajectory_ids_by_task=None
                        if trajectory_ids is None
                        else {task_name: set(trajectory_ids)},
                        show_progress=True,
                        progress_description=progress_description,
                        train_reasoning=train_reasoning,
                    )
                    if (
                        cache_allowed
                        and cache_path is not None
                        and fingerprint is not None
                    ):
                        save_examples_to_cache(
                            cache_path=cache_path,
                            fingerprint=fingerprint,
                            examples=task_examples,
                        )
                if example_filter is not None:
                    task_examples = [
                        example
                        for example in task_examples
                        if example_filter(example.sample_id)
                    ]
            finally:
                if owns_lock and lock_dir is not None:
                    try:
                        lock_dir.rmdir()
                    except OSError:
                        pass

        examples.extend(task_examples)
        _log(
            f"Finished {split_name} task {task_name}: {len(task_examples)} "
            f"examples ({len(examples)} cumulative)"
        )
    return examples


def main() -> None:
    load_dotenv_file()
    args = parse_args()
    train_reasoning = getattr(args, "train_reasoning", False)
    if args.skip_existing_artifact_image_validation:
        args.resume_existing_artifact_images = "unchecked"

    shard_args = (
        args.pretokenize_shard_index,
        args.pretokenize_num_shards,
        args.pretokenize_shard_output_dir,
    )
    shard_mode = any(value is not None for value in shard_args)
    if shard_mode and not all(value is not None for value in shard_args):
        raise ValueError(
            "--pretokenize-shard-index, --pretokenize-num-shards, and "
            "--pretokenize-shard-output-dir must be provided together."
        )
    if shard_mode and not args.pretokenize:
        raise ValueError("--pretokenize is required for pretokenization shard mode.")

    if args.artifact_image_size is not None and args.artifact_image_size < 1:
        raise ValueError("--artifact-image-size must be at least 1.")
    image_resolution = args.image_resolution
    if image_resolution is None:
        image_resolution = (
            args.artifact_image_size
            if args.artifact_image_size is not None
            else _DEFAULT_IMAGE_RESOLUTION
        )

    if args.example_build_workers < 1:
        raise ValueError("--example-build-workers must be at least 1.")
    if args.pretokenize_batch_size < 1:
        raise ValueError("--pretokenize-batch-size must be at least 1.")
    if args.pretokenize_flush_interval < 1:
        raise ValueError("--pretokenize-flush-interval must be at least 1.")
    if args.pretokenize_num_shards is not None and args.pretokenize_num_shards < 1:
        raise ValueError("--pretokenize-num-shards must be at least 1.")
    if args.pretokenize_shard_index is not None:
        if args.pretokenize_shard_index < 0:
            raise ValueError("--pretokenize-shard-index must be non-negative.")
        if (
            args.pretokenize_num_shards is not None
            and args.pretokenize_shard_index >= args.pretokenize_num_shards
        ):
            raise ValueError(
                "--pretokenize-shard-index must be less than --pretokenize-num-shards."
            )
    if image_resolution is not None and image_resolution < 1:
        raise ValueError("--image-resolution must be at least 1.")
    if args.pretokenize and not args.processor_name_or_path:
        raise ValueError(
            "--processor-name-or-path is required when --pretokenize is enabled."
        )

    dataset_root = args.dataset_root.resolve()
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        if args.resume_existing_artifact_images != "disabled":
            unexpected_entries = [
                entry
                for entry in output_dir.iterdir()
                if entry.name not in {"images", "pretokenized_shards"}
            ]
            images_dir = output_dir / "images"
            images_required = not (shard_mode and args.artifact_image_size is None)
            if unexpected_entries or (images_required and not images_dir.is_dir()):
                unexpected_text = ", ".join(
                    str(entry.relative_to(output_dir))
                    for entry in unexpected_entries[:5]
                )
                if len(unexpected_entries) > 5:
                    unexpected_text += f", ... ({len(unexpected_entries)} entries)"
                if not unexpected_text:
                    unexpected_text = "missing images/ directory"
                raise FileExistsError(
                    "Cannot resume existing artifact images because the output "
                    f"directory contains non-resumable contents: {unexpected_text}"
                )
        else:
            raise FileExistsError(
                f"Output directory already exists and is not empty: {output_dir}"
            )

    train_tasks = _resolve_task_list(args.train_tasks)
    val_tasks = _resolve_validation_task_list(
        args.val_tasks,
        train_tasks=train_tasks,
        validation_split_mode=args.validation_split_mode,
    )
    train_trajectory_ids_by_task: dict[str, list[str]] | None = None
    val_trajectory_ids_by_task: dict[str, list[str]] | None = None
    if args.validation_split_mode == "same-task":
        trajectory_split = build_same_task_trajectory_split(
            dataset_root=dataset_root,
            train_task_names=train_tasks,
            validation_task_names=val_tasks,
            validation_fraction=args.validation_trajectory_fraction,
            min_validation_trajectories_per_task=(
                args.validation_min_trajectories_per_task
            ),
            seed=args.validation_split_seed,
        )
        train_trajectory_ids_by_task = trajectory_split.train_trajectory_ids_by_task
        val_trajectory_ids_by_task = trajectory_split.validation_trajectory_ids_by_task

    training_samples_cache_dir = _resolve_training_samples_cache_dir(
        args.training_samples_cache_dir
    ).resolve()
    pretokenization_shard_sample_id_filter = None
    if args.pretokenize and shard_mode:
        pretokenization_shard_sample_id_filter = (
            _build_pretokenization_shard_sample_id_filter(
                shard_index=args.pretokenize_shard_index,
                num_shards=args.pretokenize_num_shards,
            )
        )
    artifact_image_size = (
        None
        if args.artifact_image_size is None
        else (args.artifact_image_size, args.artifact_image_size)
    )
    pretokenization_processor = None
    train_pretokenized_blobs_by_sample_id: dict[str, bytes] | None = None
    validation_pretokenized_blobs_by_sample_id: dict[str, bytes] | None = None
    pretokenization_config: dict[str, Any] | None = None
    if args.pretokenize:
        processor_name_or_path = str(args.processor_name_or_path).strip()
        _log(f"Loading processor {processor_name_or_path} for pretokenization")
        pretokenization_processor = _load_processor(
            processor_name_or_path=processor_name_or_path,
            trust_remote_code=args.trust_remote_code,
        )
        pretokenization_metadata_kwargs = {
            "processor": pretokenization_processor,
            "processor_name_or_path": processor_name_or_path,
            "max_length": args.max_length,
            "image_resolution": image_resolution,
            "trust_remote_code": args.trust_remote_code,
        }
        if shard_mode:
            pretokenization_config = build_or_load_cached_pretokenization_metadata(
                **pretokenization_metadata_kwargs,
                cache_path=(
                    args.pretokenize_shard_output_dir.resolve()
                    / "pretokenization_metadata.json"
                ),
                progress_callback=_log,
            )
        else:
            pretokenization_config = build_pretokenization_metadata(
                **pretokenization_metadata_kwargs,
            )
        if shard_mode:
            _log_process_memory_usage("after processor load")

    _log(f"Starting preprocessing into {output_dir}")
    train_examples = _build_examples_for_tasks(
        dataset_root=dataset_root,
        task_names=train_tasks,
        split_name="train",
        trajectory_ids_by_task=train_trajectory_ids_by_task,
        use_example_cache=args.use_example_cache,
        training_samples_cache_dir=training_samples_cache_dir,
        example_build_workers=args.example_build_workers,
        train_reasoning=train_reasoning,
        example_filter=pretokenization_shard_sample_id_filter,
    )
    validation_examples = _build_examples_for_tasks(
        dataset_root=dataset_root,
        train_reasoning=train_reasoning,
        task_names=val_tasks,
        split_name="validation",
        trajectory_ids_by_task=val_trajectory_ids_by_task,
        use_example_cache=args.use_example_cache,
        training_samples_cache_dir=training_samples_cache_dir,
        example_build_workers=args.example_build_workers,
        example_filter=pretokenization_shard_sample_id_filter,
    )
    train_examples_to_pretokenize = train_examples
    validation_examples_to_pretokenize = validation_examples
    examples_requiring_artifact_image_relpaths = list(train_examples) + list(
        validation_examples
    )
    if args.pretokenize and shard_mode:
        train_examples_to_pretokenize = _select_pretokenization_shard_examples(
            examples=train_examples,
            shard_index=args.pretokenize_shard_index,
            num_shards=args.pretokenize_num_shards,
        )
        validation_examples_to_pretokenize = _select_pretokenization_shard_examples(
            examples=validation_examples,
            shard_index=args.pretokenize_shard_index,
            num_shards=args.pretokenize_num_shards,
        )
        examples_requiring_artifact_image_relpaths = list(
            train_examples_to_pretokenize
        ) + list(validation_examples_to_pretokenize)
        _log(
            "Selected pretokenization shard "
            f"{args.pretokenize_shard_index}/{args.pretokenize_num_shards}: "
            f"{len(train_examples_to_pretokenize)} train, "
            f"{len(validation_examples_to_pretokenize)} validation examples"
        )
        _log_process_memory_usage("after shard example selection")

    image_relpaths_by_source: dict[str, str] | None = None
    if (
        args.resume_existing_artifact_images != "disabled"
        and (output_dir / "images").is_dir()
    ):
        image_relpaths_by_source = existing_artifact_image_relpaths_for_examples(
            examples=examples_requiring_artifact_image_relpaths,
            output_dir=output_dir,
            validate=args.resume_existing_artifact_images == "validated",
            progress_callback=_log,
        )
    elif shard_mode and artifact_image_size is not None:
        raise FileNotFoundError(
            "Pretokenization shard mode with --artifact-image-size requires an "
            "existing output images/ directory. Run image staging first or pass "
            "--resume-existing-artifact-images unchecked."
        )
    elif artifact_image_size is not None and args.pretokenize:
        image_relpaths_by_source = stage_artifact_images_for_examples(
            examples=examples_requiring_artifact_image_relpaths,
            output_dir=output_dir,
            artifact_image_size=artifact_image_size,
            progress_callback=_log,
        )
    if args.pretokenize and shard_mode:
        _log_process_memory_usage("after artifact image relpath resolution")
    if args.pretokenize and shard_mode:
        if pretokenization_processor is None:  # pragma: no cover - defensive
            raise RuntimeError(
                "Pretokenization was requested but no processor was loaded."
            )
        if pretokenization_config is None:  # pragma: no cover - defensive
            raise RuntimeError("Pretokenization metadata was not built.")
        shard_output_dir = args.pretokenize_shard_output_dir.resolve()
        with PretokenizedShardWriter(
            shard_dir=shard_output_dir,
            shard_index=args.pretokenize_shard_index,
            num_shards=args.pretokenize_num_shards,
            pretokenization=pretokenization_config,
            progress_callback=_log,
        ) as shard_writer:
            completed_train_sample_ids = shard_writer.completed_sample_ids("train")
            if completed_train_sample_ids:
                _log(
                    "Skipping "
                    f"{len(completed_train_sample_ids)} already-flushed train "
                    "pretokenized samples"
                )
            remaining_train_examples = [
                example
                for example in train_examples_to_pretokenize
                if example.sample_id not in completed_train_sample_ids
            ]
            _pretokenize_examples(
                split_name="train",
                examples=remaining_train_examples,
                processor=pretokenization_processor,
                max_length=args.max_length,
                image_resolution=image_resolution,
                batch_size=args.pretokenize_batch_size,
                artifact_root=output_dir,
                image_relpaths_by_source=image_relpaths_by_source,
                flush_callback=lambda blobs: shard_writer.write_split_chunk(
                    split_name="train",
                    blobs_by_sample_id=blobs,
                ),
                flush_interval=args.pretokenize_flush_interval,
            )
            _log_process_memory_usage("after train pretokenization")
            completed_validation_sample_ids = shard_writer.completed_sample_ids(
                "validation"
            )
            if completed_validation_sample_ids:
                _log(
                    "Skipping "
                    f"{len(completed_validation_sample_ids)} already-flushed "
                    "validation pretokenized samples"
                )
            remaining_validation_examples = [
                example
                for example in validation_examples_to_pretokenize
                if example.sample_id not in completed_validation_sample_ids
            ]
            _pretokenize_examples(
                split_name="validation",
                examples=remaining_validation_examples,
                processor=pretokenization_processor,
                max_length=args.max_length,
                image_resolution=image_resolution,
                batch_size=args.pretokenize_batch_size,
                artifact_root=output_dir,
                image_relpaths_by_source=image_relpaths_by_source,
                flush_callback=lambda blobs: shard_writer.write_split_chunk(
                    split_name="validation",
                    blobs_by_sample_id=blobs,
                ),
                flush_interval=args.pretokenize_flush_interval,
            )
            _log_process_memory_usage("after validation pretokenization")
        _log_process_memory_usage("after shard save")
        _log("Pretokenization shard complete")
        return

    if args.pretokenize:
        if pretokenization_processor is None:  # pragma: no cover - defensive
            raise RuntimeError(
                "Pretokenization was requested but no processor was loaded."
            )
        train_pretokenized_blobs_by_sample_id = _pretokenize_examples(
            split_name="train",
            examples=train_examples_to_pretokenize,
            processor=pretokenization_processor,
            max_length=args.max_length,
            image_resolution=image_resolution,
            batch_size=args.pretokenize_batch_size,
            artifact_root=output_dir,
            image_relpaths_by_source=image_relpaths_by_source,
        )
        validation_pretokenized_blobs_by_sample_id = _pretokenize_examples(
            split_name="validation",
            examples=validation_examples_to_pretokenize,
            processor=pretokenization_processor,
            max_length=args.max_length,
            image_resolution=image_resolution,
            batch_size=args.pretokenize_batch_size,
            artifact_root=output_dir,
            image_relpaths_by_source=image_relpaths_by_source,
        )
    split_manifest = build_split_manifest(
        dataset_root=dataset_root,
        train_examples=train_examples,
        val_examples=validation_examples,
    )
    dataset_dict = build_preprocessed_dataset_dict(
        train_examples=train_examples,
        validation_examples=validation_examples,
        output_dir=output_dir,
        train_pretokenized_blobs_by_sample_id=train_pretokenized_blobs_by_sample_id,
        validation_pretokenized_blobs_by_sample_id=(
            validation_pretokenized_blobs_by_sample_id
        ),
        image_relpaths_by_source=image_relpaths_by_source,
        artifact_image_size=artifact_image_size,
        progress_callback=_log,
    )
    preprocess_config = {
        "created_at": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "command": " ".join(sys.argv),
        "dataset_root": str(dataset_root),
        "train_tasks": train_tasks,
        "val_tasks": val_tasks,
        "validation_split_mode": args.validation_split_mode,
        "validation_trajectory_fraction": args.validation_trajectory_fraction,
        "validation_min_trajectories_per_task": (
            args.validation_min_trajectories_per_task
        ),
        "validation_split_seed": args.validation_split_seed,
        "train_trajectory_ids_by_task": train_trajectory_ids_by_task,
        "val_trajectory_ids_by_task": val_trajectory_ids_by_task,
        "train_example_format": "private_history_consume_once",
        "use_example_cache": args.use_example_cache,
        "training_samples_cache_dir": str(training_samples_cache_dir),
        "example_build_workers": args.example_build_workers,
        "preprocessed_format_version": _PREPROCESSED_FORMAT_VERSION,
        "trajectory_validator_contract_version": VALIDATOR_CONTRACT_VERSION,
        "prompt_contract_version": PROMPT_CONTRACT_VERSION,
        "artifact_image_size": args.artifact_image_size,
        "pretokenized": args.pretokenize,
        "train_reasoning": train_reasoning,
        "pretokenization": pretokenization_config,
        "pretokenize_batch_size": args.pretokenize_batch_size,
        "push_to_hub": args.push_to_hub,
        "hub_revision": args.hub_revision,
        "hub_private": args.hub_private,
        "hub_data_dir": args.hub_data_dir,
    }

    _log("Saving local preprocessed artifact")
    save_preprocessed_artifact(
        output_dir=output_dir,
        dataset_dict=dataset_dict,
        split_manifest=split_manifest,
        preprocess_config=preprocess_config,
        progress_callback=_log,
    )

    if args.push_to_hub:
        _log(
            f"Pushing preprocessed artifact to https://huggingface.co/datasets/{args.push_to_hub}"
        )
        push_preprocessed_artifact_to_hub(
            output_dir=output_dir,
            repo_id=args.push_to_hub,
            split_manifest=split_manifest,
            preprocess_config=preprocess_config,
            private=args.hub_private,
            revision=args.hub_revision,
            data_dir=args.hub_data_dir,
        )

    _log("Preprocessing complete")


if __name__ == "__main__":
    main()
