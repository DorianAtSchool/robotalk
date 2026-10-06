"""Dataset loading and multimodal collation for task-level VLM SFT."""

from __future__ import annotations

import gzip
import io
import json
import math
import os
import pickle
import random
import warnings
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

try:
    import torch
except ImportError:  # pragma: no cover - optional in lightweight test envs
    torch = None

try:
    from PIL import Image
except ImportError:  # pragma: no cover - optional in lightweight test envs
    Image = None

try:
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - optional in lightweight test envs

    class Dataset:  # type: ignore[no-redef]
        """Fallback base class so example builders can be imported without torch."""

        pass


try:
    from transformers import AutoProcessor
except ImportError:  # pragma: no cover - optional in lightweight test envs
    AutoProcessor = None

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional in lightweight test envs
    tqdm = None

try:
    import zstandard
except ImportError:  # pragma: no cover - optional fallback for old envs
    zstandard = None

from robotalk.tools.subatomic_tool_specs import (
    build_model_tool_specs,
    canonicalize_model_tool_args,
)

from robotalk.training.prompting import (
    build_messages,
    build_partial_user_prompt,
    strip_history_step_indices,
)
from robotalk.training.schema_utils import (
    augment_tool_specs_with_get_image,
    build_single_step_response_schema,
    compact_json_dumps,
    validate_single_step_payload,
)
from robotalk.training.task_registry import (
    AGENT_IDS,
    get_task_metadata,
    resolve_task_name,
)
from robotalk.training.tool_calling import (
    build_tool_schemas,
)

_STAGE_MANIFEST_FILENAME = "hf_stage_manifest.json"
_TRAJECTORY_METADATA_FILENAMES = (
    "original_trajectory.json",
    "plan.json",
    "metadata.json",
)
# Examples use one fixed contract: each agent sees only its own private
# history (its actions plus messages addressed to it), step indices are never
# shown, the model is trained to request observations with get_image, and a
# get_image result feeds only that agent's next target before being discarded
# ("consume once"), so every attached image is fresh by construction.
_EXAMPLE_CACHE_FORMAT_VERSION = 4
_EXAMPLE_CACHE_BINARY_FORMAT_VERSION = 1
_EXAMPLE_CACHE_ZSTD_SUFFIX = ".pkl.zst"
_EXAMPLE_CACHE_GZIP_SUFFIX = ".pkl.gz"
_EXAMPLE_CACHE_ZSTD_LEVEL = 1
_EXAMPLE_CACHE_GZIP_LEVEL = 1
_PRETOKENIZED_FIELD_NAME = "pretokenized_tensors"
# Keep cache serialization-only edits from invalidating example contents. Bump
# _EXAMPLE_CACHE_FORMAT_VERSION when dataset.py changes generated examples.
_EXAMPLE_CACHE_DEPENDENCY_PATHS = (
    Path(__file__).with_name("prompting.py").resolve(),
    Path(__file__).with_name("schema_utils.py").resolve(),
    Path(__file__).with_name("tool_calling.py").resolve(),
    Path(__file__).with_name("task_registry.py").resolve(),
)
_FALSE_ENV_VALUES = {"0", "false", "no", "off"}


def _missing_dependency_error(module_name: str) -> ImportError:
    return ImportError(
        f"Missing optional dependency {module_name!r}. "
        "Install the training dependencies first: pip install -e '.[train]'."
    )


def _require_dependency(dependency: Any, module_name: str) -> Any:
    if dependency is None:
        raise _missing_dependency_error(module_name)
    return dependency


@dataclass(frozen=True)
class AgentTurnExample:
    """One agent turn: the acting agent's private context and its next tool call."""

    sample_id: str
    task_name: str
    composite_task: str
    trajectory_id: str
    step_index: int
    agent_id: str
    task_instruction: str
    observation_views: list[str]
    image_paths: list[str]
    history_steps: list[dict[str, Any]]
    allowed_tool_specs: dict[str, dict[str, Any]]
    tool_schemas: list[dict[str, Any]]
    response_schema: dict[str, Any]
    target_payload: dict[str, Any]
    target_tool_call: dict[str, Any]
    target_text: str
    messages: list[dict[str, Any]]

    def to_manifest_entry(self) -> dict[str, Any]:
        """Returns the compact sample metadata persisted with the run."""

        return {
            "sample_id": self.sample_id,
            "task_name": self.task_name,
            "trajectory_id": self.trajectory_id,
            "step_index": self.step_index,
            "agent_id": self.agent_id,
            "num_images": len(self.image_paths),
            "num_supervised_actions": 1,
            "observation_views": list(self.observation_views),
        }

    def to_feature_dict(self) -> dict[str, Any]:
        """Builds the trainer-facing feature dict for one example."""

        return {
            "sample_id": self.sample_id,
            "task_name": self.task_name,
            "composite_task": self.composite_task,
            "trajectory_id": self.trajectory_id,
            "step_index": self.step_index,
            "agent_id": self.agent_id,
            "observation_views": list(self.observation_views),
            "image_paths": list(self.image_paths),
            "allowed_tool_specs": self.allowed_tool_specs,
            "tool_schemas": self.tool_schemas,
            "response_schema": self.response_schema,
            "target_payload": self.target_payload,
            "target_tool_call": self.target_tool_call,
            "target_text": self.target_text,
            "messages": self.messages,
        }


def estimate_example_length(
    example: AgentTurnExample,
    *,
    max_images_per_sample: int | None = None,
    estimated_visual_tokens_per_image: int = 256,
) -> int:
    """Cheap multimodal length proxy for padding-aware batch sampling.

    Exact processor tokenization would require rendering and tokenizing every
    example at startup.  JSON character count is a stable proxy for the text
    and tool-schema portion; the image term accounts for Qwen's merged visual
    tokens at the training resolution.  Only relative ordering is required by
    the length-grouped sampler.
    """

    text_characters = len(
        json.dumps(example.messages, ensure_ascii=False, separators=(",", ":"))
    ) + len(
        json.dumps(
            example.tool_schemas,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    image_count = len(example.image_paths)
    if max_images_per_sample is not None and max_images_per_sample > 0:
        image_count = min(image_count, max_images_per_sample)
    estimated_text_tokens = max(1, math.ceil(text_characters / 4))
    return estimated_text_tokens + image_count * estimated_visual_tokens_per_image




@dataclass(frozen=True)
class SameTaskTrajectorySplit:
    train_trajectory_ids_by_task: dict[str, list[str]]
    validation_trajectory_ids_by_task: dict[str, list[str]]


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_stage_manifest(dataset_root: Path) -> dict[str, Any] | None:
    manifest_path = dataset_root / _STAGE_MANIFEST_FILENAME
    if not manifest_path.is_file():
        return None
    manifest = _load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ValueError(f"Stage manifest must be a JSON object: {manifest_path}")
    return manifest


def _resolve_stage_path(raw_path: Any, *, dataset_root: Path) -> Path:
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    return (dataset_root / path).resolve()


def _stage_source_root(
    *,
    manifest: dict[str, Any],
    source: dict[str, Any],
    dataset_root: Path,
) -> Path:
    raw_root = (
        source.get("shard_output_root")
        or source.get("output_root")
        or manifest.get("output_root")
    )
    if raw_root is None:
        return dataset_root
    return _resolve_stage_path(raw_root, dataset_root=dataset_root)


def _selected_stage_source_trajectory_dirs(
    *,
    manifest: dict[str, Any],
    source: dict[str, Any],
    dataset_root: Path,
) -> list[tuple[str, Path]]:
    source_root = _stage_source_root(
        manifest=manifest,
        source=source,
        dataset_root=dataset_root,
    )
    trajectory_dirs: list[tuple[str, Path]] = []
    for raw_task_name in source.get("tasks", []):
        task_name = str(raw_task_name)
        task_root = source_root / task_name
        if not task_root.is_dir():
            raise FileNotFoundError(f"Missing staged task directory: {task_root}")
        trajectory_dirs.extend(
            (task_name, path) for path in sorted(task_root.iterdir()) if path.is_dir()
        )

    expected_count = int(source.get("num_episodes", len(trajectory_dirs)))
    if len(trajectory_dirs) < expected_count:
        repo_id = source.get("repo_id", "<unknown>")
        raise ValueError(
            f"{repo_id} manifest reports {expected_count} episodes "
            f"but only {len(trajectory_dirs)} trajectory dirs were found."
        )
    return trajectory_dirs[:expected_count]


def _manifest_trajectory_dirs_for_task(
    *,
    dataset_root: Path,
    task_name: str,
) -> list[Path] | None:
    manifest = _load_stage_manifest(dataset_root)
    if manifest is None:
        return None
    sources = manifest.get("sources")
    if not isinstance(sources, list):
        return None

    trajectory_dirs: list[Path] = []
    for source in sources:
        if not isinstance(source, dict):
            continue
        source_task_names = {str(raw_task) for raw_task in source.get("tasks", [])}
        if task_name not in source_task_names:
            continue
        trajectory_dirs.extend(
            path
            for source_task_name, path in _selected_stage_source_trajectory_dirs(
                manifest=manifest,
                source=source,
                dataset_root=dataset_root,
            )
            if source_task_name == task_name
        )
    return sorted(trajectory_dirs, key=lambda path: path.name)


def _trajectory_dirs_for_task(*, dataset_root: Path, task_name: str) -> list[Path]:
    task_metadata = get_task_metadata(task_name)
    normalized_task_name = task_metadata.dataset_name
    manifest_dirs = _manifest_trajectory_dirs_for_task(
        dataset_root=dataset_root,
        task_name=normalized_task_name,
    )
    if manifest_dirs is not None:
        return manifest_dirs

    task_root = dataset_root / normalized_task_name
    if not task_root.is_dir():
        raise FileNotFoundError(f"Task directory does not exist: {task_root}")
    return sorted(path for path in task_root.iterdir() if path.is_dir())


def list_available_task_names(dataset_root: Path) -> list[str]:
    """Returns supported task names available in a staged dataset root."""

    task_names: set[str] = set()
    manifest = _load_stage_manifest(dataset_root)
    sources = manifest.get("sources") if manifest is not None else None
    if isinstance(sources, list):
        for source in sources:
            if not isinstance(source, dict):
                continue
            for raw_task_name in source.get("tasks", []):
                try:
                    task_names.add(resolve_task_name(str(raw_task_name)))
                except ValueError:
                    continue

    if dataset_root.is_dir():
        for path in dataset_root.iterdir():
            if not path.is_dir():
                continue
            try:
                task_names.add(resolve_task_name(path.name))
            except ValueError:
                continue

    return sorted(task_names)


def _filter_trajectory_dirs(
    trajectory_dirs: list[Path],
    *,
    include_trajectory_ids: set[str] | None = None,
) -> list[Path]:
    if include_trajectory_ids is not None:
        return [path for path in trajectory_dirs if path.name in include_trajectory_ids]
    return trajectory_dirs


def _iter_trajectory_dirs(
    trajectory_dirs: list[Path],
    *,
    show_progress: bool,
    progress_description: str | None,
):
    if not show_progress or tqdm is None:
        return trajectory_dirs
    return tqdm(
        trajectory_dirs,
        desc=progress_description,
        total=len(trajectory_dirs),
        dynamic_ncols=True,
        leave=False,
        unit="traj",
    )


def list_task_trajectory_ids(*, dataset_root: Path, task_name: str) -> list[str]:
    """Returns sorted trajectory directory names for one dataset task."""

    trajectory_dirs = _trajectory_dirs_for_task(
        dataset_root=dataset_root,
        task_name=task_name,
    )
    return sorted(path.name for path in trajectory_dirs)


def task_split_seed(seed: int, task_name: str) -> int:
    """Derives one deterministic per-task RNG seed for trajectory holdout."""

    digest = sha256(f"{seed}:{task_name}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def select_held_out_trajectory_ids(
    *,
    dataset_root: Path,
    val_tasks: list[str],
    train_tasks: list[str],
    trajectories_per_task: int,
    trajectory_fraction: float,
    seed: int,
) -> dict[str, set[str]]:
    """Selects deterministic held-out validation trajectory ids per task.

    This is the single source of truth for the trajectory holdout: both
    training (main.py) and eval manifest construction must call it with the
    same (dataset_root, seed, fraction/count) so their splits provably match.
    """

    if trajectories_per_task <= 0 and trajectory_fraction <= 0.0:
        return {}

    train_task_set = set(train_tasks)
    selected_by_task: dict[str, set[str]] = {}
    for task_name in sorted(val_tasks):
        task_trajectory_ids = list_task_trajectory_ids(
            dataset_root=dataset_root,
            task_name=task_name,
        )
        max_selectable = len(task_trajectory_ids)
        if task_name in train_task_set:
            max_selectable = max(max_selectable - 1, 0)
        if trajectories_per_task > 0:
            selected_count = min(trajectories_per_task, max_selectable)
        else:
            requested_count = math.ceil(len(task_trajectory_ids) * trajectory_fraction)
            selected_count = min(max(requested_count, 1), max_selectable)
        if selected_count <= 0:
            continue

        sampler = random.Random(task_split_seed(seed, task_name))
        sampler.shuffle(task_trajectory_ids)
        selected_by_task[task_name] = set(task_trajectory_ids[:selected_count])

    return selected_by_task


def build_same_task_trajectory_split(
    *,
    dataset_root: Path,
    train_task_names: list[str],
    validation_task_names: list[str],
    validation_fraction: float,
    min_validation_trajectories_per_task: int,
    seed: int = 42,
) -> SameTaskTrajectorySplit:
    """Builds a reproducible, per-task shuffled same-task validation split."""

    if validation_fraction < 0.0 or validation_fraction > 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")
    if min_validation_trajectories_per_task < 0:
        raise ValueError("min_validation_trajectories_per_task must be non-negative.")

    train_task_set = set(train_task_names)
    validation_task_set = set(validation_task_names)
    all_task_names = sorted(train_task_set.union(validation_task_set))
    selected_validation_ids_by_task: dict[str, set[str]] = {}

    for task_name in all_task_names:
        trajectory_ids = list(
            list_task_trajectory_ids(
                dataset_root=dataset_root,
                task_name=task_name,
            )
        )
        if task_name not in validation_task_set:
            selected_validation_ids_by_task[task_name] = set()
            continue

        requested_count = math.ceil(len(trajectory_ids) * validation_fraction)
        if trajectory_ids:
            requested_count = max(
                requested_count,
                min_validation_trajectories_per_task,
            )
        max_validation_count = len(trajectory_ids)
        if task_name in train_task_set:
            max_validation_count = max(max_validation_count - 1, 0)
        selected_count = min(requested_count, max_validation_count)
        sampler = random.Random(task_split_seed(seed, task_name))
        sampler.shuffle(trajectory_ids)
        selected_validation_ids_by_task[task_name] = set(
            trajectory_ids[:selected_count] if selected_count > 0 else []
        )

    train_trajectory_ids_by_task: dict[str, list[str]] = {}
    for task_name in train_task_names:
        selected_validation_ids = selected_validation_ids_by_task.get(
            task_name,
            set(),
        )
        train_trajectory_ids_by_task[task_name] = [
            trajectory_id
            for trajectory_id in list_task_trajectory_ids(
                dataset_root=dataset_root,
                task_name=task_name,
            )
            if trajectory_id not in selected_validation_ids
        ]

    validation_trajectory_ids_by_task: dict[str, list[str]] = {}
    for task_name in validation_task_names:
        selected_validation_ids = selected_validation_ids_by_task.get(
            task_name,
            set(),
        )
        validation_trajectory_ids_by_task[task_name] = [
            trajectory_id
            for trajectory_id in list_task_trajectory_ids(
                dataset_root=dataset_root,
                task_name=task_name,
            )
            if trajectory_id in selected_validation_ids
        ]

    return SameTaskTrajectorySplit(
        train_trajectory_ids_by_task=train_trajectory_ids_by_task,
        validation_trajectory_ids_by_task=validation_trajectory_ids_by_task,
    )


def _normalize_history_step(step: dict[str, Any]) -> dict[str, Any]:
    return {
        "step": step["step"],
        "agent": step["agent"],
        "tool": step["tool"],
        "args": dict(step["args"]),
    }


def _find_latest_same_agent_observation(
    plan_steps: list[dict[str, Any]],
    *,
    before_index: int,
    agent_id: str,
) -> tuple[list[str], list[str]]:
    for step in reversed(plan_steps[:before_index]):
        if step.get("tool") != "get_image":
            continue
        metadata = step.get("metadata", {})
        if metadata.get("source_agent") != agent_id:
            continue
        args = step.get("args", {})
        image_paths = list(args.get("image_paths", ()))
        views = list(args.get("views", ()))
        if not image_paths:
            continue
        return image_paths, views
    raise ValueError(
        f"Missing preceding observation image for agent {agent_id!r} before step {before_index}."
    )


def _order_raw_steps_as_executed(
    *,
    task_name: str,
    trajectory_id: str,
    raw_steps: list[dict[str, Any]],
    plan_steps: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """``raw_steps`` permuted into the order the sim actually ran them.

    ``original_trajectory.json`` is always stored in file order, but the sweep
    runs the plan in concurrent-executor order, so
    ``plan.json`` and ``metadata.json`` come back permuted. Zipping the three
    positionally would pair each raw step with a different step's execution
    result and its images. ``plan.json`` carries the source index of every call,
    which is what makes the join possible; a file-order render is the identity
    permutation and takes the same path.

    Examples are then built in EXECUTED order, which is the order the model sees
    at evaluation time.
    """

    raw_by_step: dict[Any, dict[str, Any]] = {}
    for raw_step in raw_steps:
        step_index = raw_step.get("step")
        if step_index in raw_by_step:
            raise ValueError(
                f"Duplicate raw step index {step_index} in "
                f"{task_name}/{trajectory_id}."
            )
        raw_by_step[step_index] = raw_step

    ordered: list[dict[str, Any]] = []
    for position, plan_step in enumerate(plan_steps):
        source = plan_step.get("metadata", {}).get("step_index")
        if source not in raw_by_step:
            raise ValueError(
                f"Plan step {position} of {task_name}/{trajectory_id} names "
                f"source step {source!r}, which is not in the trajectory."
            )
        ordered.append(raw_by_step[source])

    if len(ordered) != len(raw_steps):
        raise ValueError(
            f"Plan for {task_name}/{trajectory_id} covers {len(ordered)} of "
            f"{len(raw_steps)} steps."
        )
    return ordered


def _validate_alignment(
    *,
    task_name: str,
    trajectory_id: str,
    raw_steps: list[dict[str, Any]],
    plan_steps: list[dict[str, Any]],
    executed_steps: list[dict[str, Any]],
) -> None:
    """Checks the three files agree, with ``raw_steps`` already in run order."""

    if not (len(raw_steps) == len(plan_steps) == len(executed_steps)):
        raise ValueError(
            f"Step count mismatch for {task_name}/{trajectory_id}: "
            f"raw={len(raw_steps)} plan={len(plan_steps)} executed={len(executed_steps)}."
        )
    source_indices = [
        plan_step.get("metadata", {}).get("step_index") for plan_step in plan_steps
    ]
    if sorted(source_indices, key=lambda v: (v is None, v)) != list(
        range(len(plan_steps))
    ):
        raise ValueError(
            f"Plan for {task_name}/{trajectory_id} is not a permutation of the "
            f"trajectory's steps: {source_indices}."
        )
    for index, (raw_step, plan_step, executed_step) in enumerate(
        zip(raw_steps, plan_steps, executed_steps, strict=True)
    ):
        source = source_indices[index]
        if raw_step.get("step") != source:
            raise ValueError(
                f"Raw/plan step mismatch in {task_name}/{trajectory_id} at "
                f"position {index}: raw={raw_step.get('step')} plan={source}."
            )
        executed_index = executed_step.get("step_index")
        if executed_index != index:
            raise ValueError(
                f"Unexpected executed step index in {task_name}/{trajectory_id}: "
                f"expected {index}, got {executed_index}."
            )
        # Written since the reorder landed; absent in older renders, where the
        # executed position and the source index are the same number anyway.
        executed_source = executed_step.get("source_step_index")
        if executed_source is not None and executed_source != source:
            raise ValueError(
                f"Plan/metadata source mismatch in {task_name}/{trajectory_id} "
                f"at position {index}: plan={source} metadata={executed_source}."
            )
        if raw_step.get("tool") != plan_step.get("tool"):
            raise ValueError(
                f"Tool mismatch at {task_name}/{trajectory_id} position {index} "
                f"(source step {source}): raw={raw_step.get('tool')} "
                f"plan={plan_step.get('tool')}."
            )


def _ensure_image_paths_exist(
    *,
    task_name: str,
    trajectory_id: str,
    image_paths: list[str],
) -> None:
    validate_paths = os.environ.get("ROBOCASA_VALIDATE_IMAGE_PATHS", "true")
    if validate_paths.strip().lower() in _FALSE_ENV_VALUES:
        return

    missing_images = [
        image_path for image_path in image_paths if not Path(image_path).exists()
    ]
    if not missing_images:
        return
    missing = ", ".join(missing_images)
    raise FileNotFoundError(
        f"Missing rendered images for {task_name}/{trajectory_id}: {missing}"
    )


def _canonicalize_image_paths(
    *,
    trajectory_dir: Path,
    trajectory_id: str,
    image_paths: list[str],
) -> list[str]:
    """Remaps image paths recorded at generation/staging time onto the
    trajectory's current on-disk location.

    Staged trajectories store absolute paths from the machine/directory they
    were staged on; after relocation (tar → node-local extraction) only
    ``<trajectory_dir>/images/<trajectory_id>/<filename>`` is valid.
    """

    canonical: list[str] = []
    for image_path in image_paths:
        recorded = Path(image_path)
        local = trajectory_dir / "images" / trajectory_id / recorded.name
        canonical.append(str(local) if local.exists() else image_path)
    return canonical


def _record_latest_observation(
    *,
    latest_observations_by_agent: dict[str, tuple[list[str], list[str]]],
    plan_step: dict[str, Any],
    task_name: str,
    trajectory_id: str,
    trajectory_dir: Path,
) -> None:
    if plan_step.get("tool") != "get_image":
        return

    metadata = plan_step.get("metadata", {})
    source_agent = metadata.get("source_agent")
    if source_agent is None:
        return

    args = plan_step.get("args", {})
    image_paths = list(args.get("image_paths", ()))
    views = list(args.get("views", ()))
    if not image_paths:
        return

    image_paths = _canonicalize_image_paths(
        trajectory_dir=trajectory_dir,
        trajectory_id=trajectory_id,
        image_paths=image_paths,
    )
    _ensure_image_paths_exist(
        task_name=task_name,
        trajectory_id=trajectory_id,
        image_paths=image_paths,
    )
    latest_observations_by_agent[source_agent] = (image_paths, views)


def _require_latest_observation(
    *,
    latest_observations_by_agent: dict[str, tuple[list[str], list[str]]],
    agent_id: str,
    before_index: int,
) -> tuple[list[str], list[str]]:
    try:
        return latest_observations_by_agent[agent_id]
    except KeyError as exc:
        raise ValueError(
            f"Missing preceding observation image for agent {agent_id!r} "
            f"before step {before_index}."
        ) from exc



def _raw_step_payload(
    raw_step: dict[str, Any],
) -> dict[str, Any]:
    step = {
        "step": raw_step["step"],
        "agent": raw_step["agent"],
        "tool": raw_step["tool"],
        "args": dict(raw_step["args"]),
    }
    return {"steps": [step]}




def _build_target_metadata(
    *,
    raw_step: dict[str, Any],
    allowed_tool_specs: dict[str, dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], str]:
    # The target is expert ground truth, so a tool it uses is valid by
    # definition. A few tasks' verified specs omit a tool the generated data
    # actually uses (e.g. prepare_cheese_station opens a cabinet with
    # open_hinged_part, absent from its allowed_tool_specs). Rather than reject
    # the demonstration, fall back to the raw-step target so example-building
    # never crashes on such spec/data mismatches.
    try:
        target_payload = validate_single_step_payload(
            _raw_step_payload(raw_step),
            agent_ids=AGENT_IDS,
            allowed_tool_specs=allowed_tool_specs,
        )
        target_tool_call = {
            "name": target_payload["steps"][0]["tool"],
            "arguments": dict(target_payload["steps"][0]["args"]),
        }
    except ValueError:
        target_payload = _raw_step_payload(raw_step)
        target_tool_call = {
            "name": raw_step["tool"],
            "arguments": dict(raw_step["args"]),
        }
    target_text = compact_json_dumps(target_tool_call)
    return target_payload, target_tool_call, target_text


def _build_agent_turn_examples_for_trajectory(
    *,
    task_name: str,
    trajectory_dir: Path,
    response_schema: dict[str, Any],
    tool_schemas: list[dict[str, Any]],
    model_tool_specs: dict[str, dict[str, Any]],
    train_reasoning: bool = False,
) -> list[AgentTurnExample]:
    task_metadata = get_task_metadata(task_name)
    allowed_tool_specs = augment_tool_specs_with_get_image(
        task_metadata.allowed_tool_specs
    )
    original_trajectory_path = trajectory_dir / "original_trajectory.json"
    original_trajectory = _load_json(original_trajectory_path)
    from robotalk.tasks.shared.validation_contract import (
        require_current_validation,
    )

    require_current_validation(original_trajectory, source=str(original_trajectory_path))
    plan_steps = _load_json(trajectory_dir / "plan.json")
    metadata = _load_json(trajectory_dir / "metadata.json")

    executed_steps = list(metadata["steps"])
    trajectory_id = str(original_trajectory["trajectory_id"])
    # The stored trajectory is in file order; the sim ran it in concurrent
    # order. Line the raw steps up with the run before anything zips them.
    raw_steps = _order_raw_steps_as_executed(
        task_name=task_name,
        trajectory_id=trajectory_id,
        raw_steps=list(original_trajectory["steps"]),
        plan_steps=plan_steps,
    )

    _validate_alignment(
        task_name=task_name,
        trajectory_id=trajectory_id,
        raw_steps=raw_steps,
        plan_steps=plan_steps,
        executed_steps=executed_steps,
    )

    examples: list[AgentTurnExample] = []
    private_history: dict[str, list[dict[str, Any]]] = {
        agent_id: [] for agent_id in AGENT_IDS
    }
    latest_observations_by_agent: dict[str, tuple[list[str], list[str]]] = {}
    # Unconsumed observation per agent (consume-once semantics).
    pending_observation_by_agent: dict[str, tuple[list[str], list[str]] | None] = {}
    tick_rows = original_trajectory.get("tick_rows")
    if not isinstance(tick_rows, list):
        raise ValueError(f"{original_trajectory_path} has no tick_rows (concurrent ticks)")
    tick_by_step: dict[int, int] = {}
    flat_index = 0
    for tick_index, row in enumerate(tick_rows):
        if not isinstance(row, dict):
            continue
        for agent_id in AGENT_IDS:
            action = row.get(agent_id)
            if isinstance(action, dict) and action.get("tool"):
                tick_by_step[flat_index] = tick_index
                flat_index += 1
    pending_tick_history: list[dict[str, Any]] = []
    current_tick: int | None = None

    def flush_tick_history() -> None:
        """Deliver one atomic tick only after all of its targets were built."""

        for normalized in pending_tick_history:
            owner = normalized["agent"]
            private_history[owner].append(normalized)
            if normalized["tool"] == "communicate":
                recipient = (normalized.get("args") or {}).get("to")
                if recipient and recipient != owner and recipient in private_history:
                    private_history[recipient].append(normalized)
        pending_tick_history.clear()

    for raw_step, plan_step, executed_step in zip(
        raw_steps,
        plan_steps,
        executed_steps,
        strict=True,
    ):
        raw_tick = tick_by_step.get(raw_step.get("step"))
        if raw_tick is None:
            raise ValueError(
                f"{original_trajectory_path}: step {raw_step.get('step')} is in no tick"
            )
        if raw_tick != current_tick:
            flush_tick_history()
            current_tick = raw_tick

        effective_raw_step = deepcopy(raw_step)
        effective_raw_step["args"] = canonicalize_model_tool_args(
            str(effective_raw_step.get("tool", "")),
            effective_raw_step.get("args") or {},
        )

        if not executed_step.get("success", False):
            continue

        pending = pending_observation_by_agent.get(raw_step["agent"])
        image_paths, observation_views = pending if pending is not None else ([], [])

        target_payload, target_tool_call, target_text = _build_target_metadata(
            raw_step=effective_raw_step,
            allowed_tool_specs=allowed_tool_specs,
        )
        agent_history = private_history[effective_raw_step["agent"]]
        user_prompt = build_partial_user_prompt(
            composite_task=task_metadata.composite_task,
            task_instruction=getattr(task_metadata, "task_goal", metadata["task"]),
            agent_id=effective_raw_step["agent"],
            observation_views=observation_views,
            history_steps=agent_history,
            allowed_tool_specs=model_tool_specs,
            coordinator_id=original_trajectory.get("coordinator_id"),
            initial_state=original_trajectory.get("initial_state"),
        )
        sample_id = (
            f"{task_metadata.dataset_name}/{trajectory_id}/"
            f"step_{raw_step['step']:06d}"
        )
        examples.append(
            AgentTurnExample(
                sample_id=sample_id,
                task_name=task_metadata.dataset_name,
                composite_task=task_metadata.composite_task,
                trajectory_id=trajectory_id,
                step_index=raw_step["step"],
                agent_id=effective_raw_step["agent"],
                task_instruction=metadata["task"],
                observation_views=observation_views,
                image_paths=list(image_paths),
                history_steps=strip_history_step_indices(agent_history),
                allowed_tool_specs=model_tool_specs,
                tool_schemas=tool_schemas,
                response_schema=response_schema,
                target_payload=target_payload,
                target_tool_call=target_tool_call,
                target_text=target_text,
                messages=build_messages(
                    user_prompt=user_prompt,
                    num_images=len(image_paths),
                    reasoning_text=(
                        effective_raw_step.get("reasoning")
                        if train_reasoning
                        else None
                    ),
                    target_tool_call=target_tool_call,
                ),
            )
        )
        pending_tick_history.append(_normalize_history_step(effective_raw_step))
        _record_latest_observation(
            latest_observations_by_agent=latest_observations_by_agent,
            plan_step=plan_step, task_name=task_name,
            trajectory_id=trajectory_id, trajectory_dir=trajectory_dir)
        owner = raw_step["agent"]
        if raw_step["tool"] == "get_image":
            pending_observation_by_agent[owner] = latest_observations_by_agent.get(
                owner
            )
        else:
            # Consumed by this decision; the agent must look again.
            pending_observation_by_agent[owner] = None

    flush_tick_history()
    return examples



def build_agent_turn_examples(
    *,
    dataset_root: Path,
    task_names: list[str],
    show_progress: bool = False,
    progress_description: str | None = None,
    trajectory_ids_by_task: dict[str, set[str]] | None = None,
    example_build_workers: int = 1,
    train_reasoning: bool = False,
) -> list[AgentTurnExample]:
    """Builds one supervised example per executed agent call.

    Each example's prompt is the acting agent's private context; the target
    is that agent's next tool call. With train_reasoning, each assistant turn
    is supervised with a `<think>{reasoning}</think>` prefix sourced from the
    trajectory's per-step rationale.
    """

    example_build_workers = max(example_build_workers, 1)
    examples: list[AgentTurnExample] = []

    for task_name in task_names:
        task_metadata = get_task_metadata(task_name)
        trajectory_dirs = _trajectory_dirs_for_task(
            dataset_root=dataset_root,
            task_name=task_metadata.dataset_name,
        )

        model_tool_specs = build_model_tool_specs(include_get_image=True)
        response_schema = build_single_step_response_schema(
            agent_ids=AGENT_IDS,
            allowed_tool_specs=model_tool_specs,
        )
        tool_schemas = build_tool_schemas(
            agent_ids=AGENT_IDS,
            allowed_tool_specs=model_tool_specs,
        )
        include_trajectory_ids = None
        if trajectory_ids_by_task is not None:
            include_trajectory_ids = trajectory_ids_by_task.get(
                task_metadata.dataset_name, set()
            )

        selected_trajectory_dirs = _filter_trajectory_dirs(
            trajectory_dirs,
            include_trajectory_ids=include_trajectory_ids,
        )

        def build_one(trajectory_dir: Path) -> list[AgentTurnExample]:
            return _build_agent_turn_examples_for_trajectory(
                task_name=task_metadata.dataset_name,
                trajectory_dir=trajectory_dir,
                response_schema=response_schema,
                tool_schemas=tool_schemas,
                model_tool_specs=model_tool_specs,
                train_reasoning=train_reasoning,
            )

        if example_build_workers > 1 and len(selected_trajectory_dirs) > 1:
            with ThreadPoolExecutor(max_workers=example_build_workers) as executor:
                task_results = executor.map(build_one, selected_trajectory_dirs)
                if show_progress and tqdm is not None:
                    task_results = tqdm(
                        task_results,
                        desc=progress_description or task_metadata.dataset_name,
                        total=len(selected_trajectory_dirs),
                        dynamic_ncols=True,
                        leave=False,
                        unit="traj",
                    )
                for trajectory_examples in task_results:
                    examples.extend(trajectory_examples)
            continue

        for trajectory_dir in _iter_trajectory_dirs(
            selected_trajectory_dirs,
            show_progress=show_progress,
            progress_description=progress_description or task_metadata.dataset_name,
        ):
            examples.extend(build_one(trajectory_dir))

    return examples


def _file_signature(path: Path) -> dict[str, int]:
    stat_result = path.stat()
    return {
        "mtime_ns": stat_result.st_mtime_ns,
        "size": stat_result.st_size,
    }


def build_example_cache_fingerprint(
    *,
    dataset_root: Path,
    task_name: str,
    trajectory_ids: list[str] | set[str] | None = None,
    train_reasoning: bool = False,
) -> dict[str, Any]:
    """Builds a fingerprint that invalidates cached task examples when inputs change."""

    task_metadata = get_task_metadata(task_name)
    selected_trajectory_ids = None if trajectory_ids is None else set(trajectory_ids)
    trajectory_dirs = _trajectory_dirs_for_task(
        dataset_root=dataset_root,
        task_name=task_metadata.dataset_name,
    )
    if selected_trajectory_ids is not None:
        trajectory_dirs = [
            path for path in trajectory_dirs if path.name in selected_trajectory_ids
        ]
    from robotalk.tasks.shared.validation_contract import (
        require_current_validation,
    )

    # Validate source records even when a serialized example cache is reused.
    # A cache is an optimization, never an alternate trust boundary.
    for trajectory_dir in trajectory_dirs:
        original_path = trajectory_dir / "original_trajectory.json"
        require_current_validation(
            _load_json(original_path), source=str(original_path)
        )
    fingerprint: dict[str, Any] = {
        "cache_format_version": _EXAMPLE_CACHE_FORMAT_VERSION,
        "dataset_root": str(dataset_root.resolve()),
        "task_name": task_metadata.dataset_name,
        "example_format": "private_history_consume_once",
        "agent_ids": list(AGENT_IDS),
        "task_metadata": {
            "dataset_name": task_metadata.dataset_name,
            "composite_task": task_metadata.composite_task,
            "allowed_tool_specs": task_metadata.allowed_tool_specs,
        },
        "builder_dependencies": {
            dependency_path.name: _file_signature(dependency_path)
            for dependency_path in _EXAMPLE_CACHE_DEPENDENCY_PATHS
        },
        "trajectories": [
            {
                "trajectory_id": trajectory_dir.name,
                "files": {
                    file_name: _file_signature(trajectory_dir / file_name)
                    for file_name in _TRAJECTORY_METADATA_FILENAMES
                },
            }
            for trajectory_dir in trajectory_dirs
        ],
    }
    if train_reasoning:
        fingerprint["train_reasoning"] = True
    return fingerprint


def build_example_cache_path(
    *,
    cache_dir: Path,
    dataset_root: Path,
    task_name: str,
) -> Path:
    """Returns the on-disk cache path for one task's example cache."""

    dataset_key = sha256(str(dataset_root.resolve()).encode("utf-8")).hexdigest()[:16]
    normalized_task_name = get_task_metadata(task_name).dataset_name
    safe_task_name = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in normalized_task_name
    )
    return cache_dir / dataset_key / "agent_turns" / f"{safe_task_name}.json"


def _example_cache_path_with_suffix(cache_path: Path, suffix: str) -> Path:
    """Returns the sibling cache path for a concrete serialization suffix."""

    if cache_path.suffix:
        return cache_path.with_suffix(suffix)
    return cache_path.with_name(f"{cache_path.name}{suffix}")


def _compressed_example_cache_candidate_paths(cache_path: Path) -> list[Path]:
    zstd_path = _example_cache_path_with_suffix(cache_path, _EXAMPLE_CACHE_ZSTD_SUFFIX)
    gzip_path = _example_cache_path_with_suffix(cache_path, _EXAMPLE_CACHE_GZIP_SUFFIX)
    if zstandard is not None:
        return [zstd_path, gzip_path]
    return [gzip_path, zstd_path]


def _preferred_compressed_example_cache_path(cache_path: Path) -> Path:
    suffix = (
        _EXAMPLE_CACHE_ZSTD_SUFFIX
        if zstandard is not None
        else _EXAMPLE_CACHE_GZIP_SUFFIX
    )
    return _example_cache_path_with_suffix(cache_path, suffix)


def _load_compressed_example_cache_payload(cache_path: Path) -> dict[str, Any] | None:
    if not cache_path.is_file():
        return None
    try:
        if cache_path.name.endswith(_EXAMPLE_CACHE_ZSTD_SUFFIX):
            if zstandard is None:
                return None
            with cache_path.open("rb") as raw_handle:
                with zstandard.ZstdDecompressor().stream_reader(raw_handle) as reader:
                    with io.BufferedReader(reader) as buffered_reader:
                        payload = pickle.load(buffered_reader)
        elif cache_path.name.endswith(_EXAMPLE_CACHE_GZIP_SUFFIX):
            with gzip.open(cache_path, "rb") as handle:
                payload = pickle.load(handle)
        else:
            return None
    except (
        AttributeError,
        EOFError,
        ImportError,
        OSError,
        pickle.PickleError,
        TypeError,
        ValueError,
    ):
        return None
    except Exception as exc:
        if zstandard is not None and isinstance(exc, zstandard.ZstdError):
            return None
        raise
    if not isinstance(payload, dict):
        return None
    if payload.get("binary_format_version") != _EXAMPLE_CACHE_BINARY_FORMAT_VERSION:
        return None
    return payload


def _save_compressed_example_cache_payload(
    *,
    cache_path: Path,
    fingerprint: dict[str, Any],
    examples: list[AgentTurnExample],
) -> Path:
    payload = {
        "cache_format_version": _EXAMPLE_CACHE_FORMAT_VERSION,
        "binary_format_version": _EXAMPLE_CACHE_BINARY_FORMAT_VERSION,
        "fingerprint": fingerprint,
        "examples": examples,
    }
    output_path = _preferred_compressed_example_cache_path(cache_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f"{output_path.name}.tmp")
    if output_path.name.endswith(_EXAMPLE_CACHE_ZSTD_SUFFIX):
        if zstandard is None:  # pragma: no cover - guarded by preferred path helper
            raise ImportError(
                "zstandard is required to write zstd-compressed example caches."
            )
        with temp_path.open("wb") as raw_handle:
            with zstandard.ZstdCompressor(
                level=_EXAMPLE_CACHE_ZSTD_LEVEL
            ).stream_writer(raw_handle) as handle:
                pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with gzip.open(
            temp_path,
            "wb",
            compresslevel=_EXAMPLE_CACHE_GZIP_LEVEL,
        ) as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temp_path.replace(output_path)
    return output_path


def _examples_from_cache_payload(
    payload: dict[str, Any],
    *,
    expected_fingerprint: dict[str, Any] | None,
) -> list[AgentTurnExample] | None:
    if payload.get("cache_format_version") != _EXAMPLE_CACHE_FORMAT_VERSION:
        return None
    if expected_fingerprint is not None and not _example_cache_fingerprints_match(
        payload.get("fingerprint"),
        expected_fingerprint,
    ):
        return None

    raw_examples = payload.get("examples")
    if not isinstance(raw_examples, list):
        return None

    if all(isinstance(raw_example, AgentTurnExample) for raw_example in raw_examples):
        return list(raw_examples)

    try:
        return [AgentTurnExample(**raw_example) for raw_example in raw_examples]
    except TypeError:
        return None


def _normalized_example_cache_fingerprint(
    fingerprint: dict[str, Any],
) -> dict[str, Any]:
    normalized = dict(fingerprint)
    builder_dependencies = normalized.get("builder_dependencies")
    if isinstance(builder_dependencies, dict):
        normalized_dependencies = dict(builder_dependencies)
        normalized_dependencies.pop(Path(__file__).name, None)
        normalized["builder_dependencies"] = normalized_dependencies
    return normalized


def _example_cache_fingerprints_match(
    actual_fingerprint: Any,
    expected_fingerprint: dict[str, Any],
) -> bool:
    if actual_fingerprint == expected_fingerprint:
        return True
    if not isinstance(actual_fingerprint, dict):
        return False
    return _normalized_example_cache_fingerprint(
        actual_fingerprint
    ) == _normalized_example_cache_fingerprint(expected_fingerprint)


def _filter_cached_examples(
    examples: list[AgentTurnExample],
    sample_id_filter: Any | None,
) -> list[AgentTurnExample]:
    if sample_id_filter is None:
        return examples
    return [example for example in examples if sample_id_filter(example.sample_id)]


def load_examples_from_cache(
    *,
    cache_path: Path,
    expected_fingerprint: dict[str, Any] | None,
    sample_id_filter: Any | None = None,
) -> list[AgentTurnExample] | None:
    """Loads cached task examples, optionally requiring a fingerprint match."""

    for compressed_cache_path in _compressed_example_cache_candidate_paths(cache_path):
        payload = _load_compressed_example_cache_payload(compressed_cache_path)
        if payload is None:
            continue
        examples = _examples_from_cache_payload(
            payload,
            expected_fingerprint=expected_fingerprint,
        )
        if examples is not None:
            return _filter_cached_examples(examples, sample_id_filter)

    try:
        payload = _load_json(cache_path)
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None

    examples = _examples_from_cache_payload(
        payload,
        expected_fingerprint=expected_fingerprint,
    )
    if examples is None:
        return None

    preferred_cache_path = _preferred_compressed_example_cache_path(cache_path)
    if not preferred_cache_path.exists() and isinstance(
        payload.get("fingerprint"),
        dict,
    ):
        try:
            _save_compressed_example_cache_payload(
                cache_path=cache_path,
                fingerprint=payload["fingerprint"],
                examples=examples,
            )
        except (OSError, ImportError):
            pass

    return _filter_cached_examples(examples, sample_id_filter)


def save_examples_to_cache(
    *,
    cache_path: Path,
    fingerprint: dict[str, Any],
    examples: list[AgentTurnExample],
) -> None:
    """Persists task examples to the fastest available compressed binary cache."""

    _save_compressed_example_cache_payload(
        cache_path=cache_path,
        fingerprint=fingerprint,
        examples=examples,
    )


class AgentTurnDataset(Dataset):
    """Thin PyTorch dataset wrapper around agent-turn training examples."""

    def __init__(self, examples: list[AgentTurnExample]) -> None:
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.examples[index].to_feature_dict()


def _num_supervised_actions(example: AgentTurnExample) -> int:
    return 1


def build_split_manifest(
    *,
    dataset_root: Path,
    train_examples: list[AgentTurnExample],
    val_examples: list[AgentTurnExample],
) -> dict[str, Any]:
    """Builds a compact manifest describing the train/validation split."""

    def summarize_split(examples: list[AgentTurnExample]) -> dict[str, Any]:
        counts_by_task: dict[str, dict[str, Any]] = {}
        for example in examples:
            entry = counts_by_task.setdefault(
                example.task_name,
                {
                    "num_samples": 0,
                    "num_supervised_actions": 0,
                    "trajectory_ids": set(),
                },
            )
            entry["num_samples"] += 1
            entry["num_supervised_actions"] += _num_supervised_actions(example)
            entry["trajectory_ids"].add(example.trajectory_id)

        return {
            "num_samples": len(examples),
            "num_supervised_actions": sum(
                _num_supervised_actions(example) for example in examples
            ),
            "tasks": {
                task_name: {
                    "num_samples": task_summary["num_samples"],
                    "num_supervised_actions": task_summary["num_supervised_actions"],
                    "num_trajectories": len(task_summary["trajectory_ids"]),
                }
                for task_name, task_summary in sorted(counts_by_task.items())
            },
            "samples": [example.to_manifest_entry() for example in examples],
        }

    return {
        "dataset_root": str(dataset_root.resolve()),
        "train": summarize_split(train_examples),
        "validation": summarize_split(val_examples),
    }


def image_resolution_to_pixels(image_resolution: int | None) -> int | None:
    """Converts a square image-resolution setting to a processor pixel budget."""

    if image_resolution is None:
        return None
    if image_resolution < 1:
        raise ValueError("image_resolution must be at least 1.")
    return image_resolution * image_resolution


def serialize_pretokenized_tensors(tensors: dict[str, Any]) -> bytes:
    """Serializes one pretokenized tensor payload for Arrow storage."""

    return pickle.dumps(tensors, protocol=pickle.HIGHEST_PROTOCOL)


def deserialize_pretokenized_tensors(blob: bytes) -> dict[str, Any]:
    """Deserializes one pretokenized tensor payload from Arrow storage."""

    value = pickle.loads(blob)
    if not isinstance(value, dict):
        raise TypeError("Pretokenized tensor payload must deserialize to a dict.")
    if torch is not None:
        value = {
            key: torch.tensor(item) if isinstance(item, list) else item
            for key, item in value.items()
        }
    return value


def _get_size_edge(size, edge_name: str):
    """Reads one edge from a processor size, which may be a dict or a
    SizeDict-style attribute object."""

    if isinstance(size, dict):
        return size.get(edge_name)
    return getattr(size, edge_name, None)


def _set_size_edge(size, edge_name: str, value) -> None:
    if isinstance(size, dict):
        size[edge_name] = value
    else:
        setattr(size, edge_name, value)


def _set_processor_image_resolution(processor, image_resolution: int | None):
    image_pixels = image_resolution_to_pixels(image_resolution)
    if image_pixels is None:
        return None

    image_processor = getattr(processor, "image_processor", None)
    if image_processor is None:
        return None

    # Pin EVERY pixel-area knob the processor exposes to exactly image_pixels so
    # the effective resolution is identical regardless of which family the
    # installed transformers version consults (min/max_pixels vs
    # size.shortest_edge/longest_edge), matching the historical exact-area
    # behavior. Both edges go to image_pixels (not min(...)) so the two
    # families cannot disagree.
    previous_values = {}
    for attribute_name in ("min_pixels", "max_pixels"):
        if getattr(image_processor, attribute_name, None) is not None:
            previous_values[attribute_name] = getattr(image_processor, attribute_name)
            setattr(image_processor, attribute_name, image_pixels)

    size = getattr(image_processor, "size", None)
    if size is not None:
        for edge_name in ("shortest_edge", "longest_edge"):
            if _get_size_edge(size, edge_name) is not None:
                previous_values[f"size.{edge_name}"] = _get_size_edge(size, edge_name)
                _set_size_edge(size, edge_name, image_pixels)

    if not previous_values:
        # Fixed-resolution processors (e.g. size={'height','width'}) have no
        # pixel-area budget to tune. Warn and leave the processor untouched
        # rather than crashing mid-run inside a DataLoader worker; the model
        # runs at its native resolution.
        warnings.warn(
            f"image_resolution={image_resolution} was requested but "
            f"{type(image_processor).__name__} exposes no pixel-area knob "
            "(min/max_pixels or size.*_edge); leaving native resolution.",
            RuntimeWarning,
            stacklevel=2,
        )
        return None
    return image_processor, previous_values


def _restore_processor_image_resolution(state) -> None:
    if state is None:
        return
    image_processor, previous_values = state
    for attribute_name, value in previous_values.items():
        if attribute_name.startswith("size."):
            _set_size_edge(
                image_processor.size, attribute_name.removeprefix("size."), value
            )
        else:
            setattr(image_processor, attribute_name, value)


def build_batched_pretokenized_tensors(
    *,
    processor,
    features: list[dict[str, Any]],
    max_length: int | None,
    image_resolution: int | None,
) -> list[dict[str, Any]]:
    """Builds per-example tensor payloads using the same masking as training."""

    collator = LazyVisionSFTCollator(
        processor_name_or_path="",
        max_length=max_length,
        trust_remote_code=False,
    )
    collator._processor = processor
    resolution_state = _set_processor_image_resolution(processor, image_resolution)
    try:
        batch = collator(features)
    finally:
        _restore_processor_image_resolution(resolution_state)

    batch_size = len(features)
    tensor_payloads: list[dict[str, Any]] = []
    for row_index in range(batch_size):
        row_payload: dict[str, Any] = {}
        for key, value in batch.items():
            if hasattr(value, "__getitem__") and hasattr(value, "shape"):
                row_payload[key] = value[row_index].detach().cpu()
            elif isinstance(value, list):
                row_payload[key] = value[row_index]
            else:
                row_payload[key] = value
        tensor_payloads.append(row_payload)
    return tensor_payloads


class LazyVisionSFTCollator:
    """Collates multimodal SFT batches and masks loss to assistant tokens only."""

    def __init__(
        self,
        *,
        processor_name_or_path: str,
        max_length: int | None,
        max_images_per_sample: int | None = None,
        supervise_last_assistant_turn_only: bool = False,
        trust_remote_code: bool,
        image_resolution: int | None = None,
    ) -> None:
        self.processor_name_or_path = processor_name_or_path
        self.max_length = max_length
        self.max_images_per_sample = (
            max_images_per_sample
            if max_images_per_sample and max_images_per_sample > 0
            else None
        )
        self.supervise_last_assistant_turn_only = supervise_last_assistant_turn_only
        self.trust_remote_code = trust_remote_code
        self.image_resolution = image_resolution
        self._processor = None

    def _get_processor(self):
        processor_cls = _require_dependency(AutoProcessor, "transformers")
        if self._processor is None:
            processor = processor_cls.from_pretrained(
                self.processor_name_or_path,
                trust_remote_code=self.trust_remote_code,
            )
            tokenizer = getattr(processor, "tokenizer", None)
            if tokenizer is not None:
                tokenizer.padding_side = "right"
                tokenizer.truncation_side = "left"
            if self.image_resolution and self.image_resolution > 0:
                _set_processor_image_resolution(processor, self.image_resolution)
            self._processor = processor
        return self._processor

    @staticmethod
    def _load_images(image_paths: list[str]) -> list[Any]:
        image_module = _require_dependency(Image, "Pillow")
        images: list[Any] = []
        for image_path in image_paths:
            with image_module.open(image_path) as image:
                images.append(image.convert("RGB"))
        return images

    @staticmethod
    def _message_text_for_plain_fallback(
        template_owner,
        message: dict[str, Any],
    ) -> str:
        tokenizer = getattr(template_owner, "tokenizer", None)
        image_token = (
            getattr(template_owner, "image_token", None)
            or getattr(tokenizer, "image_token", None)
            or "<|image|>"
        )
        audio_token = (
            getattr(template_owner, "audio_token", None)
            or getattr(tokenizer, "audio_token", None)
            or "<|audio|>"
        )

        content = message.get("content", "")
        if isinstance(content, str):
            return content.strip()
        if not isinstance(content, list):
            return str(content).strip()

        chunks: list[str] = []
        for item in content:
            if not isinstance(item, dict):
                continue
            item_type = item.get("type")
            if item_type == "text":
                chunks.append(str(item.get("text", "")).strip())
            elif item_type == "image":
                chunks.append(image_token)
            elif item_type == "audio":
                chunks.append(audio_token)
        return "".join(chunks).strip()

    def _apply_chat_template(
        self,
        template_owner,
        *,
        feature: dict[str, Any],
        messages: list[dict[str, Any]],
        **kwargs,
    ):
        template_kwargs = dict(kwargs)
        template_kwargs["tools"] = feature["tool_schemas"]
        return template_owner.apply_chat_template(messages, **template_kwargs)

    def _tokenize_texts(
        self,
        processor,
        *,
        texts: list[str],
        images_by_sample: list[list[Any]],
    ):
        processor_kwargs: dict[str, Any] = {
            "text": texts,
            "padding": True,
            "return_tensors": "pt",
        }
        if any(images_by_sample):
            processor_kwargs["images"] = images_by_sample
        if self.max_length is not None:
            processor_kwargs["max_length"] = self.max_length
            processor_kwargs["truncation"] = True

        batch = processor(**processor_kwargs)
        batch.pop("token_type_ids", None)
        return batch

    @staticmethod
    def _count_images_in_message(message: dict[str, Any]) -> int:
        content = message.get("content")
        if not isinstance(content, list):
            return 0
        return sum(
            1
            for item in content
            if isinstance(item, dict) and item.get("type") == "image"
        )

    def _trim_feature_to_image_budget(self, feature: dict[str, Any]) -> dict[str, Any]:
        if self.max_images_per_sample is None:
            return feature

        image_paths = list(feature["image_paths"])
        if len(image_paths) <= self.max_images_per_sample:
            return feature

        messages = list(feature["messages"])
        leading_message_count = 0
        while (
            leading_message_count < len(messages)
            and messages[leading_message_count].get("role") in {"system", "developer"}
            and self._count_images_in_message(messages[leading_message_count]) == 0
        ):
            leading_message_count += 1

        selected_start = len(messages)
        selected_image_count = 0
        for message_index in range(len(messages) - 1, leading_message_count - 1, -1):
            message_image_count = self._count_images_in_message(messages[message_index])
            if (
                message_image_count > 0
                and selected_image_count + message_image_count
                > self.max_images_per_sample
            ):
                break
            selected_image_count += message_image_count
            selected_start = message_index

        while (
            selected_start < len(messages)
            and messages[selected_start].get("role") == "assistant"
        ):
            selected_start += 1

        selected_image_count = sum(
            self._count_images_in_message(message)
            for message in messages[selected_start:]
        )
        if selected_start >= len(messages) or selected_image_count == 0:
            return feature

        dropped_image_count = sum(
            self._count_images_in_message(message)
            for message in messages[:selected_start]
        )
        selected_image_paths = image_paths[
            dropped_image_count : dropped_image_count + selected_image_count
        ]
        if len(selected_image_paths) != selected_image_count:
            return feature

        trimmed_feature = dict(feature)
        trimmed_feature["messages"] = (
            messages[:leading_message_count] + messages[selected_start:]
        )
        trimmed_feature["image_paths"] = selected_image_paths
        return trimmed_feature

    def _trim_feature_to_last_assistant_turn(
        self,
        feature: dict[str, Any],
    ) -> dict[str, Any]:
        if not self.supervise_last_assistant_turn_only:
            return feature

        messages = list(feature["messages"])
        assistant_indices = [
            index
            for index, message in enumerate(messages)
            if message.get("role") == "assistant"
        ]
        if len(assistant_indices) <= 1:
            return feature

        final_assistant_index = assistant_indices[-1]
        user_index = None
        for index in range(final_assistant_index - 1, -1, -1):
            if messages[index].get("role") == "user":
                user_index = index
                break
        if user_index is None:
            return feature

        leading_messages = [
            message
            for message in messages[:user_index]
            if message.get("role") in {"system", "developer"}
            and self._count_images_in_message(message) == 0
        ]
        kept_messages = (
            leading_messages + messages[user_index : final_assistant_index + 1]
        )

        image_paths = list(feature["image_paths"])
        dropped_image_count = sum(
            self._count_images_in_message(message) for message in messages[:user_index]
        )
        kept_image_count = sum(
            self._count_images_in_message(message) for message in kept_messages
        )
        kept_image_paths = image_paths[
            dropped_image_count : dropped_image_count + kept_image_count
        ]
        if len(kept_image_paths) != kept_image_count:
            return feature

        trimmed_feature = dict(feature)
        trimmed_feature["messages"] = kept_messages
        trimmed_feature["image_paths"] = kept_image_paths
        return trimmed_feature

    @staticmethod
    def _has_single_final_assistant_message(messages: list[dict[str, Any]]) -> bool:
        assistant_indices = [
            index
            for index, message in enumerate(messages)
            if message.get("role") == "assistant"
        ]
        return len(assistant_indices) == 1 and assistant_indices[0] == len(messages) - 1

    @staticmethod
    def _normalize_template_mask(mask: Any) -> list[bool] | None:
        if mask is None:
            return None
        if hasattr(mask, "tolist"):
            mask = mask.tolist()
        if isinstance(mask, list) and mask and isinstance(mask[0], list):
            mask = mask[0]
        if not isinstance(mask, list):
            return None
        return [bool(value) for value in mask]

    def _build_assistant_mask_from_template(
        self,
        *,
        processor,
        feature: dict[str, Any],
        full_sequence_length: int,
        sequence_width: int,
    ):
        torch_module = _require_dependency(torch, "torch")
        template_owners = (processor, getattr(processor, "tokenizer", None))
        for template_owner in template_owners:
            if template_owner is None or not hasattr(
                template_owner, "apply_chat_template"
            ):
                continue

            try:
                template_output = self._apply_chat_template(
                    template_owner,
                    feature=feature,
                    messages=feature["messages"],
                    tokenize=True,
                    add_generation_prompt=False,
                    return_dict=True,
                    return_assistant_tokens_mask=True,
                )
            except TypeError:
                continue
            except Exception:
                continue

            candidate_mask = None
            for key in (
                "assistant_masks",
                "assistant_mask",
                "assistant_tokens_mask",
                "assistant_token_mask",
            ):
                try:
                    candidate_mask = template_output[key]
                except Exception:
                    continue
                if candidate_mask is not None:
                    break

            normalized_mask = self._normalize_template_mask(candidate_mask)
            if normalized_mask is None or len(normalized_mask) != full_sequence_length:
                continue

            assistant_mask = torch_module.zeros(sequence_width, dtype=torch_module.bool)
            assistant_mask[:full_sequence_length] = torch_module.tensor(
                normalized_mask,
                dtype=torch_module.bool,
            )
            return assistant_mask
        return None

    def _build_assistant_mask_from_prefixes(
        self,
        *,
        processor,
        feature: dict[str, Any],
        images: list[Any],
        full_sequence_length: int,
        sequence_width: int,
    ):
        torch_module = _require_dependency(torch, "torch")
        assistant_mask = torch_module.zeros(sequence_width, dtype=torch_module.bool)

        prefix_messages: list[dict[str, Any]] = []
        prefix_image_count = 0
        previous_length = 0
        for message in feature["messages"]:
            prefix_messages.append(message)
            prefix_image_count += self._count_images_in_message(message)
            prefix_text = self._apply_chat_template(
                processor,
                feature=feature,
                messages=prefix_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            prefix_batch = self._tokenize_texts(
                processor,
                texts=[prefix_text],
                images_by_sample=[images[:prefix_image_count]],
            )
            current_length = min(
                full_sequence_length,
                int(prefix_batch["attention_mask"][0].sum().item()),
            )
            if message.get("role") == "assistant" and current_length > previous_length:
                assistant_mask[previous_length:current_length] = True
            previous_length = current_length
            if previous_length >= full_sequence_length:
                break

        return assistant_mask

    def _build_assistant_mask(
        self,
        *,
        processor,
        feature: dict[str, Any],
        images: list[Any],
        full_sequence_length: int,
        sequence_width: int,
    ):
        assistant_mask = self._build_assistant_mask_from_template(
            processor=processor,
            feature=feature,
            full_sequence_length=full_sequence_length,
            sequence_width=sequence_width,
        )
        if assistant_mask is not None:
            return assistant_mask

        return self._build_assistant_mask_from_prefixes(
            processor=processor,
            feature=feature,
            images=images,
            full_sequence_length=full_sequence_length,
            sequence_width=sequence_width,
        )

    @staticmethod
    def _raise_if_any_empty_labels(labels, features: list[dict[str, Any]]) -> None:
        supervised_counts = (labels != -100).sum(dim=1).tolist()
        empty_sample_ids = [
            str(features[row_index].get("sample_id", row_index))
            for row_index, count in enumerate(supervised_counts)
            if int(count) == 0
        ]
        if empty_sample_ids:
            sample_text = ", ".join(empty_sample_ids[:8])
            raise ValueError(
                "Collator produced no supervised assistant tokens after "
                f"truncation for samples: {sample_text}. Reduce "
                "--max-images-per-sample or increase --max-length."
            )

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        _require_dependency(torch, "torch")
        processor = self._get_processor()
        features = [self._trim_feature_to_image_budget(feature) for feature in features]
        features = [
            self._trim_feature_to_last_assistant_turn(feature) for feature in features
        ]
        messages = [feature["messages"] for feature in features]
        images = [self._load_images(feature["image_paths"]) for feature in features]

        full_texts = [
            self._apply_chat_template(
                processor,
                feature=feature,
                messages=message,
                tokenize=False,
                add_generation_prompt=False,
            )
            for message, feature in zip(messages, features, strict=True)
        ]
        batch = self._tokenize_texts(
            processor,
            texts=full_texts,
            images_by_sample=images,
        )

        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100

        if all(
            self._has_single_final_assistant_message(feature["messages"])
            for feature in features
        ):
            prompt_messages = [message[:-1] for message in messages]
            prompt_texts = [
                self._apply_chat_template(
                    processor,
                    feature=feature,
                    messages=message,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                for message, feature in zip(prompt_messages, features, strict=True)
            ]
            prompt_batch = self._tokenize_texts(
                processor,
                texts=prompt_texts,
                images_by_sample=images,
            )
            prompt_lengths = prompt_batch["attention_mask"].sum(dim=1).tolist()
            for row_index, prompt_length in enumerate(prompt_lengths):
                labels[row_index, :prompt_length] = -100
            self._raise_if_any_empty_labels(labels, features)
            batch["labels"] = labels
            return batch

        sequence_width = batch["input_ids"].shape[1]
        for row_index, feature in enumerate(features):
            full_sequence_length = int(batch["attention_mask"][row_index].sum().item())
            assistant_mask = self._build_assistant_mask(
                processor=processor,
                feature=feature,
                images=images[row_index],
                full_sequence_length=full_sequence_length,
                sequence_width=sequence_width,
            )
            valid_positions = assistant_mask & batch["attention_mask"][row_index].bool()
            labels[row_index, ~valid_positions] = -100

        self._raise_if_any_empty_labels(labels, features)
        batch["labels"] = labels
        return batch
