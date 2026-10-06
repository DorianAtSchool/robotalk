"""Portable preprocessed dataset artifacts for task-level BC VLM training."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import importlib
import json
from hashlib import sha256
import os
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import time
from typing import Any, Callable

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional in lightweight test envs
    tqdm = None

from robotalk.training.dataset import (
    AgentTurnExample,
    _PRETOKENIZED_FIELD_NAME,
    deserialize_pretokenized_tensors,
)
from robotalk.training.schema_utils import compact_json_dumps

_PREPROCESSED_DATASET_DIRNAME = "dataset"
_PREPROCESSED_IMAGES_DIRNAME = "images"
_PREPROCESSED_MANIFEST_FILENAME = "manifest.json"
_PREPROCESSED_CONFIG_FILENAME = "preprocess_config.json"
_PREPROCESSED_README_FILENAME = "README.md"
_PRETOKENIZED_BLOB_COLUMN = "pretokenized_blob"
_PRETOKENIZED_SHARD_MANIFEST_PREFIX = "manifest"
_PRETOKENIZED_SHARD_TRAIN_PREFIX = "train"
_PRETOKENIZED_SHARD_VALIDATION_PREFIX = "validation"
_PREPROCESSED_FORMAT_VERSION = 6
_ARTIFACT_IMAGE_PROGRESS_INTERVAL = 1000
_ROW_SERIALIZATION_PROGRESS_INTERVAL = 5000
_MAX_COPY_WORKERS = max(1, min(16, os.cpu_count() or 1))
_HF_DATASET_CLASS: Any | None = None
_HF_DATASET_DICT_CLASS: Any | None = None
_HF_FEATURES_CLASS: Any | None = None
_HF_VALUE_CLASS: Any | None = None
_HF_LOAD_FROM_DISK: Any | None = None
_HF_API_CLASS: Any | None = None
_HF_SNAPSHOT_DOWNLOAD: Any | None = None
ProgressCallback = Callable[[str], None]


class Dataset:
    """Lightweight dataset base so preprocessing can import this module cheaply."""

    pass


def _missing_dependency_error(module_name: str) -> ImportError:
    return ImportError(
        f"Missing optional dependency {module_name!r}. "
        "Install the training dependencies first: pip install -e '.[train]'."
    )


def _require_dependency(dependency: Any, module_name: str) -> Any:
    if dependency is None:
        raise _missing_dependency_error(module_name)
    return dependency


def _import_optional_module(module_name: str, dependency_name: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise _missing_dependency_error(dependency_name) from exc


def _load_datasets_symbols() -> None:
    global _HF_DATASET_CLASS
    global _HF_DATASET_DICT_CLASS
    global _HF_FEATURES_CLASS
    global _HF_VALUE_CLASS
    global _HF_LOAD_FROM_DISK
    if (
        _HF_DATASET_CLASS is not None
        and _HF_DATASET_DICT_CLASS is not None
        and _HF_FEATURES_CLASS is not None
        and _HF_VALUE_CLASS is not None
        and _HF_LOAD_FROM_DISK is not None
    ):
        return
    datasets_module = _import_optional_module("datasets", "datasets")
    _HF_DATASET_CLASS = _require_dependency(
        getattr(datasets_module, "Dataset", None),
        "datasets",
    )
    _HF_DATASET_DICT_CLASS = _require_dependency(
        getattr(datasets_module, "DatasetDict", None),
        "datasets",
    )
    _HF_FEATURES_CLASS = _require_dependency(
        getattr(datasets_module, "Features", None),
        "datasets",
    )
    _HF_VALUE_CLASS = _require_dependency(
        getattr(datasets_module, "Value", None),
        "datasets",
    )
    _HF_LOAD_FROM_DISK = _require_dependency(
        getattr(datasets_module, "load_from_disk", None),
        "datasets",
    )


def _get_hf_dataset_class() -> Any:
    _load_datasets_symbols()
    return _HF_DATASET_CLASS


def _get_hf_dataset_dict_class() -> Any:
    _load_datasets_symbols()
    return _HF_DATASET_DICT_CLASS


def _get_hf_features_class() -> Any:
    _load_datasets_symbols()
    return _HF_FEATURES_CLASS


def _get_hf_value_class() -> Any:
    _load_datasets_symbols()
    return _HF_VALUE_CLASS


def _get_load_from_disk() -> Any:
    _load_datasets_symbols()
    return _HF_LOAD_FROM_DISK


def _load_hf_hub_symbols() -> None:
    global _HF_API_CLASS
    global _HF_SNAPSHOT_DOWNLOAD
    if _HF_API_CLASS is not None and _HF_SNAPSHOT_DOWNLOAD is not None:
        return
    hub_module = _import_optional_module("huggingface_hub", "huggingface_hub")
    _HF_API_CLASS = _require_dependency(
        getattr(hub_module, "HfApi", None),
        "huggingface_hub",
    )
    _HF_SNAPSHOT_DOWNLOAD = _require_dependency(
        getattr(hub_module, "snapshot_download", None),
        "huggingface_hub",
    )


def _get_hf_api_class() -> Any:
    _load_hf_hub_symbols()
    return _HF_API_CLASS


def _get_snapshot_download() -> Any:
    _load_hf_hub_symbols()
    return _HF_SNAPSHOT_DOWNLOAD


@dataclass(frozen=True)
class LoadedPreprocessedArtifact:
    train_split: Any
    validation_split: Any
    split_manifest: dict[str, Any]
    preprocess_config: dict[str, Any] | None
    artifact_root: Path
    source_label: str


@dataclass(frozen=True)
class _ArtifactImageEntry:
    source_key: str
    source_path: Path
    relpath: str
    destination: Path


@dataclass(frozen=True)
class _StagedArtifactImages:
    image_relpaths_by_source: dict[str, str]
    num_hardlinked: int
    num_copied: int


@dataclass(frozen=True)
class PretokenizedShard:
    shard_index: int
    num_shards: int
    pretokenization: dict[str, Any]
    train_blobs_by_sample_id: dict[str, bytes]
    validation_blobs_by_sample_id: dict[str, bytes]
    manifest: dict[str, Any]


def _num_supervised_actions(example: AgentTurnExample) -> int:
    return 1


def _dataset_dir(output_dir: Path) -> Path:
    return output_dir / _PREPROCESSED_DATASET_DIRNAME


def _images_dir(output_dir: Path) -> Path:
    return output_dir / _PREPROCESSED_IMAGES_DIRNAME


def _sidecar_path(output_dir: Path, filename: str) -> Path:
    return output_dir / filename


def _repo_sidecar_path(filename: str, data_dir: str | None) -> str:
    if not data_dir:
        return filename
    return str(PurePosixPath(data_dir) / filename)


def _emit_progress(
    progress_callback: ProgressCallback | None,
    message: str,
) -> None:
    if progress_callback is not None:
        progress_callback(message)


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


def _pretokenization_processor_family(processor) -> str | None:
    class_name = processor.__class__.__name__.lower()
    module_name = processor.__class__.__module__.lower()
    if "qwen" in class_name or "qwen" in module_name:
        return "qwen"
    return None


def _ensure_supported_pretokenization_processor(processor) -> str:
    family = _pretokenization_processor_family(processor)
    if family is None:
        raise ValueError(
            "Pretokenization currently supports Qwen processors only; got "
            f"{processor.__class__.__module__}.{processor.__class__.__name__}."
        )
    return family


def _hash_directory_contents(directory: Path) -> str:
    digest = sha256()
    for path in sorted(path for path in directory.rglob("*") if path.is_file()):
        digest.update(str(path.relative_to(directory).as_posix()).encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _fingerprint_loaded_processor(processor) -> str:
    with tempfile.TemporaryDirectory() as tempdir:
        processor.save_pretrained(tempdir)
        return _hash_directory_contents(Path(tempdir))


def normalize_pretokenization_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(metadata)
    if "image_resolution" not in normalized:
        min_pixels = normalized.get("image_min_pixels")
        max_pixels = normalized.get("image_max_pixels")
        if min_pixels == max_pixels and min_pixels is not None:
            resolution = int(int(min_pixels) ** 0.5)
            if resolution * resolution == int(min_pixels):
                normalized["image_resolution"] = resolution
    normalized.pop("image_min_pixels", None)
    normalized.pop("image_max_pixels", None)
    return normalized


def _build_pretokenization_metadata_base(
    *,
    processor,
    processor_name_or_path: str,
    max_length: int | None,
    image_resolution: int | None,
    trust_remote_code: bool,
) -> dict[str, Any]:
    family = _ensure_supported_pretokenization_processor(processor)
    return {
        "processor_family": family,
        "processor_class": processor.__class__.__name__,
        "processor_name_or_path": processor_name_or_path,
        "max_length": max_length,
        "image_resolution": image_resolution,
        "trust_remote_code": trust_remote_code,
    }


def _is_matching_pretokenization_metadata_cache(
    cached_metadata: dict[str, Any] | None,
    *,
    expected_base_metadata: dict[str, Any],
) -> bool:
    if cached_metadata is None:
        return False
    processor_snapshot_hash = cached_metadata.get("processor_snapshot_hash")
    if (
        not isinstance(processor_snapshot_hash, str)
        or len(processor_snapshot_hash) != 64
    ):
        return False
    cached_metadata = normalize_pretokenization_metadata(cached_metadata)
    return all(
        cached_metadata.get(field_name) == expected_value
        for field_name, expected_value in expected_base_metadata.items()
    )


def build_pretokenization_metadata(
    *,
    processor,
    processor_name_or_path: str,
    max_length: int | None,
    image_resolution: int | None,
    trust_remote_code: bool,
) -> dict[str, Any]:
    metadata = _build_pretokenization_metadata_base(
        processor=processor,
        processor_name_or_path=processor_name_or_path,
        max_length=max_length,
        image_resolution=image_resolution,
        trust_remote_code=trust_remote_code,
    )
    metadata["processor_snapshot_hash"] = _fingerprint_loaded_processor(processor)
    return metadata


def build_or_load_cached_pretokenization_metadata(
    *,
    processor,
    processor_name_or_path: str,
    max_length: int | None,
    image_resolution: int | None,
    trust_remote_code: bool,
    cache_path: Path,
    progress_callback: ProgressCallback | None = None,
    lock_timeout_seconds: float = 1800.0,
) -> dict[str, Any]:
    """Builds pretokenization metadata once and reuses it across shard jobs."""

    expected_base_metadata = _build_pretokenization_metadata_base(
        processor=processor,
        processor_name_or_path=processor_name_or_path,
        max_length=max_length,
        image_resolution=image_resolution,
        trust_remote_code=trust_remote_code,
    )
    cached_metadata = _load_json_if_exists(cache_path)
    if _is_matching_pretokenization_metadata_cache(
        cached_metadata,
        expected_base_metadata=expected_base_metadata,
    ):
        _emit_progress(
            progress_callback,
            f"Loaded cached pretokenization metadata from {cache_path}",
        )
        return cached_metadata

    lock_dir = cache_path.with_name(f"{cache_path.name}.lock")
    deadline = time.monotonic() + lock_timeout_seconds
    while True:
        try:
            lock_dir.mkdir(parents=True)
            break
        except FileExistsError:
            cached_metadata = _load_json_if_exists(cache_path)
            if _is_matching_pretokenization_metadata_cache(
                cached_metadata,
                expected_base_metadata=expected_base_metadata,
            ):
                _emit_progress(
                    progress_callback,
                    f"Loaded cached pretokenization metadata from {cache_path}",
                )
                return cached_metadata
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "Timed out waiting for pretokenization metadata cache lock: "
                    f"{lock_dir}"
                )
            time.sleep(1.0)

    try:
        cached_metadata = _load_json_if_exists(cache_path)
        if _is_matching_pretokenization_metadata_cache(
            cached_metadata,
            expected_base_metadata=expected_base_metadata,
        ):
            _emit_progress(
                progress_callback,
                f"Loaded cached pretokenization metadata from {cache_path}",
            )
            return cached_metadata

        _emit_progress(
            progress_callback,
            f"Building pretokenization metadata cache at {cache_path}",
        )
        metadata = dict(expected_base_metadata)
        metadata["processor_snapshot_hash"] = _fingerprint_loaded_processor(processor)
        _atomic_write_json(cache_path, metadata)
        return metadata
    finally:
        try:
            lock_dir.rmdir()
        except OSError:
            pass


def _should_report_progress(
    completed: int,
    total: int,
    *,
    interval: int,
) -> bool:
    if total <= 0:
        return False
    return completed == total or completed % max(1, interval) == 0


def _nearest_existing_ancestor(path: Path) -> Path:
    current = path.resolve()
    while not current.exists() and current.parent != current:
        current = current.parent
    return current


def _artifact_image_relpath(image_path: str) -> str:
    source_path = Path(image_path).resolve()
    suffix = source_path.suffix.lower()
    digest = sha256(str(source_path).encode("utf-8")).hexdigest()
    filename = f"{digest}{suffix}"
    return str(PurePosixPath(_PREPROCESSED_IMAGES_DIRNAME) / filename)


def _collect_artifact_image_entries(
    *,
    examples: list[AgentTurnExample],
    output_dir: Path,
) -> list[_ArtifactImageEntry]:
    image_entries: list[_ArtifactImageEntry] = []
    image_relpaths_by_source: dict[str, str] = {}
    for example in examples:
        for image_path in example.image_paths:
            if image_path in image_relpaths_by_source:
                continue
            relpath = _artifact_image_relpath(image_path)
            image_relpaths_by_source[image_path] = relpath
            image_entries.append(
                _ArtifactImageEntry(
                    source_key=image_path,
                    source_path=Path(image_path).resolve(),
                    relpath=relpath,
                    destination=output_dir / relpath,
                )
            )
    return image_entries


def _hardlink_artifact_image(entry: _ArtifactImageEntry) -> None:
    entry.destination.parent.mkdir(parents=True, exist_ok=True)
    os.link(entry.source_path, entry.destination)


def _copy_artifact_image(entry: _ArtifactImageEntry) -> None:
    entry.destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(entry.source_path, entry.destination)


def _resize_artifact_image(
    entry: _ArtifactImageEntry,
    *,
    image_size: tuple[int, int],
) -> None:
    image_module = _import_optional_module("PIL.Image", "Pillow")
    resampling = getattr(image_module, "Resampling", image_module)
    entry.destination.parent.mkdir(parents=True, exist_ok=True)
    with image_module.open(entry.source_path) as image:
        resized_image = image.convert("RGB").resize(
            image_size,
            resample=getattr(resampling, "LANCZOS"),
        )
        resized_image.save(entry.destination)


def _copy_artifact_images_parallel(
    image_entries: list[_ArtifactImageEntry],
    *,
    artifact_image_size: tuple[int, int] | None,
    progress_callback: ProgressCallback | None,
    total_images: int,
    staged_so_far: int,
) -> int:
    if not image_entries:
        return 0

    worker_count = min(_MAX_COPY_WORKERS, len(image_entries))
    _emit_progress(
        progress_callback,
        f"Copying {len(image_entries)} artifact images with {worker_count} workers",
    )

    copied = 0
    progress_bar = _create_progress_bar(
        total=len(image_entries),
        description="stage images",
        unit="image",
    )
    try:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            if artifact_image_size is None:
                submit_entry = lambda entry: executor.submit(
                    _copy_artifact_image, entry
                )
            else:
                submit_entry = lambda entry: executor.submit(
                    _resize_artifact_image,
                    entry,
                    image_size=artifact_image_size,
                )
            future_to_entry = {submit_entry(entry): entry for entry in image_entries}
            for future in as_completed(future_to_entry):
                future.result()
                copied += 1
                staged_count = staged_so_far + copied
                if progress_bar is not None:
                    progress_bar.update(1)
                elif _should_report_progress(
                    staged_count,
                    total_images,
                    interval=_ARTIFACT_IMAGE_PROGRESS_INTERVAL,
                ):
                    _emit_progress(
                        progress_callback,
                        f"Staged artifact images: {staged_count}/{total_images}",
                    )
    finally:
        if progress_bar is not None:
            progress_bar.close()
    return copied


def _stage_artifact_images(
    *,
    image_entries: list[_ArtifactImageEntry],
    output_dir: Path,
    artifact_image_size: tuple[int, int] | None,
    progress_callback: ProgressCallback | None,
) -> _StagedArtifactImages:
    total_images = len(image_entries)
    if total_images == 0:
        return _StagedArtifactImages(
            image_relpaths_by_source={},
            num_hardlinked=0,
            num_copied=0,
        )

    output_device = _nearest_existing_ancestor(output_dir).stat().st_dev
    hardlink_candidates: list[_ArtifactImageEntry] = []
    copy_candidates: list[_ArtifactImageEntry] = []
    image_relpaths_by_source = {
        entry.source_key: entry.relpath for entry in image_entries
    }

    if artifact_image_size is not None:
        width, height = artifact_image_size
        _emit_progress(
            progress_callback,
            f"Resizing {total_images} unique artifact images to {width}x{height}",
        )
        num_copied = _copy_artifact_images_parallel(
            image_entries,
            artifact_image_size=artifact_image_size,
            progress_callback=progress_callback,
            total_images=total_images,
            staged_so_far=0,
        )
        _emit_progress(
            progress_callback,
            "Finished staging artifact images: "
            f"{total_images}/{total_images} "
            f"(0 hardlinked, {num_copied} resized/copied)",
        )
        return _StagedArtifactImages(
            image_relpaths_by_source=image_relpaths_by_source,
            num_hardlinked=0,
            num_copied=num_copied,
        )

    for entry in image_entries:
        if entry.source_path.stat().st_dev == output_device:
            hardlink_candidates.append(entry)
        else:
            copy_candidates.append(entry)

    if hardlink_candidates and not copy_candidates:
        _emit_progress(
            progress_callback,
            f"Staging {total_images} unique artifact images via hardlinks",
        )
    elif hardlink_candidates:
        _emit_progress(
            progress_callback,
            "Staging "
            f"{total_images} unique artifact images "
            f"({len(hardlink_candidates)} hardlink candidates, "
            f"{len(copy_candidates)} cross-device copies)",
        )
    else:
        _emit_progress(
            progress_callback,
            f"Staging {total_images} unique artifact images via copy fallback",
        )

    num_hardlinked = 0
    fallback_copy_entries: list[_ArtifactImageEntry] = list(copy_candidates)
    progress_bar = _create_progress_bar(
        total=len(hardlink_candidates),
        description="hardlink images",
        unit="image",
    )
    try:
        for entry in hardlink_candidates:
            try:
                _hardlink_artifact_image(entry)
            except OSError:
                fallback_copy_entries.append(entry)
            else:
                num_hardlinked += 1
                if progress_bar is None and _should_report_progress(
                    num_hardlinked,
                    total_images,
                    interval=_ARTIFACT_IMAGE_PROGRESS_INTERVAL,
                ):
                    _emit_progress(
                        progress_callback,
                        f"Staged artifact images: {num_hardlinked}/{total_images}",
                    )
            if progress_bar is not None:
                progress_bar.update(1)
    finally:
        if progress_bar is not None:
            progress_bar.close()

    num_copied = _copy_artifact_images_parallel(
        fallback_copy_entries,
        artifact_image_size=None,
        progress_callback=progress_callback,
        total_images=total_images,
        staged_so_far=num_hardlinked,
    )

    _emit_progress(
        progress_callback,
        "Finished staging artifact images: "
        f"{total_images}/{total_images} "
        f"({num_hardlinked} hardlinked, {num_copied} copied)",
    )
    return _StagedArtifactImages(
        image_relpaths_by_source=image_relpaths_by_source,
        num_hardlinked=num_hardlinked,
        num_copied=num_copied,
    )


def stage_artifact_images_for_examples(
    *,
    examples: list[AgentTurnExample],
    output_dir: Path,
    artifact_image_size: tuple[int, int] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, str]:
    image_entries = _collect_artifact_image_entries(
        examples=examples,
        output_dir=output_dir,
    )
    staged_images = _stage_artifact_images(
        image_entries=image_entries,
        output_dir=output_dir,
        artifact_image_size=artifact_image_size,
        progress_callback=progress_callback,
    )
    return staged_images.image_relpaths_by_source


def existing_artifact_image_relpaths_for_examples(
    *,
    examples: list[AgentTurnExample],
    output_dir: Path,
    validate: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, str]:
    image_entries = _collect_artifact_image_entries(
        examples=examples,
        output_dir=output_dir,
    )
    image_relpaths_by_source = {
        entry.source_key: entry.relpath for entry in image_entries
    }
    if not validate:
        _emit_progress(
            progress_callback,
            "Reusing "
            f"{len(image_entries)} existing staged artifact images without validation",
        )
        return image_relpaths_by_source

    progress_bar = _create_progress_bar(
        total=len(image_entries),
        description="reuse images",
        unit="image",
    )
    missing_paths: list[Path] = []
    try:
        for index, entry in enumerate(image_entries, start=1):
            if not entry.destination.is_file():
                missing_paths.append(entry.destination)
            if progress_bar is not None:
                progress_bar.update(1)
            elif _should_report_progress(
                index,
                len(image_entries),
                interval=_ARTIFACT_IMAGE_PROGRESS_INTERVAL,
            ):
                _emit_progress(
                    progress_callback,
                    f"Validated existing staged artifact images: {index}/{len(image_entries)}",
                )
    finally:
        if progress_bar is not None:
            progress_bar.close()
    if missing_paths:
        preview = ", ".join(str(path) for path in missing_paths[:5])
        suffix = (
            "" if len(missing_paths) <= 5 else f", ... ({len(missing_paths)} missing)"
        )
        raise FileNotFoundError(
            "Cannot resume from existing artifact images because expected "
            f"staged image files are missing: {preview}{suffix}"
        )
    _emit_progress(
        progress_callback,
        f"Reusing {len(image_entries)} existing staged artifact images",
    )
    return image_relpaths_by_source


def _pretokenized_shard_filename(prefix: str, shard_index: int) -> str:
    return f"{prefix}-{shard_index:06d}.arrow"


def _pretokenized_shard_manifest_filename(shard_index: int) -> str:
    return f"{_PRETOKENIZED_SHARD_MANIFEST_PREFIX}-{shard_index:06d}.json"


def pretokenized_shard_paths(
    *,
    shard_dir: Path,
    shard_index: int,
) -> dict[str, Path]:
    return {
        "train": shard_dir
        / _pretokenized_shard_filename(
            _PRETOKENIZED_SHARD_TRAIN_PREFIX,
            shard_index,
        ),
        "validation": shard_dir
        / _pretokenized_shard_filename(
            _PRETOKENIZED_SHARD_VALIDATION_PREFIX,
            shard_index,
        ),
        "manifest": shard_dir / _pretokenized_shard_manifest_filename(shard_index),
    }


def discover_pretokenized_num_shards(shard_dir: Path) -> int:
    manifests = sorted(shard_dir.glob(f"{_PRETOKENIZED_SHARD_MANIFEST_PREFIX}-*.json"))
    if not manifests:
        raise FileNotFoundError(f"No pretokenized shard manifests found in {shard_dir}")
    shard_indices = []
    for manifest_path in manifests:
        stem_suffix = manifest_path.stem.removeprefix(
            f"{_PRETOKENIZED_SHARD_MANIFEST_PREFIX}-"
        )
        try:
            shard_indices.append(int(stem_suffix))
        except ValueError as exc:
            raise ValueError(
                f"Unexpected pretokenized shard manifest name: {manifest_path.name}"
            ) from exc
    expected_indices = list(range(max(shard_indices) + 1))
    if sorted(shard_indices) != expected_indices:
        raise FileNotFoundError(
            "Pretokenized shard manifests are not contiguous from zero: "
            f"found {sorted(shard_indices)}, expected {expected_indices}."
        )
    return len(expected_indices)


def load_pretokenized_shard_metadata(
    *,
    shard_dir: Path,
    num_shards: int | None = None,
) -> tuple[int, dict[str, Any]]:
    resolved_num_shards = (
        discover_pretokenized_num_shards(shard_dir)
        if num_shards is None
        else int(num_shards)
    )
    if resolved_num_shards < 1:
        raise ValueError("num_shards must be at least 1.")

    pretokenization: dict[str, Any] | None = None
    for shard_index in range(resolved_num_shards):
        paths = pretokenized_shard_paths(
            shard_dir=shard_dir,
            shard_index=shard_index,
        )
        manifest = _load_json_if_exists(paths["manifest"])
        if manifest is None:
            raise FileNotFoundError(
                f"Missing pretokenized shard manifest: {paths['manifest']}"
            )
        if int(manifest["shard_index"]) != shard_index:
            raise ValueError(
                f"Shard manifest index mismatch for {shard_index}: "
                f"{manifest['shard_index']}"
            )
        if int(manifest["num_shards"]) != resolved_num_shards:
            raise ValueError(
                f"Shard {shard_index} reports {manifest['num_shards']} shards; "
                f"expected {resolved_num_shards}."
            )
        for split_name in ("train", "validation"):
            split_path = paths[split_name]
            if not split_path.is_file():
                raise FileNotFoundError(
                    f"Missing pretokenized shard file: {split_path}"
                )
        shard_pretokenization = normalize_pretokenization_metadata(
            dict(manifest["pretokenization"])
        )
        if pretokenization is None:
            pretokenization = shard_pretokenization
        elif shard_pretokenization != pretokenization:
            raise ValueError(
                f"Shard {shard_index} pretokenization metadata differs from "
                "previous shards."
            )

    if pretokenization is None:  # pragma: no cover - num_shards validation covers this
        raise ValueError("No pretokenization shards were loaded.")
    return resolved_num_shards, pretokenization


def _atomic_write_pretokenized_arrow(
    *,
    blobs_by_sample_id: dict[str, bytes],
    destination: Path,
) -> None:
    pyarrow_module = _import_optional_module("pyarrow", "pyarrow")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    if temp_path.exists():
        temp_path.unlink()
    sample_ids = list(blobs_by_sample_id)
    table = pyarrow_module.table(
        {
            "sample_id": pyarrow_module.array(sample_ids, type=pyarrow_module.string()),
            _PRETOKENIZED_BLOB_COLUMN: pyarrow_module.array(
                [blobs_by_sample_id[sample_id] for sample_id in sample_ids],
                type=pyarrow_module.binary(),
            ),
        }
    )
    try:
        with pyarrow_module.OSFile(str(temp_path), "wb") as sink:
            with pyarrow_module.ipc.RecordBatchStreamWriter(
                sink,
                table.schema,
            ) as writer:
                writer.write_table(table)
        temp_path.replace(destination)
    finally:
        if temp_path.exists():
            temp_path.unlink()


class PretokenizedShardWriter:
    """Incrementally writes one pretokenized shard without retaining all blobs."""

    def __init__(
        self,
        *,
        shard_dir: Path,
        shard_index: int,
        num_shards: int,
        pretokenization: dict[str, Any],
        progress_callback: ProgressCallback | None = None,
        resume_existing: bool = True,
    ) -> None:
        self.shard_dir = shard_dir
        self.shard_index = shard_index
        self.num_shards = num_shards
        self.pretokenization = pretokenization
        self.progress_callback = progress_callback
        self.resume_existing = resume_existing
        self.paths = pretokenized_shard_paths(
            shard_dir=shard_dir,
            shard_index=shard_index,
        )
        self._pyarrow = None
        self._schema = None
        self._temp_paths: dict[str, Path] = {}
        self._sinks: dict[str, Any] = {}
        self._writers: dict[str, Any] = {}
        self._sample_ids_by_split: dict[str, list[str]] = {
            "train": [],
            "validation": [],
        }

    def __enter__(self) -> "PretokenizedShardWriter":
        pyarrow_module = _import_optional_module("pyarrow", "pyarrow")
        self._pyarrow = pyarrow_module
        self._schema = pyarrow_module.schema(
            [
                ("sample_id", pyarrow_module.string()),
                (_PRETOKENIZED_BLOB_COLUMN, pyarrow_module.binary()),
            ]
        )
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        _emit_progress(
            self.progress_callback,
            "Writing pretokenized shard "
            f"{self.shard_index}/{self.num_shards} incrementally to {self.shard_dir}",
        )
        for split_name in ("train", "validation"):
            destination = self.paths[split_name]
            temp_path = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
            if temp_path.exists():
                temp_path.unlink()
            partial_paths = []
            if self.resume_existing:
                partial_paths = sorted(
                    destination.parent.glob(f".{destination.name}.tmp-*")
                )
            self._temp_paths[split_name] = temp_path
            sink = pyarrow_module.OSFile(str(temp_path), "wb")
            writer = pyarrow_module.ipc.RecordBatchStreamWriter(
                sink,
                self._schema,
            )
            self._sinks[split_name] = sink
            self._writers[split_name] = writer
            if partial_paths:
                self._copy_existing_partial_split(
                    split_name=split_name,
                    partial_paths=partial_paths,
                    writer=writer,
                )
            for stale_temp_path in partial_paths:
                if stale_temp_path.exists():
                    stale_temp_path.unlink()
        return self

    def _copy_existing_partial_split(
        self,
        *,
        split_name: str,
        partial_paths: list[Path],
        writer,
    ) -> None:
        if self._pyarrow is None or self._schema is None:
            raise RuntimeError("PretokenizedShardWriter is not open.")
        source_path = max(partial_paths, key=lambda path: path.stat().st_mtime)
        copied_count = 0
        try:
            with self._pyarrow.memory_map(str(source_path), "r") as source:
                reader = self._pyarrow.ipc.open_stream(source)
                for batch in reader:
                    writer.write_batch(batch)
                    sample_ids = [
                        str(value.as_py()) for value in batch.column("sample_id")
                    ]
                    self._sample_ids_by_split[split_name].extend(sample_ids)
                    copied_count += len(sample_ids)
        except Exception as exc:
            _emit_progress(
                self.progress_callback,
                "Ignoring unreadable partial pretokenized "
                f"{split_name} shard {source_path}: {exc}",
            )
            return
        if copied_count:
            _emit_progress(
                self.progress_callback,
                "Resumed "
                f"{copied_count} existing {split_name} pretokenized samples "
                f"from {source_path}",
            )

    def completed_sample_ids(self, split_name: str) -> set[str]:
        if split_name not in self._sample_ids_by_split:
            raise ValueError(f"Unsupported pretokenized split: {split_name}")
        return set(self._sample_ids_by_split[split_name])

    def write_split_chunk(
        self,
        *,
        split_name: str,
        blobs_by_sample_id: dict[str, bytes],
    ) -> None:
        if split_name not in self._writers:
            raise ValueError(f"Unsupported pretokenized split: {split_name}")
        if not blobs_by_sample_id:
            return
        if self._pyarrow is None or self._schema is None:
            raise RuntimeError("PretokenizedShardWriter is not open.")
        sample_ids = list(blobs_by_sample_id)
        table = self._pyarrow.table(
            {
                "sample_id": self._pyarrow.array(
                    sample_ids,
                    type=self._pyarrow.string(),
                ),
                _PRETOKENIZED_BLOB_COLUMN: self._pyarrow.array(
                    [blobs_by_sample_id[sample_id] for sample_id in sample_ids],
                    type=self._pyarrow.binary(),
                ),
            },
            schema=self._schema,
        )
        self._writers[split_name].write_table(table)
        sink = self._sinks[split_name]
        if hasattr(sink, "flush"):
            sink.flush()
        self._sample_ids_by_split[split_name].extend(sample_ids)
        _emit_progress(
            self.progress_callback,
            "Wrote "
            f"{len(sample_ids)} {split_name} pretokenized samples "
            f"({len(self._sample_ids_by_split[split_name])} total)",
        )

    def close(self, *, commit: bool) -> dict[str, Any] | None:
        for writer in self._writers.values():
            writer.close()
        for sink in self._sinks.values():
            sink.close()
        self._writers.clear()
        self._sinks.clear()

        if not commit:
            for temp_path in self._temp_paths.values():
                if temp_path.exists():
                    temp_path.unlink()
            return None

        for split_name, temp_path in self._temp_paths.items():
            temp_path.replace(self.paths[split_name])
        manifest = {
            "format_version": _PREPROCESSED_FORMAT_VERSION,
            "shard_index": self.shard_index,
            "num_shards": self.num_shards,
            "pretokenization": self.pretokenization,
            "splits": {
                "train": {
                    "num_samples": len(self._sample_ids_by_split["train"]),
                    "sample_ids": list(self._sample_ids_by_split["train"]),
                    "path": self.paths["train"].name,
                },
                "validation": {
                    "num_samples": len(self._sample_ids_by_split["validation"]),
                    "sample_ids": list(self._sample_ids_by_split["validation"]),
                    "path": self.paths["validation"].name,
                },
            },
        }
        _atomic_write_json(self.paths["manifest"], manifest)
        return manifest

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.close(commit=exc_type is None)
        return False


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temp_path.replace(path)


def save_pretokenized_shard(
    *,
    shard_dir: Path,
    shard_index: int,
    num_shards: int,
    train_blobs_by_sample_id: dict[str, bytes],
    validation_blobs_by_sample_id: dict[str, bytes],
    pretokenization: dict[str, Any],
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    paths = pretokenized_shard_paths(
        shard_dir=shard_dir,
        shard_index=shard_index,
    )
    _emit_progress(
        progress_callback,
        "Writing pretokenized shard " f"{shard_index}/{num_shards} to {shard_dir}",
    )
    _atomic_write_pretokenized_arrow(
        blobs_by_sample_id=train_blobs_by_sample_id,
        destination=paths["train"],
    )
    _atomic_write_pretokenized_arrow(
        blobs_by_sample_id=validation_blobs_by_sample_id,
        destination=paths["validation"],
    )
    manifest = {
        "format_version": _PREPROCESSED_FORMAT_VERSION,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "pretokenization": pretokenization,
        "splits": {
            "train": {
                "num_samples": len(train_blobs_by_sample_id),
                "sample_ids": list(train_blobs_by_sample_id),
                "path": paths["train"].name,
            },
            "validation": {
                "num_samples": len(validation_blobs_by_sample_id),
                "sample_ids": list(validation_blobs_by_sample_id),
                "path": paths["validation"].name,
            },
        },
    }
    _atomic_write_json(paths["manifest"], manifest)
    return manifest


def _load_pretokenized_shard_dataset(path: Path) -> dict[str, bytes]:
    if not path.is_file():
        raise FileNotFoundError(f"Missing pretokenized shard file: {path}")
    dataset_cls = _get_hf_dataset_class()
    from_file = getattr(dataset_cls, "from_file", None)
    if from_file is None:  # pragma: no cover - depends on datasets version
        raise ImportError(
            "The installed datasets package does not support Dataset.from_file, "
            "which is required for pretokenized shard merge."
        )
    dataset = from_file(str(path))
    blobs_by_sample_id: dict[str, bytes] = {}
    for sample_id, blob in zip(
        dataset["sample_id"],
        dataset[_PRETOKENIZED_BLOB_COLUMN],
        strict=True,
    ):
        blobs_by_sample_id[str(sample_id)] = bytes(blob)
    return blobs_by_sample_id


def iter_pretokenized_shard_split_records(
    *,
    shard_dir: Path,
    shard_index: int,
    split_name: str,
):
    paths = pretokenized_shard_paths(
        shard_dir=shard_dir,
        shard_index=shard_index,
    )
    path = paths[split_name]
    if not path.is_file():
        raise FileNotFoundError(f"Missing pretokenized shard file: {path}")
    pyarrow_module = _import_optional_module("pyarrow", "pyarrow")
    with pyarrow_module.memory_map(str(path), "r") as source:
        reader = pyarrow_module.ipc.open_stream(source)
        for batch in reader:
            sample_id_column = batch.column("sample_id")
            blob_column = batch.column(_PRETOKENIZED_BLOB_COLUMN)
            for sample_id, blob in zip(sample_id_column, blob_column, strict=True):
                yield str(sample_id.as_py()), bytes(blob.as_py())


def load_pretokenized_shard(
    *,
    shard_dir: Path,
    shard_index: int,
) -> PretokenizedShard:
    paths = pretokenized_shard_paths(
        shard_dir=shard_dir,
        shard_index=shard_index,
    )
    manifest = _load_json_if_exists(paths["manifest"])
    if manifest is None:
        raise FileNotFoundError(
            f"Missing pretokenized shard manifest: {paths['manifest']}"
        )
    train_blobs_by_sample_id = _load_pretokenized_shard_dataset(paths["train"])
    validation_blobs_by_sample_id = _load_pretokenized_shard_dataset(
        paths["validation"]
    )
    expected_train_sample_ids = list(
        manifest.get("splits", {}).get("train", {}).get("sample_ids", [])
    )
    expected_validation_sample_ids = list(
        manifest.get("splits", {}).get("validation", {}).get("sample_ids", [])
    )
    if expected_train_sample_ids != list(train_blobs_by_sample_id):
        raise ValueError(
            f"Train sample ids in {paths['manifest']} do not match {paths['train']}."
        )
    if expected_validation_sample_ids != list(validation_blobs_by_sample_id):
        raise ValueError(
            "Validation sample ids in "
            f"{paths['manifest']} do not match {paths['validation']}."
        )
    return PretokenizedShard(
        shard_index=int(manifest["shard_index"]),
        num_shards=int(manifest["num_shards"]),
        pretokenization=normalize_pretokenization_metadata(
            dict(manifest["pretokenization"])
        ),
        train_blobs_by_sample_id=train_blobs_by_sample_id,
        validation_blobs_by_sample_id=validation_blobs_by_sample_id,
        manifest=manifest,
    )


def _feature_json_from_feature(
    feature: dict[str, Any],
    *,
    image_relpaths_by_source: dict[str, str],
) -> str:
    feature["image_paths"] = [
        image_relpaths_by_source[image_path]
        for image_path in feature.get("image_paths", ())
    ]
    feature.pop("images", None)
    return compact_json_dumps(feature)


def _row_from_example(
    example: AgentTurnExample,
    *,
    feature: dict[str, Any],
    image_relpaths_by_source: dict[str, str],
    pretokenized_blob: bytes | None = None,
) -> dict[str, Any]:
    image_paths = list(feature.get("image_paths", ()))
    return {
        "sample_id": example.sample_id,
        "task_name": example.task_name,
        "trajectory_id": example.trajectory_id,
        "agent_id": example.agent_id,
        "num_images": len(image_paths),
        "num_supervised_actions": _num_supervised_actions(example),
        "manifest_entry_json": compact_json_dumps(example.to_manifest_entry()),
        "feature_json": _feature_json_from_feature(
            dict(feature),
            image_relpaths_by_source=image_relpaths_by_source,
        ),
        _PRETOKENIZED_BLOB_COLUMN: pretokenized_blob or b"",
    }


def _serialize_split_rows(
    *,
    split_name: str,
    examples: list[AgentTurnExample],
    image_relpaths_by_source: dict[str, str],
    pretokenized_blobs_by_sample_id: dict[str, bytes] | None,
    progress_callback: ProgressCallback | None,
) -> list[dict[str, Any]]:
    total_examples = len(examples)
    _emit_progress(
        progress_callback,
        f"Serializing {split_name} rows ({total_examples} examples)",
    )
    rows: list[dict[str, Any]] = []
    progress_bar = _create_progress_bar(
        total=total_examples,
        description=f"serialize {split_name}",
        unit="row",
    )
    try:
        for index, example in enumerate(examples, start=1):
            feature = example.to_feature_dict()
            rows.append(
                _row_from_example(
                    example,
                    feature=feature,
                    image_relpaths_by_source=image_relpaths_by_source,
                    pretokenized_blob=None
                    if pretokenized_blobs_by_sample_id is None
                    else pretokenized_blobs_by_sample_id.get(example.sample_id),
                )
            )
            if progress_bar is not None:
                progress_bar.update(1)
            elif _should_report_progress(
                index,
                total_examples,
                interval=_ROW_SERIALIZATION_PROGRESS_INTERVAL,
            ):
                _emit_progress(
                    progress_callback,
                    f"Serialized {split_name} rows: {index}/{total_examples}",
                )
    finally:
        if progress_bar is not None:
            progress_bar.close()
    if progress_bar is not None:
        _emit_progress(
            progress_callback,
            f"Serialized {split_name} rows: {total_examples}/{total_examples}",
        )
    return rows


def _dataset_features():
    features_cls = _get_hf_features_class()
    value_cls = _get_hf_value_class()
    return features_cls(
        {
            "sample_id": value_cls("string"),
            "task_name": value_cls("string"),
            "trajectory_id": value_cls("string"),
            "agent_id": value_cls("string"),
            "num_images": value_cls("int32"),
            "num_supervised_actions": value_cls("int32"),
            "manifest_entry_json": value_cls("large_string"),
            "feature_json": value_cls("large_string"),
            _PRETOKENIZED_BLOB_COLUMN: value_cls("binary"),
        }
    )


def _preprocessed_row_arrow_schema():
    pyarrow_module = _import_optional_module("pyarrow", "pyarrow")
    return pyarrow_module.schema(
        [
            ("sample_id", pyarrow_module.string()),
            ("task_name", pyarrow_module.string()),
            ("trajectory_id", pyarrow_module.string()),
            ("agent_id", pyarrow_module.string()),
            ("num_images", pyarrow_module.int32()),
            ("num_supervised_actions", pyarrow_module.int32()),
            ("manifest_entry_json", pyarrow_module.large_string()),
            ("feature_json", pyarrow_module.large_string()),
            (_PRETOKENIZED_BLOB_COLUMN, pyarrow_module.binary()),
        ]
    )


def _streamed_dataset_split_path(dataset_dir: Path, split_name: str) -> Path:
    return dataset_dir / f"{split_name}.arrow"


class StreamedPreprocessedArtifactWriter:
    """Writes final preprocessed split Arrow files incrementally."""

    def __init__(
        self,
        *,
        output_dir: Path,
        progress_callback: ProgressCallback | None = None,
        resume_existing: bool = True,
    ) -> None:
        self.output_dir = output_dir
        self.dataset_dir = _dataset_dir(output_dir)
        self.progress_callback = progress_callback
        self.resume_existing = resume_existing
        self._pyarrow = None
        self._schema = None
        self._temp_paths: dict[str, Path] = {}
        self._sinks: dict[str, Any] = {}
        self._writers: dict[str, Any] = {}
        self._sample_ids_by_split: dict[str, list[str]] = {
            "train": [],
            "validation": [],
        }

    def __enter__(self) -> "StreamedPreprocessedArtifactWriter":
        pyarrow_module = _import_optional_module("pyarrow", "pyarrow")
        self._pyarrow = pyarrow_module
        self._schema = _preprocessed_row_arrow_schema()
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        for split_name in ("train", "validation"):
            destination = _streamed_dataset_split_path(self.dataset_dir, split_name)
            temp_path = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
            if temp_path.exists():
                temp_path.unlink()
            partial_paths = []
            if self.resume_existing:
                partial_paths = sorted(
                    destination.parent.glob(f".{destination.name}.tmp-*")
                )
            self._temp_paths[split_name] = temp_path
            sink = pyarrow_module.OSFile(str(temp_path), "wb")
            writer = pyarrow_module.ipc.RecordBatchStreamWriter(
                sink,
                self._schema,
            )
            self._sinks[split_name] = sink
            self._writers[split_name] = writer
            if partial_paths:
                self._copy_existing_partial_split(
                    split_name=split_name,
                    partial_paths=partial_paths,
                    writer=writer,
                )
            for stale_temp_path in partial_paths:
                if stale_temp_path.exists():
                    stale_temp_path.unlink()
        return self

    def _copy_existing_partial_split(
        self,
        *,
        split_name: str,
        partial_paths: list[Path],
        writer,
    ) -> None:
        if self._pyarrow is None or self._schema is None:
            raise RuntimeError("StreamedPreprocessedArtifactWriter is not open.")
        source_path = max(partial_paths, key=lambda path: path.stat().st_mtime)
        copied_count = 0
        try:
            with self._pyarrow.memory_map(str(source_path), "r") as source:
                reader = self._pyarrow.ipc.open_stream(source)
                for batch in reader:
                    writer.write_batch(batch)
                    sample_ids = [
                        str(value.as_py()) for value in batch.column("sample_id")
                    ]
                    self._sample_ids_by_split[split_name].extend(sample_ids)
                    copied_count += len(sample_ids)
        except Exception as exc:
            _emit_progress(
                self.progress_callback,
                "Ignoring unreadable partial preprocessed "
                f"{split_name} split {source_path}: {exc}",
            )
            return
        if copied_count:
            _emit_progress(
                self.progress_callback,
                "Resumed "
                f"{copied_count} existing {split_name} preprocessed rows "
                f"from {source_path}",
            )

    def completed_sample_ids(self, split_name: str) -> set[str]:
        if split_name not in self._sample_ids_by_split:
            raise ValueError(f"Unsupported preprocessed split: {split_name}")
        return set(self._sample_ids_by_split[split_name])

    def write_split_rows(
        self,
        *,
        split_name: str,
        rows: list[dict[str, Any]],
    ) -> None:
        if split_name not in self._writers:
            raise ValueError(f"Unsupported preprocessed split: {split_name}")
        if not rows:
            return
        if self._pyarrow is None or self._schema is None:
            raise RuntimeError("StreamedPreprocessedArtifactWriter is not open.")
        table = self._pyarrow.Table.from_pylist(rows, schema=self._schema)
        self._writers[split_name].write_table(table)
        sink = self._sinks[split_name]
        if hasattr(sink, "flush"):
            sink.flush()
        sample_ids = [str(row["sample_id"]) for row in rows]
        self._sample_ids_by_split[split_name].extend(sample_ids)
        _emit_progress(
            self.progress_callback,
            "Wrote "
            f"{len(rows)} {split_name} preprocessed rows "
            f"({len(self._sample_ids_by_split[split_name])} total)",
        )

    def close(self, *, commit: bool) -> None:
        for writer in self._writers.values():
            writer.close()
        for sink in self._sinks.values():
            sink.close()
        self._writers.clear()
        self._sinks.clear()

        if not commit:
            return
        for split_name, temp_path in self._temp_paths.items():
            temp_path.replace(
                _streamed_dataset_split_path(self.dataset_dir, split_name)
            )

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.close(commit=exc_type is None)
        return False


def _build_split_dataset(rows: list[dict[str, Any]]):
    dataset_cls = _get_hf_dataset_class()
    features = _dataset_features()

    if not rows:
        empty_columns = {column_name: [] for column_name in features}
        return dataset_cls.from_dict(
            empty_columns,
            features=features,
        )

    columns = {
        column_name: [row[column_name] for row in rows] for column_name in features
    }
    return dataset_cls.from_dict(
        columns,
        features=features,
    )


def build_preprocessed_dataset_dict(
    *,
    train_examples: list[AgentTurnExample],
    validation_examples: list[AgentTurnExample],
    output_dir: Path,
    train_pretokenized_blobs_by_sample_id: dict[str, bytes] | None = None,
    validation_pretokenized_blobs_by_sample_id: dict[str, bytes] | None = None,
    image_relpaths_by_source: dict[str, str] | None = None,
    artifact_image_size: tuple[int, int] | None = None,
    progress_callback: ProgressCallback | None = None,
):
    dataset_dict_cls = _get_hf_dataset_dict_class()
    all_examples = list(train_examples) + list(validation_examples)
    if image_relpaths_by_source is None:
        image_relpaths_by_source = stage_artifact_images_for_examples(
            examples=all_examples,
            output_dir=output_dir,
            artifact_image_size=artifact_image_size,
            progress_callback=progress_callback,
        )
    train_rows = _serialize_split_rows(
        split_name="train",
        examples=train_examples,
        image_relpaths_by_source=image_relpaths_by_source,
        pretokenized_blobs_by_sample_id=train_pretokenized_blobs_by_sample_id,
        progress_callback=progress_callback,
    )
    validation_rows = _serialize_split_rows(
        split_name="validation",
        examples=validation_examples,
        image_relpaths_by_source=image_relpaths_by_source,
        pretokenized_blobs_by_sample_id=validation_pretokenized_blobs_by_sample_id,
        progress_callback=progress_callback,
    )
    _emit_progress(
        progress_callback,
        "Building in-memory preprocessed DatasetDict "
        f"({len(train_rows)} train rows, {len(validation_rows)} validation rows)",
    )
    return dataset_dict_cls(
        {
            "train": _build_split_dataset(train_rows),
            "validation": _build_split_dataset(validation_rows),
        }
    )


def _build_dataset_card(
    *,
    repo_id: str | None,
    preprocess_config: dict[str, Any],
    split_manifest: dict[str, Any],
) -> str:
    train_tasks = ", ".join(sorted(split_manifest.get("train", {}).get("tasks", {})))
    val_tasks = ", ".join(sorted(split_manifest.get("validation", {}).get("tasks", {})))
    load_snippet = (
        "from datasets import load_dataset\n"
        f'ds = load_dataset("{repo_id}", split="train")'
        if repo_id
        else "from datasets import load_from_disk\n"
        'ds = load_from_disk("/path/to/preprocessed_artifact/dataset")["train"]'
    )
    command = preprocess_config.get("command", "")
    return (
        "\n".join(
            [
                "---",
                "pretty_name: RoboCasa Task-VLM Preprocessed Examples",
                "---",
                "",
                "# RoboCasa Task-VLM Preprocessed Examples",
                "",
                "This artifact contains preprocessed task-VLM examples ready for fine-tuning.",
                "",
                "## Summary",
                "",
                f"- format_version: {preprocess_config.get('preprocessed_format_version', _PREPROCESSED_FORMAT_VERSION)}",
                f"- source dataset root: {split_manifest.get('dataset_root', 'unknown')}",
                f"- train tasks: {train_tasks or 'none'}",
                f"- validation tasks: {val_tasks or 'none'}",
                f"- train format: {preprocess_config.get('train_example_format', 'unknown')}",
                f"- pretokenized: {bool(preprocess_config.get('pretokenized', False))}",
                f"- train samples: {split_manifest.get('train', {}).get('num_samples', 0)}",
                f"- validation samples: {split_manifest.get('validation', {}).get('num_samples', 0)}",
                "",
                "## Load",
                "",
                "```python",
                load_snippet,
                "```",
                "",
                "## Training",
                "",
                "- Use `--preprocessed-data-dir` for local fine-tuning.",
                "- Use `--preprocessed-hf-dataset` for Hub-backed fine-tuning.",
                "",
                "## Provenance",
                "",
                "```bash",
                command,
                "```",
            ]
        ).rstrip()
        + "\n"
    )


def save_preprocessed_artifact(
    *,
    output_dir: Path,
    dataset_dict,
    split_manifest: dict[str, Any],
    preprocess_config: dict[str, Any],
    progress_callback: ProgressCallback | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_dir = _dataset_dir(output_dir)
    _emit_progress(progress_callback, f"Saving dataset split data to {dataset_dir}")
    dataset_dict.save_to_disk(str(dataset_dir))
    write_preprocessed_artifact_sidecars(
        output_dir=output_dir,
        split_manifest=split_manifest,
        preprocess_config=preprocess_config,
        progress_callback=progress_callback,
    )


def write_preprocessed_artifact_sidecars(
    *,
    output_dir: Path,
    split_manifest: dict[str, Any],
    preprocess_config: dict[str, Any],
    progress_callback: ProgressCallback | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    _emit_progress(progress_callback, "Writing manifest, preprocess config, and README")
    _sidecar_path(output_dir, _PREPROCESSED_MANIFEST_FILENAME).write_text(
        json.dumps(split_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _sidecar_path(output_dir, _PREPROCESSED_CONFIG_FILENAME).write_text(
        json.dumps(preprocess_config, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _sidecar_path(output_dir, _PREPROCESSED_README_FILENAME).write_text(
        _build_dataset_card(
            repo_id=None,
            preprocess_config=preprocess_config,
            split_manifest=split_manifest,
        ),
        encoding="utf-8",
    )
    _emit_progress(
        progress_callback, f"Saved local preprocessed artifact to {output_dir}"
    )


def push_preprocessed_artifact_to_hub(
    *,
    output_dir: Path,
    repo_id: str,
    split_manifest: dict[str, Any],
    preprocess_config: dict[str, Any],
    private: bool,
    revision: str | None,
    data_dir: str | None,
) -> None:
    api_cls = _get_hf_api_class()
    api = api_cls()
    api.create_repo(
        repo_id,
        repo_type="dataset",
        private=private,
        exist_ok=True,
    )
    _sidecar_path(output_dir, _PREPROCESSED_README_FILENAME).write_text(
        _build_dataset_card(
            repo_id=repo_id,
            preprocess_config=preprocess_config,
            split_manifest=split_manifest,
        ),
        encoding="utf-8",
    )
    path_in_repo = "" if not data_dir else str(PurePosixPath(data_dir))
    api.upload_folder(
        folder_path=str(output_dir),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        commit_message="Upload preprocessed artifact",
    )


def _load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def infer_split_manifest_from_preprocessed_splits(
    *,
    train_split,
    validation_split,
    source_label: str,
) -> dict[str, Any]:
    def summarize(split_dataset) -> dict[str, Any]:
        counts_by_task: dict[str, dict[str, Any]] = {}
        manifest_entries = [
            json.loads(raw_entry) for raw_entry in split_dataset["manifest_entry_json"]
        ]
        task_names = split_dataset["task_name"]
        trajectory_ids = split_dataset["trajectory_id"]
        supervised_counts = split_dataset["num_supervised_actions"]

        for task_name, trajectory_id, supervised_count in zip(
            task_names,
            trajectory_ids,
            supervised_counts,
            strict=True,
        ):
            entry = counts_by_task.setdefault(
                task_name,
                {
                    "num_samples": 0,
                    "num_supervised_actions": 0,
                    "trajectory_ids": set(),
                },
            )
            entry["num_samples"] += 1
            entry["num_supervised_actions"] += int(supervised_count)
            entry["trajectory_ids"].add(trajectory_id)

        return {
            "num_samples": len(split_dataset),
            "num_supervised_actions": sum(int(value) for value in supervised_counts),
            "tasks": {
                task_name: {
                    "num_samples": task_summary["num_samples"],
                    "num_supervised_actions": task_summary["num_supervised_actions"],
                    "num_trajectories": len(task_summary["trajectory_ids"]),
                }
                for task_name, task_summary in sorted(counts_by_task.items())
            },
            "samples": manifest_entries,
        }

    return {
        "dataset_root": source_label,
        "train": summarize(train_split),
        "validation": summarize(validation_split),
    }


def load_preprocessed_artifact_from_disk(
    output_dir: Path,
) -> LoadedPreprocessedArtifact:
    dataset_dir = _dataset_dir(output_dir)
    if (dataset_dir / "state.json").is_file() or (
        dataset_dir / "dataset_dict.json"
    ).is_file():
        loader = _get_load_from_disk()
        dataset_dict = loader(str(dataset_dir))
    else:
        dataset_dict_cls = _get_hf_dataset_dict_class()
        dataset_cls = _get_hf_dataset_class()
        dataset_dict = dataset_dict_cls(
            {
                "train": dataset_cls.from_file(
                    str(_streamed_dataset_split_path(dataset_dir, "train"))
                ),
                "validation": dataset_cls.from_file(
                    str(_streamed_dataset_split_path(dataset_dir, "validation"))
                ),
            }
        )
    manifest = _load_json_if_exists(
        _sidecar_path(output_dir, _PREPROCESSED_MANIFEST_FILENAME)
    )
    preprocess_config = _load_json_if_exists(
        _sidecar_path(output_dir, _PREPROCESSED_CONFIG_FILENAME)
    )
    source_label = str(output_dir.resolve())
    if manifest is None:
        manifest = infer_split_manifest_from_preprocessed_splits(
            train_split=dataset_dict["train"],
            validation_split=dataset_dict["validation"],
            source_label=source_label,
        )
    return LoadedPreprocessedArtifact(
        train_split=dataset_dict["train"],
        validation_split=dataset_dict["validation"],
        split_manifest=manifest,
        preprocess_config=preprocess_config,
        artifact_root=output_dir.resolve(),
        source_label=source_label,
    )


def resolve_hf_dataset_reference(
    raw_value: str,
    *,
    explicit_revision: str | None = None,
) -> tuple[str, str | None]:
    if explicit_revision:
        return raw_value, explicit_revision
    repo_id, separator, revision = raw_value.rpartition(":")
    if separator and "/" in repo_id and revision:
        return repo_id, revision
    return raw_value, None


def pretokenization_compatibility_reason(
    *,
    runtime_pretokenization: dict[str, Any],
    preprocess_config: dict[str, Any] | None,
) -> str | None:
    if not preprocess_config or not preprocess_config.get("pretokenized", False):
        return "artifact does not include pretokenized tensors"

    pretokenization = preprocess_config.get("pretokenization")
    if not isinstance(pretokenization, dict):
        return "artifact is missing pretokenization metadata"

    runtime_pretokenization = normalize_pretokenization_metadata(
        runtime_pretokenization
    )
    pretokenization = normalize_pretokenization_metadata(pretokenization)
    mismatches = []
    for field_name, expected_value in runtime_pretokenization.items():
        if pretokenization.get(field_name) != expected_value:
            mismatches.append(
                f"{field_name}={pretokenization.get(field_name)!r} "
                f"(artifact) != {expected_value!r} (run)"
            )

    if mismatches:
        return "; ".join(mismatches)
    return None


def load_preprocessed_artifact_from_hub(
    *,
    dataset_ref: str,
    revision: str | None,
    data_dir: str | None,
) -> LoadedPreprocessedArtifact:
    downloader = _get_snapshot_download()
    repo_id, resolved_revision = resolve_hf_dataset_reference(
        dataset_ref,
        explicit_revision=revision,
    )
    snapshot_dir = Path(
        downloader(
            repo_id=repo_id,
            repo_type="dataset",
            revision=resolved_revision,
        )
    )
    artifact_root = snapshot_dir if not data_dir else snapshot_dir / data_dir
    loaded_artifact = load_preprocessed_artifact_from_disk(artifact_root)
    source_label = (
        repo_id if resolved_revision is None else f"{repo_id}:{resolved_revision}"
    )
    return LoadedPreprocessedArtifact(
        train_split=loaded_artifact.train_split,
        validation_split=loaded_artifact.validation_split,
        split_manifest=loaded_artifact.split_manifest,
        preprocess_config=loaded_artifact.preprocess_config,
        artifact_root=loaded_artifact.artifact_root,
        source_label=source_label,
    )


class PreprocessedFeatureDataset(Dataset):
    """PyTorch-style dataset wrapper over serialized preprocessed HF rows."""

    def __init__(
        self,
        split_dataset,
        *,
        artifact_root: Path,
        include_pretokenized: bool = True,
    ) -> None:
        self.split_dataset = split_dataset
        self.artifact_root = artifact_root
        self.include_pretokenized = include_pretokenized

    def __len__(self) -> int:
        return len(self.split_dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.split_dataset[int(index)]
        feature = json.loads(row["feature_json"])
        if "image_paths" in feature:
            feature["image_paths"] = [
                str((self.artifact_root / image_relpath).resolve())
                for image_relpath in feature["image_paths"]
            ]
        elif "images" in row:
            feature["images"] = list(row.get("images", ()))
        pretokenized_blob = row.get(_PRETOKENIZED_BLOB_COLUMN, b"")
        if self.include_pretokenized and pretokenized_blob:
            feature[_PRETOKENIZED_FIELD_NAME] = deserialize_pretokenized_tensors(
                pretokenized_blob
            )
        return feature


def _sample_manifest_entry_from_id(sample_id: str) -> dict[str, Any]:
    parts = sample_id.split("/")
    task_name = parts[0] if len(parts) >= 1 else ""
    trajectory_id = parts[1] if len(parts) >= 2 else ""
    agent_id = parts[2] if len(parts) >= 3 else ""
    step_index = -1
    if parts:
        step_token = parts[-1]
        if step_token.startswith("step_"):
            try:
                step_index = int(step_token.removeprefix("step_"))
            except ValueError:
                step_index = -1

    return {
        "sample_id": sample_id,
        "task_name": task_name,
        "trajectory_id": trajectory_id,
        "step_index": step_index,
        "agent_id": agent_id,
        "num_supervised_actions": 1,
    }


def build_pretokenized_shard_split_manifest(
    *,
    shard_dir: Path,
    num_shards: int | None = None,
    source_label: str | None = None,
) -> dict[str, Any]:
    resolved_num_shards, _ = load_pretokenized_shard_metadata(
        shard_dir=shard_dir,
        num_shards=num_shards,
    )

    def summarize_split(split_name: str) -> dict[str, Any]:
        sample_ids: list[str] = []
        for shard_index in range(resolved_num_shards):
            manifest_path = pretokenized_shard_paths(
                shard_dir=shard_dir,
                shard_index=shard_index,
            )["manifest"]
            manifest = _load_json_if_exists(manifest_path)
            if manifest is None:
                raise FileNotFoundError(
                    f"Missing pretokenized shard manifest: {manifest_path}"
                )
            split_manifest = manifest.get("splits", {}).get(split_name, {})
            sample_ids.extend(
                str(value) for value in split_manifest.get("sample_ids", ())
            )

        entries = [
            _sample_manifest_entry_from_id(sample_id) for sample_id in sample_ids
        ]
        counts_by_task: dict[str, dict[str, Any]] = {}
        for entry in entries:
            task_name = str(entry["task_name"])
            task_summary = counts_by_task.setdefault(
                task_name,
                {
                    "num_samples": 0,
                    "num_supervised_actions": 0,
                    "trajectory_ids": set(),
                },
            )
            task_summary["num_samples"] += 1
            task_summary["num_supervised_actions"] += 1
            if entry["trajectory_id"]:
                task_summary["trajectory_ids"].add(entry["trajectory_id"])

        return {
            "num_samples": len(entries),
            "num_supervised_actions": len(entries),
            "tasks": {
                task_name: {
                    "num_samples": task_summary["num_samples"],
                    "num_supervised_actions": task_summary["num_supervised_actions"],
                    "num_trajectories": len(task_summary["trajectory_ids"]),
                }
                for task_name, task_summary in sorted(counts_by_task.items())
            },
            "samples": entries,
        }

    return {
        "dataset_root": source_label or str(shard_dir.resolve()),
        "train": summarize_split("train"),
        "validation": summarize_split("validation"),
    }


class PretokenizedShardFeatureDataset(Dataset):
    """Dataset wrapper that reads completed pretokenized shard Arrow files directly."""

    def __init__(
        self,
        examples: list[AgentTurnExample],
        *,
        shard_dir: Path,
        split_name: str,
        num_shards: int | None = None,
        include_pretokenized: bool = True,
    ) -> None:
        if split_name not in {"train", "validation"}:
            raise ValueError(f"Unsupported pretokenized split: {split_name}")
        self.examples = list(examples)
        self.shard_dir = shard_dir
        self.split_name = split_name
        self.num_shards, self.pretokenization = load_pretokenized_shard_metadata(
            shard_dir=shard_dir,
            num_shards=num_shards,
        )
        self.include_pretokenized = include_pretokenized
        self._datasets_by_shard: dict[int, Any] = {}
        self._sample_locations = self._build_sample_locations()

    def _build_sample_locations(self) -> dict[str, tuple[int, int]]:
        sample_locations: dict[str, tuple[int, int]] = {}
        for shard_index in range(self.num_shards):
            manifest_path = pretokenized_shard_paths(
                shard_dir=self.shard_dir,
                shard_index=shard_index,
            )["manifest"]
            manifest = _load_json_if_exists(manifest_path)
            if manifest is None:
                raise FileNotFoundError(
                    f"Missing pretokenized shard manifest: {manifest_path}"
                )
            split_manifest = manifest.get("splits", {}).get(self.split_name, {})
            sample_ids = list(split_manifest.get("sample_ids", ()))
            for row_index, sample_id in enumerate(sample_ids):
                sample_id = str(sample_id)
                if sample_id in sample_locations:
                    raise ValueError(
                        f"Duplicate {self.split_name} sample id in pretokenized "
                        f"shards: {sample_id}"
                    )
                sample_locations[sample_id] = (shard_index, row_index)

        requested_ids = {example.sample_id for example in self.examples}
        missing_ids = sorted(requested_ids.difference(sample_locations))
        if missing_ids:
            raise ValueError(
                f"Pretokenized shard coverage mismatch for {self.split_name}: "
                f"missing {len(missing_ids)} ids, first: {missing_ids[:5]}"
            )
        return sample_locations

    def _load_shard_dataset(self, shard_index: int):
        dataset = self._datasets_by_shard.get(shard_index)
        if dataset is not None:
            return dataset
        dataset_cls = _get_hf_dataset_class()
        from_file = getattr(dataset_cls, "from_file", None)
        if from_file is None:  # pragma: no cover - depends on datasets version
            raise ImportError(
                "The installed datasets package does not support Dataset.from_file, "
                "which is required for pretokenized shard training."
            )
        path = pretokenized_shard_paths(
            shard_dir=self.shard_dir,
            shard_index=shard_index,
        )[self.split_name]
        dataset = from_file(str(path))
        self._datasets_by_shard[shard_index] = dataset
        return dataset

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[int(index)]
        feature = example.to_feature_dict()
        if not self.include_pretokenized:
            return feature

        shard_index, row_index = self._sample_locations[example.sample_id]
        row = self._load_shard_dataset(shard_index)[row_index]
        row_sample_id = str(row["sample_id"])
        if row_sample_id != example.sample_id:
            raise ValueError(
                f"Pretokenized shard index mismatch for {example.sample_id}: "
                f"read {row_sample_id} from shard {shard_index} row {row_index}."
            )
        pretokenized_blob = row.get(_PRETOKENIZED_BLOB_COLUMN, b"")
        if pretokenized_blob:
            feature[_PRETOKENIZED_FIELD_NAME] = deserialize_pretokenized_tensors(
                bytes(pretokenized_blob)
            )
        return feature


class PretokenizedShardTensorDataset(Dataset):
    """Dataset wrapper that reads only pretokenized tensors from shard Arrow files."""

    def __init__(
        self,
        *,
        shard_dir: Path,
        split_name: str,
        num_shards: int | None = None,
    ) -> None:
        if split_name not in {"train", "validation"}:
            raise ValueError(f"Unsupported pretokenized split: {split_name}")
        self.shard_dir = shard_dir
        self.split_name = split_name
        self.num_shards, self.pretokenization = load_pretokenized_shard_metadata(
            shard_dir=shard_dir,
            num_shards=num_shards,
        )
        self._datasets_by_shard: dict[int, Any] = {}
        self._sample_locations = self._build_sample_locations()
        self.sample_ids = list(self._sample_locations)

    def _build_sample_locations(self) -> dict[str, tuple[int, int]]:
        sample_locations: dict[str, tuple[int, int]] = {}
        for shard_index in range(self.num_shards):
            manifest_path = pretokenized_shard_paths(
                shard_dir=self.shard_dir,
                shard_index=shard_index,
            )["manifest"]
            manifest = _load_json_if_exists(manifest_path)
            if manifest is None:
                raise FileNotFoundError(
                    f"Missing pretokenized shard manifest: {manifest_path}"
                )
            split_manifest = manifest.get("splits", {}).get(self.split_name, {})
            sample_ids = list(split_manifest.get("sample_ids", ()))
            for row_index, sample_id in enumerate(sample_ids):
                sample_id = str(sample_id)
                if sample_id in sample_locations:
                    raise ValueError(
                        f"Duplicate {self.split_name} sample id in pretokenized "
                        f"shards: {sample_id}"
                    )
                sample_locations[sample_id] = (shard_index, row_index)
        return sample_locations

    def _load_shard_dataset(self, shard_index: int):
        dataset = self._datasets_by_shard.get(shard_index)
        if dataset is not None:
            return dataset
        dataset_cls = _get_hf_dataset_class()
        from_file = getattr(dataset_cls, "from_file", None)
        if from_file is None:  # pragma: no cover - depends on datasets version
            raise ImportError(
                "The installed datasets package does not support Dataset.from_file, "
                "which is required for pretokenized shard training."
            )
        path = pretokenized_shard_paths(
            shard_dir=self.shard_dir,
            shard_index=shard_index,
        )[self.split_name]
        dataset = from_file(str(path))
        self._datasets_by_shard[shard_index] = dataset
        return dataset

    def __len__(self) -> int:
        return len(self.sample_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample_id = self.sample_ids[int(index)]
        shard_index, row_index = self._sample_locations[sample_id]
        row = self._load_shard_dataset(shard_index)[row_index]
        row_sample_id = str(row["sample_id"])
        if row_sample_id != sample_id:
            raise ValueError(
                f"Pretokenized shard index mismatch for {sample_id}: "
                f"read {row_sample_id} from shard {shard_index} row {row_index}."
            )
        pretokenized_blob = row.get(_PRETOKENIZED_BLOB_COLUMN, b"")
        if not pretokenized_blob:
            raise ValueError(f"Missing pretokenized tensor blob for {sample_id}.")
        return {
            "sample_id": sample_id,
            _PRETOKENIZED_FIELD_NAME: deserialize_pretokenized_tensors(
                bytes(pretokenized_blob)
            ),
        }
