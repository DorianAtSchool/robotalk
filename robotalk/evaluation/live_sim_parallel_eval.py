"""Run disjoint live-sim evaluator workers and safely aggregate their outputs.

Pass ordinary ``live_sim_eval`` arguments after ``--``. The source manifest is
partitioned at task boundaries so one worker owns every trajectory for a task
and can reuse that task's simulator session. Worker JSONLs remain authoritative
and are never rewritten; combined outputs are derived under ``aggregate/``.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
from typing import Any, Sequence


SCHEMA_VERSION = 1
EVALUATOR_MODULE = "robotalk.evaluation.live_sim_eval"
RESULTS_FILENAME = "live_sim_trajectories.jsonl"
METRICS_FILENAME = "live_sim_metrics.json"
SENSITIVE_OPTIONS = frozenset({"--vllm-api-key"})


class ParallelEvalError(RuntimeError):
    """A safety or integrity check prevented a parallel evaluation."""


@dataclass(frozen=True)
class WorkerSpec:
    """The immutable files and evaluator arguments owned by one worker."""

    index: int
    manifest_path: Path
    output_dir: Path
    task_names: tuple[str, ...]
    trajectory_keys: frozenset[tuple[str, str]]
    live_args: tuple[str, ...]


def _option_value(args: Sequence[str], name: str) -> str | None:
    """Return one CLI option value, supporting both common spellings."""

    matches: list[str] = []
    index = 0
    while index < len(args):
        token = args[index]
        if token == name:
            if index + 1 >= len(args):
                raise ParallelEvalError(f"{name} requires a value")
            matches.append(args[index + 1])
            index += 2
            continue
        prefix = name + "="
        if token.startswith(prefix):
            matches.append(token[len(prefix) :])
        index += 1
    if len(matches) > 1:
        raise ParallelEvalError(f"{name} may be provided only once")
    return matches[0] if matches else None


def _remove_option(args: Sequence[str], name: str) -> list[str]:
    """Remove one value-taking option from an argument sequence."""

    result: list[str] = []
    index = 0
    while index < len(args):
        token = args[index]
        if token == name:
            if index + 1 >= len(args):
                raise ParallelEvalError(f"{name} requires a value")
            index += 2
            continue
        if token.startswith(name + "="):
            index += 1
            continue
        result.append(token)
        index += 1
    return result


def _replace_option(args: Sequence[str], name: str, value: str) -> list[str]:
    result = _remove_option(args, name)
    result.extend((name, value))
    return result


def _redact_args(args: Sequence[str]) -> list[str]:
    """Remove secrets from persisted metadata and printed commands."""

    redacted: list[str] = []
    index = 0
    while index < len(args):
        token = args[index]
        if token in SENSITIVE_OPTIONS:
            redacted.append(token)
            if index + 1 >= len(args):
                raise ParallelEvalError(f"{token} requires a value")
            redacted.append("<redacted>")
            index += 2
            continue
        sensitive_prefix = next(
            (name + "=" for name in SENSITIVE_OPTIONS if token.startswith(name + "=")),
            None,
        )
        if sensitive_prefix is not None:
            redacted.append(sensitive_prefix + "<redacted>")
        else:
            redacted.append(token)
        index += 1
    return redacted


def _identity_args(args: Sequence[str]) -> list[str]:
    """Fingerprint secrets while retaining exact resume identity."""

    identity: list[str] = []
    index = 0
    while index < len(args):
        token = args[index]
        if token in SENSITIVE_OPTIONS:
            identity.append(token)
            if index + 1 >= len(args):
                raise ParallelEvalError(f"{token} requires a value")
            value = args[index + 1]
            identity.append("sha256:" + hashlib.sha256(value.encode()).hexdigest())
            index += 2
            continue
        sensitive_name = next(
            (name for name in SENSITIVE_OPTIONS if token.startswith(name + "=")),
            None,
        )
        if sensitive_name is not None:
            value = token.split("=", 1)[1]
            digest = hashlib.sha256(value.encode()).hexdigest()
            identity.append(f"{sensitive_name}=sha256:{digest}")
        else:
            identity.append(token)
        index += 1
    return identity


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _selected_mapping(
    manifest: dict[str, Any], tasks_value: str | None,
    *,
    cohort_split: str | None = None,
    episodes_per_task: int | None = None,
    cohort_mode: str = "sampled",
    episodes_per_config: int = 1,
    max_configs_per_task: int | None = None,
    evaluation_seed: int | None = None,
) -> dict[str, list[str]]:
    manifest_type = manifest.get("manifest_type")
    if manifest_type == "fixed_live_sim_configuration_targets":
        if cohort_split is None:
            raise ParallelEvalError(
                "fixed_live_sim_configuration_targets requires --cohort-split"
            )
        try:
            source = manifest["configurations"][cohort_split]
        except KeyError as exc:
            raise ParallelEvalError(f"manifest has no split {cohort_split!r}") from exc
        from robotalk.evaluation.fixed_cohort_selection import (
            select_configuration_episodes,
        )

        resolved_seed = manifest.get("cohort_seed") if evaluation_seed is None else evaluation_seed
        if resolved_seed is None:
            raise ParallelEvalError(
                "configuration cohort requires --evaluation-seed or cohort_seed"
            )
        count = 10 if episodes_per_task is None else episodes_per_task
        mapping = {
            str(task): [
                str(row["episode_id"])
                for row in select_configuration_episodes(
                    configurations,
                    task_name=str(task),
                    split=cohort_split,
                    mode=cohort_mode,
                    evaluation_seed=int(resolved_seed),
                    episodes_per_task=count,
                    episodes_per_config=episodes_per_config,
                    max_configs_per_task=max_configs_per_task,
                )
            ]
            for task, configurations in source.items()
        }
    else:
        try:
            raw_mapping = manifest["trajectory_ids_by_task"]
        except KeyError as exc:
            raise ParallelEvalError(
                "manifest has no trajectory_ids_by_task mapping"
            ) from exc
        mapping = {
            str(task): sorted(str(trajectory_id) for trajectory_id in ids)
            for task, ids in raw_mapping.items()
        }
    if tasks_value is not None:
        requested = {task.strip() for task in tasks_value.split(",") if task.strip()}
        missing = requested - set(mapping)
        if missing:
            raise ParallelEvalError(
                "--tasks contains names absent from the manifest: "
                + ", ".join(sorted(missing))
            )
        mapping = {task: ids for task, ids in mapping.items() if task in requested}
    mapping = {task: ids for task, ids in mapping.items() if ids}
    if not mapping:
        raise ParallelEvalError("no trajectories remain after task filtering")
    return mapping



def _task_weights(
    manifest: dict[str, Any], mapping: dict[str, list[str]]
) -> dict[str, int]:
    """Estimate each task's cost by its number of episodes."""

    return {task: len(ids) for task, ids in mapping.items()}


def shard_task_mapping(
    manifest: dict[str, Any],
    mapping: dict[str, list[str]],
    worker_count: int,
) -> list[dict[str, list[str]]]:
    """Deterministically balance whole tasks with longest-processing-time bins."""

    if worker_count < 1:
        raise ParallelEvalError("--workers must be at least 1")
    effective_workers = min(worker_count, len(mapping))
    bins: list[dict[str, Any]] = [
        {"weight": 0, "mapping": {}} for _ in range(effective_workers)
    ]
    weights = _task_weights(manifest, mapping)
    for task in sorted(mapping, key=lambda name: (-weights[name], name)):
        worker_index = min(
            range(effective_workers),
            key=lambda index: (
                bins[index]["weight"],
                len(bins[index]["mapping"]),
                index,
            ),
        )
        bins[worker_index]["mapping"][task] = mapping[task]
        bins[worker_index]["weight"] += weights[task]
    return [worker["mapping"] for worker in bins]


def build_shard_manifest(
    manifest: dict[str, Any],
    mapping: dict[str, list[str]],
    *,
    worker_index: int,
    worker_count: int,
) -> dict[str, Any]:
    """Return a self-describing manifest containing exactly one worker's keys."""

    shard = deepcopy(manifest)
    if shard.get("manifest_type") == "fixed_live_sim_configuration_targets":
        for split_name, configurations in list(shard.get("configurations", {}).items()):
            shard["configurations"][split_name] = {
                task: rows
                for task, rows in configurations.items()
                if task in mapping
            }
    else:
        shard["trajectory_ids_by_task"] = mapping
    allowed = {
        (task, trajectory_id)
        for task, trajectory_ids in mapping.items()
        for trajectory_id in trajectory_ids
    }
    summary = dict(shard.get("summary") or {})
    summary.update({"num_trajectories": len(allowed), "num_tasks": len(mapping)})
    shard["summary"] = summary
    shard["parallel_shard"] = {
        "schema_version": SCHEMA_VERSION,
        "worker_index": worker_index,
        "worker_count": worker_count,
        "task_names": sorted(mapping),
    }
    return shard


def _json_text(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _normalize_live_args(live_args: Sequence[str]) -> tuple[list[str], str | None]:
    if not live_args:
        raise ParallelEvalError("live_sim_eval arguments are required after --")
    normalized = list(live_args)
    if normalized[:1] == ["--"]:
        normalized = normalized[1:]
    manifest_value = _option_value(normalized, "--manifest")
    output_value = _option_value(normalized, "--output-dir")
    backend = _option_value(normalized, "--backend")
    gl_backend = _option_value(normalized, "--gl-backend")
    tasks_value = _option_value(normalized, "--tasks")
    if manifest_value is None:
        raise ParallelEvalError("forwarded arguments must include --manifest")
    if output_value is None:
        raise ParallelEvalError("forwarded arguments must include --output-dir")
    if backend is None:
        raise ParallelEvalError("forwarded arguments must include --backend")
    if gl_backend != "egl":
        raise ParallelEvalError(
            "parallel live-sim requires an explicit --gl-backend egl"
        )
    if _option_value(normalized, "--max-trajectories") is not None:
        raise ParallelEvalError(
            "--max-trajectories has ambiguous per-worker semantics; use a fixed manifest"
        )
    normalized = _remove_option(normalized, "--tasks")
    if "--resume" not in normalized:
        normalized.append("--resume")
    return normalized, tasks_value


def prepare_parallel_run(
    live_args: Sequence[str],
    *,
    requested_workers: int,
) -> tuple[Path, dict[str, Any], list[WorkerSpec]]:
    """Create or validate a stable parallel layout without launching workers."""

    normalized, tasks_value = _normalize_live_args(live_args)
    manifest_path = Path(_option_value(normalized, "--manifest") or "").resolve()
    output_root = Path(_option_value(normalized, "--output-dir") or "").resolve()
    if not manifest_path.is_file():
        raise ParallelEvalError(f"manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    cohort_split = _option_value(normalized, "--cohort-split")
    episodes_value = _option_value(normalized, "--episodes-per-task")
    episodes_per_task = int(episodes_value) if episodes_value is not None else None
    cohort_mode = _option_value(normalized, "--cohort-mode") or "sampled"
    episodes_per_config_value = _option_value(normalized, "--episodes-per-config")
    episodes_per_config = (
        int(episodes_per_config_value) if episodes_per_config_value is not None else 1
    )
    max_configs_value = _option_value(normalized, "--max-configs-per-task")
    max_configs_per_task = int(max_configs_value) if max_configs_value is not None else None
    evaluation_seed_value = _option_value(normalized, "--evaluation-seed")
    evaluation_seed = int(evaluation_seed_value) if evaluation_seed_value is not None else None
    selected = _selected_mapping(
        manifest,
        tasks_value,
        cohort_split=cohort_split,
        episodes_per_task=episodes_per_task,
        cohort_mode=cohort_mode,
        episodes_per_config=episodes_per_config,
        max_configs_per_task=max_configs_per_task,
        evaluation_seed=evaluation_seed,
    )
    shard_mappings = shard_task_mapping(manifest, selected, requested_workers)
    effective_workers = len(shard_mappings)

    template_args = _replace_option(normalized, "--manifest", "<worker-manifest>")
    template_args = _replace_option(template_args, "--output-dir", "<worker-output>")
    identity = {
        "schema_version": SCHEMA_VERSION,
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": _sha256(manifest_path),
        "requested_workers": requested_workers,
        "effective_workers": effective_workers,
        "selected_tasks": sorted(selected),
        "live_args_template": _identity_args(template_args),
        "shards": [
            {
                "worker_index": index,
                "trajectory_ids_by_task": mapping,
            }
            for index, mapping in enumerate(shard_mappings)
        ],
    }
    metadata_path = output_root / "parallel_run.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        existing_identity = metadata.get("identity")
        if existing_identity != identity:
            raise ParallelEvalError(
                "parallel output configuration differs from parallel_run.json; "
                "resume with the identical command and source manifest"
            )
    else:
        existing_entries = (
            [path for path in output_root.iterdir() if path.name != ".parallel_eval.lock"]
            if output_root.exists()
            else []
        )
        if existing_entries:
            raise ParallelEvalError(
                f"output root is non-empty and has no parallel_run.json: {output_root}"
            )
        output_root.mkdir(parents=True, exist_ok=True)
        metadata = {"identity": identity, "invocations": []}
        _atomic_write(metadata_path, _json_text(metadata))

    specs: list[WorkerSpec] = []
    manifests_dir = output_root / "manifests"
    workers_dir = output_root / "workers"
    for index, mapping in enumerate(shard_mappings):
        worker_name = f"worker_{index:03d}"
        worker_manifest_path = manifests_dir / f"{worker_name}.json"
        worker_output_dir = workers_dir / worker_name
        shard = build_shard_manifest(
            manifest,
            mapping,
            worker_index=index,
            worker_count=effective_workers,
        )
        expected_text = _json_text(shard)
        if worker_manifest_path.exists():
            if worker_manifest_path.read_text(encoding="utf-8") != expected_text:
                raise ParallelEvalError(
                    f"worker manifest changed unexpectedly: {worker_manifest_path}"
                )
        else:
            _atomic_write(worker_manifest_path, expected_text)
        worker_output_dir.mkdir(parents=True, exist_ok=True)
        worker_args = _replace_option(
            normalized, "--manifest", str(worker_manifest_path)
        )
        worker_args = _replace_option(
            worker_args, "--output-dir", str(worker_output_dir)
        )
        trajectory_keys = frozenset(
            (task, trajectory_id)
            for task, trajectory_ids in mapping.items()
            for trajectory_id in trajectory_ids
        )
        specs.append(
            WorkerSpec(
                index=index,
                manifest_path=worker_manifest_path,
                output_dir=worker_output_dir,
                task_names=tuple(sorted(mapping)),
                trajectory_keys=trajectory_keys,
                live_args=tuple(worker_args),
            )
        )
    return output_root, metadata, specs


class OutputLock:
    """Prevent two coordinators from mutating one parallel output root."""

    def __init__(self, output_root: Path):
        self.path = output_root / ".parallel_eval.lock"

    def __enter__(self) -> "OutputLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            try:
                payload = json.loads(self.path.read_text(encoding="utf-8"))
                pid = int(payload["pid"])
            except FileNotFoundError:
                pass
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                raise ParallelEvalError(
                    f"parallel output has an unreadable lock; inspect {self.path}"
                ) from exc
            else:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    self.path.unlink(missing_ok=True)
                except PermissionError as exc:
                    raise ParallelEvalError(
                        f"parallel output is locked by PID {pid}"
                    ) from exc
                else:
                    raise ParallelEvalError(
                        f"parallel output is locked by PID {pid}"
                    )
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        descriptor = os.open(self.path, flags, 0o644)
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump({"pid": os.getpid(), "created_at": _utc_now()}, stream)
            stream.write("\n")
        return self

    def __exit__(self, *_args: object) -> None:
        self.path.unlink(missing_ok=True)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _terminate_processes(processes: dict[int, subprocess.Popen[Any]]) -> None:
    running = [process for process in processes.values() if process.poll() is None]
    for process in running:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 10.0
    while running and time.monotonic() < deadline:
        running = [process for process in running if process.poll() is None]
        if running:
            time.sleep(0.1)
    for process in running:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def run_workers(
    specs: Sequence[WorkerSpec],
    *,
    python_executable: str,
    poll_interval: float,
    fail_fast: bool,
) -> tuple[float, dict[int, int]]:
    """Launch all workers, terminating peers if fail-fast observes a failure."""

    processes: dict[int, subprocess.Popen[Any]] = {}
    seen_record_counts = _preflight_worker_records(specs)
    logs: list[Any] = []
    started = time.monotonic()
    try:
        for spec in specs:
            log_path = spec.output_dir / "worker.log"
            log = log_path.open("a", encoding="utf-8")
            logs.append(log)
            command = [python_executable, "-m", EVALUATOR_MODULE, *spec.live_args]
            print(
                f"[parallel] launch worker {spec.index}: "
                + shlex.join(_redact_args(command)),
                flush=True,
            )
            environment = os.environ.copy()
            environment.update(
                {
                    "MUJOCO_GL": "egl",
                    "PYOPENGL_PLATFORM": "egl",
                    "PYTHONUNBUFFERED": "1",
                }
            )
            processes[spec.index] = subprocess.Popen(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
                start_new_session=True,
            )

        unfinished = set(processes)
        exit_codes: dict[int, int] = {}
        while unfinished:
            fatal_issues = _scan_new_worker_records(specs, seen_record_counts)
            if fatal_issues:
                _terminate_processes(processes)
                raise ParallelEvalError(
                    "live worker validation failed: " + "; ".join(fatal_issues)
                )
            for index in sorted(tuple(unfinished)):
                return_code = processes[index].poll()
                if return_code is None:
                    continue
                unfinished.remove(index)
                exit_codes[index] = return_code
                print(
                    f"[parallel] worker {index} exited with {return_code}",
                    flush=True,
                )
                if return_code != 0 and fail_fast:
                    _terminate_processes(processes)
                    for other_index in unfinished:
                        exit_codes[other_index] = (
                            processes[other_index].wait()
                            if processes[other_index].poll() is None
                            else int(processes[other_index].returncode)
                        )
                    unfinished.clear()
                    break
            if unfinished:
                time.sleep(poll_interval)
        return time.monotonic() - started, exit_codes
    except BaseException:
        _terminate_processes(processes)
        raise
    finally:
        for log in logs:
            log.close()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ParallelEvalError(f"worker results are missing: {path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ParallelEvalError(
                f"malformed JSON in {path} at line {line_number}: {exc}"
            ) from exc
        if not isinstance(record, dict):
            raise ParallelEvalError(
                f"non-object JSON record in {path} at line {line_number}"
            )
        records.append(record)
    return records


def _record_key(record: dict[str, Any]) -> tuple[str, str] | None:
    key = (record.get("task_name"), record.get("trajectory_id"))
    if all(isinstance(part, str) and part for part in key):
        return key
    return None


def _preflight_worker_records(specs: Sequence[WorkerSpec]) -> dict[int, int]:
    """Validate existing append-only records before starting a resumed run."""

    counts: dict[int, int] = {}
    for spec in specs:
        path = spec.output_dir / RESULTS_FILENAME
        if not path.exists():
            counts[spec.index] = 0
            continue
        records = _read_jsonl(path)
        for record in records:
            key = _record_key(record)
            if key is None:
                raise ParallelEvalError(
                    f"worker {spec.index} has a record without a valid key"
                )
            if key not in spec.trajectory_keys:
                raise ParallelEvalError(
                    f"worker {spec.index} has an unowned trajectory: {key}"
                )
        counts[spec.index] = len(records)
    return counts


def _fatal_record_issues(
    worker_index: int, record: dict[str, Any]
) -> list[str]:
    key = _record_key(record)
    label = f"worker {worker_index} {key or '<invalid-key>'}"
    issues: list[str] = []
    if key is None:
        issues.append(f"{label} has no valid trajectory key")
    if record.get("termination") == "harness_error":
        issues.append(f"{label} ended in harness_error")
    if record.get("native_success") is None:
        issues.append(f"{label} has null native_success")
    if record.get("fsm_goal_satisfied") is None:
        issues.append(f"{label} has null fsm_goal_satisfied")
    if record.get("native_error"):
        issues.append(f"{label} has native_error: {record['native_error']}")
    return issues


def _scan_new_worker_records(
    specs: Sequence[WorkerSpec], seen_record_counts: dict[int, int]
) -> list[str]:
    """Inspect only records appended during this coordinator invocation."""

    issues: list[str] = []
    for spec in specs:
        path = spec.output_dir / RESULTS_FILENAME
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        lines = text.splitlines()
        if text and not text.endswith("\n"):
            lines = lines[:-1]
        seen = seen_record_counts[spec.index]
        if len(lines) < seen:
            raise ParallelEvalError(
                f"worker {spec.index} results were truncated during evaluation"
            )
        for line_number, line in enumerate(lines[seen:], start=seen + 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ParallelEvalError(
                    f"malformed JSON in {path} at line {line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise ParallelEvalError(
                    f"non-object JSON record in {path} at line {line_number}"
                )
            key = _record_key(record)
            if key is not None and key not in spec.trajectory_keys:
                issues.append(
                    f"worker {spec.index} emitted unowned trajectory {key}"
                )
            issues.extend(_fatal_record_issues(spec.index, record))
        seen_record_counts[spec.index] = len(lines)
    return issues


def aggregate_worker_outputs(
    output_root: Path,
    specs: Sequence[WorkerSpec],
    *,
    parallel_wall_elapsed_s: float,
    parallel_wall_elapsed_s_this_invocation: float | None = None,
) -> dict[str, Any]:
    """Validate immutable worker JSONLs and create a derived aggregate."""

    from robotalk.evaluation.live_sim_eval import _write_metrics

    combined_records: list[dict[str, Any]] = []
    latest_global: dict[tuple[str, str], dict[str, Any]] = {}
    structural_issues: list[str] = []
    for spec in specs:
        records = _read_jsonl(spec.output_dir / RESULTS_FILENAME)
        latest_worker: dict[tuple[str, str], dict[str, Any]] = {}
        for record in records:
            key = _record_key(record)
            if key is None:
                structural_issues.append(
                    f"worker {spec.index} has a record without a valid task/trajectory key"
                )
                continue
            if key not in spec.trajectory_keys:
                structural_issues.append(
                    f"worker {spec.index} emitted unowned trajectory {key}"
                )
            latest_worker[key] = record
        missing = spec.trajectory_keys - set(latest_worker)
        if missing:
            structural_issues.append(
                f"worker {spec.index} is missing {len(missing)} trajectories: "
                + ", ".join(f"{task}/{trajectory}" for task, trajectory in sorted(missing))
            )
        overlap = set(latest_global) & set(latest_worker)
        if overlap:
            structural_issues.append(
                f"duplicate ownership across workers: {sorted(overlap)}"
            )
        latest_global.update(latest_worker)
        combined_records.extend(records)
    if structural_issues:
        raise ParallelEvalError("; ".join(structural_issues))

    aggregate_dir = output_root / "aggregate"
    aggregate_results = aggregate_dir / RESULTS_FILENAME
    rendered_jsonl = "".join(
        json.dumps(record, ensure_ascii=True) + "\n" for record in combined_records
    )
    _atomic_write(aggregate_results, rendered_jsonl)
    worker_metrics = {
        str(spec.index): json.loads(
            (spec.output_dir / METRICS_FILENAME).read_text(encoding="utf-8")
        )
        for spec in specs
    }
    policy_load_values = [
        float(metrics["policy_load_s"])
        for metrics in worker_metrics.values()
        if metrics.get("policy_load_s") is not None
    ]
    _write_metrics(
        aggregate_results,
        aggregate_dir,
        policy_load_s=max(policy_load_values) if policy_load_values else None,
        total_elapsed_s=round(parallel_wall_elapsed_s, 3),
    )
    aggregate_metrics_path = aggregate_dir / METRICS_FILENAME
    aggregate_metrics = json.loads(aggregate_metrics_path.read_text(encoding="utf-8"))
    aggregate_metrics["parallel"] = {
        "effective_workers": len(specs),
        "parallel_wall_elapsed_s": round(parallel_wall_elapsed_s, 3),
        "parallel_wall_elapsed_s_this_invocation": round(
            parallel_wall_elapsed_s_this_invocation
            if parallel_wall_elapsed_s_this_invocation is not None
            else parallel_wall_elapsed_s,
            3,
        ),
        "worker_total_elapsed_s": {
            index: metrics.get("total_elapsed_s")
            for index, metrics in worker_metrics.items()
        },
        "worker_output_dirs": {
            str(spec.index): str(spec.output_dir) for spec in specs
        },
    }
    _atomic_write(aggregate_metrics_path, _json_text(aggregate_metrics))

    outcome_issues: list[str] = []
    for key, record in sorted(latest_global.items()):
        if record.get("termination") == "harness_error":
            outcome_issues.append(f"{key} ended in harness_error")
        if record.get("native_success") is None:
            outcome_issues.append(f"{key} has null native_success")
        if record.get("fsm_goal_satisfied") is None:
            outcome_issues.append(f"{key} has null fsm_goal_satisfied")
        if record.get("native_error"):
            outcome_issues.append(f"{key} has native_error: {record['native_error']}")
    validation = {
        "num_worker_jsonl_records": len(combined_records),
        "num_latest_trajectories": len(latest_global),
        "issues": outcome_issues,
        "valid": not outcome_issues,
    }
    _atomic_write(aggregate_dir / "validation.json", _json_text(validation))
    if outcome_issues:
        raise ParallelEvalError("; ".join(outcome_issues))
    return aggregate_metrics


def _record_invocation(
    metadata_path: Path,
    *,
    started_at: str,
    elapsed_s: float,
    status: str,
    exit_codes: dict[int, int],
    error: str | None = None,
) -> None:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    invocation: dict[str, Any] = {
        "started_at": started_at,
        "finished_at": _utc_now(),
        "elapsed_s": round(elapsed_s, 3),
        "status": status,
        "worker_exit_codes": {str(key): value for key, value in exit_codes.items()},
    }
    if error is not None:
        invocation["error"] = error
    metadata.setdefault("invocations", []).append(invocation)
    _atomic_write(metadata_path, _json_text(metadata))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Let other workers finish after one exits nonzero.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write and validate worker manifests without launching evaluators.",
    )
    parser.add_argument("live_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.poll_interval <= 0:
        parser.error("--poll-interval must be positive")
    return args


def main() -> None:
    args = parse_args()
    normalized, _tasks_value = _normalize_live_args(args.live_args)
    output_value = _option_value(normalized, "--output-dir")
    assert output_value is not None
    output_root = Path(output_value).resolve()
    with OutputLock(output_root):
        output_root, _metadata, specs = prepare_parallel_run(
            args.live_args,
            requested_workers=args.workers,
        )
        print(
            f"[parallel] prepared {len(specs)} workers under {output_root}",
            flush=True,
        )
        for spec in specs:
            print(
                f"[parallel] worker {spec.index}: {len(spec.trajectory_keys)} "
                f"trajectories across {len(spec.task_names)} tasks",
                flush=True,
            )
        if args.dry_run:
            print("[parallel] dry run complete; no evaluator launched", flush=True)
            return

        started_at = _utc_now()
        invocation_started = time.monotonic()
        exit_codes: dict[int, int] = {}
        try:
            elapsed_s, exit_codes = run_workers(
                specs,
                python_executable=args.python,
                poll_interval=args.poll_interval,
                fail_fast=not args.keep_going,
            )
            failures = {
                index: code for index, code in exit_codes.items() if code != 0
            }
            if failures:
                raise ParallelEvalError(f"worker process failures: {failures}")
            aggregate_metrics = aggregate_worker_outputs(
                output_root,
                specs,
                parallel_wall_elapsed_s=(
                    sum(
                        float(invocation.get("elapsed_s", 0.0))
                        for invocation in _metadata.get("invocations", [])
                    )
                    + elapsed_s
                ),
                parallel_wall_elapsed_s_this_invocation=elapsed_s,
            )
        except BaseException as exc:
            elapsed_s = time.monotonic() - invocation_started
            _record_invocation(
                output_root / "parallel_run.json",
                started_at=started_at,
                elapsed_s=elapsed_s,
                status="failed",
                exit_codes=exit_codes,
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        _record_invocation(
            output_root / "parallel_run.json",
            started_at=started_at,
            elapsed_s=elapsed_s,
            status="complete",
            exit_codes=exit_codes,
        )
        print(json.dumps(aggregate_metrics, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    try:
        main()
    except ParallelEvalError as exc:
        raise SystemExit(f"parallel live-sim error: {exc}") from exc
