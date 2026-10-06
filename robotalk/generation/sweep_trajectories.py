#!/usr/bin/env python3
"""Sweep all trajectories in a dataset directory through the sim tool executor.

Discovers all traj_*.json files under <input_dir>/<task_name>/trajectories/,
executes each one across all (layout, style, seed) combinations, and writes
outputs: adapted trajectory, execution metadata, and images rendered by
get_image tool calls to their specified paths.

Usage:
    # Render every trajectory once in a certified scene, as for the dataset:
    python -m robotalk.generation.sweep_trajectories \
        --input-dir data/robotalk_image \
        --output-dir data/robotalk_rendered

    # Limit to specific tasks or trajectory indices; --dry-run lists the jobs:
    python -m robotalk.generation.sweep_trajectories \
        --input-dir data/robotalk_image \
        --output-dir data/robotalk_rendered \
        --tasks hot_dog_setup prepare_coffee --indices 0 1 2 --dry-run

    # Render in every combination of chosen layouts, styles and seeds instead:
    python -m robotalk.generation.sweep_trajectories \
        --input-dir data/robotalk_image \
        --output-dir data/robotalk_sweep --scene-compatibility-cache none \
        --workers 4 --layouts 11 56 --styles 34 42 --seeds 42 99
"""

from __future__ import annotations

import atexit
import argparse
import contextlib
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from functools import partial
import io
import itertools
import json
import logging
import math
import multiprocessing
import os
import queue
import re
import signal
import shutil
import sys
import textwrap
import threading
import time
import traceback
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

if TYPE_CHECKING:
    import datasets

# Make repo-root imports work when this file is executed as a script.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robotalk.generation.runtime.render_env import normalize_mujoco_render_env


def _bootstrap_mujoco_gl() -> None:
    """Prevent inherited unsupported GL backends from breaking imports."""

    normalize_mujoco_render_env(os.environ)


_bootstrap_mujoco_gl()

from robotalk.generation.raw import progress as raw_progress
from robotalk.generation.runtime.client import load_dotenv_file
from robocasa.utils.trajectory_pruning import build_trajectory_pruning_config


DEFAULT_MAP_DPI = 60
DEFAULT_MAP_RENDERER = "raster"

BarColumn = raw_progress.BarColumn
Console = raw_progress.Console
MofNCompleteColumn = raw_progress.MofNCompleteColumn
OVERALL_PROGRESS_COLOR = raw_progress.OVERALL_PROGRESS_COLOR
PROGRESS_BAR_WIDTH = raw_progress.PROGRESS_BAR_WIDTH
ProgressColumn = raw_progress.ProgressColumn
RichProgress = raw_progress.RichProgress
RichTaskProgressAdapter = raw_progress.RichTaskProgressAdapter
StaticQueuedTimeElapsedColumn = raw_progress.StaticQueuedTimeElapsedColumn
TaskProgressColumn = raw_progress.TaskProgressColumn
Text = raw_progress.Text
TextColumn = raw_progress.TextColumn

CLI_EPILOG = textwrap.dedent(
    """\
    Examples:
      # One certified scene per trajectory (how the dataset was rendered):
      python -m robotalk.generation.sweep_trajectories \\
        --input-dir data/robotalk_image \\
        --output-dir data/robotalk_rendered

      # Every combination of chosen layouts, styles and seeds:
      python -m robotalk.generation.sweep_trajectories \\
        --input-dir data/robotalk_image \\
        --output-dir data/robotalk_sweep \\
        --scene-compatibility-cache none \\
        --tasks hot_dog_setup prepare_coffee \\
        --layouts 11 42 56 --styles 34 42 --seeds 1 2 3
    """
)

QUIET_DIAGNOSTIC_PATTERN = re.compile(
    r"\b(warn(?:ing)?|error|exception|traceback|critical|fatal)\b",
    re.IGNORECASE,
)
_WORKER_EXECUTOR_CACHE: dict[
    tuple[int, int],
    tuple[tuple[Any, ...], Any],
] = {}


def _worker_cache_slot() -> tuple[int, int]:
    """Return the process-local worker slot used for executor reuse."""
    return (os.getpid(), threading.get_ident())


def clear_executor_cache() -> None:
    """Close and clear all cached executors in the current process."""
    for _, executor in list(_WORKER_EXECUTOR_CACHE.values()):
        try:
            executor.close()
        except Exception:
            pass
    _WORKER_EXECUTOR_CACHE.clear()


atexit.register(clear_executor_cache)


def _stable_pruning_signature(
    *,
    update_fxtr_cfg_dict: dict[str, dict[str, Any]] | None,
    trajectory_object_names: list[str] | tuple[str, ...] | None,
    trajectory_object_types: list[str] | tuple[str, ...] | None,
    trajectory_object_specs: dict | list | None,
) -> str:
    """Serialize pruning inputs into a stable cache signature."""

    payload = {
        "update_fxtr_cfg_dict": update_fxtr_cfg_dict,
        "trajectory_object_names": list(trajectory_object_names or []),
        "trajectory_object_types": list(trajectory_object_types or []),
        "trajectory_object_specs": trajectory_object_specs,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def _executor_cache_key(
    *,
    task_name: str,
    robots: int,
    layout: int,
    style: int,
    seed: int,
    placement: str,
    cell_size: float,
    robot_spawn: str,
    gl_backend: str,
    render_width: int = 512,
    render_height: int = 512,
    map_dpi: int = DEFAULT_MAP_DPI,
    map_renderer: str = DEFAULT_MAP_RENDERER,
    pruning_signature: str = "",
) -> tuple[Any, ...]:
    """Build the simulator reuse key for one worker-local executor."""

    return (
        task_name,
        robots,
        layout,
        style,
        seed,
        placement,
        cell_size,
        robot_spawn,
        gl_backend,
        render_width,
        render_height,
        map_dpi,
        map_renderer,
        pruning_signature,
    )


def _get_or_create_cached_executor(
    *,
    task_name: str,
    robots: int,
    layout: int,
    style: int,
    seed: int,
    placement: str,
    cell_size: float,
    robot_spawn: str,
    gl_backend: str,
    render_width: int = 512,
    render_height: int = 512,
    map_dpi: int = DEFAULT_MAP_DPI,
    map_renderer: str = DEFAULT_MAP_RENDERER,
    update_fxtr_cfg_dict: dict[str, dict[str, Any]] | None = None,
    trajectory_object_names: list[str] | tuple[str, ...] | None = None,
    trajectory_object_types: list[str] | tuple[str, ...] | None = None,
    trajectory_object_specs: dict | list | None = None,
    executor_factory: Callable[..., Any] | None = None,
) -> Any:
    """Reuse one live executor per worker thread when the env config matches."""

    cache_slot = _worker_cache_slot()
    pruning_signature = _stable_pruning_signature(
        update_fxtr_cfg_dict=update_fxtr_cfg_dict,
        trajectory_object_names=trajectory_object_names,
        trajectory_object_types=trajectory_object_types,
        trajectory_object_specs=trajectory_object_specs,
    )
    cache_key = _executor_cache_key(
        task_name=task_name,
        robots=robots,
        layout=layout,
        style=style,
        seed=seed,
        placement=placement,
        cell_size=cell_size,
        robot_spawn=robot_spawn,
        gl_backend=gl_backend,
        render_width=render_width,
        render_height=render_height,
        map_dpi=map_dpi,
        map_renderer=map_renderer,
        pruning_signature=pruning_signature,
    )
    cached_entry = _WORKER_EXECUTOR_CACHE.get(cache_slot)
    if cached_entry is not None:
        cached_key, cached_executor = cached_entry
        if cached_key == cache_key:
            return cached_executor
        cached_executor.close()
        del _WORKER_EXECUTOR_CACHE[cache_slot]

    if executor_factory is None:
        from robocasa.utils.sim_tool_executor import SimToolExecutor

        executor_factory = SimToolExecutor

    executor = executor_factory(
        task_name=task_name,
        robots=robots,
        layout=layout,
        style=style,
        seed=seed,
        placement=placement,
        cell_size=cell_size,
        robot_spawn=robot_spawn,
        update_fxtr_cfg_dict=update_fxtr_cfg_dict,
        trajectory_object_names=trajectory_object_names,
        trajectory_object_types=trajectory_object_types,
        trajectory_object_specs=trajectory_object_specs,
        gl_backend=gl_backend,
        render_width=render_width,
        render_height=render_height,
        map_dpi=map_dpi,
        map_renderer=map_renderer,
    )
    _WORKER_EXECUTOR_CACHE[cache_slot] = (cache_key, executor)
    return executor

def _executor_processes(executor: Any) -> list[Any]:
    """Return the worker processes owned by one executor when exposed."""

    processes = getattr(executor, "_processes", None)
    if processes is None:
        return []
    if isinstance(processes, dict):
        return [process for process in processes.values() if process is not None]
    values = getattr(processes, "values", None)
    if callable(values):
        return [process for process in values() if process is not None]
    return []


def _terminate_executor_processes(processes: list[Any]) -> None:
    """Terminate then kill worker processes for a cancelled sweep executor."""

    for process in processes:
        is_alive = getattr(process, "is_alive", None)
        if callable(is_alive):
            try:
                if not is_alive():
                    continue
            except Exception:
                pass
        terminate = getattr(process, "terminate", None)
        if callable(terminate):
            try:
                terminate()
            except Exception:
                pass

    for process in processes:
        join = getattr(process, "join", None)
        if callable(join):
            try:
                join(timeout=0.2)
            except Exception:
                pass
        is_alive = getattr(process, "is_alive", None)
        if callable(is_alive):
            try:
                if not is_alive():
                    continue
            except Exception:
                pass
        kill = getattr(process, "kill", None)
        if callable(kill):
            try:
                kill()
            except Exception:
                pass


def _shutdown_sweep_executor(
    executor: Any | None,
    *,
    cancel_running: bool,
) -> None:
    """Shut down one sweep executor and terminate workers on cancellation."""

    if executor is None:
        return

    executor_processes = _executor_processes(executor) if cancel_running else []
    if cancel_running:
        terminate_workers = getattr(executor, "terminate_workers", None)
        if callable(terminate_workers):
            try:
                terminate_workers()
                return
            except Exception:
                pass

    shutdown = getattr(executor, "shutdown", None)
    if callable(shutdown):
        try:
            if cancel_running:
                shutdown(wait=False, cancel_futures=True)
            else:
                shutdown(wait=True)
        except TypeError:
            try:
                shutdown(wait=not cancel_running)
            except TypeError:
                try:
                    shutdown()
                except Exception:
                    pass
        except Exception:
            pass

    if cancel_running:
        _terminate_executor_processes(executor_processes)


class SweepCancellationController:
    """Track the active sweep executor so signals can stop it promptly."""

    def __init__(self) -> None:
        self._active_executor: Any | None = None

    def attach_executor(self, executor: Any) -> None:
        """Register the currently active sweep executor."""

        self._active_executor = executor

    def detach_executor(self, executor: Any) -> None:
        """Clear the active sweep executor when it exits."""

        if self._active_executor is executor:
            self._active_executor = None

    def request_cancel(self) -> None:
        """Stop the active sweep executor immediately on termination."""

        _shutdown_sweep_executor(self._active_executor, cancel_running=True)


@contextlib.contextmanager
def _install_sweep_signal_handlers(
    cancellation_controller: SweepCancellationController | None,
) -> Any:
    """Translate SIGINT and SIGTERM into KeyboardInterrupt during one sweep."""

    if cancellation_controller is None:
        yield
        return
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    handled_signals = [signal.SIGINT]
    if hasattr(signal, "SIGTERM"):
        handled_signals.append(signal.SIGTERM)

    previous_handlers = {signum: signal.getsignal(signum) for signum in handled_signals}

    def _handle_signal(signum: int, _frame: Any) -> None:
        """Cancel the active sweep executor before unwinding the main thread."""

        cancellation_controller.request_cancel()
        raise KeyboardInterrupt(f"received signal {signum}")

    try:
        for signum in handled_signals:
            signal.signal(signum, _handle_signal)
        yield
    finally:
        for signum, previous_handler in previous_handlers.items():
            signal.signal(signum, previous_handler)


class SweepOverallEtaColumn(ProgressColumn):
    """Shows ETA only for the overall runs row."""

    def __init__(self, eta_seconds_getter: Callable[[], float | None]) -> None:
        """Initializes the shared trajectory-average ETA column."""

        if Text is None:
            raise RuntimeError("rich progress support is unavailable")
        super().__init__()
        self._eta_seconds_getter = eta_seconds_getter

    def render(self, task: Any) -> Any:
        """Renders ETA for the overall row and blanks for worker rows."""

        if not getattr(task, "fields", {}).get("show_eta", False):
            return Text("")
        eta_seconds = self._eta_seconds_getter()
        if eta_seconds is None:
            return Text("--:--:--")
        rounded_seconds = 0 if eta_seconds <= 0 else math.ceil(eta_seconds)
        hours, remainder = divmod(rounded_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return Text(f"{hours}:{minutes:02d}:{seconds:02d}")


class SweepAverageTrajectoryTimeColumn(ProgressColumn):
    """Shows average trajectory duration only for the overall runs row."""

    def __init__(
        self,
        average_seconds_getter: Callable[[], float | None],
    ) -> None:
        """Initializes the shared average-duration column."""

        if Text is None:
            raise RuntimeError("rich progress support is unavailable")
        super().__init__()
        self._average_seconds_getter = average_seconds_getter

    def render(self, task: Any) -> Any:
        """Renders average trajectory time for the overall row."""

        if not getattr(task, "fields", {}).get("show_eta", False):
            return Text("")
        average_seconds = self._average_seconds_getter()
        if average_seconds is None:
            return Text("avg --:--:--")
        rounded_seconds = 0 if average_seconds <= 0 else math.ceil(average_seconds)
        hours, remainder = divmod(rounded_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return Text(f"avg {hours}:{minutes:02d}:{seconds:02d}")


class SweepTrajectoryRateColumn(ProgressColumn):
    """Shows completed trajectory throughput only for the overall runs row."""

    def __init__(self, rate_getter: Callable[[], float | None]) -> None:
        """Initializes the shared trajectory-throughput column."""

        if Text is None:
            raise RuntimeError("rich progress support is unavailable")
        super().__init__()
        self._rate_getter = rate_getter

    def render(self, task: Any) -> Any:
        """Renders trajectory throughput for the overall row."""

        if not getattr(task, "fields", {}).get("show_eta", False):
            return Text("")
        rate = self._rate_getter()
        if rate is None:
            return Text("tpm --.-")
        return Text(f"tpm {rate:.1f}")


class SweepRichProgressDisplay:
    """Owns the Rich progress layout used by sweep execution."""

    def __init__(self, *, total_runs: int, worker_count: int) -> None:
        """Initializes the shared overall row used by sweep execution."""

        if RichProgress is None or Console is None:
            raise RuntimeError("rich progress support is unavailable")
        # The scheduler still tracks worker slots internally, but the terminal
        # now renders only the overall runs bar.
        self._worker_count = worker_count
        self._total_runs = total_runs
        self._total_trajectories: int | None = None
        self._completed_trajectories = 0
        self._timed_completed_trajectories = 0
        self._completed_trajectory_seconds = 0.0
        self._first_trajectory_started_at: float | None = None
        self._active_trajectory_starts: dict[int, float] = {}
        self._eta_lock = threading.Lock()

        self.console = Console(stderr=True)
        self._progress = RichProgress(
            TextColumn("[bold]{task.description}[/bold]"),
            BarColumn(bar_width=PROGRESS_BAR_WIDTH),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            StaticQueuedTimeElapsedColumn(),
            SweepOverallEtaColumn(self._estimate_remaining_seconds),
            SweepAverageTrajectoryTimeColumn(self._average_trajectory_seconds),
            SweepTrajectoryRateColumn(self._trajectories_per_minute),
            TextColumn("[dim]{task.fields[status]}"),
            console=self.console,
            transient=False,
            expand=False,
        )
        self._progress.start()
        overall_task_id = self._progress.add_task(
            f"[{OVERALL_PROGRESS_COLOR}]runs[/{OVERALL_PROGRESS_COLOR}]",
            total=total_runs,
            status="running",
            show_eta=True,
        )
        self.overall_progress = RichTaskProgressAdapter(
            self._progress,
            overall_task_id,
            total_runs,
            status="running",
        )

    def assign_worker(
        self,
        worker_slot: int,
        *,
        task_name: str,
        traj_idx: int,
        total_runs: int,
    ) -> None:
        """Keeps compatibility with the sweep scheduler's worker-slot hooks."""

        del task_name, traj_idx
        with self._eta_lock:
            if total_runs > 0:
                started_at = time.monotonic()
                if self._total_trajectories is None:
                    self._total_trajectories = max(self._total_runs // total_runs, 1)
                if self._first_trajectory_started_at is None:
                    self._first_trajectory_started_at = started_at
                self._active_trajectory_starts[worker_slot] = started_at
        self._progress.refresh()

    def record_run_completion(
        self,
        worker_slot: int,
        *,
        result: dict[str, Any],
    ) -> None:
        """Advances the overall bar by one completed run."""

        del worker_slot, result
        self.overall_progress.update(1)

    def finalize_worker(
        self,
        worker_slot: int,
        *,
        error_count: int,
        missing_runs: int,
    ) -> None:
        """Backfills the overall bar if a worker exits without progress events."""

        del error_count
        if missing_runs > 0:
            self.overall_progress.update(missing_runs)
        with self._eta_lock:
            started_at = self._active_trajectory_starts.pop(worker_slot, None)
            total_trajectories = self._total_trajectories
            if total_trajectories is None:
                self._completed_trajectories += 1
            else:
                self._completed_trajectories = min(
                    self._completed_trajectories + 1,
                    total_trajectories,
                )
            if started_at is not None:
                self._completed_trajectory_seconds += max(
                    time.monotonic() - started_at,
                    0.0,
                )
                self._timed_completed_trajectories += 1
        self._progress.refresh()

    def write_log_line(self, log_line: str) -> None:
        """Prints one log line without corrupting the active Rich display."""

        self.console.print(log_line, markup=False, highlight=False)

    def close(self) -> None:
        """Stops the Rich progress display."""

        self._progress.stop()

    def _estimate_remaining_seconds(self) -> float | None:
        """Estimate completion time from the average completed trajectory."""

        with self._eta_lock:
            total_trajectories = self._total_trajectories
            completed_trajectories = self._completed_trajectories
            timed_completed_trajectories = self._timed_completed_trajectories
            completed_trajectory_seconds = self._completed_trajectory_seconds
            active_starts = list(self._active_trajectory_starts.values())

        if total_trajectories is None or timed_completed_trajectories <= 0:
            return None
        if completed_trajectories >= total_trajectories and not active_starts:
            return 0.0

        average_trajectory_seconds = (
            completed_trajectory_seconds / timed_completed_trajectories
        )
        now = time.monotonic()
        worker_finish_times = [
            max(average_trajectory_seconds - max(now - started_at, 0.0), 0.0)
            for started_at in active_starts
        ]
        worker_finish_times.extend(
            [0.0] * max(self._worker_count - len(worker_finish_times), 0)
        )
        if not worker_finish_times:
            worker_finish_times = [0.0]

        remaining_queued_trajectories = max(
            total_trajectories - completed_trajectories - len(active_starts),
            0,
        )
        for _ in range(remaining_queued_trajectories):
            next_worker_index = min(
                range(len(worker_finish_times)),
                key=worker_finish_times.__getitem__,
            )
            worker_finish_times[next_worker_index] += average_trajectory_seconds
        return max(worker_finish_times)

    def _average_trajectory_seconds(self) -> float | None:
        """Return the mean duration for completed trajectories."""

        with self._eta_lock:
            if self._timed_completed_trajectories <= 0:
                return None
            return (
                self._completed_trajectory_seconds / self._timed_completed_trajectories
            )

    def _trajectories_per_minute(self) -> float | None:
        """Return the completed-trajectory throughput per minute."""

        with self._eta_lock:
            completed_trajectories = self._completed_trajectories
            first_trajectory_started_at = self._first_trajectory_started_at

        if completed_trajectories <= 0 or first_trajectory_started_at is None:
            return None
        elapsed_seconds = max(time.monotonic() - first_trajectory_started_at, 0.0)
        if elapsed_seconds <= 0:
            return None
        return completed_trajectories / (elapsed_seconds / 60.0)


def _create_sweep_progress_display(
    *,
    total_runs: int,
    worker_count: int,
    progress_factory: Callable[..., Any] | None = None,
) -> Any | None:
    """Builds the interactive sweep progress display when possible."""

    if total_runs <= 0 or worker_count <= 0:
        return None
    if progress_factory is not None:
        return progress_factory(total_runs=total_runs, worker_count=worker_count)
    if not os.isatty(2) or RichProgress is None or Console is None:
        return None
    return SweepRichProgressDisplay(total_runs=total_runs, worker_count=worker_count)


def _record_local_progress_event(
    progress_display: Any,
    slot_completed_runs: dict[int, int],
    worker_slot: int,
    event: dict[str, Any],
) -> None:
    """Applies one run-completed event directly in the main process."""

    _apply_progress_event(
        progress_display,
        slot_completed_runs,
        {"worker_slot": worker_slot, **event},
    )


def _enqueue_progress_event(
    progress_queue: Any,
    worker_slot: int,
    event: dict[str, Any],
) -> None:
    """Queues one run-completed event from a background worker."""

    progress_queue.put({"worker_slot": worker_slot, **event})


def _apply_progress_event(
    progress_display: Any | None,
    slot_completed_runs: dict[int, int],
    event: dict[str, Any],
) -> None:
    """Applies one queued run-completed event to the active display."""

    if progress_display is None:
        return
    worker_slot = int(event["worker_slot"])
    slot_completed_runs[worker_slot] = slot_completed_runs.get(worker_slot, 0) + 1
    progress_display.record_run_completion(
        worker_slot,
        result={
            "status": event["status"],
            "layout": event["layout"],
            "style": event["style"],
            "seed": event["seed"],
        },
    )


def _drain_progress_events(
    progress_queue: Any | None,
    progress_display: Any | None,
    slot_completed_runs: dict[int, int],
) -> None:
    """Consumes every pending background progress event."""

    if progress_queue is None or progress_display is None:
        return
    while True:
        try:
            event = progress_queue.get_nowait()
        except queue.Empty:
            return
        _apply_progress_event(progress_display, slot_completed_runs, event)


def _create_progress_event_queue(
    *,
    executor_factory: Callable[..., Any],
    mp_context: Any | None = None,
) -> tuple[Any, Any | None]:
    """Builds the queue used to stream run completions back to the main process."""

    if executor_factory is ThreadPoolExecutor:
        return queue.Queue(), None
    progress_manager = (
        mp_context.Manager() if mp_context is not None else multiprocessing.Manager()
    )
    return progress_manager.Queue(), progress_manager


def _assign_progress_worker(
    progress_display: Any | None,
    slot_completed_runs: dict[int, int],
    *,
    worker_slot: int,
    entry: dict[str, Any],
    combo_count: int,
) -> None:
    """Resets one worker row for a newly scheduled trajectory entry."""

    if progress_display is None:
        return
    slot_completed_runs[worker_slot] = 0
    progress_display.assign_worker(
        worker_slot,
        task_name=str(entry["task_dir_name"]),
        traj_idx=int(entry["traj_idx"]),
        total_runs=combo_count,
    )


def _finalize_progress_worker(
    progress_display: Any | None,
    slot_completed_runs: dict[int, int],
    *,
    worker_slot: int,
    results: list[dict[str, Any]],
) -> None:
    """Completes one worker row once its trajectory entry finishes."""

    if progress_display is None:
        return
    completed_runs = slot_completed_runs.get(worker_slot, 0)
    total_runs = len(results)
    error_count = sum(1 for result in results if result["status"] == "error")
    progress_display.finalize_worker(
        worker_slot,
        error_count=error_count,
        missing_runs=max(total_runs - completed_runs, 0),
    )
    slot_completed_runs[worker_slot] = 0


def _emit_sweep_log_line(
    log_line: str,
    *,
    log_writer: Callable[[str], None] | None,
    use_stderr: bool,
) -> None:
    """Write one log line without corrupting an active progress display."""

    if log_writer is not None:
        log_writer(log_line)
        return
    output_stream = sys.stderr if use_stderr else sys.stdout
    print(log_line, file=output_stream, flush=True)


def _quiet_diagnostic_lines(stderr_text: str) -> list[str]:
    """Keep only warning and error stderr lines for quiet sweep runs."""

    diagnostics: list[str] = []
    for raw_line in stderr_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if QUIET_DIAGNOSTIC_PATTERN.search(line):
            diagnostics.append(line)
    return diagnostics


@contextlib.contextmanager
def _quiet_run_output_context(*, suppress_output: bool) -> Any:
    """Suppress normal run chatter while keeping warning/error diagnostics."""

    if not suppress_output:
        yield None
        return

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    robosuite_logger = None
    original_level = None

    try:
        from robosuite.utils.log_utils import ROBOSUITE_DEFAULT_LOGGER

        robosuite_logger = ROBOSUITE_DEFAULT_LOGGER
        original_level = robosuite_logger.level
    except Exception:
        robosuite_logger = None
        original_level = None

    with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(
        stderr_buffer
    ):
        if robosuite_logger is not None:
            robosuite_logger.setLevel(logging.WARNING)
        try:
            yield stderr_buffer
        finally:
            if robosuite_logger is not None and original_level is not None:
                robosuite_logger.setLevel(original_level)


@contextlib.contextmanager
def _temporary_environment(overrides: dict[str, str | None]) -> Any:
    """Apply environment overrides for the duration of one context."""

    previous_values = {key: os.environ.get(key) for key in overrides}
    for key, value in overrides.items():
        if value is None:
            os.environ.pop(key, None)
            continue
        os.environ[key] = value
    try:
        yield
    finally:
        for key, previous_value in previous_values.items():
            if previous_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = previous_value


def _emit_trajectory_log_lines(
    trajectory_result: dict[str, Any],
    *,
    log_writer: Callable[[str], None] | None,
    log_run_completions: bool,
) -> None:
    """Print the selected run logs from one finished trajectory entry."""

    for diagnostic_line in trajectory_result.get("diagnostic_lines", []):
        _emit_sweep_log_line(
            diagnostic_line,
            log_writer=log_writer,
            use_stderr=True,
        )

    for result, log_line in zip(
        trajectory_result["results"],
        trajectory_result["log_lines"],
    ):
        use_stderr = result["status"] == "error"
        if log_run_completions or use_stderr:
            _emit_sweep_log_line(
                log_line,
                log_writer=log_writer,
                use_stderr=use_stderr,
            )


def discover_trajectories(
    input_dir: Path,
    task_filter: list[str] | None = None,
    indices: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Find all trajectory JSONs under input_dir/<task>/trajectories/."""

    entries = []
    for task_dir in sorted(input_dir.iterdir()):
        if not task_dir.is_dir():
            continue
        if task_filter and task_dir.name not in task_filter:
            continue
        traj_dir = task_dir / "trajectories"
        if not traj_dir.is_dir():
            continue
        for traj_file in sorted(traj_dir.glob("traj_*.json")):
            traj_idx = int(traj_file.stem.split("_")[-1])
            if indices is not None and traj_idx not in indices:
                continue
            entries.append(
                {
                    "task_dir_name": task_dir.name,
                    "traj_file": traj_file,
                    "traj_idx": traj_idx,
                }
            )
    return entries


def select_trajectory_shard(
    entries: list[dict[str, Any]],
    *,
    num_shards: int | None = None,
    shard_index: int | None = None,
) -> list[dict[str, Any]]:
    """Select one deterministic round-robin shard from the discovered entries."""

    if num_shards is None and shard_index is None:
        return list(entries)
    if num_shards is None or shard_index is None:
        raise ValueError("--num-shards and --shard-index must be provided together.")
    if num_shards <= 0:
        raise ValueError("--num-shards must be greater than 0.")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("--shard-index must be in [0, --num-shards).")
    if num_shards == 1:
        return list(entries)
    return list(entries[shard_index::num_shards])


def _resolve_run_output_dir(
    output_root: Path,
    *,
    task_name: str,
    traj_idx: int,
    combo_count: int,
    layout: int,
    style: int,
    seed: int,
) -> tuple[Path, str]:
    """Resolve the output directory and display label for one run."""

    if combo_count == 1:
        return output_root / task_name / f"traj_{traj_idx:06d}", ""

    combo_dir = f"L{layout}_S{style}_sd{seed}"
    combo_label = _resolve_combo_label(
        combo_count=combo_count,
        layout=layout,
        style=style,
        seed=seed,
    )
    return output_root / task_name / f"traj_{traj_idx:06d}" / combo_dir, combo_label


def _resolve_combo_label(
    *,
    combo_count: int,
    layout: int,
    style: int,
    seed: int,
) -> str:
    """Build the display label for one scene combo."""

    if combo_count == 1:
        return ""
    return f" L{layout}/S{style}/sd{seed}"


def _format_run_log_line(
    *,
    run_num: int,
    total_runs: int,
    task_name: str,
    traj_idx: int,
    combo_label: str,
    result: dict[str, Any],
) -> str:
    """Build the CLI log line for one completed sweep run."""

    label = f"[{run_num}/{total_runs}] {task_name}/traj_{traj_idx:06d}{combo_label}"
    elapsed_seconds = float(result["elapsed_s"])
    if result["status"] == "ok":
        return (
            f"{label} "
            f"{result['steps_succeeded']}/{result['steps_total']} steps ok, "
            f"{result['images_rendered']} images ({elapsed_seconds:.1f}s)"
        )
    return f"{label} ERROR: {result['error']} ({elapsed_seconds:.1f}s)"


def run_one(
    traj_file: Path,
    output_dir: Path,
    layout: int,
    style: int,
    seed: int,
    robots: int,
    placement: str,
    cell_size: float,
    robot_spawn: str = "sim",
    skip_videos: bool = True,
    gl_backend: str = "osmesa",
    render_width: int = 512,
    render_height: int = 512,
    map_dpi: int = DEFAULT_MAP_DPI,
    map_renderer: str = DEFAULT_MAP_RENDERER,
) -> dict:
    """Execute a single trajectory and return summary info."""
    from robocasa.utils.trajectory_adapter import execute_trajectory

    with open(traj_file) as f:
        trajectory = json.load(f)
    from robotalk.tasks.shared.validation_contract import (
        require_current_validation,
    )

    require_current_validation(trajectory, source=str(traj_file))

    task_name = trajectory.get("composite_task", "Kitchen")
    pruning_config = build_trajectory_pruning_config(trajectory, layout=layout)
    used_pruning_fallback = False
    pruning_fallback_reason: str | None = None

    should_try_pruning = any(
        (
            pruning_config.get("update_fxtr_cfg_dict"),
            pruning_config.get("trajectory_object_names"),
            pruning_config.get("trajectory_object_types"),
            pruning_config.get("trajectory_object_specs"),
        )
    )

    if should_try_pruning:
        try:
            executor = _get_or_create_cached_executor(
                task_name=task_name,
                robots=robots,
                layout=layout,
                style=style,
                seed=seed,
                placement=placement,
                cell_size=cell_size,
                robot_spawn=robot_spawn,
                gl_backend=gl_backend,
                render_width=render_width,
                render_height=render_height,
                map_dpi=map_dpi,
                map_renderer=map_renderer,
                update_fxtr_cfg_dict=pruning_config["update_fxtr_cfg_dict"],
                trajectory_object_names=pruning_config["trajectory_object_names"],
                trajectory_object_types=pruning_config["trajectory_object_types"],
                trajectory_object_specs=pruning_config["trajectory_object_specs"],
            )
        except Exception as exc:
            pruning_fallback_reason = (
                f"trajectory pruning init failed: {type(exc).__name__}: {exc}"
            )
            executor = _get_or_create_cached_executor(
                task_name=task_name,
                robots=robots,
                layout=layout,
                style=style,
                seed=seed,
                placement=placement,
                cell_size=cell_size,
                robot_spawn=robot_spawn,
                gl_backend=gl_backend,
                render_width=render_width,
                render_height=render_height,
                map_dpi=map_dpi,
                map_renderer=map_renderer,
            )
            used_pruning_fallback = True
    else:
        executor = _get_or_create_cached_executor(
            task_name=task_name,
            robots=robots,
            layout=layout,
            style=style,
            seed=seed,
            placement=placement,
            cell_size=cell_size,
            robot_spawn=robot_spawn,
            gl_backend=gl_backend,
            render_width=render_width,
            render_height=render_height,
            map_dpi=map_dpi,
            map_renderer=map_renderer,
        )

    executor.restore_baseline_state()
    output_dir.mkdir(parents=True, exist_ok=True)
    # The source trajectory is copied verbatim -- reordering below changes only
    # the order the sim runs the steps in, never what is stored.
    shutil.copy2(traj_file, output_dir / "original_trajectory.json")

    # Run the steps in the order the concurrent executor applies them.
    from robotalk.tasks.shared.render_order import reorder_for_concurrent_render

    trajectory, order = reorder_for_concurrent_render(trajectory)
    reordered_steps = sum(1 for i, j in enumerate(order) if i != j)
    with open(output_dir / "render_order.json", "w") as f:
        json.dump(
            {"step_order": "concurrent", "order": order, "reordered_steps": reordered_steps},
            f,
        )

    metadata = execute_trajectory(
        executor=executor,
        trajectory=trajectory,
        output_dir=str(output_dir),
        skip_videos=skip_videos,
    )
    steps = metadata.get("steps", [])
    action_steps = [s for s in steps if s.get("tool") != "get_image"]
    n_success = sum(1 for s in action_steps if s.get("success"))
    n_total = len(action_steps)
    n_images = sum(1 for s in steps if s.get("tool") == "get_image")

    return {
        "status": "ok",
        "task": task_name,
        "steps_succeeded": n_success,
        "steps_total": n_total,
        "images_rendered": n_images,
        "used_pruning_fallback": used_pruning_fallback,
        "pruning_fallback_reason": pruning_fallback_reason,
        "reordered_steps": reordered_steps,
    }


def run_trajectory_entry(
    entry: dict[str, Any],
    *,
    entry_index: int,
    total_runs: int,
    combos: tuple[tuple[int, int, int], ...],
    output_root: Path,
    robots: int,
    placement: str,
    cell_size: float,
    robot_spawn: str = "sim",
    skip_videos: bool = True,
    suppress_stdout: bool = False,
    progress_reporter: Callable[[dict[str, Any]], None] | None = None,
    gpu_id: int | None = None,
    gl_backend: str = "osmesa",
    render_width: int = 512,
    render_height: int = 512,
    map_dpi: int = DEFAULT_MAP_DPI,
    map_renderer: str = DEFAULT_MAP_RENDERER,
) -> dict[str, Any]:
    """Execute one discovered trajectory across every requested scene combo."""

    runtime_environment_overrides: dict[str, str | None] = {
        "MUJOCO_GL": gl_backend,
    }
    if gpu_id is not None:
        # Constrain each worker invocation to one GPU before constructing any
        # simulator state so concurrent entries can be spread across GPUs.
        runtime_environment_overrides.update(
            {
                "CUDA_VISIBLE_DEVICES": str(gpu_id),
                "GPUS": str(gpu_id),
                "MUJOCO_EGL_DEVICE_ID": str(gpu_id),
            }
        )
    elif gl_backend != "egl":
        # Clear stale EGL routing when the caller explicitly requested a
        # non-EGL backend in a process that may have run GPU work earlier.
        runtime_environment_overrides["MUJOCO_EGL_DEVICE_ID"] = None

    with _temporary_environment(runtime_environment_overrides):
        task_name = str(entry["task_dir_name"])
        traj_idx = int(entry["traj_idx"])
        traj_file = Path(entry["traj_file"])
        entry_combos = (
            (tuple(entry["scene_combo"]),)
            if entry.get("scene_combo") is not None else combos
        )
        combo_count = len(entry_combos)
        results: list[dict[str, Any]] = []
        diagnostic_lines: list[str] = []
        log_lines: list[str] = []

        for combo_offset, (layout, style, seed) in enumerate(entry_combos):
            run_num = entry_index * combo_count + combo_offset + 1
            traj_output_dir, combo_label = _resolve_run_output_dir(
                output_root,
                task_name=task_name,
                traj_idx=traj_idx,
                combo_count=combo_count,
                layout=layout,
                style=style,
                seed=seed,
            )

            started_at = time.time()
            quiet_stderr_buffer = None
            try:
                with _quiet_run_output_context(
                    suppress_output=suppress_stdout
                ) as quiet_stderr_buffer:
                    result = run_one(
                        traj_file=traj_file,
                        output_dir=traj_output_dir,
                        layout=layout,
                        style=style,
                        seed=seed,
                        robots=robots,
                        placement=placement,
                        cell_size=cell_size,
                        robot_spawn=robot_spawn,
                        skip_videos=skip_videos,
                        gl_backend=gl_backend,
                        render_width=render_width,
                        render_height=render_height,
                        map_dpi=map_dpi,
                        map_renderer=map_renderer,
                    )
                elapsed_seconds = time.time() - started_at
                result["elapsed_s"] = round(elapsed_seconds, 1)
            except Exception as exc:  # pragma: no cover - exercised via callers
                elapsed_seconds = time.time() - started_at
                result = {
                    "status": "error",
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "elapsed_s": round(elapsed_seconds, 1),
                }
            if quiet_stderr_buffer is not None:
                diagnostic_lines.extend(
                    _quiet_diagnostic_lines(quiet_stderr_buffer.getvalue())
                )

            result["task_dir"] = task_name
            result["traj_idx"] = traj_idx
            result["traj_file"] = str(traj_file)
            result["layout"] = layout
            result["style"] = style
            result["seed"] = seed
            results.append(result)
            if progress_reporter is not None:
                progress_reporter(
                    {
                        "status": result["status"],
                        "layout": layout,
                        "style": style,
                        "seed": seed,
                    }
                )
            log_lines.append(
                _format_run_log_line(
                    run_num=run_num,
                    total_runs=total_runs,
                    task_name=task_name,
                    traj_idx=traj_idx,
                    combo_label=combo_label,
                    result=result,
                )
            )

    return {
        "diagnostic_lines": diagnostic_lines,
        "entry_index": entry_index,
        "results": results,
        "log_lines": log_lines,
    }


def _build_trajectory_crash_results(
    entry: dict[str, Any],
    *,
    entry_index: int,
    total_runs: int,
    combos: tuple[tuple[int, int, int], ...],
    error_message: str,
    traceback_text: str,
) -> dict[str, Any]:
    """Build error records when a worker fails before returning any run data."""

    task_name = str(entry["task_dir_name"])
    traj_idx = int(entry["traj_idx"])
    traj_file = Path(entry["traj_file"])
    entry_combos = (
        (tuple(entry["scene_combo"]),)
        if entry.get("scene_combo") is not None else combos
    )
    combo_count = len(entry_combos)
    results: list[dict[str, Any]] = []
    log_lines: list[str] = []

    for combo_offset, (layout, style, seed) in enumerate(entry_combos):
        run_num = entry_index * combo_count + combo_offset + 1
        combo_label = _resolve_combo_label(
            combo_count=combo_count,
            layout=layout,
            style=style,
            seed=seed,
        )
        result = {
            "status": "error",
            "error": error_message,
            "traceback": traceback_text,
            "elapsed_s": 0.0,
            "task_dir": task_name,
            "traj_idx": traj_idx,
            "traj_file": str(traj_file),
            "layout": layout,
            "style": style,
            "seed": seed,
        }
        results.append(result)
        log_lines.append(
            _format_run_log_line(
                run_num=run_num,
                total_runs=total_runs,
                task_name=task_name,
                traj_idx=traj_idx,
                combo_label=combo_label,
                result=result,
            )
        )

    return {
        "entry_index": entry_index,
        "results": results,
        "log_lines": log_lines,
    }


def get_gpu_allocation(
    num_workers: int,
    gpu_ids: list[int] | None,
    procs_per_gpu: list[int] | None = None,
) -> list[int] | None:
    """Resolve one GPU assignment per concurrent worker slot."""

    if gpu_ids is None:
        if procs_per_gpu is not None:
            raise ValueError("--procs-per-gpu requires --gpu-ids.")
        return None
    if not gpu_ids:
        raise ValueError("--gpu-ids must include at least one GPU id.")
    if num_workers <= 0:
        return []

    if procs_per_gpu is None:
        return [gpu_ids[i % len(gpu_ids)] for i in range(num_workers)]

    if len(procs_per_gpu) != len(gpu_ids):
        raise ValueError("--procs-per-gpu must have the same length as --gpu-ids.")

    adjusted_counts = list(procs_per_gpu)
    total_allocated = sum(adjusted_counts)
    if total_allocated < num_workers:
        raise ValueError(
            f"Sum of --procs-per-gpu ({total_allocated}) must be at least the "
            f"number of concurrent workers ({num_workers})."
        )

    while total_allocated > num_workers:
        gpu_index = max(range(len(adjusted_counts)), key=adjusted_counts.__getitem__)
        adjusted_counts[gpu_index] -= 1
        total_allocated -= 1

    gpu_allocation: list[int] = []
    for gpu_id, worker_count in zip(gpu_ids, adjusted_counts):
        gpu_allocation.extend([gpu_id] * worker_count)
    return gpu_allocation


def execute_sweep(
    entries: list[dict[str, Any]],
    *,
    combos: tuple[tuple[int, int, int], ...],
    output_root: Path,
    workers: int,
    robots: int,
    placement: str,
    cell_size: float,
    robot_spawn: str = "sim",
    skip_videos: bool = True,
    executor_factory: Callable[..., Any] | None = None,
    show_progress: bool = True,
    log_run_completions: bool = True,
    suppress_run_stdout: bool = False,
    progress_factory: Callable[..., Any] | None = None,
    gpu_ids: list[int] | None = None,
    procs_per_gpu: list[int] | None = None,
    max_tasks_per_child: int | None = None,
    gl_backend: str = "osmesa",
    render_width: int = 512,
    render_height: int = 512,
    map_dpi: int = DEFAULT_MAP_DPI,
    map_renderer: str = DEFAULT_MAP_RENDERER,
    cancellation_controller: SweepCancellationController | None = None,
) -> list[dict[str, Any]]:
    """Execute the discovered trajectories and preserve summary ordering."""

    total_runs = len(entries) * len(combos)
    if not entries or total_runs == 0:
        return []

    ordered_results: list[list[dict[str, Any]] | None] = [None] * len(entries)
    combo_count = len(combos)
    max_workers = min(workers, len(entries)) if entries else 0
    progress_worker_count = max_workers
    gpu_allocation = get_gpu_allocation(max_workers, gpu_ids, procs_per_gpu)
    progress_display = None
    slot_completed_runs: dict[int, int] = {}
    sweep_executor_factory = executor_factory or ProcessPoolExecutor
    process_pool_context = None
    active_executor = None
    if sweep_executor_factory is ProcessPoolExecutor:
        # Reused MuJoCo / rendering workers are more stable under a spawned
        # multiprocessing context than the Linux default forked context.
        process_pool_context = multiprocessing.get_context("spawn")

    try:
        if show_progress and total_runs > 0:
            progress_display = _create_sweep_progress_display(
                total_runs=total_runs,
                worker_count=progress_worker_count,
                progress_factory=progress_factory,
            )
        log_writer = (
            progress_display.write_log_line if progress_display is not None else None
        )

        if workers == 1 or len(entries) == 1:
            worker_slot = 0
            for entry_index, entry in enumerate(entries):
                _assign_progress_worker(
                    progress_display,
                    slot_completed_runs,
                    worker_slot=worker_slot,
                    entry=entry,
                    combo_count=combo_count,
                )
                trajectory_result = run_trajectory_entry(
                    entry,
                    entry_index=entry_index,
                    total_runs=total_runs,
                    combos=combos,
                    output_root=output_root,
                    robots=robots,
                    placement=placement,
                    cell_size=cell_size,
                    robot_spawn=robot_spawn,
                    skip_videos=skip_videos,
                    suppress_stdout=suppress_run_stdout,
                    gpu_id=gpu_allocation[worker_slot] if gpu_allocation else None,
                    gl_backend=gl_backend,
                    render_width=render_width,
                    render_height=render_height,
                    map_dpi=map_dpi,
                    map_renderer=map_renderer,
                    progress_reporter=(
                        partial(
                            _record_local_progress_event,
                            progress_display,
                            slot_completed_runs,
                            worker_slot,
                        )
                        if progress_display is not None
                        else None
                    ),
                )
                ordered_results[entry_index] = trajectory_result["results"]
                _finalize_progress_worker(
                    progress_display,
                    slot_completed_runs,
                    worker_slot=worker_slot,
                    results=trajectory_result["results"],
                )
                _emit_trajectory_log_lines(
                    trajectory_result,
                    log_writer=log_writer,
                    log_run_completions=log_run_completions,
                )
        else:
            with contextlib.ExitStack() as exit_stack:
                progress_queue = None
                progress_manager = None
                if progress_display is not None:
                    progress_queue, progress_manager = _create_progress_event_queue(
                        executor_factory=sweep_executor_factory,
                        mp_context=process_pool_context,
                    )
                    if progress_manager is not None:
                        exit_stack.enter_context(progress_manager)
                executor_kwargs = {"max_workers": max_workers}
                if process_pool_context is not None:
                    executor_kwargs["mp_context"] = process_pool_context
                    # Recycling worker processes can reclaim simulator memory
                    # between trajectory entries when the caller prefers lower
                    # peak RAM over maximum reuse speed.
                    if max_tasks_per_child is not None:
                        executor_kwargs["max_tasks_per_child"] = max_tasks_per_child
                with sweep_executor_factory(**executor_kwargs) as executor:
                    active_executor = executor
                    if cancellation_controller is not None:
                        cancellation_controller.attach_executor(executor)
                    try:
                        pending_entries = iter(enumerate(entries))
                        future_to_context: dict[
                            Any, tuple[int, int, dict[str, Any]]
                        ] = {}

                        def submit_entry(worker_slot: int) -> bool:
                            """Schedules the next entry on the requested worker slot."""

                            try:
                                entry_index, entry = next(pending_entries)
                            except StopIteration:
                                return False
                            _assign_progress_worker(
                                progress_display,
                                slot_completed_runs,
                                worker_slot=worker_slot,
                                entry=entry,
                                combo_count=combo_count,
                            )
                            progress_reporter = None
                            if progress_queue is not None:
                                progress_reporter = partial(
                                    _enqueue_progress_event,
                                    progress_queue,
                                    worker_slot,
                                )
                            future = executor.submit(
                                run_trajectory_entry,
                                entry,
                                entry_index=entry_index,
                                total_runs=total_runs,
                                combos=combos,
                                output_root=output_root,
                                robots=robots,
                                placement=placement,
                                cell_size=cell_size,
                                robot_spawn=robot_spawn,
                                skip_videos=skip_videos,
                                suppress_stdout=suppress_run_stdout,
                                gpu_id=(
                                    gpu_allocation[worker_slot]
                                    if gpu_allocation is not None
                                    else None
                                ),
                                gl_backend=gl_backend,
                                render_width=render_width,
                                render_height=render_height,
                                map_dpi=map_dpi,
                                map_renderer=map_renderer,
                                progress_reporter=progress_reporter,
                            )
                            future_to_context[future] = (
                                entry_index,
                                worker_slot,
                                entry,
                            )
                            return True

                        for worker_slot in range(max_workers):
                            if not submit_entry(worker_slot):
                                break

                        while future_to_context:
                            _drain_progress_events(
                                progress_queue,
                                progress_display,
                                slot_completed_runs,
                            )
                            done, _ = wait(
                                tuple(future_to_context),
                                timeout=0.1,
                                return_when=FIRST_COMPLETED,
                            )
                            if not done:
                                continue
                            _drain_progress_events(
                                progress_queue,
                                progress_display,
                                slot_completed_runs,
                            )
                            for future in done:
                                entry_index, worker_slot, entry = future_to_context.pop(
                                    future
                                )
                                try:
                                    trajectory_result = future.result()
                                except (
                                    Exception
                                ) as exc:  # pragma: no cover - defensive path
                                    trajectory_result = _build_trajectory_crash_results(
                                        entry,
                                        entry_index=entry_index,
                                        total_runs=total_runs,
                                        combos=combos,
                                        error_message=str(exc),
                                        traceback_text=traceback.format_exc(),
                                    )

                                _drain_progress_events(
                                    progress_queue,
                                    progress_display,
                                    slot_completed_runs,
                                )
                                ordered_results[entry_index] = trajectory_result[
                                    "results"
                                ]
                                _finalize_progress_worker(
                                    progress_display,
                                    slot_completed_runs,
                                    worker_slot=worker_slot,
                                    results=trajectory_result["results"],
                                )
                                _emit_trajectory_log_lines(
                                    trajectory_result,
                                    log_writer=log_writer,
                                    log_run_completions=log_run_completions,
                                )
                                submit_entry(worker_slot)
                        _drain_progress_events(
                            progress_queue,
                            progress_display,
                            slot_completed_runs,
                        )
                    finally:
                        if cancellation_controller is not None:
                            cancellation_controller.detach_executor(executor)
                active_executor = None
    except KeyboardInterrupt:
        _shutdown_sweep_executor(active_executor, cancel_running=True)
        raise
    finally:
        if progress_display is not None:
            progress_display.close()
        clear_executor_cache()

    flattened_results: list[dict[str, Any]] = []
    for entry_results in ordered_results:
        if entry_results is None:
            continue
        flattened_results.extend(entry_results)
    return flattened_results


def _load_sweep_summary(output_root: Path) -> dict:
    summary_path = output_root / "sweep_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"No sweep_summary.json in {output_root}")
    with open(summary_path) as f:
        return json.load(f)


def _iter_completed_runs(output_root: Path):
    summary = _load_sweep_summary(output_root)
    combo_count = summary.get("scene_combos", 1)

    for run_result in summary["results"]:
        if run_result.get("status") != "ok":
            continue

        task_dir = run_result["task_dir"]
        traj_idx = run_result["traj_idx"]
        layout = run_result["layout"]
        style = run_result["style"]
        seed = run_result["seed"]

        # When called with a task-level dir (output_root.name == task_dir),
        # the task_dir component is already baked in — don't add it again.
        task_root = output_root if output_root.name == task_dir else output_root / task_dir

        if combo_count == 1:
            run_dir = task_root / f"traj_{traj_idx:06d}"
        else:
            run_dir = task_root / f"traj_{traj_idx:06d}" / f"L{layout}_S{style}_sd{seed}"

        meta_path = run_dir / "trajectory_execution_metadata.json"
        if not meta_path.exists():
            continue

        with open(meta_path) as f:
            metadata = json.load(f)

        episode_id = f"{task_dir}/traj_{traj_idx:06d}/L{layout}_S{style}_sd{seed}"
        task_name = metadata.get("composite_task") or metadata.get("task") or task_dir
        rel_run_dir = run_dir.relative_to(output_root)

        yield {
            "episode_id": episode_id,
            "task": task_name,
            "task_dir": task_dir,
            "traj_idx": traj_idx,
            "layout": layout,
            "style": style,
            "seed": seed,
            "run_dir": run_dir,
            "run_dir_rel": str(rel_run_dir),
            "metadata": metadata,
            "adapted_trajectory_path": str(rel_run_dir / "adapted_trajectory.json"),
            "original_trajectory_path": str(rel_run_dir / "original_trajectory.json"),
            "execution_metadata_path": str(
                rel_run_dir / "trajectory_execution_metadata.json"
            ),
        }


def iter_sweep_metadata_paths(output_root: Path):
    """Yield repo-relative JSON artifact paths that should accompany the dataset."""

    seen = {Path("sweep_summary.json")}
    yield output_root / "sweep_summary.json", "sweep_summary.json"

    for run in _iter_completed_runs(output_root):
        for key in (
            "adapted_trajectory_path",
            "original_trajectory_path",
            "execution_metadata_path",
        ):
            rel_path = Path(run[key])
            if rel_path in seen:
                continue
            abs_path = output_root / rel_path
            if not abs_path.exists():
                continue
            seen.add(rel_path)
            yield abs_path, str(rel_path)


def upload_sweep_metadata_files(repo_id: str, output_root: Path) -> None:
    """Upload referenced episode JSON files alongside the parquet dataset."""

    from huggingface_hub import HfApi

    HfApi().upload_folder(
        repo_id=repo_id,
        repo_type="dataset",
        folder_path=output_root,
        allow_patterns=[
            "sweep_summary.json",
            "**/adapted_trajectory.json",
            "**/original_trajectory.json",
            "**/trajectory_execution_metadata.json",
        ],
        commit_message="Upload sweep metadata files",
    )


def _resolve_step_images(
    image_paths: list[str] | None, image_columns: list[str]
) -> dict[str, dict | None]:
    images = {col: None for col in image_columns}
    for img_path_str in image_paths or []:
        img_path = Path(img_path_str)
        if not img_path.exists() and img_path.suffix == ".png":
            img_path = img_path.with_suffix(".jpg")
        if not img_path.exists():
            continue
        fname = img_path.stem
        for view_token in image_columns:
            if f"_{view_token}_" in f"_{fname}_":
                images[view_token] = {"bytes": img_path.read_bytes(), "path": None}
                break
    return images


def _read_compact_json(path: Path) -> str:
    if not path.exists():
        return ""
    with open(path) as f:
        return json.dumps(json.load(f), separators=(",", ":"))


def _build_step_level_dataset(output_root: Path) -> "datasets.Dataset":
    """Convert sweep output directory into a flat step-level dataset."""

    from datasets import Dataset, Features, Image as HFImage, Value

    image_columns = [
        "room_view",
        "top_view",
        "map",
        "agentview_center",
        "agentview_left",
        "agentview_right",
        "wrist",
    ]

    rows: list[dict[str, Any]] = []
    for run in _iter_completed_runs(output_root):
        metadata = run["metadata"]
        num_steps = len(metadata.get("steps", []))

        for step in metadata.get("steps", []):
            step_idx = step.get("step_index", 0)
            tool = step.get("tool", "")
            robot_idx = step.get("robot_idx", 0)
            args = step.get("args", {})
            success = step.get("success", False)
            images = _resolve_step_images(step.get("image_paths"), image_columns)

            args_clean = {k: v for k, v in args.items() if k != "image_paths"}

            rows.append(
                {
                    "episode_id": run["episode_id"],
                    "task": run["task"],
                    "task_dir": run["task_dir"],
                    "layout": run["layout"],
                    "style": run["style"],
                    "seed": run["seed"],
                    "num_steps": num_steps,
                    "run_dir": run["run_dir_rel"],
                    "adapted_trajectory_path": run["adapted_trajectory_path"],
                    "original_trajectory_path": run["original_trajectory_path"],
                    "execution_metadata_path": run["execution_metadata_path"],
                    "step_index": step_idx,
                    "tool_name": tool,
                    "tool_args": json.dumps(args_clean, separators=(",", ":")),
                    "robot_idx": robot_idx,
                    "success": success,
                    **images,
                }
            )

    features = Features(
        {
            "episode_id": Value("string"),
            "task": Value("string"),
            "task_dir": Value("string"),
            "layout": Value("int32"),
            "style": Value("int32"),
            "seed": Value("int32"),
            "num_steps": Value("int32"),
            "run_dir": Value("string"),
            "adapted_trajectory_path": Value("string"),
            "original_trajectory_path": Value("string"),
            "execution_metadata_path": Value("string"),
            "step_index": Value("int32"),
            "tool_name": Value("string"),
            "tool_args": Value("string"),
            "robot_idx": Value("int32"),
            "success": Value("bool"),
            **{col: HFImage() for col in image_columns},
        }
    )

    ds = Dataset.from_list(rows, features=features)
    print(
        f"Built dataset: {len(ds)} step rows across {len(set(ds['episode_id']))} episodes"
    )
    return ds


def _build_trajectory_run_row(
    run: dict[str, Any],
    image_columns: list[str],
) -> dict[str, Any]:
    """Build one trajectory-level dataset row from a completed run."""

    metadata = run["metadata"]
    row: dict[str, Any] = {
        "episode_id": run["episode_id"],
        "task": run["task"],
        "task_dir": run["task_dir"],
        "layout": run["layout"],
        "style": run["style"],
        "seed": run["seed"],
        "num_steps": len(metadata.get("steps", [])),
        "run_dir": run["run_dir_rel"],
        "adapted_trajectory": _read_compact_json(
            run["run_dir"] / "adapted_trajectory.json"
        ),
        "original_trajectory": _read_compact_json(
            run["run_dir"] / "original_trajectory.json"
        ),
        "execution_metadata": json.dumps(metadata, separators=(",", ":")),
        "step_index": [],
        "tool_name": [],
        "tool_args": [],
        "robot_idx": [],
        "success": [],
        **{col: [] for col in image_columns},
    }
    for step in metadata.get("steps", []):
        images = _resolve_step_images(step.get("image_paths"), image_columns)
        args_clean = {
            k: v for k, v in (step.get("args") or {}).items() if k != "image_paths"
        }
        row["step_index"].append(step.get("step_index", 0))
        row["tool_name"].append(step.get("tool", ""))
        row["tool_args"].append(json.dumps(args_clean, separators=(",", ":")))
        row["robot_idx"].append(step.get("robot_idx", 0))
        row["success"].append(step.get("success", False))
        for col in image_columns:
            row[col].append(images[col])
    return row


def _build_trajectory_level_dataset(
    output_root: Path,
    *,
    num_proc: int = 1,
) -> "datasets.Dataset":
    """Convert sweep output directory into a trajectory-level dataset."""

    from concurrent.futures import ThreadPoolExecutor
    from functools import partial

    from datasets import Dataset, Features, Image as HFImage, Sequence, Value

    image_columns = [
        "room_view",
        "top_view",
        "map",
        "agentview_center",
        "agentview_left",
        "agentview_right",
        "wrist",
    ]

    runs = list(_iter_completed_runs(output_root))
    build_row = partial(_build_trajectory_run_row, image_columns=image_columns)

    if num_proc > 1:
        with ThreadPoolExecutor(max_workers=num_proc) as pool:
            rows = list(pool.map(build_row, runs))
    else:
        rows = [build_row(run) for run in runs]

    features = Features(
        {
            "episode_id": Value("string"),
            "task": Value("string"),
            "task_dir": Value("string"),
            "layout": Value("int32"),
            "style": Value("int32"),
            "seed": Value("int32"),
            "num_steps": Value("int32"),
            "run_dir": Value("string"),
            "adapted_trajectory": Value("large_string"),
            "original_trajectory": Value("large_string"),
            "execution_metadata": Value("large_string"),
            "step_index": Sequence(Value("int32")),
            "tool_name": Sequence(Value("string")),
            "tool_args": Sequence(Value("string")),
            "robot_idx": Sequence(Value("int32")),
            "success": Sequence(Value("bool")),
            **{col: Sequence(HFImage()) for col in image_columns},
        }
    )

    ds = Dataset.from_list(rows, features=features)
    print(f"Built dataset: {len(ds)} trajectory rows")
    return ds


def sweep_output_to_dataset(
    output_root: Path,
    *,
    row_granularity: str = "trajectory",
    num_proc: int = 1,
) -> "datasets.Dataset":
    """Convert sweep output into a dataset with configurable row granularity."""

    if row_granularity == "step":
        return _build_step_level_dataset(output_root)
    if row_granularity == "trajectory":
        return _build_trajectory_level_dataset(output_root, num_proc=num_proc)
    raise ValueError(f"Unsupported row granularity: {row_granularity}")


def build_dataset_card(
    repo_id: str,
    ds: "datasets.Dataset",
    *,
    row_granularity: str = "trajectory",
) -> str:
    """Build a readable HuggingFace dataset card."""

    tasks = sorted(set(ds["task"]))
    episode_ids = ds["episode_id"]
    episodes = len(set(episode_ids))
    if row_granularity == "step":
        avg_steps = len(ds) / max(episodes, 1)
        intro = (
            "This dataset contains flat RoboCasa step rows with sidecar episode JSON."
        )
        row_text = "Each row is one tool step."
        episode_json_text = textwrap.dedent(
            """\
            Episode-level JSON is not duplicated into parquet. Instead, each row
            carries repo-relative references:

            - `adapted_trajectory_path`
            - `original_trajectory_path`
            - `execution_metadata_path`
            """
        ).strip()
        notes_tail = (
            "- Episode JSON sidecars are available in the repo files at the paths "
            "referenced by `*_path` columns."
        )
    else:
        avg_steps = sum(ds["num_steps"]) / max(len(ds), 1)
        intro = "This dataset contains one row per RoboCasa trajectory / episode."
        row_text = "Each row is one trajectory / episode."
        episode_json_text = textwrap.dedent(
            """\
            Episode-level JSON is stored inline:

            - `adapted_trajectory`
            - `original_trajectory`
            - `execution_metadata`

            Step-level data is stored in aligned sequence columns:

            - `step_index`
            - `tool_name`
            - `tool_args`
            - `robot_idx`
            - `success`
            """
        ).strip()
        notes_tail = (
            "- This layout is self-contained under `load_dataset()`, but nested "
            "sequence columns are less friendly for the HF table viewer."
        )
    task_list = ", ".join(tasks) if tasks else "Unknown"
    return textwrap.dedent(
        f"""\
        ---
        pretty_name: RoboCasa Trajectories Single
        configs:
        - config_name: default
          data_files:
          - split: train
            path: data/train-*
        ---

        # RoboCasa Trajectories Single

        {intro}

        ## Structure

        {row_text}

        {episode_json_text}

        Image columns stay inline and viewable in the dataset table:

        - `room_view`
        - `top_view`
        - `map`
        - `agentview_center`
        - `agentview_left`
        - `agentview_right`
        - `wrist`

        ## Summary

        - rows: {len(ds)}
        - episodes: {episodes}
        - tasks: {task_list}
        - average steps per episode: {avg_steps:.1f}

        ## Load

        ```python
        from datasets import load_dataset

        ds = load_dataset("{repo_id}", split="train")
        ```

        ## Notes

        - Camera renders are stored as JPEG. Maps remain PNG.
        - MP4 videos are not included in the dataset.
        - Debug `initial` / `pre_initial_state` camera frames are not part of the dataset.
        - Row granularity: `{row_granularity}`.
        {notes_tail}
        """
    )


def upload_dataset_card(
    repo_id: str,
    ds: "datasets.Dataset",
    *,
    row_granularity: str = "trajectory",
) -> None:
    """Overwrite the auto-generated Hub README with a readable dataset card."""

    from io import BytesIO

    from huggingface_hub import HfApi

    HfApi().upload_file(
        path_or_fileobj=BytesIO(
            build_dataset_card(repo_id, ds, row_granularity=row_granularity).encode(
                "utf-8"
            )
        ),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Update dataset card",
    )


def main() -> None:
    load_dotenv_file()

    parser = argparse.ArgumentParser(
        description=(
            "Sweep trajectories through the sim executor and optionally publish "
            "the dataset."
        ),
        epilog=CLI_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--input-dir", type=str, required=True, help="Dataset root dir")
    parser.add_argument("--output-dir", type=str, required=True, help="Output root dir")
    parser.add_argument(
        "--summary-path",
        type=str,
        default=None,
        help=(
            "Optional path for the sweep summary JSON. Defaults to "
            "<output-dir>/sweep_summary.json."
        ),
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=None,
        help="Filter to specific task dir names",
    )
    parser.add_argument(
        "--indices",
        type=int,
        nargs="+",
        default=None,
        help="Filter to specific traj indices",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=None,
        help=(
            "Optional number of deterministic round-robin shards to split the "
            "discovered trajectories across."
        ),
    )
    parser.add_argument(
        "--shard-index",
        type=int,
        default=None,
        help=(
            "Zero-based shard index to run when --num-shards is set. Useful for "
            "multi-node launches."
        ),
    )
    parser.add_argument(
        "--layouts",
        type=int,
        nargs="+",
        default=[11],
        help="Kitchen layout ids for --scene-compatibility-cache none (default: 11)",
    )
    parser.add_argument(
        "--styles",
        type=int,
        nargs="+",
        default=[34],
        help="Kitchen style ids for --scene-compatibility-cache none (default: 34)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="Environment seeds for --scene-compatibility-cache none (default: 42)",
    )
    parser.add_argument(
        "--scene-compatibility-cache",
        default=str(REPO_ROOT / "configs/eval/scene_compatibility_cache_v2.json"),
        help=(
            "Render each trajectory in one certified scene, sampled from the "
            "scenes this cache lists as compatible with its task and initial "
            "configuration (how the dataset was rendered). Pass 'none' to "
            "render every trajectory in each --layouts x --styles x --seeds "
            "combination instead."
        ),
    )
    parser.add_argument("--scene-sampling-seed", type=int, default=20260819)
    parser.add_argument("--robots", type=int, default=2)
    parser.add_argument("--placement", choices=["grid"], default="grid")
    parser.add_argument("--cell-size", type=float, default=0.05)
    parser.add_argument(
        "--render-width",
        type=int,
        default=512,
        help=(
            "Offscreen render width in pixels. Lower values reduce VRAM and "
            "render time."
        ),
    )
    parser.add_argument(
        "--render-height",
        type=int,
        default=512,
        help=(
            "Offscreen render height in pixels. Lower values reduce VRAM and "
            "render time."
        ),
    )
    parser.add_argument(
        "--map-dpi",
        type=int,
        default=DEFAULT_MAP_DPI,
        help=(
            "Placement-map DPI (default: 60, matching live evaluation). "
            "The former legacy path used 300 DPI and produced 6000x4800 maps."
        ),
    )
    parser.add_argument(
        "--map-renderer",
        choices=("legacy", "raster"),
        default=DEFAULT_MAP_RENDERER,
        help="Placement-map renderer (default: raster, matching live evaluation).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Maximum parallel trajectory workers (default: 1).",
    )
    parser.add_argument(
        "--max-tasks-per-child",
        type=int,
        default=None,
        help=(
            "Optional number of trajectory entries to run before replacing a "
            "worker process. Use this to reclaim simulator RAM between runs."
        ),
    )
    parser.add_argument(
        "--gpu-ids",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional GPU IDs to assign across concurrent workers. When set, "
            "workers are distributed across these GPUs."
        ),
    )
    parser.add_argument(
        "--procs-per-gpu",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Optional worker counts per GPU. Must match --gpu-ids in length and "
            "sum to at least the number of concurrent workers."
        ),
    )
    parser.add_argument(
        "--gl-backend",
        choices=["osmesa", "egl"],
        default="egl",
        help="OpenGL backend: 'egl' (headless GPU rendering) or 'osmesa' (CPU).",
    )
    parser.add_argument(
        "--robot-spawn",
        choices=["sim", "trajectory"],
        default="trajectory",
        help=(
            "Robot initial placement source. 'sim': all robots at "
            "init_robot_base_ref. 'trajectory' (default): each robot at its "
            "trajectory location."
        ),
    )
    parser.add_argument(
        "--row-granularity",
        choices=["step", "trajectory"],
        default="trajectory",
        help="Dataset row shape when exporting or pushing (default: trajectory)",
    )
    parser.add_argument(
        "--videos",
        action="store_true",
        help="Record per-camera MP4 videos for each run",
    )
    parser.add_argument(
        "--push-to-hub",
        type=str,
        default=None,
        metavar="REPO_ID",
        help=(
            "Push dataset to HuggingFace Hub "
            "(e.g. 'username/robocasa-trajectories')"
        ),
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print what would run without executing"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Show the run progress bar but suppress normal informational output.",
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_root = Path(args.output_dir)
    summary_path = (
        Path(args.summary_path)
        if args.summary_path is not None
        else output_root / "sweep_summary.json"
    )

    if args.workers <= 0:
        parser.error("--workers must be greater than 0.")
    if args.max_tasks_per_child is not None and args.max_tasks_per_child <= 0:
        parser.error("--max-tasks-per-child must be greater than 0.")
    if args.render_width <= 0:
        parser.error("--render-width must be greater than 0.")
    if args.render_height <= 0:
        parser.error("--render-height must be greater than 0.")
    if args.map_dpi <= 0:
        parser.error("--map-dpi must be greater than 0.")
    if (args.num_shards is None) != (args.shard_index is None):
        parser.error("--num-shards and --shard-index must be provided together.")
    if args.num_shards is not None and args.num_shards <= 0:
        parser.error("--num-shards must be greater than 0.")
    if args.num_shards is not None and (
        args.shard_index < 0 or args.shard_index >= args.num_shards
    ):
        parser.error("--shard-index must be in [0, --num-shards).")

    os.environ["ROBOCASA_SWEEP_VERBOSE"] = "0" if args.quiet else "1"

    discovered_entries = discover_trajectories(
        input_dir, task_filter=args.tasks, indices=args.indices
    )
    if not discovered_entries:
        print("No trajectories found.", file=sys.stderr, flush=True)
        sys.exit(1)
    try:
        entries = select_trajectory_shard(
            discovered_entries,
            num_shards=args.num_shards,
            shard_index=args.shard_index,
        )
    except ValueError as exc:
        parser.error(str(exc))

    combos = list(itertools.product(args.layouts, args.styles, args.seeds))
    if str(args.scene_compatibility_cache).lower() == "none":
        args.scene_compatibility_cache = None
    if args.scene_compatibility_cache is not None:
        from robotalk.scene_sampling import (
            load_compatibility_cache,
            sample_compatible_scene,
        )
        cache = load_compatibility_cache(args.scene_compatibility_cache)
        for entry in entries:
            trajectory = json.loads(Path(entry["traj_file"]).read_text())
            physical_signature = trajectory.get(
                "physical_configuration_signature"
            )
            if not physical_signature:
                parser.error(
                    f"{entry['traj_file']} lacks physical_configuration_signature; "
                    "certified random-scene rendering requires new-format data"
                )
            task = trajectory.get("composite_task") or trajectory.get("task")
            scene = sample_compatible_scene(
                cache,
                task=task,
                physical_configuration_signature=physical_signature,
                sampling_seed=args.scene_sampling_seed,
                sample_index=int(entry["traj_idx"]),
            )
            entry["scene_combo"] = (
                scene["layout"], scene["style"], scene["seed"]
            )
        # Each entry now carries exactly one independently sampled scene.
        combos = [(0, 0, 0)]
    total_runs = len(entries) * len(combos)
    concurrent_workers = min(args.workers, len(entries))
    resolved_gl_backend = args.gl_backend or ("egl" if args.gpu_ids else "osmesa")
    if args.gpu_ids is not None and resolved_gl_backend != "egl":
        parser.error("--gpu-ids requires --gl-backend egl.")
    try:
        gpu_allocation = get_gpu_allocation(
            concurrent_workers,
            args.gpu_ids,
            args.procs_per_gpu,
        )
    except ValueError as exc:
        parser.error(str(exc))

    if not args.quiet:
        print(
            f"Found {len(entries)} trajectories x {len(combos)} scene combos = {total_runs} runs"
        )
        if args.num_shards is not None:
            print(
                "Running shard "
                f"{args.shard_index + 1}/{args.num_shards} "
                f"from {len(discovered_entries)} discovered trajectories"
            )
        if args.workers > 1:
            print(f"Using {concurrent_workers} trajectory workers")
        if args.max_tasks_per_child is not None:
            print(
                "Recycling worker processes after "
                f"{args.max_tasks_per_child} trajectory entries"
            )
        if gpu_allocation is not None:
            print(f"Using GL backend: {resolved_gl_backend}")
            print(f"GPU allocation by worker slot: {gpu_allocation}")
        print(f"Render size: {args.render_width}x{args.render_height}")
        print(f"Map rendering: {args.map_renderer}@{args.map_dpi}dpi")
        if len(combos) > 1:
            print(f"  layouts: {args.layouts}")
            print(f"  styles:  {args.styles}")
            print(f"  seeds:   {args.seeds}")
        print()

    if args.dry_run:
        if not args.quiet:
            for entry in entries:
                entry_combos = (
                    [tuple(entry["scene_combo"])]
                    if entry.get("scene_combo") is not None else combos
                )
                for layout, style, seed in entry_combos:
                    task_name = entry["task_dir_name"]
                    traj_idx = entry["traj_idx"]
                    out, combo_label = _resolve_run_output_dir(
                        output_root,
                        task_name=task_name,
                        traj_idx=traj_idx,
                        combo_count=len(combos),
                        layout=layout,
                        style=style,
                        seed=seed,
                    )
                    if combo_label:
                        print(f"  {task_name}/traj_{traj_idx:06d}{combo_label} -> {out}")
                    else:
                        print(f"  {task_name}/traj_{traj_idx:06d} -> {out}")
            print(f"\n{total_runs} runs (dry run, nothing executed)")
        return

    cancellation_controller = SweepCancellationController()
    with _install_sweep_signal_handlers(cancellation_controller):
        results = execute_sweep(
            entries,
            combos=tuple(combos),
            output_root=output_root,
            workers=args.workers,
            robots=args.robots,
            placement=args.placement,
            cell_size=args.cell_size,
            robot_spawn=args.robot_spawn,
            skip_videos=not args.videos,
            log_run_completions=not args.quiet,
            show_progress=True,
            suppress_run_stdout=args.quiet,
            gpu_ids=args.gpu_ids,
            procs_per_gpu=args.procs_per_gpu,
            max_tasks_per_child=args.max_tasks_per_child,
            gl_backend=resolved_gl_backend,
            render_width=args.render_width,
            render_height=args.render_height,
            map_dpi=args.map_dpi,
            map_renderer=args.map_renderer,
            cancellation_controller=cancellation_controller,
        )

    output_root.mkdir(parents=True, exist_ok=True)
    summary = {
        "input_dir": str(input_dir),
        "layouts": args.layouts,
        "styles": args.styles,
        "seeds": args.seeds,
        "scene_combos": len(combos),
        "trajectories": len(entries),
        "total": len(results),
        "succeeded": sum(1 for r in results if r["status"] == "ok"),
        "failed": sum(1 for r in results if r["status"] == "error"),
        "gl_backend": resolved_gl_backend,
        "gpu_ids": args.gpu_ids,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "max_tasks_per_child": args.max_tasks_per_child,
        "procs_per_gpu": args.procs_per_gpu,
        "render_width": args.render_width,
        "render_height": args.render_height,
        "map_dpi": args.map_dpi,
        "map_renderer": args.map_renderer,
        "results": results,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"Done: {summary['succeeded']}/{summary['total']} succeeded")
    print(f"Summary: {summary_path}")

    if summary["failed"] > 0:
        print("\nFailed runs:")
        for r in results:
            if r["status"] == "error":
                print(
                    f"  {r['task_dir']}/traj_{r['traj_idx']:06d} "
                    f"L{r['layout']}/S{r['style']}/sd{r['seed']}: {r['error']}"
                )

    if args.push_to_hub:
        if not args.quiet:
            print("\nConverting sweep output to HuggingFace dataset...")
        ds = sweep_output_to_dataset(output_root, row_granularity=args.row_granularity)
        if not args.quiet:
            print(f"Pushing to {args.push_to_hub}...")
        ds.push_to_hub(args.push_to_hub)
        if args.row_granularity == "step":
            if not args.quiet:
                print("Uploading sweep metadata sidecars...")
            upload_sweep_metadata_files(args.push_to_hub, output_root)
        if not args.quiet:
            print("Uploading dataset card...")
        upload_dataset_card(args.push_to_hub, ds, row_granularity=args.row_granularity)
        if not args.quiet:
            print(
                "Done! Dataset pushed to "
                f"https://huggingface.co/datasets/{args.push_to_hub}"
            )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
