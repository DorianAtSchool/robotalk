"""Shared rich progress-bar components used by the render sweep."""

from __future__ import annotations

from typing import Any

try:
    from rich.console import Console
    from rich.text import Text
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        ProgressColumn,
        Progress as RichProgress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
    )
except ImportError:  # pragma: no cover
    Console = None
    BarColumn = None
    MofNCompleteColumn = None
    ProgressColumn = None
    RichProgress = None
    SpinnerColumn = None
    TaskProgressColumn = None
    TextColumn = None
    Text = None
    TimeElapsedColumn = None

OVERALL_PROGRESS_COLOR = "cyan"
PROGRESS_BAR_WIDTH = 30
_RICH_PROGRESS_COLUMN_BASE = ProgressColumn if ProgressColumn is not None else object


class StaticQueuedTimeElapsedColumn(_RICH_PROGRESS_COLUMN_BASE):
    """Shows a fixed zero timer for queued tasks and real elapsed time once started."""

    def __init__(self) -> None:
        """Initializes the shared queued-aware elapsed column."""

        if TimeElapsedColumn is None or ProgressColumn is None:
            raise RuntimeError("rich progress support is unavailable")
        super().__init__()
        self._delegate = TimeElapsedColumn()

    def render(self, task: Any) -> Any:
        """Renders zero elapsed time until the task leaves the queued state."""

        if getattr(task, "fields", {}).get("queued", False):
            return Text("0:00:00")
        return self._delegate.render(task)


class RichTaskProgressAdapter:
    """Adapts one Rich progress task to the shared progress-bar interface."""

    def __init__(
        self,
        progress: RichProgress,
        task_id: int,
        total: int,
        *,
        started: bool = True,
        status: str = "",
    ):
        """Caches task state so Rich resets preserve custom fields like status."""

        self._progress = progress
        self._task_id = task_id
        self._total = total
        self._completed = 0
        self._started = started
        self._status = status

    def start(self) -> None:
        """Starts elapsed-time tracking only when generation actually begins."""

        if self._started:
            return
        self._progress.reset(
            self._task_id,
            start=True,
            total=self._total,
            completed=0,
            queued=False,
            status=self._status,
        )
        self._started = True

    @property
    def total(self) -> int:
        return self._total

    @total.setter
    def total(self, value: int) -> None:
        self._total = value
        self._completed = min(self._completed, value)
        self._progress.update(
            self._task_id,
            total=value,
            completed=self._completed,
        )

    def update(self, amount: int = 1) -> None:
        self.start()
        self._completed += amount
        self._progress.advance(self._task_id, amount)

    def set_postfix_str(self, text: str) -> None:
        self._status = text
        self._progress.update(self._task_id, status=text)

    def refresh(self) -> None:
        self._progress.refresh()

    def complete(self, total: int) -> None:
        """Marks the task complete at the requested total."""

        self.start()
        self._total = total
        self._progress.update(
            self._task_id,
            total=total,
            completed=total,
        )

    def close(self) -> None:
        return

