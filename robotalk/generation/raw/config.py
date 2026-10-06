"""Define shared generation configuration and package-wide constants."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# This module lives under robotalk/generation/raw, so the repository root is
# three parents above this file.
REPO_ROOT = Path(__file__).resolve().parents[3]
TRAJECTORY_ID_DIGITS = 6
INTERRUPTED_EXIT_CODE = 130
INTERRUPTED_MESSAGE = (
    "Interrupted. Exiting immediately. Queued trajectories were cancelled; "
    "requests already in flight may still be billed."
)


@dataclass(frozen=True)
class RuntimeConfig:
    """Settings for one demonstration-generation request."""

    composite_task: str | None
    num_runs: int
    model: str
    sdk: str
    project: str | None
    location: str
    temperature: float
    random_start_location: bool = True
    # Deterministically sample open/closed states for eligible cabinets,
    # refrigerators, and drawers. Eligibility is inferred from symbolic state
    # plus an exact opening tool; appliance/object state is never randomized.
    random_access_state: bool = False
    sampling: str = "structured_random"
    thinking_level: str | None = None
