"""Define mutable task-level symbolic runtime state containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentRuntimeState:
    """Tracks the mutable symbolic state for one agent during FSM validation."""

    location: str | None
    held_object: str | None


@dataclass
class TaskRuntimeState:
    """Tracks shared mutable symbolic state while the FSM walks each trajectory."""

    agents: dict[str, AgentRuntimeState]  # Current symbolic state for each agent.
    objects: dict[str, dict[str, Any]]  # Mutable symbolic state for movable objects.
    fixtures: dict[
        str, dict[str, Any]
    ]  # Mutable symbolic state for fixtures and parts.
    machine_state: dict[str, dict[str, Any]]  # Task-local machine or appliance flags.
    communicated_agents: set[str] = field(
        default_factory=set
    )  # Agents that have coordinated so far.
    public_state: dict[str, Any] = field(
        default_factory=dict
    )  # Extra task-specific summary fields for output.
