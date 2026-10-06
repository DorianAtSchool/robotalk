"""Register task definitions and expose task-level validation error types."""

from __future__ import annotations

from robotalk.tasks.shared.errors import (
    CommunicationStepSemanticValidationError,
    DuplicateTrajectoryValidationError,
    HeldObjectSemanticValidationError,
    InsufficientValidUniqueTrajectoriesDuplicateError,
    InsufficientValidUniqueTrajectoriesInvalidError,
    InsufficientValidUniqueTrajectoriesMixedError,
    InsufficientValidUniqueTrajectoriesValidationError,
    MissingInitialCommunicationSemanticValidationError,
    MissingTaskActionSemanticValidationError,
    NavigationSemanticValidationError,
    ObjectStateSemanticValidationError,
    ObservationSequenceSemanticValidationError,
    PlacementDestinationSemanticValidationError,
    PostGoalActionSemanticValidationError,
    ResponseFormatValidationError,
    TaskPreconditionSemanticValidationError,
    TaskSemanticValidationError,
    ToolArgumentSemanticValidationError,
    TrajectoryStructureValidationError,
    TrajectoryValidationError,
    UnexpectedStepIndexSemanticValidationError,
    UnsupportedToolSemanticValidationError,
    UnsatisfiedGoalSemanticValidationError,
)
from robotalk.tasks.shared.types import TaskDefinition, TaskValidator

__all__ = [
    "CommunicationStepSemanticValidationError",
    "DuplicateTrajectoryValidationError",
    "HeldObjectSemanticValidationError",
    "InsufficientValidUniqueTrajectoriesDuplicateError",
    "InsufficientValidUniqueTrajectoriesInvalidError",
    "InsufficientValidUniqueTrajectoriesMixedError",
    "InsufficientValidUniqueTrajectoriesValidationError",
    "MissingInitialCommunicationSemanticValidationError",
    "MissingTaskActionSemanticValidationError",
    "NavigationSemanticValidationError",
    "ObjectStateSemanticValidationError",
    "ObservationSequenceSemanticValidationError",
    "PlacementDestinationSemanticValidationError",
    "PostGoalActionSemanticValidationError",
    "ResponseFormatValidationError",
    "TaskPreconditionSemanticValidationError",
    "TaskDefinition",
    "TaskSemanticValidationError",
    "TaskValidator",
    "ToolArgumentSemanticValidationError",
    "TrajectoryStructureValidationError",
    "TrajectoryValidationError",
    "UnexpectedStepIndexSemanticValidationError",
    "UnsupportedToolSemanticValidationError",
    "UnsatisfiedGoalSemanticValidationError",
    "get_task_definition",
    "supported_task_names",
]


def _get_task_registry() -> dict[str, TaskDefinition]:
    from robotalk.tasks.specs.runtime import SPEC_TASK_REGISTRY

    return dict(SPEC_TASK_REGISTRY)


def get_task_definition(composite_task: str) -> TaskDefinition | None:
    return _get_task_registry().get(composite_task)


def supported_task_names() -> tuple[str, ...]:
    return tuple(sorted(_get_task_registry()))
