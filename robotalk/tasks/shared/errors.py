"""Define shared task-level validation error types."""

from __future__ import annotations

from typing import Any


class TrajectoryValidationError(ValueError):
    """Base class for task-level validation failures."""

    def __init__(
        self,
        message: str,
        *,
        step: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        """Stores optional step and machine-readable details for one failure."""

        super().__init__(message)
        self.step = step
        self.details = dict(details or {})

    @property
    def error_type(self) -> str:
        """Returns the concrete exception class name used for reporting."""

        return type(self).__name__

    @property
    def error_base_type(self) -> str:
        """Returns the top-level validation family for aggregation."""

        for cls in type(self).mro():
            if TrajectoryValidationError in cls.__bases__:
                return cls.__name__
        return self.error_type

    def with_step(self, step: int | None) -> TrajectoryValidationError:
        """Clones the exception with a step number when one is missing."""

        if self.step is not None or step is None:
            return self
        return type(self)(str(self), step=step, details=self.details)


class ResponseFormatValidationError(TrajectoryValidationError):
    """Raised when the model output does not match the JSON contract."""


class DuplicateTrajectoryValidationError(TrajectoryValidationError):
    """Raised when a candidate duplicates an existing saved trajectory."""


class InsufficientValidUniqueTrajectoriesValidationError(TrajectoryValidationError):
    """Raised when a multi-sample run yields too few distinct valid trajectories."""


class InsufficientValidUniqueTrajectoriesInvalidError(
    InsufficientValidUniqueTrajectoriesValidationError
):
    """Raised when too many multi-sample candidates fail validation checks."""


class InsufficientValidUniqueTrajectoriesDuplicateError(
    InsufficientValidUniqueTrajectoriesValidationError
):
    """Raised when too many multi-sample candidates duplicate existing trajectories."""


class InsufficientValidUniqueTrajectoriesMixedError(
    InsufficientValidUniqueTrajectoriesValidationError
):
    """Raised when multi-sample candidates fail from both invalid and duplicate causes."""


class TrajectoryStructureValidationError(TrajectoryValidationError):
    """Raised when a candidate fails structural schema-like checks."""


class TaskSemanticValidationError(TrajectoryValidationError):
    """Raised when a candidate violates task semantics or FSM transitions."""


class UnexpectedStepIndexSemanticValidationError(TaskSemanticValidationError):
    """Raised when step numbering diverges from the replay order."""


class PostGoalActionSemanticValidationError(TaskSemanticValidationError):
    """Raised when a non-observation action appears after the goal is reached."""


class MissingInitialCommunicationSemanticValidationError(TaskSemanticValidationError):
    """Raised when agents skip required coordination before acting."""


class UnsupportedToolSemanticValidationError(TaskSemanticValidationError):
    """Raised when a trajectory uses a tool outside the task tool registry."""


class MissingTaskActionSemanticValidationError(TaskSemanticValidationError):
    """Raised when a trajectory never performs a real task action."""


class UnsatisfiedGoalSemanticValidationError(TaskSemanticValidationError):
    """Raised when replay ends before the task goal is satisfied."""


class CommunicationStepSemanticValidationError(TaskSemanticValidationError):
    """Raised when a communicate step uses invalid symbolic arguments."""


class WaitSignalSemanticValidationError(TaskSemanticValidationError):
    """Raised when a wait_for_signal step is malformed or never released."""


class ObservationSequenceSemanticValidationError(TaskSemanticValidationError):
    """Raised when required observation bracketing is missing."""


class ObjectStateSemanticValidationError(TaskSemanticValidationError):
    """Raised when an object's symbolic state conflicts with the action."""


class HeldObjectSemanticValidationError(TaskSemanticValidationError):
    """Raised when held-object constraints are violated."""


class NavigationSemanticValidationError(TaskSemanticValidationError):
    """Raised when an agent acts at the wrong symbolic location."""


class ToolArgumentSemanticValidationError(TaskSemanticValidationError):
    """Raised when a tool argument is missing, malformed, or disallowed."""


class PlacementDestinationSemanticValidationError(TaskSemanticValidationError):
    """Raised when a placement tool omits its symbolic destination."""


class TaskPreconditionSemanticValidationError(TaskSemanticValidationError):
    """Raised when task-local symbolic preconditions are not met."""


class ResourceConflictSemanticValidationError(TaskSemanticValidationError):
    """Raised when two agents hold the same resource at the same instant.

    This is what a missing wait actually looks like once the plan is run on a
    clock rather than read down the page: nothing is out of order in the file,
    but the two streams reach the same object or the same exclusive fixture
    while neither has been told to hold off.
    """


class DeadlockSemanticValidationError(TaskSemanticValidationError):
    """Raised when every remaining agent is blocked on a wait nothing releases."""
