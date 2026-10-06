"""Define the shared sampling strategy interface and base sampling behavior."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from robotalk.tasks import TaskDefinition
from robotalk.tasks.shared.types import PreflightTokenEstimate

if TYPE_CHECKING:
    from robotalk.generation.raw.config import RuntimeConfig
    from robotalk.tasks.shared.types import TaskInstance


@dataclass(frozen=True)
class SampledTrajectoryCandidate:
    """Stores one parsed candidate trajectory emitted by a sampling strategy."""

    candidate: dict[str, Any]
    raw_output: Any
    probability: float | None = None
    sampling_configuration: dict[str, str] | None = None
    sampling_seed: int | None = None
    sampling_attempt_number: int | None = None


class SamplingStrategy(Protocol):
    """Protocol implemented by task-level response sampling strategies."""

    name: str

    def trajectories_per_run(self, runtime_config: RuntimeConfig) -> int:
        """Returns how many saved trajectories one successful run yields."""

    def build_prompt(
        self,
        *,
        task_definition: TaskDefinition,
        runtime_config: RuntimeConfig,
        task_instance: TaskInstance,
        variation_key: str,
        retry_feedback: str | None = None,
    ) -> str:
        """Builds the prompt for one run."""

    def response_schema(
        self,
        *,
        task_definition: TaskDefinition,
        runtime_config: RuntimeConfig,
    ) -> dict[str, Any]:
        """Builds the expected model response schema for one run."""

    def preflight_token_estimate(
        self,
        *,
        task_definition: TaskDefinition,
        runtime_config: RuntimeConfig,
    ) -> PreflightTokenEstimate:
        """Returns the manual token estimate for one run."""

    def extract_candidates(
        self,
        *,
        raw_response: Any,
        task_definition: TaskDefinition,
        runtime_config: RuntimeConfig,
        variation_key: str | None = None,
    ) -> list[SampledTrajectoryCandidate]:
        """Parses one model response into saved candidate trajectories."""


class BaseSamplingStrategy:
    """Preserves the existing single-trajectory generation contract."""

    name = "base"

    def trajectories_per_run(self, runtime_config: RuntimeConfig) -> int:
        """Returns the single saved trajectory emitted by base sampling."""

        return 1

    def build_prompt(
        self,
        *,
        task_definition: TaskDefinition,
        runtime_config: RuntimeConfig,
        task_instance: TaskInstance,
        variation_key: str,
        retry_feedback: str | None = None,
    ) -> str:
        """Builds the unchanged task prompt for one base-sampled run."""

        return task_definition.build_prompt(
            variation_key,
            task_instance=task_instance,
            retry_feedback=retry_feedback,
        )

    def response_schema(
        self,
        *,
        task_definition: TaskDefinition,
        runtime_config: RuntimeConfig,
    ) -> dict[str, Any]:
        """Uses the task's tick response schema with explicit blocked markers."""

        del runtime_config
        tick_schema = task_definition.explicit_blocked_tick_response_schema
        if tick_schema is not None:
            return tick_schema
        return task_definition.response_schema

    def preflight_token_estimate(
        self,
        *,
        task_definition: TaskDefinition,
        runtime_config: RuntimeConfig,
    ) -> PreflightTokenEstimate:
        """Uses the task's original manual token estimate."""

        del runtime_config
        return task_definition.preflight_token_estimate

    def extract_candidates(
        self,
        *,
        raw_response: Any,
        task_definition: TaskDefinition,
        runtime_config: RuntimeConfig,
        variation_key: str | None = None,
    ) -> list[SampledTrajectoryCandidate]:
        """Parses the unchanged single-trajectory response payload."""

        del task_definition
        del runtime_config
        del variation_key
        # Reuse the shared JSON extraction helper so base sampling inherits the
        # same fence-stripping behavior as the runtime validation path.
        from robotalk.generation.raw.runtime_support import (
            extract_json_candidate,
        )

        candidate = extract_json_candidate(raw_response)
        return [
            SampledTrajectoryCandidate(candidate=candidate, raw_output=raw_response)
        ]
