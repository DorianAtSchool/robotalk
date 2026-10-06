"""Implement seed-driven structured random sampling over base trajectories."""

from __future__ import annotations

import random
import re
from typing import TYPE_CHECKING

from robotalk.generation.sampling.base import (
    BaseSamplingStrategy,
    SampledTrajectoryCandidate,
)
from robotalk.utils import stable_json_sha256

if TYPE_CHECKING:
    from robotalk.generation.raw.config import RuntimeConfig
    from robotalk.tasks import TaskDefinition
    from robotalk.tasks.shared.types import TaskInstance


STRUCTURED_RANDOM_RANKS = ("low", "medium", "high")
STRUCTURED_RANDOM_FIELDS = (
    "communication_message_length",
    "communication_message_complexity",
    "tool_call_diversity",
)


class StructuredRandomSamplingStrategy(BaseSamplingStrategy):
    """Uses base sampling with a code-selected random seed and configuration."""

    name = "structured_random"

    def build_prompt(
        self,
        *,
        task_definition: TaskDefinition,
        runtime_config: RuntimeConfig,
        task_instance: TaskInstance,
        variation_key: str,
        retry_feedback: str | None = None,
    ) -> str:
        """Builds a base prompt conditioned on a structured random configuration."""

        base_prompt = super().build_prompt(
            task_definition=task_definition,
            runtime_config=runtime_config,
            task_instance=task_instance,
            variation_key=variation_key,
            retry_feedback=retry_feedback,
        )
        _, configuration = _structured_random_seed_and_configuration(
            task_definition=task_definition,
            variation_key=variation_key,
        )
        configuration_text = ", ".join(
            f"{field_name}={configuration[field_name]}"
            for field_name in STRUCTURED_RANDOM_FIELDS
        )
        return (
            f"Structured random configuration: {configuration_text}\n\n"
            "Use the fixed configuration above for this single trajectory.\n"
            "- Return exactly one trajectory using the normal single-trajectory "
            "JSON format requested below.\n"
            "- Do not return a samples array, a responses array, or multiple "
            "trajectory candidates.\n"
            "- communication_message_length controls whether communicate "
            "messages are terse, sentence-length, or detailed.\n"
            "- communication_message_complexity controls whether communicate "
            "messages are simple status/request messages, include task "
            "constraints, or include multi-part rationale and contingency "
            "planning.\n"
            "- tool_call_diversity controls whether the trajectory uses a "
            "narrow, moderate, or broad task-relevant mix of allowed "
            "non-communication tools.\n"
            "- Keep all tool calls task-relevant; do not add unnecessary "
            "actions just to increase diversity.\n\n"
            f"{base_prompt}"
        )

    def extract_candidates(
        self,
        *,
        raw_response: object,
        task_definition: TaskDefinition,
        runtime_config: RuntimeConfig,
        variation_key: str | None = None,
    ) -> list[SampledTrajectoryCandidate]:
        """Parses one candidate and attaches its selected configuration."""

        sampled_candidates = super().extract_candidates(
            raw_response=raw_response,
            task_definition=task_definition,
            runtime_config=runtime_config,
            variation_key=variation_key,
        )
        if variation_key is None:
            return sampled_candidates
        seed, configuration = _structured_random_seed_and_configuration(
            task_definition=task_definition,
            variation_key=variation_key,
        )
        attempt_match = re.search(r"-attempt-(\d+)$", variation_key)
        attempt_number = int(attempt_match.group(1)) + 1 if attempt_match else None
        return [
            SampledTrajectoryCandidate(
                candidate=sampled_candidate.candidate,
                raw_output=sampled_candidate.raw_output,
                probability=sampled_candidate.probability,
                sampling_configuration=configuration,
                sampling_seed=seed,
                sampling_attempt_number=attempt_number,
            )
            for sampled_candidate in sampled_candidates
        ]


def _structured_random_seed_and_configuration(
    *,
    task_definition: TaskDefinition,
    variation_key: str,
) -> tuple[int, dict[str, str]]:
    """Selects a reproducible seed and configuration for one run attempt."""

    seed_hex = stable_json_sha256(
        {
            "sampling": "structured_random",
            "task": task_definition.composite_task,
            # The variation key contains both run and retry-attempt identity.
            # Including the full key prevents a difficult configuration from
            # being pinned across every retry while preserving reproducibility.
            "run_attempt": variation_key,
        }
    )[:16]
    seed = int(seed_hex, 16)
    rng = random.Random(seed)
    configuration = {
        field_name: rng.choice(STRUCTURED_RANDOM_RANKS)
        for field_name in STRUCTURED_RANDOM_FIELDS
    }
    return seed, configuration
