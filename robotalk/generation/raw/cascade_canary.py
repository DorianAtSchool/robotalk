"""Run the bounded low/temperature/medium/critic/Pro generation cascade."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from robotalk.generation.raw.config import RuntimeConfig
from robotalk.generation.raw.llm_critic_repair import (
    CRITIC_SCHEMA,
    _critic_prompt,
    _repair_prompt,
)
from robotalk.generation.raw.runtime_support import (
    _sampling_strategy_for_runtime,
    _unwrap_generation_response,
    _validate_candidate,
    format_trajectory_variation_key,
)
from robotalk.generation.runtime.client import build_generation_client
from robotalk.tasks import get_task_definition


FLASH_MODEL = "gemini-3-flash-preview"
PRO_MODEL = "gemini-3.1-pro-preview"


def _config(
    args: argparse.Namespace,
    *,
    model: str,
    thinking: str,
    temperature: float | None = None,
) -> RuntimeConfig:
    return RuntimeConfig(
        composite_task=args.task,
        num_runs=args.num_runs,
        model=model,
        sdk="google-genai",
        project=None,
        location=args.location,
        temperature=args.temperature if temperature is None else temperature,
        random_start_location=True,
        random_access_state=True,
        sampling="structured_random",
        thinking_level=thinking,
    )


def _usage(value: Any) -> dict[str, Any] | None:
    return asdict(value) if value is not None else None


def _validation_error(exc: Exception) -> dict[str, Any]:
    return {
        "is_valid": False,
        "error_type": type(exc).__name__,
        "error": str(exc),
    }


def _direct_prompt(
    rules: str,
    *,
    validation_history: list[dict[str, Any]] | None = None,
) -> str:
    sections = [rules]
    if validation_history:
        # Carry facts, not speculative repairs, through every later cascade
        # stage.  Deduplicating repeated failures keeps the prompt compact while
        # still preventing a temperature/model switch from forgetting an error.
        observations: list[str] = []
        seen: set[tuple[str, str]] = set()
        for validation in validation_history:
            if validation.get("is_valid") is True:
                continue
            error_type = str(validation.get("error_type") or "ValidationError")
            error = str(validation.get("error") or "candidate was rejected")
            key = (error_type, error)
            if key in seen:
                continue
            seen.add(key)
            observations.append(f"- {error_type}: {error}")
        if observations:
            sections.append(
                "Prior validator observations from this same configuration. "
                "Avoid these rejected patterns; they are factual errors, not a "
                "required work assignment:\n" + "\n".join(observations)
            )
    return "\n\n".join(sections)


def run(args: argparse.Namespace) -> dict[str, Any]:
    low = _config(args, model=FLASH_MODEL, thinking="low")
    low_cool = _config(args, model=FLASH_MODEL, thinking="low", temperature=0.3)
    low_hot = _config(args, model=FLASH_MODEL, thinking="low", temperature=0.9)
    medium = _config(args, model=FLASH_MODEL, thinking="medium")
    critic = _config(args, model=FLASH_MODEL, thinking="high")
    pro = _config(args, model=PRO_MODEL, thinking="high")
    definition = get_task_definition(args.task)
    instance = definition.build_task_instance(args.run_index, low)
    validator = definition.validator_factory(instance)
    strategy = _sampling_strategy_for_runtime(low)
    client = build_generation_client(
        sdk="google-genai", project=None, location=args.location, timeout_sec=600
    )
    attempts: list[dict[str, Any]] = []
    latest_candidate: dict[str, Any] | None = None
    latest_validation: dict[str, Any] = {
        "is_valid": False,
        "error_type": "NoAttemptYet",
        "error": "No generation attempt has run.",
    }
    validation_history: list[dict[str, Any]] = []
    accepted: dict[str, Any] | None = None
    accepted_raw: dict[str, Any] | None = None
    accepted_stage: str | None = None
    ordinal = 0

    def direct(stage: str, config: RuntimeConfig) -> bool:
        nonlocal ordinal, latest_candidate, latest_validation, accepted, accepted_raw, accepted_stage
        variation_key = format_trajectory_variation_key(args.run_index, ordinal)
        ordinal += 1
        rules = strategy.build_prompt(
            task_definition=definition,
            runtime_config=config,
            task_instance=instance,
            variation_key=variation_key,
        )
        try:
            raw = client.generate(
                model=config.model,
                prompt=_direct_prompt(
                    rules,
                    validation_history=validation_history,
                ),
                response_schema=strategy.response_schema(
                    task_definition=definition, runtime_config=config
                ),
                temperature=config.temperature,
                thinking_level=config.thinking_level,
            )
            payload, usage = _unwrap_generation_response(raw)
            sampled = strategy.extract_candidates(
                raw_response=payload,
                task_definition=definition,
                runtime_config=config,
                variation_key=variation_key,
            )[0]
            latest_candidate = sampled.candidate
            latest_validation, normalized = _validate_candidate(
                latest_candidate, validator, enforce_validation=False
            )
            validation_history.append(latest_validation)
            attempts.append(
                {
                    "stage": stage,
                    "model": config.model,
                    "thinking_level": config.thinking_level,
                    "variation_key": variation_key,
                    "usage": _usage(usage),
                    "validation": latest_validation,
                }
            )
            if latest_validation.get("is_valid") is True:
                accepted, accepted_raw, accepted_stage = normalized, latest_candidate, stage
                return True
        except Exception as exc:
            latest_validation = _validation_error(exc)
            validation_history.append(latest_validation)
            attempts.append(
                {
                    "stage": stage,
                    "model": config.model,
                    "thinking_level": config.thinking_level,
                    "variation_key": variation_key,
                    "validation": latest_validation,
                }
            )
        return False

    def critic_repair() -> bool:
        """Critique the latest failure, then regenerate a complete trajectory."""

        nonlocal ordinal, latest_candidate, latest_validation, accepted, accepted_raw, accepted_stage
        if latest_candidate is None:
            return False
        variation_key = format_trajectory_variation_key(args.run_index, ordinal)
        ordinal += 1
        rules = strategy.build_prompt(
            task_definition=definition,
            runtime_config=low,
            task_instance=instance,
            variation_key=variation_key,
        )
        rejected_candidate = latest_candidate
        rejected_validation = latest_validation
        try:
            critic_raw = client.generate(
                model=critic.model,
                prompt=_critic_prompt(
                    rules=rules,
                    candidate=rejected_candidate,
                    validation=rejected_validation,
                    validation_history=validation_history,
                ),
                response_schema=CRITIC_SCHEMA,
                temperature=critic.temperature,
                thinking_level=critic.thinking_level,
            )
            critique, critic_usage = _unwrap_generation_response(critic_raw)
            repair_raw = client.generate(
                model=low.model,
                prompt=_repair_prompt(
                    rules=rules,
                    candidate=rejected_candidate,
                    validation=rejected_validation,
                    critique=critique,
                    validation_history=validation_history,
                ),
                response_schema=strategy.response_schema(
                    task_definition=definition, runtime_config=low
                ),
                temperature=low.temperature,
                thinking_level=low.thinking_level,
            )
            payload, repair_usage = _unwrap_generation_response(repair_raw)
            sampled = strategy.extract_candidates(
                raw_response=payload,
                task_definition=definition,
                runtime_config=low,
                variation_key=variation_key,
            )[0]
            latest_candidate = sampled.candidate
            latest_validation, normalized = _validate_candidate(
                latest_candidate, validator, enforce_validation=False
            )
            validation_history.append(latest_validation)
            attempts.append(
                {
                    "stage": "flash_critic_assisted_regeneration",
                    "critic_model": critic.model,
                    "critic_thinking_level": critic.thinking_level,
                    "generator_model": low.model,
                    "generator_thinking_level": low.thinking_level,
                    "variation_key": variation_key,
                    "critique": critique,
                    "critic_usage": _usage(critic_usage),
                    "usage": _usage(repair_usage),
                    "validation": latest_validation,
                }
            )
            if latest_validation.get("is_valid") is True:
                accepted = normalized
                accepted_raw = latest_candidate
                accepted_stage = "flash_critic_assisted_regeneration"
                return True
        except Exception as exc:
            latest_validation = _validation_error(exc)
            validation_history.append(latest_validation)
            attempts.append(
                {
                    "stage": "flash_critic_assisted_regeneration",
                    "critic_model": critic.model,
                    "critic_thinking_level": critic.thinking_level,
                    "generator_model": low.model,
                    "generator_thinking_level": low.thinking_level,
                    "variation_key": variation_key,
                    "validation": latest_validation,
                }
            )
        return False

    for _ in range(3):
        if direct("flash_low_direct", low):
            break

    if accepted is None and not args.low_direct_only:
        direct("flash_low_temp_0.3", low_cool)
    if accepted is None and not args.low_direct_only:
        direct("flash_low_temp_0.9", low_hot)

    if accepted is None and not args.low_direct_only:
        for _ in range(2):
            if direct("flash_medium_direct", medium):
                break

    if accepted is None and not args.low_direct_only:
        for _ in range(2):
            if critic_repair():
                break
    if accepted is None and not args.low_direct_only:
        for _ in range(2):
            if direct("pro_high_direct", pro):
                break

    result = {
        "task": args.task,
        "run_index": args.run_index,
        "is_valid": accepted is not None,
        "accepted_stage": accepted_stage,
        "accepted_raw_candidate": accepted_raw,
        "accepted_candidate": accepted,
        "latest_candidate": latest_candidate,
        "final_validation": latest_validation,
        "attempts": attempts,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--run-index", type=int, required=True)
    parser.add_argument("--num-runs", type=int, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--location", default="global")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--low-direct-only", action="store_true")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({k: result[k] for k in ("task", "run_index", "is_valid", "accepted_stage")}))
    return 0 if result["is_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
