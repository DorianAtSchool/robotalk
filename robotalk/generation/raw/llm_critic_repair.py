"""Experimental validator-gated LLM critic -> trajectory repair canary."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

from robotalk.generation.raw.config import RuntimeConfig
from robotalk.generation.raw.runtime_support import (
    _sampling_strategy_for_runtime,
    _unwrap_generation_response,
    _validate_candidate,
    format_trajectory_variation_key,
)
from robotalk.generation.runtime.client import build_generation_client
from robotalk.tasks import get_task_definition


CRITIC_SCHEMA = {
    "type": "object",
    "properties": {
        "mistakes": {"type": "array", "items": {"type": "string"}},
        "requirements_for_repair": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["mistakes", "requirements_for_repair"],
}


def _compact_validation_history(
    validations: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Keep cumulative failure evidence without repeating bulky state snapshots."""

    history: list[dict[str, Any]] = []
    for attempt_number, validation in enumerate(validations, start=1):
        history.append(
            {
                "attempt_number": attempt_number,
                "error_type": validation.get("error_type"),
                "error": validation.get("error"),
                "step": validation.get("step"),
                "error_details": {
                    key: value
                    for key, value in (validation.get("error_details") or {}).items()
                    if key not in {"final_state", "state_before_step"}
                },
            }
        )
    return history


def _validator_protocol_facts(validation: dict[str, Any]) -> list[str]:
    """Translate protocol error classes into authoritative temporal facts."""

    error_type = str(validation.get("error_type") or "")
    error_text = str(validation.get("error") or "")
    facts: list[str] = []
    if error_type == "WaitSignalSemanticValidationError":
        facts.extend(
            [
                "A waiter must communicate its request to the current holder on the tick immediately before it calls wait_for_signal.",
                "For an occupied exclusive fixture, the waiter calls wait_for_signal while the holder calls give_space; entering during that give_space tick is forbidden.",
                "The holder communicates releases=<the exact waited-for id> on a later tick, while the waiter remains blocked.",
                "The waiter may enter or use the released fixture only on a tick after the matching release.",
                "There is no idle or no-op tool; use only a real allowed tool call or an explicit blocked marker for an already-blocked agent.",
            ]
        )
    if error_type == "ResourceConflictSemanticValidationError":
        facts.append(
            "An exclusive fixture has at most one occupant, and a give_space departure takes effect only after its atomic tick finishes."
        )
        if "give_space" in error_text:
            facts.append(
                "A partner must not navigate to or use the fixture during the holder's give_space tick; entry is possible only on a later tick."
            )
    return facts


def _critic_prompt(
    *,
    rules: str,
    candidate: dict[str, Any],
    validation: dict[str, Any],
    validation_history: list[dict[str, Any]] | None = None,
) -> str:
    history = _compact_validation_history(validation_history or [validation])
    protocol_facts = _validator_protocol_facts(validation)
    return (
        "You are auditing a two-agent robot trajectory. Treat the validator error "
        "and the rules as authoritative. Find every mistake relevant to repairing "
        "the trajectory, including downstream consequences. Do not write a replacement "
        "trajectory and do not invent task facts. Return JSON only.\n\n"
        f"Authoritative generation rules and task state:\n{rules}\n\n"
        "Rejected trajectory:\n"
        f"{json.dumps(candidate, indent=2, sort_keys=True)}\n\n"
        "Cumulative validator error history (oldest to newest):\n"
        f"{json.dumps(history, indent=2, sort_keys=True)}\n\n"
        "Authoritative validator-derived protocol facts for the newest error:\n"
        f"{json.dumps(protocol_facts, indent=2, sort_keys=True)}\n\n"
        "Validator result:\n"
        f"{json.dumps(validation, indent=2, sort_keys=True)}"
    )


def _repair_prompt(
    *,
    rules: str,
    candidate: dict[str, Any],
    validation: dict[str, Any],
    critique: Any,
    validation_history: list[dict[str, Any]] | None = None,
) -> str:
    history = _compact_validation_history(validation_history or [validation])
    protocol_facts = _validator_protocol_facts(validation)
    return (
        f"{rules}\n\n"
        "An independent LLM critic reviewed a rejected attempt. Its report is advisory; "
        "the authoritative rules and validator still control acceptance. Reconsider the "
        "whole schedule, correct every genuine mistake, and return one COMPLETE replacement "
        "trajectory from tick 0. Do not return a patch, diagnosis, or markdown.\n\n"
        "Rejected trajectory:\n"
        f"{json.dumps(candidate, indent=2, sort_keys=True)}\n\n"
        "Validator result:\n"
        f"{json.dumps(validation, indent=2, sort_keys=True)}\n\n"
        "Cumulative validator error history (oldest to newest):\n"
        f"{json.dumps(history, indent=2, sort_keys=True)}\n\n"
        "Authoritative validator-derived protocol facts for the newest error:\n"
        f"{json.dumps(protocol_facts, indent=2, sort_keys=True)}\n\n"
        "Critic report:\n"
        f"{json.dumps(critique, indent=2, sort_keys=True)}"
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = RuntimeConfig(
        composite_task=args.task,
        num_runs=args.num_runs,
        model=args.model,
        sdk="google-genai",
        project=None,
        location=args.location,
        temperature=args.temperature,
        random_start_location=True,
        random_access_state=True,
        sampling="structured_random",
        thinking_level=args.thinking_level,
    )
    definition = get_task_definition(args.task)
    instance = definition.build_task_instance(args.run_index, config)
    validator = definition.validator_factory(instance)
    strategy = _sampling_strategy_for_runtime(config)
    client = build_generation_client(
        sdk=config.sdk,
        project=config.project,
        location=config.location,
        timeout_sec=600,
    )
    source = json.loads(Path(args.rejected_json).read_text(encoding="utf-8"))
    candidate = source["candidate"]
    validation = source["validation"]
    validation_history = [validation]
    attempts: list[dict[str, Any]] = []

    for repair_index in range(args.max_repairs):
        variation_key = format_trajectory_variation_key(
            args.run_index, args.variation_offset + repair_index
        )
        rules = strategy.build_prompt(
            task_definition=definition,
            runtime_config=config,
            task_instance=instance,
            variation_key=variation_key,
        )
        critic_raw = client.generate(
            model=args.model,
            prompt=_critic_prompt(
                rules=rules,
                candidate=candidate,
                validation=validation,
                validation_history=validation_history,
            ),
            response_schema=CRITIC_SCHEMA,
            temperature=args.temperature,
            thinking_level=args.thinking_level,
        )
        critique, critic_usage = _unwrap_generation_response(critic_raw)
        repair_raw = client.generate(
            model=args.model,
            prompt=_repair_prompt(
                rules=rules,
                candidate=candidate,
                validation=validation,
                critique=critique,
                validation_history=validation_history,
            ),
            response_schema=strategy.response_schema(
                task_definition=definition, runtime_config=config
            ),
            temperature=args.temperature,
            thinking_level=args.thinking_level,
        )
        repair_payload, repair_usage = _unwrap_generation_response(repair_raw)
        extracted = strategy.extract_candidates(
            raw_response=repair_payload,
            task_definition=definition,
            runtime_config=config,
            variation_key=variation_key,
        )
        candidate = extracted[0].candidate
        validation, normalized = _validate_candidate(
            candidate, validator, enforce_validation=False
        )
        validation_history.append(validation)
        attempts.append(
            {
                "repair_number": repair_index + 1,
                "critique": critique,
                "critic_usage": asdict(critic_usage) if critic_usage else None,
                "candidate": candidate,
                "validation": validation,
                "repair_usage": asdict(repair_usage) if repair_usage else None,
            }
        )
        if validation.get("is_valid") is True:
            candidate = normalized
            break

    result = {
        "task": args.task,
        "run_index": args.run_index,
        "model": args.model,
        "is_valid": validation.get("is_valid") is True,
        "final_candidate": candidate,
        "final_validation": validation,
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
    parser.add_argument("--num-runs", type=int, default=12)
    parser.add_argument("--rejected-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default="gemini-3-flash-preview")
    parser.add_argument("--location", default="global")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--thinking-level", default="high")
    parser.add_argument("--max-repairs", type=int, default=3)
    parser.add_argument("--variation-offset", type=int, default=100)
    args = parser.parse_args()
    result = run(args)
    print(json.dumps({k: result[k] for k in ("task", "run_index", "is_valid")}))
    return 0 if result["is_valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
