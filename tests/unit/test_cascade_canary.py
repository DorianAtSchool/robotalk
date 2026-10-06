from robotalk.generation.raw.cascade_canary import _direct_prompt
from robotalk.tasks.shared.prompting import format_concurrency_facts


def test_direct_prompt_carries_deduplicated_validator_errors():
    history = [
        {"is_valid": False, "error_type": "LocationError", "error": "wrong place"},
        {"is_valid": False, "error_type": "LocationError", "error": "wrong place"},
        {"is_valid": False, "error_type": "WaitError", "error": "missing request"},
    ]

    prompt = _direct_prompt(
        "base rules",
        validation_history=history,
    )

    assert prompt.count("LocationError: wrong place") == 1
    assert prompt.count("WaitError: missing request") == 1
    assert "factual errors, not a required work assignment" in prompt
    assert "Validated feasibility example" not in prompt


def test_direct_prompt_without_history_is_unchanged():
    assert _direct_prompt("base rules") == "base rules"


def test_concurrency_facts_name_exact_exclusive_and_shared_fixture_ids():
    facts = format_concurrency_facts(
        {
            "fixtures": {
                "stove": {"fixture_type": "stove"},
                "fridge": {"fixture_type": "fridge"},
                "counter": {"fixture_type": "counter"},
            }
        }
    )

    assert "Exclusive, one agent at a time: fridge, stove" in facts
    assert "Roomy/shared for simultaneous work on different objects: counter" in facts
