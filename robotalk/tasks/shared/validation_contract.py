"""One fail-closed validity contract for every trajectory consumer."""

from __future__ import annotations

from typing import Any, Mapping


VALIDATOR_CONTRACT_VERSION = "transactional_tick_v3_native_goal_parity"


def current_validation_error(trajectory: Mapping[str, Any]) -> str | None:
    """Return why a trajectory is unusable, or ``None`` when currently valid."""

    validation = trajectory.get("validation")
    if not isinstance(validation, Mapping):
        return "missing validation metadata"
    if validation.get("is_valid") is not True:
        return f"validation.is_valid is {validation.get('is_valid')!r}, not true"
    actual_version = validation.get("validator_contract_version")
    if actual_version != VALIDATOR_CONTRACT_VERSION:
        return (
            "validator contract is stale or missing: "
            f"{actual_version!r} != {VALIDATOR_CONTRACT_VERSION!r}"
        )
    return None


def require_current_validation(
    trajectory: Mapping[str, Any], *, source: str = "trajectory"
) -> None:
    """Raise rather than silently accepting an invalid or stale trajectory."""

    error = current_validation_error(trajectory)
    if error is not None:
        trajectory_id = trajectory.get("trajectory_id", "<unknown>")
        raise ValueError(f"{source} ({trajectory_id}): {error}")
