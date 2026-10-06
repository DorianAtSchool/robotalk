from __future__ import annotations

import pytest

from robotalk.tasks.shared.validation_contract import (
    VALIDATOR_CONTRACT_VERSION,
    current_validation_error,
    require_current_validation,
)


def _trajectory(*, valid=True, version=VALIDATOR_CONTRACT_VERSION):
    return {
        "trajectory_id": "traj_000001",
        "validation": {
            "is_valid": valid,
            "validator_contract_version": version,
        },
    }


def test_current_contract_accepts_only_current_valid_record():
    assert current_validation_error(_trajectory()) is None


@pytest.mark.parametrize(
    "record",
    [
        {"trajectory_id": "missing"},
        _trajectory(valid=False),
        _trajectory(version=None),
        _trajectory(version="legacy"),
    ],
)
def test_current_contract_fails_closed(record):
    assert current_validation_error(record)
    with pytest.raises(ValueError):
        require_current_validation(record, source="test")
