"""Test layout: tests/unit runs on any CPU without RoboCasa assets; tests/sim
builds MuJoCo scenes and needs the downloaded kitchen assets."""

from pathlib import Path

import pytest

SIM_DIR = Path(__file__).parent / "sim"


def pytest_collection_modifyitems(config, items):
    for item in items:
        if SIM_DIR in Path(str(item.fspath)).parents:
            item.add_marker(pytest.mark.sim)
