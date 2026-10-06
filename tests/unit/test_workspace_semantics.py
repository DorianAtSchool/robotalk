from robotalk.tasks.shared.constants import EXCLUSIVE_FIXTURE_TYPES
from robotalk.tasks.shared.workspace_semantics import (
    accepted_agent_workspaces,
    canonical_agent_workspace,
    canonicalize_agent_locations,
)


STATE = {
    "agents": {
        "agent_0": {"location": "cab"},
        "agent_1": {"location": "toaster"},
    },
    "objects": {"mug": {"location": "cab"}},
    "fixtures": {
        "counter": {"fixture_type": "counter"},
        "cab": {"fixture_type": "cabinet", "parent_fixture": "counter"},
        "toaster": {"fixture_type": "toaster_oven", "parent_fixture": "counter"},
    },
}


def test_only_agent_cabinet_location_is_canonicalized():
    normalized = canonicalize_agent_locations(STATE)
    assert normalized["agents"]["agent_0"]["location"] == "counter"
    assert normalized["agents"]["agent_1"]["location"] == "toaster"
    assert normalized["objects"]["mug"]["location"] == "cab"


def test_parent_accepts_cabinet_alias_and_exclusive_child():
    accepted = accepted_agent_workspaces(
        STATE,
        "counter",
        exclusive_fixture_types=EXCLUSIVE_FIXTURE_TYPES,
    )
    assert set(accepted) == {"counter", "cab", "toaster"}


def test_cabinet_navigation_canonicalizes_but_child_does_not():
    assert canonical_agent_workspace(STATE, "cab") == "counter"
    assert canonical_agent_workspace(STATE, "toaster") == "toaster"
    assert "cabinet" not in EXCLUSIVE_FIXTURE_TYPES
    assert "toaster_oven" in EXCLUSIVE_FIXTURE_TYPES
