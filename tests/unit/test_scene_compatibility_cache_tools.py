import json
import sys

import pytest

from robotalk.scene_sampling import (
    SCENE_POLICY_VERSION,
    scene_signature,
)
from robotalk.evaluation import build_scene_compatibility_cache
from robotalk.evaluation import merge_shared_workspace_compatibility


def _scene():
    scene = {"layout": 11, "style": 14, "seed": 42}
    return {**scene, "scene_signature": scene_signature(scene)}


def _write(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_cache_builder_ignores_generated_json_in_audit_directory(tmp_path, monkeypatch):
    scene = _scene()
    audit = tmp_path / "layout_11_style_14_seed_42.json"
    _write(
        audit,
        {
            "scene_policy_version": SCENE_POLICY_VERSION,
            "scene": scene,
            "records": [
                {
                    "task": "Task",
                    "physical_configuration_signature": "physical",
                    "physical_configuration": {},
                    "representative_run_index": 0,
                    "compatible": True,
                }
            ],
            "task_errors": [],
        },
    )
    output = tmp_path / "generated_cache.json"
    argv = [
        "build_scene_compatibility_cache.py",
        "--input-dir", str(tmp_path),
        "--output", str(output),
        "--require-scene-count", "1",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    build_scene_compatibility_cache.main()
    # A second build must not count its own previous output as another scene.
    build_scene_compatibility_cache.main()
    assert json.loads(output.read_text())["summary"]["scene_count"] == 1


def test_workspace_merge_accepts_current_policy_without_legacy_control(
    tmp_path, monkeypatch
):
    scene = _scene()
    initialization_cache = tmp_path / "initialization.json"
    _write(
        initialization_cache,
        {
            "schema_version": 1,
            "scene_policy_version": SCENE_POLICY_VERSION,
            "candidate_scenes": [scene],
            "summary": {},
            "tasks": {
                "Task": {
                    "configurations": {
                        "physical": {
                            "physical_configuration": {},
                            "representative_run_index": 0,
                            "compatible_scenes": [scene],
                            "incompatible_scenes": [],
                        }
                    }
                }
            },
        },
    )
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write(
        workspace_dir / "layout_11_style_14_seed_42.json",
        {"scene": scene, "records": [], "task_errors": []},
    )
    output = tmp_path / "merged.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "merge_shared_workspace_compatibility.py",
            "--initialization-cache", str(initialization_cache),
            "--workspace-audit-dir", str(workspace_dir),
            "--output", str(output),
            "--require-workspace-scene-count", "1",
        ],
    )
    merge_shared_workspace_compatibility.main()
    merged = json.loads(output.read_text())
    assert merged["workspace_compatibility_gate"]["positive_control"] == {
        "required": False,
        "task": None,
        "layout": None,
        "detections": [],
    }
    assert merged["tasks"]["Task"]["configurations"]["physical"][
        "compatible_scenes"
    ] == [scene]


def test_workspace_merge_rejects_a_different_scene_set(tmp_path, monkeypatch):
    expected_scene = _scene()
    actual_scene = {"layout": 15, "style": 14, "seed": 42}
    actual_scene["scene_signature"] = scene_signature(actual_scene)
    initialization_cache = tmp_path / "initialization.json"
    _write(
        initialization_cache,
        {
            "scene_policy_version": SCENE_POLICY_VERSION,
            "candidate_scenes": [expected_scene],
            "summary": {},
            "tasks": {},
        },
    )
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    _write(
        workspace_dir / "layout_15_style_14_seed_42.json",
        {"scene": actual_scene, "records": [], "task_errors": []},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "merge_shared_workspace_compatibility.py",
            "--initialization-cache", str(initialization_cache),
            "--workspace-audit-dir", str(workspace_dir),
            "--output", str(tmp_path / "merged.json"),
            "--require-workspace-scene-count", "1",
        ],
    )
    with pytest.raises(ValueError, match="scene sets differ"):
        merge_shared_workspace_compatibility.main()
