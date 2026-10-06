from __future__ import annotations

import json

import pytest

from robotalk.evaluation.live_sim_parallel_eval import (
    METRICS_FILENAME,
    RESULTS_FILENAME,
    OutputLock,
    ParallelEvalError,
    _preflight_worker_records,
    _scan_new_worker_records,
    aggregate_worker_outputs,
    build_shard_manifest,
    prepare_parallel_run,
    shard_task_mapping,
)


def _manifest() -> dict:
    return {
        "split": "test",
        "trajectory_ids_by_task": {
            "task_a": ["a1", "a2"],
            "task_b": ["b1"],
            "task_c": ["c1"],
        },
        "summary": {"stale_field": "preserved"},
    }


def _live_args(manifest_path, output_root, *extra: str) -> list[str]:
    return [
        "--backend",
        "vllm",
        "--vllm-model",
        "robocasa-v2",
        "--manifest",
        str(manifest_path),
        "--dataset-root",
        "dataset",
        "--output-dir",
        str(output_root),
        "--gl-backend",
        "egl",
        *extra,
    ]


def _record(task: str, trajectory_id: str, *, native=False, fsm=False) -> dict:
    return {
        "task_name": task,
        "trajectory_id": trajectory_id,
        "termination": "goal_satisfied" if native else "budget_exhausted",
        "native_success": native,
        "fsm_goal_satisfied": fsm,
        "declared_complete": False,
        "steps_used": 1,
        "expert_steps": 1,
        "partial_goal_fraction": float(fsm),
        "step_efficiency_ratio": 1.0,
        "rejected_steps": 0,
        "agent_turns": ["agent_0"],
        "elapsed_s": 0.1,
        "steps": [],
    }


def _write_complete_worker_outputs(specs) -> dict:
    original_text = {}
    for spec in specs:
        records = [
            _record(task, trajectory_id)
            for task, trajectory_id in sorted(spec.trajectory_keys)
        ]
        text = "".join(json.dumps(record) + "\n" for record in records)
        results_path = spec.output_dir / RESULTS_FILENAME
        results_path.write_text(text, encoding="utf-8")
        (spec.output_dir / METRICS_FILENAME).write_text(
            json.dumps({"policy_load_s": 0.0, "total_elapsed_s": 1.0}),
            encoding="utf-8",
        )
        original_text[results_path] = text
    return original_text


def test_sharding_balances_whole_tasks_deterministically():
    manifest = _manifest()
    mapping = manifest["trajectory_ids_by_task"]

    first = shard_task_mapping(manifest, mapping, 2)
    second = shard_task_mapping(manifest, mapping, 2)

    assert first == second
    assert first == [
        {"task_a": ["a1", "a2"]},
        {"task_b": ["b1"], "task_c": ["c1"]},
    ]
    assert set().union(*(set(shard) for shard in first)) == set(mapping)
    assert sum(len(shard) for shard in first) == len(mapping)


def test_shard_manifest_keeps_only_its_tasks_and_recomputes_summary():
    shard = build_shard_manifest(
        _manifest(),
        {"task_b": ["b1"]},
        worker_index=1,
        worker_count=2,
    )

    assert shard["trajectory_ids_by_task"] == {"task_b": ["b1"]}
    assert shard["summary"] == {
        "num_trajectories": 1,
        "num_tasks": 1,
        "stale_field": "preserved",
    }
    assert shard["parallel_shard"]["worker_index"] == 1


def test_prepare_is_resume_stable_and_rejects_changed_arguments(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
    output_root = tmp_path / "parallel"
    args = _live_args(manifest_path, output_root)

    first_root, first_metadata, first_specs = prepare_parallel_run(
        args, requested_workers=2
    )
    second_root, second_metadata, second_specs = prepare_parallel_run(
        args, requested_workers=2
    )

    assert first_root == second_root == output_root.resolve()
    assert first_metadata == second_metadata
    assert [spec.manifest_path for spec in first_specs] == [
        spec.manifest_path for spec in second_specs
    ]
    assert all("--resume" in spec.live_args for spec in first_specs)
    assert len({spec.output_dir for spec in first_specs}) == 2
    assert not (set(first_specs[0].trajectory_keys) & set(first_specs[1].trajectory_keys))

    with pytest.raises(ParallelEvalError, match="configuration differs"):
        prepare_parallel_run(
            _live_args(manifest_path, output_root, "--map-dpi", "60"),
            requested_workers=2,
        )


def test_prepare_fingerprints_api_key_without_persisting_it(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
    output_root = tmp_path / "parallel"
    secret = "not-for-metadata"

    prepare_parallel_run(
        _live_args(manifest_path, output_root, "--vllm-api-key", secret),
        requested_workers=2,
    )

    metadata_text = (output_root / "parallel_run.json").read_text()
    assert secret not in metadata_text
    assert "sha256:" in metadata_text
    with pytest.raises(ParallelEvalError, match="configuration differs"):
        prepare_parallel_run(
            _live_args(
                manifest_path,
                output_root,
                "--vllm-api-key",
                "different-secret",
            ),
            requested_workers=2,
        )


def test_prepare_rejects_ambiguous_limit(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")

    with pytest.raises(ParallelEvalError, match="ambiguous per-worker"):
        prepare_parallel_run(
            _live_args(
                manifest_path,
                tmp_path / "limited",
                "--max-trajectories",
                "2",
            ),
            requested_workers=2,
        )


def test_output_lock_rejects_a_second_coordinator(tmp_path):
    output_root = tmp_path / "parallel"
    with OutputLock(output_root):
        with pytest.raises(ParallelEvalError, match="locked by PID"):
            with OutputLock(output_root):
                pass
    assert not (output_root / ".parallel_eval.lock").exists()


def test_output_lock_never_deletes_unreadable_lock(tmp_path):
    output_root = tmp_path / "parallel"
    output_root.mkdir()
    lock_path = output_root / ".parallel_eval.lock"
    lock_path.write_text("{", encoding="utf-8")

    with pytest.raises(ParallelEvalError, match="unreadable lock"):
        with OutputLock(output_root):
            pass
    assert lock_path.read_text(encoding="utf-8") == "{"


def test_aggregation_is_derived_and_does_not_edit_worker_jsonl(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(_manifest()), encoding="utf-8")
    output_root, _metadata, specs = prepare_parallel_run(
        _live_args(manifest_path, tmp_path / "parallel"),
        requested_workers=2,
    )
    original_text = _write_complete_worker_outputs(specs)

    metrics = aggregate_worker_outputs(
        output_root,
        specs,
        parallel_wall_elapsed_s=2.5,
        parallel_wall_elapsed_s_this_invocation=1.25,
    )

    assert metrics["num_trajectories"] == 4
    assert metrics["harness_error_rate"] == 0.0
    assert metrics["parallel"]["effective_workers"] == 2
    assert metrics["parallel"]["parallel_wall_elapsed_s"] == 2.5
    assert (
        metrics["parallel"]["parallel_wall_elapsed_s_this_invocation"] == 1.25
    )
    for path, text in original_text.items():
        assert path.read_text(encoding="utf-8") == text
    aggregate_lines = (
        output_root / "aggregate" / RESULTS_FILENAME
    ).read_text(encoding="utf-8").splitlines()
    assert len(aggregate_lines) == 4
    validation = json.loads(
        (output_root / "aggregate" / "validation.json").read_text()
    )
    assert validation["valid"] is True


def test_aggregation_rejects_malformed_json_without_rewriting_it(tmp_path):
    manifest = {
        "trajectory_ids_by_task": {"task_a": ["a1"]},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output_root, _metadata, specs = prepare_parallel_run(
        _live_args(manifest_path, tmp_path / "parallel"),
        requested_workers=1,
    )
    results_path = specs[0].output_dir / RESULTS_FILENAME
    malformed = '{"task_name":"task_a"\n'
    results_path.write_text(malformed, encoding="utf-8")
    (specs[0].output_dir / METRICS_FILENAME).write_text(
        json.dumps({"policy_load_s": 0.0, "total_elapsed_s": 1.0}),
        encoding="utf-8",
    )

    with pytest.raises(ParallelEvalError, match="malformed JSON"):
        aggregate_worker_outputs(output_root, specs, parallel_wall_elapsed_s=1.0)

    assert results_path.read_text(encoding="utf-8") == malformed
    assert not (output_root / "aggregate" / RESULTS_FILENAME).exists()


def test_aggregation_reports_native_null_as_invalid(tmp_path):
    manifest = {"trajectory_ids_by_task": {"task_a": ["a1"]}}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output_root, _metadata, specs = prepare_parallel_run(
        _live_args(manifest_path, tmp_path / "parallel"),
        requested_workers=1,
    )
    record = _record("task_a", "a1")
    record["termination"] = "harness_error"
    record["native_success"] = None
    (specs[0].output_dir / RESULTS_FILENAME).write_text(
        json.dumps(record) + "\n", encoding="utf-8"
    )
    (specs[0].output_dir / METRICS_FILENAME).write_text(
        json.dumps({"policy_load_s": 0.0, "total_elapsed_s": 1.0}),
        encoding="utf-8",
    )

    with pytest.raises(ParallelEvalError, match="harness_error"):
        aggregate_worker_outputs(output_root, specs, parallel_wall_elapsed_s=1.0)

    validation = json.loads(
        (output_root / "aggregate" / "validation.json").read_text()
    )
    assert validation["valid"] is False
    assert any("null native_success" in issue for issue in validation["issues"])


def test_live_monitor_checks_only_newly_appended_records(tmp_path):
    manifest = {"trajectory_ids_by_task": {"task_a": ["a1"]}}
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _root, _metadata, specs = prepare_parallel_run(
        _live_args(manifest_path, tmp_path / "parallel"),
        requested_workers=1,
    )
    spec = specs[0]
    results_path = spec.output_dir / RESULTS_FILENAME
    old_harness_error = _record("task_a", "a1")
    old_harness_error["termination"] = "harness_error"
    old_harness_error["native_success"] = None
    results_path.write_text(json.dumps(old_harness_error) + "\n", encoding="utf-8")

    seen = _preflight_worker_records(specs)
    assert seen == {0: 1}
    assert _scan_new_worker_records(specs, seen) == []

    successful_retry = _record("task_a", "a1")
    with results_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(successful_retry) + "\n")
    assert _scan_new_worker_records(specs, seen) == []
    assert seen == {0: 2}

    new_harness_error = _record("task_a", "a1")
    new_harness_error["termination"] = "harness_error"
    new_harness_error["native_success"] = None
    with results_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(new_harness_error) + "\n")
    issues = _scan_new_worker_records(specs, seen)
    assert any("harness_error" in issue for issue in issues)
    assert any("null native_success" in issue for issue in issues)


# --- observations are free on the virtual clock, up to a cap -----------------
#
# The concurrent validator makes get_image cost 0.0; the executor charged it
# 0.25 (1.0 under --uniform-durations), so a plan the gate certified could still
# drift apart in sim. These pin the executor to the gate's semantics.


def _obs(clock, n, cap, multiplier=1.0):
    from robotalk.evaluation.live_sim_eval import observation_ready_at

    return observation_ready_at(
        clock,
        consecutive_obs=n,
        max_free_observations=cap,
        multiplier=multiplier,
    )


def test_observation_does_not_advance_the_clock_below_the_cap():
    for n in range(4):
        ready_at, count, capped = _obs(7.5, n, 4)
        assert ready_at == 7.5, "an observation must not move the agent's clock"
        assert count == n + 1
        assert capped is False


def test_observation_beyond_the_cap_is_charged_and_resets():
    ready_at, count, capped = _obs(7.5, 4, 4)
    assert ready_at > 7.5, "past the cap the clock must advance or it freezes"
    assert capped is True
    assert count == 0, "the counter resets so the agent gets a fresh allowance"


def test_zero_cap_restores_the_old_always_charged_behaviour():
    ready_at, count, capped = _obs(0.0, 0, 0)
    assert ready_at > 0.0
    assert (count, capped) == (0, False)


def test_free_observations_keep_two_agents_on_the_same_schedule():
    """The drift that broke 47 corpus trajectories, reproduced and removed.

    agent_0 takes 3 actions, agent_1 takes 1, and the image pass gives each
    action two observations. Charging the cameras puts the busier agent ahead by
    more than a whole action, which is how a release lands before its waiter
    blocks.
    """

    from robotalk.evaluation.live_sim_eval import _tool_duration

    def finish(actions, cap):
        clock, n = 0.0, 0
        for _ in range(actions):
            for _ in range(2):  # the injector's two observations per action
                clock, n, _capped = _obs(clock, n, cap)
            clock += _tool_duration("pick", sim_steps=None, multiplier=1.0)
            n = 0
        return clock

    charged = finish(3, 0) - finish(1, 0)
    free = finish(3, 4) - finish(1, 4)
    assert free < charged, "free observations must shrink the drift"
    assert free == 2 * _tool_duration("pick", sim_steps=None, multiplier=1.0), (
        "with cameras free the gap is exactly the two extra actions -- the "
        "schedule the validator certified"
    )
