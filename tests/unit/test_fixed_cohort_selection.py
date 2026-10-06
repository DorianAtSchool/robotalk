from robotalk.evaluation.fixed_cohort_selection import select_configuration_episodes


def _configs(count: int):
    return [
        {"configuration_signature": f"{index:064x}", "trajectory_id": f"traj_{index}"}
        for index in range(count)
    ]


def test_sampled_defaults_can_select_ten_without_duplicates_when_available():
    rows = select_configuration_episodes(
        _configs(12), task_name="task", split="train_task_types",
        evaluation_seed=7,
    )
    assert len(rows) == 10
    assert len({row["configuration_signature"] for row in rows}) == 10


def test_sampled_cycles_uniformly_when_more_episodes_than_configurations():
    rows = select_configuration_episodes(
        _configs(3), task_name="task", split="train_task_types",
        evaluation_seed=7, episodes_per_task=8,
    )
    counts = {
        signature: sum(row["configuration_signature"] == signature for row in rows)
        for signature in {row["configuration_signature"] for row in rows}
    }
    assert sorted(counts.values()) == [2, 3, 3]


def test_full_config_supports_repetitions_and_cap():
    rows = select_configuration_episodes(
        _configs(8), task_name="task", split="heldout_task_types",
        evaluation_seed=9, mode="full_config", episodes_per_config=2,
        max_configs_per_task=3,
    )
    assert len(rows) == 6
    assert len({row["configuration_signature"] for row in rows}) == 3
