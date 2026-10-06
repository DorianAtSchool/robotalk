from robotalk.scene_sampling import (
    SCENE_POLICY_VERSION,
    candidate_scenes_v1,
    sample_compatible_scene,
)


def test_v1_candidates_are_unique_and_exclude_layout18():
    scenes = candidate_scenes_v1()
    assert len(scenes) == 60
    assert len({row["scene_signature"] for row in scenes}) == 60
    assert {row["layout"] for row in scenes} == {11, 15, 40, 50}


def test_compatible_scene_sampling_is_reproducible():
    cache = {
        "scene_policy_version": SCENE_POLICY_VERSION,
        "tasks": {
            "Task": {
                "configurations": {
                    "cfg": {
                        "compatible_scenes": list(candidate_scenes_v1()[:4])
                    }
                }
            }
        },
    }
    first = sample_compatible_scene(
        cache, task="Task", physical_configuration_signature="cfg",
        sampling_seed=3, sample_index=1,
    )
    second = sample_compatible_scene(
        cache, task="Task", physical_configuration_signature="cfg",
        sampling_seed=3, sample_index=1,
    )
    assert first == second
