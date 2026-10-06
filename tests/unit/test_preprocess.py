from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from robotalk.training import dataset
from robotalk.training import preprocess
from robotalk.training import preprocessed_data


class _FakeTqdm:
    instances: list["_FakeTqdm"] = []
    writes: list[str] = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.updates: list[int] = []
        self.closed = False
        type(self).instances.append(self)

    def update(self, amount: int = 1) -> None:
        self.updates.append(amount)

    def close(self) -> None:
        self.closed = True

    @classmethod
    def reset(cls) -> None:
        cls.instances = []
        cls.writes = []

    @staticmethod
    def write(message: str, file=None) -> None:
        _FakeTqdm.writes.append(message)


class _FakeExample:
    def __init__(self, sample_id: str, *, image_paths: list[str] | None = None):
        self.sample_id = sample_id
        self.task_name = "fake_task"
        self.trajectory_id = "traj_000001"
        self.agent_id = "agent_0"
        self.image_paths = list(image_paths or [])

    def to_feature_dict(self) -> dict[str, object]:
        return {
            "image_paths": list(self.image_paths),
            "prompt": f"prompt-{self.sample_id}",
        }

    def to_manifest_entry(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "task_name": self.task_name,
        }


class PreprocessProgressTests(unittest.TestCase):
    def setUp(self) -> None:
        _FakeTqdm.reset()

    def test_log_process_memory_usage_reports_current_and_peak_rss(self):
        messages: list[str] = []

        with (
            mock.patch.object(
                preprocess,
                "_read_current_rss_bytes",
                return_value=512 * 1024 * 1024,
            ),
            mock.patch.object(
                preprocess,
                "_read_peak_rss_bytes",
                return_value=1024 * 1024 * 1024,
            ),
            mock.patch.object(preprocess, "_log", side_effect=messages.append),
        ):
            preprocess._log_process_memory_usage("after train pretokenization")

        self.assertEqual(
            messages,
            [
                "Memory usage after train pretokenization: "
                "current_rss=512.0 MiB, peak_rss=1.0 GiB"
            ],
        )

    def test_pretokenize_examples_uses_tqdm_when_available(self):
        examples = [_FakeExample("sample-1"), _FakeExample("sample-2")]

        with (
            mock.patch.object(preprocess, "tqdm", _FakeTqdm),
            mock.patch.object(
                preprocess,
                "build_batched_pretokenized_tensors",
                return_value=[
                    {"input_ids": [1, 2, 3]},
                    {"input_ids": [1, 2, 3]},
                ],
            ) as build_tensors,
            mock.patch.object(
                preprocess,
                "serialize_pretokenized_tensors",
                side_effect=lambda tensors: bytes(tensors["input_ids"]),
            ) as serialize_tensors,
        ):
            blobs_by_sample_id = preprocess._pretokenize_examples(
                split_name="train",
                examples=examples,
                processor=object(),
                max_length=1024,
                image_resolution=256,
                batch_size=2,
            )

        self.assertEqual(
            blobs_by_sample_id,
            {
                "sample-1": b"\x01\x02\x03",
                "sample-2": b"\x01\x02\x03",
            },
        )
        build_tensors.assert_called()
        self.assertEqual(build_tensors.call_count, 1)
        self.assertEqual(serialize_tensors.call_count, 2)

        self.assertEqual(len(_FakeTqdm.instances), 1)
        progress_bar = _FakeTqdm.instances[0]
        self.assertEqual(progress_bar.kwargs["total"], 2)
        self.assertEqual(progress_bar.kwargs["desc"], "pretokenize train")
        self.assertEqual(progress_bar.kwargs["unit"], "example")
        self.assertEqual(progress_bar.updates, [2])
        self.assertTrue(progress_bar.closed)
        self.assertTrue(
            any(
                "Pretokenizing train examples (2 samples)" in message
                for message in _FakeTqdm.writes
            )
        )
        self.assertTrue(
            any(
                "Pretokenized train examples: 2/2" in message
                for message in _FakeTqdm.writes
            )
        )

    def test_pretokenize_examples_flushes_serialized_chunks(self):
        examples = [_FakeExample(f"sample-{index}") for index in range(5)]
        flushed_chunks: list[dict[str, bytes]] = []

        with (
            mock.patch.object(preprocess, "tqdm", None),
            mock.patch.object(
                preprocess,
                "build_batched_pretokenized_tensors",
                side_effect=[
                    [{"input_ids": [1]}],
                    [{"input_ids": [2]}],
                    [{"input_ids": [3]}],
                    [{"input_ids": [4]}],
                    [{"input_ids": [5]}],
                ],
            ),
            mock.patch.object(
                preprocess,
                "serialize_pretokenized_tensors",
                side_effect=lambda tensors: bytes(tensors["input_ids"]),
            ),
        ):
            blobs_by_sample_id = preprocess._pretokenize_examples(
                split_name="train",
                examples=examples,
                processor=object(),
                max_length=1024,
                image_resolution=256,
                batch_size=1,
                flush_callback=flushed_chunks.append,
                flush_interval=3,
            )

        self.assertEqual(blobs_by_sample_id, {})
        self.assertEqual(
            flushed_chunks,
            [
                {
                    "sample-0": b"\x01",
                    "sample-1": b"\x02",
                    "sample-2": b"\x03",
                },
                {
                    "sample-3": b"\x04",
                    "sample-4": b"\x05",
                },
            ],
        )

    def test_shard_mode_reuses_artifact_images_for_selected_examples_only(self):
        temp_root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        dataset_root = temp_root / "dataset"
        dataset_root.mkdir()
        output_dir = temp_root / "artifact"
        (output_dir / "images").mkdir(parents=True)
        shard_dir = output_dir / "pretokenized_shards"
        train_examples = [
            _FakeExample("train-keep", image_paths=["source/train-keep.png"]),
            _FakeExample("train-skip", image_paths=["source/train-skip.png"]),
        ]
        validation_examples = [
            _FakeExample("val-keep", image_paths=["source/val-keep.png"]),
            _FakeExample("val-skip", image_paths=["source/val-skip.png"]),
        ]
        args = argparse.Namespace(
            dataset_root=dataset_root,
            train_tasks="hot_dog_setup",
            val_tasks="prepare_coffee",
            validation_split_mode="none",
            validation_trajectory_fraction=0.1,
            validation_min_trajectories_per_task=1,
            output_dir=output_dir,
            use_example_cache=True,
            training_samples_cache_dir=None,
            example_build_workers=1,
            pretokenize=True,
            processor_name_or_path="Qwen/Qwen3.5-0.8B",
            max_length=4096,
            image_resolution=None,
            artifact_image_size=256,
            resume_existing_artifact_images="unchecked",
            skip_existing_artifact_image_validation=False,
            pretokenize_batch_size=4,
            pretokenize_flush_interval=400,
            pretokenize_shard_index=0,
            pretokenize_num_shards=2,
            pretokenize_shard_output_dir=shard_dir,
            trust_remote_code=False,
            push_to_hub=None,
            hub_revision=None,
            hub_private=False,
            hub_data_dir=None,
        )

        def select_first(*, examples, shard_index, num_shards):
            del shard_index, num_shards
            return examples[:1]

        with (
            mock.patch.object(preprocess, "parse_args", return_value=args),
            mock.patch.object(preprocess, "_load_processor", return_value=object()),
            mock.patch.object(
                preprocess,
                "build_or_load_cached_pretokenization_metadata",
                return_value={"processor_family": "qwen"},
            ),
            mock.patch.object(
                preprocess,
                "_build_examples_for_tasks",
                side_effect=[train_examples, validation_examples],
            ) as build_examples,
            mock.patch.object(
                preprocess,
                "_select_pretokenization_shard_examples",
                side_effect=select_first,
            ),
            mock.patch.object(
                preprocess,
                "existing_artifact_image_relpaths_for_examples",
                return_value={
                    "source/train-keep.png": "images/train-keep.png",
                    "source/val-keep.png": "images/val-keep.png",
                },
            ) as reuse_images,
            mock.patch.object(
                preprocess,
                "_pretokenize_examples",
                return_value={},
            ) as pretokenize_examples,
            mock.patch.object(
                preprocess,
                "PretokenizedShardWriter",
            ) as shard_writer_cls,
            mock.patch.object(preprocess, "build_split_manifest") as build_manifest,
        ):
            shard_writer = shard_writer_cls.return_value.__enter__.return_value
            preprocess.main()

        reuse_images.assert_called_once()
        reused_examples = reuse_images.call_args.kwargs["examples"]
        self.assertEqual(
            [example.sample_id for example in reused_examples],
            ["train-keep", "val-keep"],
        )
        pretokenized_counts = [
            len(call.kwargs["examples"]) for call in pretokenize_examples.call_args_list
        ]
        self.assertEqual(pretokenized_counts, [1, 1])
        for call in pretokenize_examples.call_args_list:
            self.assertEqual(call.kwargs["flush_interval"], 400)
            self.assertIsNotNone(call.kwargs["flush_callback"])
        self.assertEqual(build_examples.call_count, 2)
        for call in build_examples.call_args_list:
            example_filter = call.kwargs["example_filter"]
            self.assertIsNotNone(example_filter)
            self.assertEqual(
                example_filter("train-keep"),
                (preprocess.stable_sample_rank("train-keep") % 2 == 0),
            )
        shard_writer_cls.assert_called_once_with(
            shard_dir=shard_dir.resolve(),
            shard_index=0,
            num_shards=2,
            pretokenization={"processor_family": "qwen"},
            progress_callback=preprocess._log,
        )
        self.assertIsNotNone(shard_writer)
        build_manifest.assert_not_called()

    def test_load_examples_from_cache_filters_cached_examples(self):
        temp_root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        cache_path = temp_root / "examples.json"
        fingerprint = {"dataset": "fake"}

        def raw_example(sample_id: str) -> dict[str, object]:
            return {
                "sample_id": sample_id,
                "task_name": "fake_task",
                "composite_task": "fake_task",
                "trajectory_id": "traj_000001",
                "step_index": 0,
                "agent_id": "agent_0",
                "task_instruction": "fake instruction",
                "observation_views": ["front"],
                "image_paths": [f"{sample_id}.png"],
                "history_steps": [],
                "allowed_tool_specs": {},
                "tool_schemas": [],
                "response_schema": {},
                "target_payload": {},
                "target_tool_call": {},
                "target_text": "{}",
                "messages": [],
            }

        cache_path.write_text(
            json.dumps(
                {
                    "cache_format_version": dataset._EXAMPLE_CACHE_FORMAT_VERSION,
                    "fingerprint": fingerprint,
                    "examples": [
                        raw_example("sample-keep"),
                        raw_example("sample-skip"),
                    ],
                }
            ),
            encoding="utf-8",
        )

        examples = dataset.load_examples_from_cache(
            cache_path=cache_path,
            expected_fingerprint=fingerprint,
            sample_id_filter=lambda sample_id: sample_id.endswith("keep"),
        )

        self.assertIsNotNone(examples)
        self.assertEqual([example.sample_id for example in examples], ["sample-keep"])


class PreprocessedDataProgressTests(unittest.TestCase):
    def setUp(self) -> None:
        _FakeTqdm.reset()

    def test_serialize_split_rows_uses_tqdm_when_available(self):
        examples = [
            _FakeExample("sample-1", image_paths=["source/a.png"]),
            _FakeExample("sample-2", image_paths=["source/a.png"]),
        ]
        progress_messages: list[str] = []

        with mock.patch.object(preprocessed_data, "tqdm", _FakeTqdm):
            rows = preprocessed_data._serialize_split_rows(
                split_name="train",
                examples=examples,
                image_relpaths_by_source={"source/a.png": "images/000000.png"},
                pretokenized_blobs_by_sample_id=None,
                progress_callback=progress_messages.append,
            )

        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["num_images"], 1)
        self.assertEqual(len(_FakeTqdm.instances), 1)
        progress_bar = _FakeTqdm.instances[0]
        self.assertEqual(progress_bar.kwargs["total"], 2)
        self.assertEqual(progress_bar.kwargs["desc"], "serialize train")
        self.assertEqual(progress_bar.kwargs["unit"], "row")
        self.assertEqual(progress_bar.updates, [1, 1])
        self.assertTrue(progress_bar.closed)
        self.assertIn("Serializing train rows (2 examples)", progress_messages)
        self.assertIn("Serialized train rows: 2/2", progress_messages)


if __name__ == "__main__":
    unittest.main()
