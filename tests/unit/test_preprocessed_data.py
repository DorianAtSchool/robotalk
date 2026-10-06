from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

from robotalk.training.preprocessed_data import (
    build_or_load_cached_pretokenization_metadata,
    build_pretokenization_metadata,
    pretokenization_compatibility_reason,
    resolve_hf_dataset_reference,
)


HAS_DATASETS = importlib.util.find_spec("datasets") is not None
HAS_PILLOW = importlib.util.find_spec("PIL") is not None



class PretokenizationCompatibilityTests(unittest.TestCase):
    def test_returns_none_when_artifact_matches_run_config(self):
        reason = pretokenization_compatibility_reason(
            runtime_pretokenization={
                "processor_family": "qwen",
                "processor_class": "FakeQwenProcessor",
                "processor_name_or_path": "processor-a",
                "processor_snapshot_hash": "hash-a",
                "max_length": 4096,
                "image_resolution": 512,
                "trust_remote_code": False,
            },
            preprocess_config={
                "pretokenized": True,
                "pretokenization": {
                    "processor_family": "qwen",
                    "processor_class": "FakeQwenProcessor",
                    "processor_name_or_path": "processor-a",
                    "processor_snapshot_hash": "hash-a",
                    "max_length": 4096,
                    "image_resolution": 512,
                    "trust_remote_code": False,
                },
            },
        )

        self.assertIsNone(reason)

    def test_reports_mismatched_pretokenization_settings(self):
        reason = pretokenization_compatibility_reason(
            runtime_pretokenization={
                "processor_family": "qwen",
                "processor_class": "FakeQwenProcessor",
                "processor_name_or_path": "processor-a",
                "processor_snapshot_hash": "hash-a",
                "max_length": 4096,
                "image_resolution": 512,
                "trust_remote_code": False,
            },
            preprocess_config={
                "pretokenized": True,
                "pretokenization": {
                    "processor_family": "qwen",
                    "processor_class": "FakeQwenProcessor",
                    "processor_name_or_path": "processor-b",
                    "processor_snapshot_hash": "hash-b",
                    "max_length": 2048,
                    "image_resolution": 256,
                    "trust_remote_code": True,
                },
            },
        )

        self.assertIn("processor_name_or_path", reason)
        self.assertIn("processor_snapshot_hash", reason)
        self.assertIn("max_length", reason)
        self.assertIn("image_resolution", reason)
        self.assertIn("trust_remote_code", reason)

    def test_build_pretokenization_metadata_rejects_non_qwen_processors(self):
        class FakeProcessor:
            def save_pretrained(self, path):
                Path(path, "tokenizer.json").write_text("fake", encoding="utf-8")

        with self.assertRaisesRegex(ValueError, "Qwen processors only"):
            build_pretokenization_metadata(
                processor=FakeProcessor(),
                processor_name_or_path="processor-a",
                max_length=None,
                image_resolution=512,
                trust_remote_code=False,
            )

    def test_build_pretokenization_metadata_hashes_saved_qwen_processor(self):
        class FakeQwenProcessor:
            def save_pretrained(self, path):
                Path(path, "tokenizer.json").write_text(
                    "fake-tokenizer", encoding="utf-8"
                )
                Path(path, "chat_template.jinja").write_text(
                    "template", encoding="utf-8"
                )

        metadata = build_pretokenization_metadata(
            processor=FakeQwenProcessor(),
            processor_name_or_path="processor-a",
            max_length=4096,
            image_resolution=512,
            trust_remote_code=False,
        )

        self.assertEqual(metadata["processor_family"], "qwen")
        self.assertEqual(metadata["processor_class"], "FakeQwenProcessor")
        self.assertEqual(metadata["processor_name_or_path"], "processor-a")
        self.assertEqual(metadata["max_length"], 4096)
        self.assertRegex(metadata["processor_snapshot_hash"], r"^[0-9a-f]{64}$")

    def test_cached_pretokenization_metadata_avoids_rehashing_processor(self):
        class FakeQwenProcessor:
            save_calls = 0

            def save_pretrained(self, path):
                type(self).save_calls += 1
                Path(path, "tokenizer.json").write_text(
                    "fake-tokenizer", encoding="utf-8"
                )

        temp_root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        cache_path = temp_root / "pretokenization_metadata.json"
        processor = FakeQwenProcessor()

        first_metadata = build_or_load_cached_pretokenization_metadata(
            processor=processor,
            processor_name_or_path="processor-a",
            max_length=4096,
            image_resolution=256,
            trust_remote_code=False,
            cache_path=cache_path,
        )
        second_metadata = build_or_load_cached_pretokenization_metadata(
            processor=processor,
            processor_name_or_path="processor-a",
            max_length=4096,
            image_resolution=256,
            trust_remote_code=False,
            cache_path=cache_path,
        )

        self.assertEqual(first_metadata, second_metadata)
        self.assertEqual(FakeQwenProcessor.save_calls, 1)
        self.assertRegex(second_metadata["processor_snapshot_hash"], r"^[0-9a-f]{64}$")

    def test_resolve_hf_dataset_reference_parses_revision_suffix(self):
        self.assertEqual(
            resolve_hf_dataset_reference("user/repo:branch-name"),
            ("user/repo", "branch-name"),
        )
        self.assertEqual(
            resolve_hf_dataset_reference(
                "user/repo:branch-name",
                explicit_revision="explicit-branch",
            ),
            ("user/repo:branch-name", "explicit-branch"),
        )


if __name__ == "__main__":
    unittest.main()
