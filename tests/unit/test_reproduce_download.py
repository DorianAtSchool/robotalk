"""reproduce.py resumes Hugging Face downloads after rate limiting."""

import importlib.util
from pathlib import Path

import huggingface_hub
import pytest
import requests
from huggingface_hub.errors import HfHubHTTPError

SPEC = importlib.util.spec_from_file_location(
    "reproduce", Path(__file__).resolve().parents[2] / "scripts" / "reproduce.py"
)
reproduce = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reproduce)


def _error(status: int) -> HfHubHTTPError:
    response = requests.Response()
    response.status_code = status
    response.request = requests.Request("GET", "https://huggingface.co").prepare()
    return HfHubHTTPError("rate limited", response=response)


def test_a_rate_limited_download_waits_and_resumes(monkeypatch):
    calls, sleeps = [], []

    def download(repo_id, **kwargs):
        calls.append(repo_id)
        if len(calls) < 3:
            raise _error(429)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    monkeypatch.setattr(reproduce.time, "sleep", sleeps.append)
    reproduce._snapshot_download("DorianAtSchool/RoboTalk", repo_type="dataset")
    assert len(calls) == 3 and sleeps == [300, 300]


def test_the_lookup_error_of_a_throttled_file_check_is_retried(monkeypatch):
    from huggingface_hub.errors import LocalEntryNotFoundError

    calls = []

    def download(repo_id, **kwargs):
        calls.append(kwargs.get("revision"))
        if len(calls) < 2:
            raise LocalEntryNotFoundError("cannot locate the file on the Hub")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    monkeypatch.setattr(reproduce.time, "sleep", lambda s: None)
    reproduce._snapshot_download("DorianAtSchool/RoboTalk", repo_type="dataset")
    assert len(calls) == 2


def test_other_errors_are_not_retried(monkeypatch):
    def download(repo_id, **kwargs):
        raise _error(404)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    monkeypatch.setattr(reproduce.time, "sleep", lambda s: pytest.fail("should not wait"))
    with pytest.raises(HfHubHTTPError):
        reproduce._snapshot_download("DorianAtSchool/RoboTalk", repo_type="dataset")
