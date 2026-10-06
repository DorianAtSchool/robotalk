"""`robotalk.training.main` defaults are the paper's training settings."""

import argparse
from pathlib import Path

import pytest
import yaml

pytest.importorskip("torch")  # needs the `train` extra

from robotalk.training import main as training_main  # noqa: E402

PAPER = yaml.safe_load(
    (Path(__file__).resolve().parents[2] / "configs/experiments.yaml").read_text()
)["training"]["main_args"]


def _parser(monkeypatch) -> argparse.ArgumentParser:
    captured = {}

    def capture(self, *args, **kwargs):
        captured["parser"] = self
        raise SystemExit

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", capture)
    with pytest.raises(SystemExit):
        training_main.parse_args()
    return captured["parser"]


@pytest.mark.parametrize("flag", sorted(PAPER))
def test_default_matches_the_paper(monkeypatch, flag):
    actions = {option: action for action in _parser(monkeypatch)._actions for option in action.option_strings}
    value, action = PAPER[flag], actions[flag]
    if flag == "--validation-split-seed":
        assert action.default is None  # falls back to --seed, which is 42
    elif flag.startswith("--no-"):
        assert action.default is False
    elif value is True:
        assert action.default is True
    else:
        assert type(value)(action.default) == value
