---
configs:
- config_name: trajectories
  data_files: data/trajectories.parquet
- config_name: ticks
  data_files: data/ticks.parquet
license: apache-2.0
language:
- en
pretty_name: RoboTalk
size_categories:
- 1K<n<10K
tags:
- robotics
- multi-agent
- vision-language-model
- tool-use
- communication
---

# RoboTalk

RoboTalk is a dataset of 7,950 multimodal two-robot trajectories across 53
mobile-manipulation kitchen tasks adapted from RoboCasa365. Each trajectory
interleaves the agents' visual observations, natural-language messages,
skill-level tool calls (perception, navigation, manipulation, communication)
and one-sentence rationale traces, under a leader–follower planning protocol,
wait–release synchronization and exclusive-workspace rules.

- Paper: *RoboTalk: Learning Multi-Robot Communication and Coordination from
  Multimodal Demonstrations* ([arXiv:2609.23997](https://arxiv.org/abs/2609.23997))
- Code: [DorianAtSchool/robotalk](https://github.com/DorianAtSchool/robotalk)
- Fine-tuned adapters: `DorianAtSchool/RoboTalk-Qwen3-VL-8B-*`

## Contents

| Path | Contents |
|---|---|
| `data/trajectories.parquet` (`trajectories` config) | One row per trajectory: task, instruction, coordinator, initial state, physical configuration, generation stage, summary counts. |
| `data/ticks.parquet` (`ticks` config) | One row per concurrent tick: both agents' calls, arguments, rationales, messages, observation calls and the latest image available to each agent. |
| `raw/<task>/traj_XXXXXX.json` | The generated trajectory record (source of truth). |
| `media_archives/<task>__traj_XXXXXX.tar` | The rendered images of that trajectory; image maps in `ticks` name members of this archive. |

```python
from datasets import load_dataset

trajectories = load_dataset("DorianAtSchool/RoboTalk", "trajectories", split="train")
ticks = load_dataset("DorianAtSchool/RoboTalk", "ticks", split="train")
```

Training with the RoboTalk code uses a per-trajectory file layout that is
derived deterministically from `raw/` and `media_archives/`:

```bash
python -m robotalk.release.materialize_training_layout \
    --hf-dir <local copy of this repository> --output data/robotalk_rendered
```

## How the data was made

1. **Tasks.** 53 RoboCasa365 kitchen tasks with verified symbolic
   specifications; 150 trajectories per task, spread evenly over each task's
   valid initial configurations (agent start locations, open/closed fixtures)
   and alternating the coordinating agent.
2. **Generation.** Gemini 3 Flash writes a complete two-agent trajectory as
   concurrent ticks, with structured-random variation of message length,
   message complexity and tool-call diversity.
3. **Validation.** A concurrent finite-state machine checks every candidate
   against action preconditions, the planning protocol, exclusive-workspace
   access and wait–release semantics. Rejected candidates are retried through a
   bounded cascade (Flash at several temperatures and thinking levels, a
   critic-assisted repair, then Gemini 3.1 Pro).
4. **Observations and rendering.** Perception calls (`get_image`) with
   templated rationales are inserted programmatically, and each trajectory is
   replayed in a RoboCasa365 scene with two robots to render the requested
   camera views.

## Versions

- **v1.0**: the dataset used for every result in the paper. The released
  adapters were trained on v1.0.
- **v1.1** (current): 95 trajectories (1.2%) were replaced, either re-rendered or
  regenerated for the same task and initial configuration. Identifiers and the
  150-per-task balance are unchanged. The replaced share by split:

  | Split | Replaced |
  |---|---:|
  | 43 trained tasks | 65 / 6,450 (1.0%) |
  | 10 held-out tasks | 30 / 1,500 (2.0%) |
  | Training subsets of 30 / 60 / 90 / 120 / 150 trajectories per task | 1.1% / 1.0% / 1.0% / 1.0% / 1.0% |

## Intended use and limitations

Intended for research on learned communication and coordination for embodied
agents, e.g. supervised fine-tuning of vision-language models as decentralized
policies. Skills are executed by scripted simulator tools, so the data covers
high-level coordination and planning, not low-level motor control. Success is
judged by a symbolic finite-state machine that abstracts some physical details.
Messages and rationales are written by an LLM and inherit its style; the task
suite and scene set are finite.

## License and attribution

Released under Apache-2.0. Images are renders of RoboCasa365 scenes and assets,
which are licensed CC BY 4.0 by the RoboCasa team; please credit RoboCasa365
when using the images. Trajectory text was generated with Google
Gemini; users are responsible for complying with the applicable model terms of
service.

## Citation

```bibtex
@article{robotalk2026,
  title   = {RoboTalk: Learning Multi-Robot Communication and Coordination from Multimodal Demonstrations},
  author  = {Benhamou Goldfajn, Dorian and Nakamura, Mason and Mahmud, Saaduddin and Svegliato, Justin and Wray, Kyle H. and Zilberstein, Shlomo},
  journal = {arXiv preprint arXiv:2609.23997},
  year    = {2026}
}
```
