# RoboTalk

**Learning Multi-Robot Communication and Coordination from Multimodal Demonstrations**

Dorian Benhamou Goldfajn\*, Mason Nakamura\*, Saaduddin Mahmud, Justin Svegliato, Kyle H. Wray, Shlomo Zilberstein
(\* equal contribution)

[Paper (arXiv:2609.23997)](https://arxiv.org/abs/2609.23997) ·
[Dataset](https://huggingface.co/datasets/DorianAtSchool/RoboTalk) ·
[Explorer](https://huggingface.co/spaces/DorianAtSchool/RoboTalk-Explorer) ·
[Models](https://huggingface.co/DorianAtSchool) ·
[PDF in this repo](docs/paper/RoboTalk_arxiv.pdf)

RoboTalk is a synthetic data-generation pipeline and dataset for training small
vision-language models (VLMs) to act as decentralized policies for two mobile
manipulators. It contains 7,950 two-robot trajectories across 53 kitchen tasks
adapted from [RoboCasa365](https://robocasa.ai). Each trajectory interleaves
both robots' camera observations, natural-language messages, skill-level tool
calls (perception, navigation, manipulation and communication) and one-sentence
rationales. Coordination follows a leader–follower planning protocol,
wait–release synchronization and exclusive-workspace rules, and a
finite-state machine (FSM) validates every trajectory.

## Contents

- [Installation](#installation)
- [Quickstart](#quickstart)
- [Repository structure](#repository-structure)
- [Results and reproduction](#results-and-reproduction)
- [Generating new data](#generating-new-data)
- [Training new models](#training-new-models)
- [Evaluating new models](#evaluating-new-models)
- [Limitations](#limitations)
- [License](#license) · [Citation](#citation) · [Acknowledgements](#acknowledgements)

## Installation

Linux and Python 3.11, installed with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/DorianAtSchool/robotalk.git
cd robotalk
uv sync --all-extras
source .venv/bin/activate

python -m robocasa.scripts.setup_macros
python -m robocasa.scripts.download_kitchen_assets   # RoboCasa365 assets: ~10 GB download, 23 GB on disk
cp .env.example .env    # data/output folders, Gemini key, vLLM launcher (all optional)
```

`--all-extras` installs everything. To install only part, pick extras:

| Extra | For |
|---|---|
| (none) | Simulator, FSM validator, dataset tools, live-simulation evaluation |
| `train` | LoRA fine-tuning (`torch`, `transformers`, `peft`, `accelerate`) |
| `flash` | FlashAttention-2, used for training |
| `gen` | Generating new data and the Gemini evaluations (`google-genai`) |
| `dev` | `pytest` |

robosuite is installed from
[DorianAtSchool/robosuite](https://github.com/DorianAtSchool/robosuite):
upstream robosuite plus one fix that lets two mobile manipulators share a
scene.

**FlashAttention.** Nothing extra to do: `uv sync --all-extras` installs a
prebuilt FlashAttention-2 wheel, and training uses it by default. The wheel
exists for Linux x86-64 with Python 3.11 only. Elsewhere, add
`--attn-implementation sdpa` to the training command.

**vLLM** (needed only to evaluate open-weight models). The models are served by
a local [vLLM](https://github.com/vllm-project/vllm) 0.27.1 server running
from the official container. vLLM brings its own torch and transformers, so it
does not go in the environment above. With
[Apptainer](https://apptainer.org) (or Singularity):

```bash
apptainer pull vllm-openai-v0.27.1.sif docker://vllm/vllm-openai:v0.27.1   # ~8 GB
echo 'VLLM_LAUNCHER="apptainer exec --nv /path/to/vllm-openai-v0.27.1.sif"' >> .env
```

The evaluation scripts start and stop the server themselves. Apptainer gives
the container your home directory and the current directory. If your
Hugging Face cache (`HF_HOME`) or outputs (`ROBOTALK_OUTPUT_ROOT`) live
elsewhere, add `--bind /that/path` after `--nv`.

> [!IMPORTANT]
> **GPU requirements.**
> - **Training:** `torch` comes from PyTorch's CUDA 12.8 builds, as in the
>   paper's environment. These run on NVIDIA GPUs from Turing to Blackwell
>   (compute capability 7.5–12.0, e.g. T4, A100, H100, RTX PRO 6000, B200)
>   with a driver for CUDA 12.8 (570 or newer). Volta (V100) and older GPUs
>   are not supported.
> - **Evaluation:** the vLLM 0.27.1 container is built for CUDA 13.0 and needs
>   a driver of version 580 or newer.
>
> The paper trained on 4× B200 and evaluated on single RTX PRO 6000 GPUs.

## Quickstart

**Load the dataset.** It has two tables (Hugging Face "configs"):

```python
from datasets import load_dataset

trajectories = load_dataset("DorianAtSchool/RoboTalk", "trajectories", split="train")
ticks = load_dataset("DorianAtSchool/RoboTalk", "ticks", split="train")
```

- `trajectories` has one row per trajectory (7,950 rows). It holds the task and
  its instruction, the coordinating robot, the initial state, how the
  trajectory was generated, and summary counts (ticks, messages, physical
  actions per robot).
- `ticks` has one row per time step of a trajectory (66,592 rows). It holds
  both robots' tool calls, arguments, rationales and messages at that step,
  and the camera images each robot had seen by then.
- The two tables join on `episode_key`. Here `split="train"` is simply the name
  of each table, not the paper's train/held-out task split.

**Replay demonstrations in the simulator.** This checks that the simulator is
installed correctly, in three steps.

1. Download one task (`prepare_coffee`) from the dataset, about 255 MB:

   ```bash
   hf download DorianAtSchool/RoboTalk --repo-type dataset --revision v1.1 \
       --include "data/trajectories.parquet" "raw/prepare_coffee/*" "media_archives/prepare_coffee__*" \
       --local-dir data/hf
   ```

   - `data/trajectories.parquet` is the trajectory table, which holds each
     task's instruction.
   - `raw/prepare_coffee/*` contains the task's 150 trajectory records: the
     tool calls, messages and rationales.
   - `media_archives/prepare_coffee__*` contains one archive of rendered
     camera images per trajectory.

2. Unpack it into the per-trajectory layout that training and evaluation read:

   ```bash
   python -m robotalk.release.materialize_training_layout \
       --hf-dir data/hf --output data/robotalk_rendered --tasks prepare_coffee
   ```

   This writes `data/robotalk_rendered/<task>/<trajectory>/` for each
   trajectory. Each folder holds:
   - the trajectory, with its observation calls inserted;
   - the calls in the order the simulator executes them, with image paths;
   - the execution record;
   - the extracted images.

3. Replay two of the trajectories in the simulator:

   ```bash
   python -m robotalk.evaluation.live_sim_eval --backend oracle \
       --manifest configs/eval/oracle_smoke.json --tasks prepare_coffee \
       --dataset-root data/robotalk_rendered --output-dir outputs/oracle
   ```

   | Argument | Meaning |
   |---|---|
   | `--backend oracle` | Replay the recorded demonstration calls instead of querying a model. The calls go through the same concurrent scheduler, validator and simulator used to evaluate models, so every replayed trajectory should reach its goal. |
   | `--manifest configs/eval/oracle_smoke.json` | Which trajectories to replay: two per task, for four tasks. |
   | `--tasks prepare_coffee` | Only the task downloaded in step 1. |
   | `--dataset-root`, `--output-dir` | Where to read trajectories and write results. |

   Outcomes are written to `outputs/oracle/live_sim_trajectories.jsonl`.
   Videos of each robot's cameras for the first successful and the first
   failed episode of each task go to `outputs/oracle/recordings/`. Use
   `--no-record-firsts` to turn videos off, or `--save-frames` to keep camera
   images of every episode instead. Rendering uses EGL (headless GPU);
   `--gl-backend osmesa` renders on the CPU instead.

**Run a fine-tuned model** for one episode on each of two tasks. This needs a
GPU and vLLM:

```bash
python scripts/reproduce.py fig6 --models instruct_s30 \
    --tasks prepare_coffee,arrange_bread_bowl --episodes-per-task 1
```

## Repository structure

```text
robotalk/
  tasks/        53 task specifications, symbolic state, concurrent FSM validator
  tools/        the tool interface shared by generation, training and evaluation
  generation/   Gemini trajectory generation, observation insertion, rendering
  training/     example construction, preprocessing, LoRA fine-tuning
  evaluation/   live-simulation evaluation (vLLM, Gemini and oracle backends)
  analysis/     paper results and figures
  release/      dataset export, training-layout rebuild, explorer Space
robocasa/       RoboCasa365 with two-robot support and the skill executor (docs/FORK_CHANGES.md)
configs/        experiment settings, task split, training subsets, evaluation cohort
scripts/        reproduce.py and a SLURM wrapper
tests/          unit tests (tests/unit) and simulator tests (tests/sim)
docs/           paper PDF and technical notes
```

### Task split

The paper trains on 43 tasks and holds out 10 to test generalization to new
tasks. The split is `configs/splits/43_train_10_heldout.json`. The held-out
tasks are `arrange_bread_bowl`, `beverage_organization`,
`cluster_items_for_clearing`, `distribute_chicken`, `garnish_cake`,
`hot_dog_setup`, `meat_skewer_assembly`, `prepare_cheese_station`,
`prepare_sandwich_station` and `serve_meal_juice`. Training uses the 43 by
default, and evaluation reports both groups separately. To use another split,
see [Training new models](#training-new-models) and
[Evaluating new models](#evaluating-new-models).

Technical notes:
- [concurrency semantics](docs/concurrency_semantics.md)
- [tool interface](docs/tool_interface.md)
- [data format](docs/data_format.md)
- [evaluation](docs/evaluation.md)

## Results and reproduction

Every number is closed-loop **error-free FSM success**: the episode reaches the
task goal with no rejected or failed tool call, within a budget of four times
the length of the task's example solution. Each model is evaluated on 430
episodes of the 43 training tasks and 100 episodes of the 10 held-out tasks
(10 per task, fixed cohort). Error bars in the paper are 95% Wilson
intervals.

`scripts/reproduce.py` runs a figure end to end:
- it downloads the dataset (v1.1) and the released models;
- it starts vLLM and evaluates every model on both task splits;
- it plots the figure into `outputs/figures/43_10/`.

Finished evaluations are skipped, so an interrupted run resumes when you rerun
the same command. `--dry-run` prints the commands without running them, and
`--train` retrains the models instead of downloading them.
`scripts/slurm/reproduce.sbatch` wraps it for SLURM. The settings of every
model are in `configs/experiments.yaml`.

### Fig. 3: task phases

| Phase (stages) | 43 training tasks | 10 held-out tasks |
|---|---:|---:|
| Phase 2 (2–3) | 12 | 2 |
| Phase 3 (4–5) | 12 | 2 |
| Phase 4 (6+) | 19 | 6 |

```bash
python scripts/reproduce.py fig3        # seconds, CPU only
```

### Fig. 5: communication guidance, untuned models

| Condition | Gemini 3 Flash (train / held-out) | Qwen3-VL-8B-Instruct (train / held-out) |
|---|---:|---:|
| None | 7.7% / 5% | 0.2% / 0% |
| Unguided | 9.5% / 8% | 0.5% / 0% |
| Minimal | 43.3% / 34% | 1.2% / 0% |
| Intermediate | 46.0% / 46% | 1.6% / 0% |
| Full | 65.6% / 72% | 0.0% / 0% |

```bash
python scripts/reproduce.py fig5
```

Fig. 5 needs a Gemini API key (see `.env.example`) and one GPU for Qwen. It
runs 10 conditions of 530 episodes each.

### Fig. 6: fine-tuning with more data (trajectories per training task)

| Model | 30 | 60 | 90 | 120 | 150 |
|---|---:|---:|---:|---:|---:|
| Instruct, train tasks | 81.9% | 87.9% | 92.1% | 93.0% | 94.2% |
| Instruct, held-out tasks | 45% | 64% | 55% | 48% | 65% |
| Thinking + rationale, train tasks | 80.2% | 84.7% | 91.4% | 90.2% | 92.6% |
| Thinking + rationale, held-out tasks | 71% | 72% | 66% | 77% | 73% |

```bash
python scripts/reproduce.py fig6                         # released models
python scripts/reproduce.py fig6 --train                 # retrain first
python scripts/reproduce.py fig6 --models instruct_s30   # one model
```

**Compute:**
- **Evaluation:** about 1 hour per model on one GPU (Thinking models about 30%
  longer).
- **Training:** about 2 hours per 30 trajectories per task on 4× B200. That
  ranges from 2 h (30/task) to 9.7 h (150/task), or about 235 B200 GPU-hours
  for all ten models.

### Fig. 7: rationales, at 30 trajectories per task

| Model and targets | Train tasks | Held-out tasks |
|---|---:|---:|
| Instruct, tool calls | 81.9% | 45% |
| Instruct, rationale + tool calls | 82.6% | 58% |
| Thinking, tool calls | 71.6% | 50% |
| Thinking, rationale + tool calls | 80.2% | 71% |

```bash
python scripts/reproduce.py fig7 [--train]
```

**Dataset versions.** The released models were trained on dataset v1.0. v1.1
replaces 95 trajectories (1.2%) that failed in simulation (see the dataset
card). Retraining on v1.1 should match the paper within its confidence
intervals.

## Generating new data

Generation needs a Gemini API key (`GOOGLE_API_KEY`) or Vertex AI credentials.
It runs in three stages, shown here for one task:

```bash
# 1. Generate and validate trajectories with Gemini (150 per task in the dataset).
python -m robotalk.generation.raw.production_cascade --task PrepareCoffee --num-runs 150 \
    --output-root data/my_raw

# 2. Insert the observation (get_image) calls and revalidate.
python -m robotalk.generation.image.cli --revalidate \
    --dataset data/my_raw/prepare_coffee/summary.json \
    --output-dataset data/my_image/prepare_coffee/summary.json

# 3. Replay each trajectory in a kitchen scene and render the camera views.
python -m robotalk.generation.sweep_trajectories --input-dir data/my_image --output-dir data/my_rendered
```

In stage 3 each trajectory is rendered in one kitchen scene. The scene is drawn
from the scenes known to work for its task and starting configuration
(`configs/eval/scene_compatibility_cache_v2.json`, seeded by
`--scene-sampling-seed`), as for the dataset. Alternatively,
`--scene-compatibility-cache none --layouts 11 40 --styles 14 34 --seeds 42`
renders every trajectory in each listed layout, style and seed combination.

The last stage writes the same per-trajectory layout as the Quickstart, so the
result can be used directly for training (`--dataset-root data/my_rendered`).
`python -m robotalk.release.export_robotalk` converts it into the published
dataset format.

## Training new models

Training defaults are the paper's settings: LoRA rank 16 (alpha 32, dropout
0.05), learning rate 1e-4 with weight decay 0.01 and a cosine schedule, one
epoch, effective batch 64 over 4 GPUs, and the 43 training tasks. Training
reads the per-trajectory layout in `data/robotalk_rendered`, as built in the
Quickstart or by `reproduce.py`. Change the folder with `--dataset-root` or
`ROBOTALK_DATA_ROOT`.

```bash
# 1. Build the training examples (CPU).
python -m robotalk.training.preprocess --output-dir outputs/preprocessed/my_run

# 2. Fine-tune with LoRA on 4 GPUs.
accelerate launch --config_file configs/train/accelerate_multigpu.yaml --num_processes 4 \
    -m robotalk.training.main --preprocessed-data-dir outputs/preprocessed/my_run \
    --model-name-or-path Qwen/Qwen3-VL-8B-Instruct --output-dir outputs/train/my_run
```

- **Rationales:** add `--train-reasoning` to both commands to train with
  rationales (`<think>…</think>` before each tool call).
- **Thinking model:** use `--model-name-or-path Qwen/Qwen3-VL-8B-Thinking`.
- **Fewer trajectories per task (Fig. 6):** step 1 builds examples from all
  150 trajectories of each task. The paper's smaller training sets are fixed,
  nested subsets listed in `configs/splits/selection_{30,60,90,120,150}.json`.
  To train on one, cut the step-1 output down to it, then pass the new folder
  to step 2:

  ```bash
  python -m robotalk.training.materialize_scale_artifacts \
      --master outputs/preprocessed/my_run --selection configs/splits/selection_30.json \
      --output outputs/preprocessed/my_run_30
  ```
- **Other tasks:** `--train-tasks a,b,c` on both commands trains on other
  tasks, for example the training half of your own split.
- **GPU memory:** the paper's 8 examples per GPU fit 180 GB B200s. On 80–96 GB
  GPUs, add `--per-device-batch-size 4 --grad-accum 4`, which keeps the
  effective batch at 64.
- **Other settings:** see `python -m robotalk.training.main --help`.

## Evaluating new models

`reproduce.py evaluate` runs any model on the paper's evaluation cohort with
the paper's settings: 10 episodes per task on both task splits.

```bash
# Your adapter (a training output folder, or a Hugging Face model repo):
python scripts/reproduce.py evaluate --adapter outputs/train/my_run

# A Thinking model trained with rationales:
python scripts/reproduce.py evaluate --adapter outputs/train/my_run \
    --base Qwen/Qwen3-VL-8B-Thinking --profile thinking

# An untuned model under another communication condition:
python scripts/reproduce.py evaluate --base Qwen/Qwen3-VL-8B-Instruct --communication-mode minimal

# Gemini (needs the API key):
python scripts/reproduce.py evaluate --profile gemini --base gemini-3-flash-preview
```

- **Profiles:** `--profile` sets decoding and vLLM's output parsers.
  - `instruct`: Instruct models.
  - `thinking`: Thinking models trained with rationales.
  - `thinking_no_rationale`: Thinking models trained without rationales.
  - `gemini`: Gemini models.
- **Results:** they go to `outputs/eval/<name>/<split>/`, and the command
  prints the error-free success counts per split.
- **Quick tests:** `--tasks a,b --episodes-per-task 1` runs a short smoke test
  instead, written to `outputs/eval_smoke/`.
- **Videos:** `--record-videos` saves videos of the first successful and the
  first failed episode of each task (off by default, as for the paper).
- **Another task split:** write a split file like
  `configs/splits/43_train_10_heldout.json`, regroup the evaluation episodes
  under it, and pass both:

  ```bash
  python -m robotalk.evaluation.resplit_cohort --task-split my_split.json --output my_cohort.json
  python scripts/reproduce.py evaluate --adapter outputs/train/my_run \
      --task-split my_split.json --cohort my_cohort.json
  ```

  A task keeps its evaluation configurations when it changes group. Its 10
  episodes are drawn anew, because the draw is seeded by the group name.

## Limitations

- Skills run through scripted simulator tools. RoboTalk covers high-level
  planning, communication and coordination, not low-level motor control.
- Success is judged by a symbolic FSM that abstracts some physical details. For
  example, placing the spatula in the pot counts as stirring in CheeseMixing.

## License

| Component | License |
|---|---|
| RoboTalk code | [Apache-2.0](LICENSE) |
| RoboCasa365 code in `robocasa/` | MIT, © the RoboCasa Team ([robocasa/LICENSE](robocasa/LICENSE)) |
| RoboCasa365 assets (downloaded separately) | CC BY 4.0 |
| Dataset and models | Apache-2.0 |

See [NOTICE](NOTICE). The trajectories were generated with Google Gemini, and
users are responsible for complying with the applicable model terms of
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

Please also cite RoboCasa365, RoboCasa and robosuite, on which the simulator
is built:

```bibtex
@inproceedings{robocasa365,
  title     = {RoboCasa365: A Large-Scale Simulation Framework for Training and Benchmarking Generalist Robots},
  author    = {Soroush Nasiriany and Sepehr Nasiriany and Abhiram Maddukuri and Yuke Zhu},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year      = {2026}
}

@inproceedings{robocasa2024,
  title     = {RoboCasa: Large-Scale Simulation of Everyday Tasks for Generalist Robots},
  author    = {Soroush Nasiriany and Abhiram Maddukuri and Lance Zhang and Adeet Parikh and Aaron Lo and Abhishek Joshi and Ajay Mandlekar and Yuke Zhu},
  booktitle = {Robotics: Science and Systems (RSS)},
  year      = {2024}
}

@article{robosuite2020,
  title   = {robosuite: A Modular Simulation Framework and Benchmark for Robot Learning},
  author  = {Yuke Zhu and Josiah Wong and Ajay Mandlekar and Roberto Mart{\'\i}n-Mart{\'\i}n and Abhishek Joshi and Soroush Nasiriany and Yifeng Zhu and Kevin Lin},
  journal = {arXiv preprint arXiv:2009.12293},
  year    = {2020}
}
```

## Acknowledgements

RoboTalk builds on [RoboCasa365](https://github.com/robocasa/robocasa) and
[robosuite](https://github.com/ARISE-Initiative/robosuite). This research was
supported in part by the National Science Foundation under Grant Nos. 2321786,
2326054 and 2416460, and by Schmidt Sciences under the AI Safety Science
program. Mason Nakamura was supported by an NSF Graduate Research Fellowship
under Grant No. 2439846. This work used Google Cloud through the CloudBank
project, which is supported by National Science Foundation grant No. 1925001.
