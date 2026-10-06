# Data format

## Published dataset (`DorianAtSchool/RoboTalk`)

| Path | Contents |
|---|---|
| `data/trajectories.parquet` (`trajectories` config) | One row per trajectory: task, instruction, coordinator, initial state, physical configuration, generation stage, summary counts. |
| `data/ticks.parquet` (`ticks` config) | One row per concurrent tick: both agents' calls, arguments, rationales and messages, their observation calls, and the latest image available to each agent. |
| `raw/<task>/traj_XXXXXX.json` | The generated trajectory record (source of truth). |
| `media_archives/<task>__traj_XXXXXX.tar` | Rendered images of that trajectory. The image maps in `ticks` name members of this archive. |

Counts:
- 53 tasks × 150 trajectories = 7,950 trajectories.
- Trajectories run 4–18 ticks (median 7).
- The `ticks` table adds one final observation-only row per trajectory.

`train` in `load_dataset(..., split="train")` is the name of the published
table. It is not the 43/10 task split, which is defined in
`configs/splits/43_train_10_heldout.json`.

Versions are git tags on the dataset repository. `v1.0` is the data the paper
used. `v1.1` replaces 95 trajectories that failed in simulation; see the
dataset card.

### Raw trajectory record

A raw record contains:
- task and trajectory IDs, the agents and the coordinator;
- the symbolic initial state and its physical configuration signature;
- `tick_rows`, the concurrent schedule, plus `steps`, the same calls flattened;
- each call's `reasoning` (the rationale);
- the grounding map, final state and validation result;
- generation metadata: accepted cascade stage, attempt count, sampling
  variables, token usage.

## Training layout

Training and evaluation read a per-trajectory directory layout, rebuilt from
the published files by:

```bash
python -m robotalk.release.materialize_training_layout \
    --hf-dir <local copy of the dataset> --output data/robotalk_rendered
```

```text
data/robotalk_rendered/<task>/<traj>/
  original_trajectory.json   # raw record after observation insertion
  plan.json                  # calls in simulator execution order, with image paths
  metadata.json              # execution record (every published step succeeded)
  images/<traj>/*.jpg|png    # rendered views
```

`scripts/reproduce.py` runs this automatically when `ROBOTALK_DATA_ROOT`
(default `data/robotalk_rendered`) is missing.

## Training examples

Each example is one agent's next tool call given that agent's private context.
The context holds:
- the task instruction;
- the symbolic initial state;
- the derived concurrency facts;
- the agent's own call history and received messages, without step indices;
- images from its most recent `get_image` call. Each image is used for one
  example only.

The target is the tool call. With `--train-reasoning` the target also includes
the call's rationale, as `<think>…</think>`. Prompt and padding tokens are
masked from the loss.

Calls made in the same tick are added to history only after both agents'
targets for that tick are built. So neither agent sees information it could not
have had when it chose its call.
