# Evaluation

All results are closed-loop: each episode is a fresh rollout in the simulator.
Each robot is controlled by its own instance of the policy with its own
context. Open-weight models are served by a shared local vLLM server; Gemini
uses native function calling.

## Cohort

`configs/eval/fixed_live_sim_cohort.json` lists the valid initial
configurations of every task:
- agent start locations;
- open or closed fixtures;
- the coordinator.

`configs/eval/scene_compatibility_cache_v2.json` lists the scenes each
configuration works in. Scenes come from 4 layouts × 5 styles × 3 seeds = 60
candidates.

For each split, the evaluator draws 10 episodes per task (`--cohort-mode
sampled`). It cycles through the configurations in a shuffled order, seeded by
`--evaluation-seed 20260817`. The scene is drawn with `--scene-sampling-seed
20260819`.

| Split | Tasks | Episodes |
|---|---:|---:|
| `train_task_types` | 43 | 430 |
| `heldout_task_types` | 10 | 100 |

The training tasks are evaluated on fresh rollouts, but their configurations
can coincide with training configurations. Generalization is measured on the
held-out tasks.

## Running

The simplest route is `scripts/reproduce.py`, which starts the vLLM server
with the right parsers and passes the paper's settings from
`configs/experiments.yaml`:

```bash
python scripts/reproduce.py evaluate --adapter outputs/train/my_run   # any model
python scripts/reproduce.py fig6 --models instruct_s30                # a paper cell
```

Underneath, each split is one call of the parallel evaluator, which runs 4
simulator workers against the shared server:

```bash
python -m robotalk.evaluation.live_sim_parallel_eval --workers 4 -- \
    --backend vllm --manifest configs/eval/fixed_live_sim_cohort.json \
    --cohort-split heldout_task_types \
    --scene-compatibility-cache configs/eval/scene_compatibility_cache_v2.json \
    --dataset-root data/robotalk_rendered --output-dir outputs/eval/<model>/heldout_task_types \
    ...
```

`python scripts/reproduce.py <figure> --dry-run` prints the complete commands.

Backends:
- `vllm` for open-weight models, optionally with a LoRA adapter;
- `gemini`;
- `oracle`, which replays the expert trajectory and is used to check the
  installation (`configs/eval/oracle_smoke.json`).

## Episode rules

- **Rejected calls.** A rejected call is reported privately to the agent that
  made it, and the episode continues.
- **Budget.** An episode ends in any of these cases:
  - the FSM goal is reached;
  - the step budget is used up: four times the length of the task's example
    solution (`--step-budget-factor 2`, applied per agent);
  - there are 3 consecutive rejected calls (`--max-consecutive-rejections`);
  - the agents deadlock.
- **Durations.** All skills take the same simulated time
  (`--uniform-durations`).

## Metrics

With N episodes, S episodes reaching the FSM goal, and E of those without any
rejected or failed call:

- **Error-free FSM success** = E / N. This is the paper's headline metric.
- **Final FSM success** = S / N.
- **Single-worker inset (Fig. 5)**: error-free successes in which exactly one
  agent performed every task-changing physical action, divided by N.
  Communication, observation, navigation, `give_space` and waiting are not
  counted as task-changing.

Intervals are 95% Wilson intervals on these binomial rates. They do not include
variation across training seeds or prompt changes.

`robotalk.analysis.paper_results` aggregates the results:
- input: `outputs/eval/<model>/<split>/aggregate/live_sim_trajectories.jsonl`;
- output: `outputs/figures/43_10/artifact.json`, which the figure exporters
  read.
