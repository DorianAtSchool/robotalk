# Task specifications

The 53 RoboTalk tasks, one JSON file per task, named by the normalized task
slug (for example `addlemontofish.json` for `AddLemonToFish`).
`robotalk.tasks.specs` loads them into runtime task definitions.

Each specification names its RoboCasa365 composite task
(`source_python_module`) and gives:

- the symbolic initial state;
- the tools available to the agents;
- the task's preconditions, effects and goal conditions, which the
  finite-state validator checks;
- the grounding of symbolic objects and fixtures to simulator entities.
