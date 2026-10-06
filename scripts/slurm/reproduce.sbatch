#!/usr/bin/env bash
# Generic SLURM wrapper around scripts/reproduce.py. Adapt the #SBATCH lines
# (partition, account, GPUs, time) to your cluster, then e.g.:
#
#   sbatch scripts/slurm/reproduce.sbatch fig6 --models instruct_s30
#   sbatch --gres=gpu:4 scripts/slurm/reproduce.sbatch fig7 --train
#
# Evaluation of one model needs 1 GPU (vLLM) plus CPU/EGL rendering for the
# simulator workers; --train needs the GPUs set in configs/experiments.yaml
# (training.num_processes, 4 in the paper).
#SBATCH --job-name=robotalk-reproduce
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=outputs/slurm/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/../..}"
mkdir -p outputs/slurm
# Optional: run vLLM from a container, e.g.
#   export VLLM_LAUNCHER="apptainer exec --nv --bind $PWD:$PWD vllm-openai-v0.27.1.sif"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
PORT=$((18000 + ${SLURM_JOB_ID:-0} % 10000))
exec python scripts/reproduce.py "$@" --vllm-port "${PORT}"
