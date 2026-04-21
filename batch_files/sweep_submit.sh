#!/bin/bash
#SBATCH --job-name=sweep
#SBATCH --output=sweep_%A_%a.out
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=shard:1
#SBATCH --mem=16G
#SBATCH -M anansi

set -euo pipefail

: "${DATASET:?DATASET env var is required (e.g. Cora)}"
: "${CURVATURE:?CURVATURE env var is required (e.g. BFc)}"
: "${REWIRING:?REWIRING env var is required (True or False)}"
: "${RUNS_PER_AGENT:?RUNS_PER_AGENT env var is required (int)}"
: "${SWEEP_ID:?SWEEP_ID env var is required — create one first with submit_sweep.sh}"

module purge
module load CUDA/12.6.0
export PYTHONUNBUFFERED=1
export CUDA_HOME=${CUDA_HOME:-$EBROOTCUDA}
export NUMBA_CUDA_USE_NVIDIA_BINDING=1

nvidia-smi
echo "CUDA_HOME=$CUDA_HOME"
echo "SLURM_GPUS=${SLURM_GPUS_ON_NODE:-unset}"
echo "Sweep config: dataset=$DATASET curvature=$CURVATURE rewiring=$REWIRING runs_per_agent=$RUNS_PER_AGENT sweep_id=$SWEEP_ID array_task=${SLURM_ARRAY_TASK_ID:-none}"

source /scratch/brussel/101/vsc10124/venvs/revisiting_graph_curvature_rewiring/bin/activate

srun python -u scripts/sweeps/ClassificationExperiment_sweep.py \
    --dataset "$DATASET" \
    --curvature-type "$CURVATURE" \
    --rewiring-run "$REWIRING" \
    --runs-per-agent "$RUNS_PER_AGENT" \
    --sweep-id "$SWEEP_ID"
