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
#
# Submit a wandb sweep as a SLURM array of agents. Each array task runs one
# wandb agent that consumes RUNS_PER_AGENT configs.
#
# Usage (Python side auto-registers sweep_id in config/sweep_registry.json):
#
#   sbatch --array=1-4 \
#       --export=ALL,DATASET=Cora,CURVATURE=BFc,REWIRING=True,RUNS_PER_AGENT=10 \
#       batch_files/sweep.sh
#
# Optional: pin an existing sweep_id (skips registry lookup, avoids the
# first-run race where parallel array tasks each create their own sweep):
#
#   sbatch --array=1-4 \
#       --export=ALL,DATASET=Cora,CURVATURE=BFc,REWIRING=True,RUNS_PER_AGENT=10,SWEEP_ID=abc123 \
#       batch_files/sweep.sh

module purge
module load CUDA/12.6.0
export PYTHONUNBUFFERED=1
export CUDA_HOME=${CUDA_HOME:-$EBROOTCUDA}
export NUMBA_CUDA_USE_NVIDIA_BINDING=1

nvidia-smi
echo "CUDA_HOME=$CUDA_HOME"
echo "SLURM_GPUS=${SLURM_GPUS_ON_NODE:-unset}"
echo "Sweep: dataset=${DATASET} curvature=${CURVATURE} rewiring=${REWIRING} runs_per_agent=${RUNS_PER_AGENT} sweep_id=${SWEEP_ID:-<registry>} array_task=${SLURM_ARRAY_TASK_ID:-none}"

source /scratch/brussel/101/vsc10124/venvs/Revisiting-Graph-Curvature-Rewiring/bin/activate

srun python -u scripts/sweeps/ClassificationExperiment_sweep.py \
    --dataset "${DATASET:?set DATASET}" \
    --curvature-type "${CURVATURE:?set CURVATURE}" \
    --rewiring-run "${REWIRING:?set REWIRING}" \
    --runs-per-agent "${RUNS_PER_AGENT:-3}" \
    ${SWEEP_ID:+--sweep-id "$SWEEP_ID"}
