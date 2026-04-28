#!/bin/bash
#SBATCH --job-name=sweep
#SBATCH --output=logs/sweep_%A_%a.out
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=shard:1
#SBATCH --mem=16G
#SBATCH -M anansi

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
