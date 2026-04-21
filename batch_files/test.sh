#!/bin/bash
#SBATCH --job-name=test
#SBATCH --output=test_%A.out
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=shard:1
#SBATCH --mem=16G
#SBATCH -M anansi

# Modules
module purge
module load CUDA/12.6.0
export PYTHONUNBUFFERED=1
# Numba needs this — point it at the module's toolkit
export CUDA_HOME=${CUDA_HOME:-$EBROOTCUDA}   # EBROOTCUDA is the EasyBuild convention
export NUMBA_CUDA_USE_NVIDIA_BINDING=1

# Show what we've got
nvidia-smi
echo "CUDA_HOME=$CUDA_HOME"
echo "SLURM_GPUS=$SLURM_GPUS_ON_NODE"

source /scratch/brussel/101/vsc10124/venvs/Revisiting-Graph-Curvature-Rewiring/bin/activate

srun python -u scripts/sweeps/ClassificationExperiment_sweep.py \
    --dataset Texas \
    --curvature-type BFc \
    --rewiring-run True \
    --runs-per-agent 3 \