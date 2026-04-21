#!/bin/bash
# Submit a wandb sweep as a SLURM array of agents sharing one sweep_id.
#
# Usage:
#   batch_files/submit_sweep.sh \
#       --dataset Cora \
#       --curvature BFc \
#       --rewiring True \
#       --agents 4 \
#       --runs-per-agent 10 \
#       [--sweep-id <id>] \
#       [--force-new]
#
# By default the sweep id is resolved via config/sweep_registry.json: if the
# (dataset, curvature, rewiring) combo is already registered, that id is
# reused; otherwise a new sweep is created and recorded. --sweep-id overrides
# the lookup; --force-new ignores the registry and creates a fresh sweep.

set -euo pipefail

DATASET=""
CURVATURE=""
REWIRING=""
AGENTS=""
RUNS_PER_AGENT=""
SWEEP_ID=""
FORCE_NEW=0

usage() {
    sed -n '2,17p' "$0" >&2
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset)         DATASET="$2";         shift 2 ;;
        --curvature)       CURVATURE="$2";       shift 2 ;;
        --rewiring)        REWIRING="$2";        shift 2 ;;
        --agents)          AGENTS="$2";          shift 2 ;;
        --runs-per-agent)  RUNS_PER_AGENT="$2";  shift 2 ;;
        --sweep-id)        SWEEP_ID="$2";        shift 2 ;;
        --force-new)       FORCE_NEW=1;          shift 1 ;;
        -h|--help)         usage ;;
        *) echo "Unknown flag: $1" >&2; usage ;;
    esac
done

for var in DATASET CURVATURE REWIRING AGENTS RUNS_PER_AGENT; do
    if [[ -z "${!var}" ]]; then
        echo "Missing required flag for $var" >&2
        usage
    fi
done

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -z "$SWEEP_ID" ]]; then
    extra_args=()
    if [[ "$FORCE_NEW" -eq 1 ]]; then
        extra_args+=(--force-new)
    fi
    echo "Resolving sweep id for $DATASET / $CURVATURE / rewiring=$REWIRING ..." >&2
    SWEEP_ID=$(python -u scripts/sweeps/ClassificationExperiment_sweep.py \
        --dataset "$DATASET" \
        --curvature-type "$CURVATURE" \
        --rewiring-run "$REWIRING" \
        --create-only "${extra_args[@]}" | tail -n 1)
    echo "Using sweep_id=$SWEEP_ID" >&2
fi

sbatch \
    --array=1-"$AGENTS" \
    --export=ALL,DATASET="$DATASET",CURVATURE="$CURVATURE",REWIRING="$REWIRING",RUNS_PER_AGENT="$RUNS_PER_AGENT",SWEEP_ID="$SWEEP_ID" \
    batch_files/sweep_submit.sh
