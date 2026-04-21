#!/usr/bin/env bash


set -euo pipefail

export PATH="${VSC_HOME}/.local/bin:$PATH"

# --- project + venv locations ---
PROJ="${VSC_HOME}/revisiting_graph_curvature_rewiring"
export UV_PROJECT_ENVIRONMENT="/scratch/brussel/101/$USER/venvs/$(basename "${PROJ}")"

echo $PATH
echo $PROJ
echo $UV_PROJECT_ENVIRONMENT
# --- sync environment exactly to uv.lock ---
cd "${PROJ}"
uv lock
uv sync
