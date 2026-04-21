import argparse
import json
import os
import sys


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {"true", "t", "1", "yes", "y"}:
        return True
    if value.lower() in {"false", "f", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean, got: {value!r}")


parser = argparse.ArgumentParser(description="Submit or join a wandb sweep for a (dataset, curvature) combo.")
parser.add_argument("--dataset", required=True)
parser.add_argument("--curvature-type", required=True)
parser.add_argument("--rewiring-run", required=True, type=_parse_bool)
parser.add_argument("--runs-per-agent", type=int, default=3,
                    help="Value passed to wandb.agent(count=...); how many sweep configs this agent will consume.")
parser.add_argument("--sweep-id", default=None,
                    help="Attach to an existing sweep instead of consulting the registry. Empty string is treated as unset.")
parser.add_argument("--create-only", action="store_true",
                    help="Ensure a sweep exists for this config (consulting the registry), print its id to stdout, then exit.")
parser.add_argument("--force-new", action="store_true",
                    help="Ignore any registry entry and create a brand new sweep. The new id replaces the registry entry.")
parser.add_argument("--registry", default=None,
                    help=f"Path to the JSON sweep registry. Defaults to config/sweep_registry.json.")
args = parser.parse_args()

datasetname = args.dataset
curvature_type = args.curvature_type
rewiring_run = args.rewiring_run
make_undirected = True
int_node = False
sweep_id = args.sweep_id or None

project = "revisiting-graph-curvature-rewiring"

import wandb
from experiment_utils import sweep_registry

registry_path = args.registry or sweep_registry.DEFAULT_REGISTRY_PATH

if sweep_id is None and not args.force_new:
    cached = sweep_registry.lookup(datasetname, curvature_type, rewiring_run, path=registry_path)
    if cached is not None:
        sweep_id = cached
        print(f"Registry hit: {datasetname}|{curvature_type}|{rewiring_run} -> {sweep_id}", file=sys.stderr)

if args.create_only and sweep_id is not None:
    print(sweep_id)
    sys.exit(0)

# Heavy imports / dataset loading only needed when we will actually create a sweep
# or launch an agent.
import numba, torch
from numba import cuda
print('cuda available', cuda.is_available())

from experiment_utils.largestconnectedcomponent import lcc_dataset
from utils.load_datasets import load_data, data_information
from experiment_utils.training_objective import objective

dataset, data, G = load_data(datasetname)
dataset_lcc = lcc_dataset(dataset, to_undirected=make_undirected)
data_lcc = dataset_lcc[0]

data_information(dataset_lcc, data_lcc)


with open('config/hyperparameters/hyperparameters_sweep_v2.json', 'r') as file:
    sweep_configuration = json.load(file)[datasetname]

sweep_configuration["name"] = f"{datasetname}_{curvature_type}"


def main():
    wandb.init(dir="../../wandb")
    wandb.log({"dataset": datasetname, "curvature_type": curvature_type, "rewiring_run": rewiring_run})
    acc, test_acc = objective(wandb.config,
                              datasetname, dataset_lcc, data_lcc, curvature_type,
                              int_node, rewiring_run)
    wandb.log({"mean_accuracy": acc, "mean_test_accuracy": test_acc})


if sweep_id is None:
    sweep_id = wandb.sweep(sweep=sweep_configuration, project=project)
    sweep_registry.record(datasetname, curvature_type, rewiring_run, sweep_id, path=registry_path)
    print(f"Created sweep {sweep_id} and recorded in {registry_path}", file=sys.stderr)

if args.create_only:
    print(sweep_id)
    sys.exit(0)

print(f"Attaching agent to sweep {sweep_id}", file=sys.stderr)
wandb.agent(sweep_id, function=main, count=args.runs_per_agent, project=project)
