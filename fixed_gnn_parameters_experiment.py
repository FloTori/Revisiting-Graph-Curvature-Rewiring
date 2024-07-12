from experiment_utils.largestconnectedcomponent import lcc_dataset
from utils.load_datasets import load_data,data_information

from experiment_utils.training_objective import objective

import json
import wandb
import os

os.environ["WANDB_SILENT"] = "true"
os.environ["NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS"] = "false"

cluster = True
if cluster:
    import sys
    """
    Parameters for the experiment
    """

    datasetname = sys.argv[1]
    rewiring_run = eval(sys.argv[2])
    make_undirected = True
    int_node = False
    Curvature_type = sys.argv[3]
else:
    """
    Parameters for the experiment
    """

    datasetname = "Texas"

    rewiring_run = True
    make_undirected = True
    int_node = False
    curvature_type = "BFc_w4cycle"

"""
Loading the dataset
"""

dataset,data,G = load_data(datasetname)
dataset_lcc = lcc_dataset(dataset,to_undirected = make_undirected)
data_lcc = dataset_lcc[0]

data_information(dataset_lcc,data_lcc)



with open(os.path.join('experiment_utils\hyperparameters','hyperparameters_FixedGNNParameters.json'), 'r') as file:
     sweep_configuration = json.load(file)
     sweep_configuration =sweep_configuration.get(datasetname, {})

sweep_configuration["name"] = datasetname + '_' + curvature_type


def main():
    wandb.init(dir = "../../wandb")
    acc,test_acc = objective(wandb.config,rewiring_run)
    wandb.log({"mean accuracy": acc, "mean test accuracy": test_acc})

#sweep_id = "cwu2mmfw"# wandb.sweep(sweep=sweep_configuration, project="curvature")
#wandb.agent(sweep_id, project="curvature", function=main,count = 100)

sweep_id = wandb.sweep(sweep=sweep_configuration, project="Revisiting-Graph-Rewiring-Fixed_GNN_Parameters")
wandb.agent(sweep_id, function=main,count = 100)