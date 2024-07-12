from experiment_utils.largestconnectedcomponent import lcc_dataset
from utils.load_datasets import load_data,data_information
from experiment_utils.training_objective import objective

import os
import json

import wandb

os.environ["WANDB_SILENT"] = "true"
os.environ["NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS"] = "false"

#results_dir = "results"
"""
Parameters for the experiment
"""

datasetname = "Texas"

rewiring_run = True
make_undirected = True
int_node = False
curvature_type = "BFc"

"""
Loading the dataset
"""

dataset,data,G = load_data(datasetname)
dataset_lcc = lcc_dataset(dataset,to_undirected = make_undirected)
data_lcc = dataset_lcc[0]

data_information(dataset_lcc,data_lcc)


with open(os.path.join('experiment_utils\hyperparameters','hyperparameters_FixedPar_Best.json'), 'r') as file:
     sweep_configuration = json.load(file)[datasetname][curvature_type]
     print(type(sweep_configuration))
     #sweep_configuration =sweep_configuration.get(datasetname, {})

sweep_configuration["name"] = f"{datasetname}_{curvature_type}"



def main():
    wandb.init(dir = "../../wandb")
    acc,test_acc = objective(wandb.config,
                             datasetname,dataset_lcc,data_lcc,curvature_type,
                             int_node,rewiring_run)
    wandb.log({"mean accuracy": acc, "mean test accuracy": test_acc})

#sweep_id = "cwu2mmfw"# wandb.sweep(sweep=sweep_configuration, project="curvature")
#wandb.agent(sweep_id, project="curvature", function=main,count = 100)

sweep_id = wandb.sweep(sweep=sweep_configuration, project="Curvature_Neurips_FixedGNNParameters_r2")
wandb.agent(sweep_id, function=main,count = 3)