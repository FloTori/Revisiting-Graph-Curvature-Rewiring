from experiment_utils.largestconnectedcomponent import lcc_dataset
from utils.load_datasets import load_data,data_information

from experiment_utils.training_objective import objective

import os




os.environ["WANDB_SILENT"] = "true"
os.environ["NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS"] = "false"

#results_dir = "results"
"""
Parameters for the experiment
"""

datasetname = "Cornell"

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

config_hyper = {
        "learning_rate": 0.07468,
        "layers": [128],
        "dropout":0.214,
        "weight_decay":0.7837,
        "loops": 200,
        "C+": 16.881,
        "tau": 212
    } 

import time
start_time = time.time()

accuracies = objective(config_hyper,rewiring_run)
print("Average accuracy", accuracies)

print("--- %s seconds ---" % (time.time() - start_time))