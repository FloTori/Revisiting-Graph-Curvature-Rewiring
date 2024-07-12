
from utils.load_datasets import get_dataset_graphs
from experiment_utils.sdrf_cuda import sdrf_BFc,sdrf_JTc,sdrf_JLc,sdrf_AFc

from utils.seeds import val_seeds
from experiment_utils.experimentclass import GraphExperiment

from torch_geometric.data import Data
import torch
import torch.nn.functional as F
import torch_geometric
from torch_geometric.transforms import BaseTransform
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device: ", device)
import numpy as np

import os
import json

from tqdm import tqdm



"""
Parameters for the experiment
"""

datasetname = "MUTAG"
batch_size = 64
results_dir = "results"
rewiring_run = True
make_undirected = True
int_node = False
Curvature_type = "JcT"

path = ""

train_fraction = 0.6
validation_fraction = 0.2

os.environ["WANDB_SILENT"] = "true"
os.environ["NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS"] = "false"

dataset_graphs = get_dataset_graphs(datasetname)


class edge_rewiring_transform(BaseTransform):
    def __init__(self,hyperparameters,intermediate_node,remove_edges,curvaturetype: str):
        
        self.hyperparameters= hyperparameters  # parameters you need
        self.intermediate_node = intermediate_node
        self.remove_edges = remove_edges
        self.curvaturetype = curvaturetype

    def __call__(self, data: Data) -> Data:
        data.edge_index = self.create_rewired_edge_index(data)
        return data

    def create_rewired_edge_index(self,data : Data):
        
        if self.curvaturetype == "BFc_w4cycle":
            G_rewired,_ = sdrf_BFc(
                data,
                loops=self.hyperparameters["loops"],
                remove_edges=self.remove_edges,
                removal_bound=self.hyperparameters["C+"],
                tau=self.hyperparameters["tau"],
                int_node = self.intermediate_node,
                is_undirected=data.is_undirected(),
                fcc = True,
                progress_bar= False
                            )
            edge_index_rewired = torch_geometric.utils.to_undirected(torch.tensor(list(G_rewired.edges)).t())
        elif self.curvaturetype == "BFc_no4cycle":
            G_rewired,_ = sdrf_BFc(
                data,
                loops=self.hyperparameters["loops"],
                remove_edges=self.remove_edges,
                removal_bound=self.hyperparameters["C+"],
                tau=self.hyperparameters["tau"],
                int_node = self.intermediate_node,
                is_undirected=data.is_undirected(),
                fcc = False,
                progress_bar= False
                            )
            edge_index_rewired = torch_geometric.utils.to_undirected(torch.tensor(list(G_rewired.edges)).t())
        elif self.curvaturetype == "JTc":
            G_rewired,_ = sdrf_JTc(
                data,
                loops=self.hyperparameters["loops"],
                remove_edges=self.remove_edges,
                removal_bound=self.hyperparameters["C+"],
                tau=self.yperparameters["tau"],
                is_undirected=data.is_undirected(),
                progress_bar= False
                            )
            edge_index_rewired = torch_geometric.utils.to_undirected(torch.tensor(list(G_rewired.edges)).t())
        elif self.curvaturetype == "JLc":
            G_rewired,_ = sdrf_JLc(
                data,
                loops=self.hyperparameters["loops"],
                remove_edges=self.remove_edges,
                removal_bound=self.hyperparameters["C+"],
                tau=self.hyperparameters["tau"], 
                is_undirected=data.is_undirected(),
                progress_bar = False
                            )
            edge_index_rewired = torch_geometric.utils.to_undirected(torch.tensor(list(G_rewired.edges)).t())
        elif self.curvaturetype == "AFc_3":
            G_rewired,_ = sdrf_AFc(
                data,
                loops=self.hyperparameters["loops"],
                remove_edges=self.remove_edges,
                removal_bound=self.hyperparameters["C+"],
                tau=self.hyperparameters["tau"],
                is_undirected=data.is_undirected(),
                k = 3.,
                progress_bar= False
                            )
            edge_index_rewired = torch_geometric.utils.to_undirected(torch.tensor(list(G_rewired.edges)).t())
        elif self.curvaturetype == "AFc_4":
            G_rewired,_ = sdrf_AFc(
                data,
                loops=self.hyperparameters["loops"],
                remove_edges=self.remove_edges,
                removal_bound=self.hyperparameters["C+"],
                tau=self.hyperparameters["tau"],
                is_undirected=data.is_undirected(),
                k = 4,
                progress_bar= False
                            )
            edge_index_rewired = torch_geometric.utils.to_undirected(torch.tensor(list(G_rewired.edges)).t())
        return edge_index_rewired
    
def objective(config,dataset,rewire = False):
    accuracies = []
    test_acc = []
    if rewire:
        print("===Starting Rewiring===")
        
        dataset_rew = {}
        data_rew = []
        transform = torch_geometric.transforms.Compose([edge_rewiring_transform(config_hyper,False,True,"JcT")])
        for k in tqdm(range(len(dataset))):
            data_rew.append(transform(dataset[k]))
        dataset_rew["data"] = data_rew
        dataset_rew["num_classes"] =  dataset.num_classes
        print(" ")
    else:
        dataset_rew = dataset
    print(" == Starting Runs == ")
    for idx_k,k in tqdm(enumerate(val_seeds[:5])):

        dataset_size = len(dataset)
        train_size = int(train_fraction * dataset_size)
        validation_size = int(validation_fraction * dataset_size)
        test_size = dataset_size - train_size - validation_size
        development_dataset, test_dataset = random_split(dataset,[train_size + validation_size, test_size], generator=torch.Generator().manual_seed(development_seed)) 
        train_dataset, validation_dataset = random_split(development_dataset,[train_size,validation_size], generator=torch.Generator().manual_seed(k)) 

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

        
        Exp = GraphExperiment(device,dataset,config)
        counter = 0
        for epoch in range(1, Exp.epoch):
            loss = Exp.train(train_loader)
            
            val = Exp.eval(validation_loader)
            #wandb.log({"loss " + str(idx_k): loss, "val " + str(idx_k): val,"epoch": epoch})
            if epoch ==1:
                best_val = val
            elif epoch > 1 and val > best_val:
                best_val = val
                counter = 0
            else:
                counter += 1
            if counter > 100:
                break
        final_accuracy = Exp.eval(validation_loader)
        final_test_acc =  Exp.eval(test_loader)
        accuracies.append(final_accuracy)
        test_acc.append(final_test_acc)
    print("")
    return np.mean(np.array(accuracies)),np.mean(np.array(test_acc))

#def main():
#    wandb.init(dir = "")
#    acc,test_acc = objective(wandb.config,rewiring_run)
#    wandb.log({"mean accuracy": acc, "mean test accuracy": test_acc})

#sweep_id = "epj1z76g" # wandb.sweep(sweep=sweep_configuration, project="curvature")
#wandb.agent(sweep_id, project="Curvature_Neurips", function=main,count = 50)

#sweep_id = wandb.sweep(sweep=sweep_configuration, project="Curvature_Neurips")
#wandb.agent(sweep_id, function=main,count = 2)

config_hyper = {
        "learning_rate": 0.07468,
        "layers": [128],
        "dropout":0.214,
        "weight_decay":0.7837,
        "loops": 20,
        "C+": 16.881,
        "tau": 212
    } 

import time
start_time = time.time()

accuracies,test_accuracies = objective(config_hyper,dataset_graphs,rewiring_run)
print(f"Average Test accuracy: {test_accuracies*100:.2f}")

print("--- %s seconds ---" % (time.time() - start_time))