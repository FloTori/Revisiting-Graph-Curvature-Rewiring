from experiment_utils.largestconnectedcomponent import lcc_dataset
from utils.load_datasets import load_data,data_information

import torch
import networkx as nx
import torch_geometric.utils 

#from experiment_utils.sdrf_cuda import sdrf_cuda_personal,sdrf_jctopping,sdrf_JcL
from experiment_utils.curvatures_cuda import BF_curvature_undirected,JT_curvature,JL_curvature,AF_curvature
from plotting.curvaturegraphs import CurvatureGraph
from plotting.degreegraphs import DegreeGraph,RewiringdistributionGraph

import os
import json
from scipy import stats
import numpy as np

#NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS=0
datasetname = "Citeseer"
results_dir = "results"


with open(os.path.join('experiment_utils\hyperparameters','hyperparameters_rewiring_jctopping.json'), 'r') as file:
     rewiring_parameters = json.load(file)
     rewiring_parameters = rewiring_parameters.get(datasetname, {})

loops = rewiring_parameters["max_iterations"]
tau = rewiring_parameters["tau"]
C_plus = rewiring_parameters["C+"]
make_undirected = True

dataset,data,G = load_data(datasetname)

A =  nx.adjacency_matrix(G).todense()

dataset_lcc = lcc_dataset(dataset,to_undirected = make_undirected)
data_lcc = dataset_lcc[0]
G_lcc = torch_geometric.utils.to_networkx(data_lcc)

if data_lcc.is_undirected:
    G_lcc = G_lcc.to_undirected()

A_lcc = nx.adjacency_matrix(G_lcc).todense()

data_information(dataset_lcc,data_lcc)

edge_upper = np.where(np.triu(A_lcc))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Currently using device: ", device)

Initial_C_JTc = JT_curvature(torch.tensor(A_lcc, dtype = torch.float).to(device))
Initial_C_JLc = JL_curvature(torch.tensor(A_lcc, dtype = torch.float).to(device))
Initial_C_BFc_w4cycle = BF_curvature_undirected(torch.tensor(A_lcc, dtype = torch.float).to(device), data_lcc.edge_index.to(device),fcc = True)
Initial_C_BFc_no4cycle = BF_curvature_undirected(torch.tensor(A_lcc, dtype = torch.float).to(device), data_lcc.edge_index.to(device),fcc = False)
Initial_C_AF_3 = AF_curvature(torch.tensor(A_lcc, dtype = torch.float).to(device), k =3)
Initial_C_AF_4 = AF_curvature(torch.tensor(A_lcc, dtype = torch.float).to(device), k =4)


CurvatureGraph(Initial_C_BFc_w4cycle.cpu(),Initial_C_BFc_no4cycle.cpu(),Initial_C_JTc.cpu(),
               Initial_C_JLc.cpu(),Initial_C_AF_3.cpu(),Initial_C_AF_4.cpu(),
               edge_upper, 
               ["BFc","BFc_no4cycle","BFc_mod","JLc","AFc_3","AFc_4"],
               [r"$BFc$",r"$BFc_{no4cycle}$",r"$BFc_{mod}$",r"$JLc$",r"$AFc_3$",r"$AFc_4$"],
               datasetname )#+ ' (undirected_'+ str(data_lcc.is_undirected()) +')')
