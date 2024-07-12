from experiment_utils.sdrf_cudaexperiment import sdrf_BFc,sdrf_JTc,sdrf_JLc,sdrf_AFc

from experiment_utils.largestconnectedcomponent import lcc_dataset
from utils.load_datasets import load_data

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device: ", device)

import numpy as np
from tqdm import tqdm
import pandas as pd

import random as random
import os 


import json as json

os.environ['NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS']='False'

from collections import Counter

"""
Determing saving path
"""

hpc_cluster = False

if hpc_cluster:
    path_save = "/rhea/scratch/brussel/101/vsc10124/Curvature/results_data/"
else:
    path_save = "results_data/"

def calculate_agreement_percentage(list_a, list_b):
    
    """
    Calculate the agreement percentage between the content of two lists of edges.

    Args:
        list_a (list): List of edges.
        list_b (list): List of edges.

    Returns:
        float: Agreement percentage between the two lists of edges.
    """
    
    # Create Counter objects to count the occurrences of each edge
    count_a = Counter(list_a)
    count_b = Counter(list_b)


    if len(list_a) == len(list_b):
   
        # Calculate the common edge count (minimum of counts in both lists)
        common_edges_count = sum(min(count_a[edge], count_b[edge]) for edge in count_a)
    
        # Calculate the agreement percentage
        agreement_percentage = ( common_edges_count / len(list_a)) * 100

        return agreement_percentage
    else:
        raise ValueError(
            f"Edge count mismatch: {len(list_a)} in list_a, { len(list_b)} in list_b")

def comparing_rewiring_edges(datasetname,data,nr_loops_min_max,iterations):
    """
    args:
        datasetname (str): name of the dataset
        data: the dataset
        nr_loops_min_max (dict): dictionary with the minimum and maximum number of loops
        iterations (int): number of iterations to run the experiment
    
    Returns:
        dict: dictionary with the agreement percentages of the different curvature methods
    
    """
    
    total_BFc_no4cycle,total_JTc,total_JLc,total_AFc_3,total_AFc_4 = [],[],[],[],[]

    for k in tqdm(range(iterations)):
        nr_loops = random.randint(nr_loops_min_max["min"], nr_loops_min_max["max"]) 
        
        _,_,_,_,_,edges_rewired_BFc_w4cycle = sdrf_BFc(
            data,
            loops=nr_loops,
            remove_edges= False,
            tau=25000,
            int_node = False,
            is_undirected=data.is_undirected(),
            fcc = True,
            progress_bar=False
                            )
        _,_,_,_,_,edges_rewired_BFc_no4cycle = sdrf_BFc(
            data,
            loops=nr_loops,
            remove_edges= False,
            tau=25000,
            int_node = False,
            is_undirected=data.is_undirected(),
            fcc = False,
            progress_bar=False
                                    )
        _,_,edges_rewired_JTc,_,_ = sdrf_JTc(
            data,
            loops=nr_loops,
            remove_edges= False,
            tau=25000,
            is_undirected=data.is_undirected(),
            progress_bar=False,
            computespectralgap = False
                                    )
        _,_,edges_rewired_JLc,_ = sdrf_JLc(
            data,
            loops=nr_loops,
            remove_edges= False,
            tau=25000,
            is_undirected=data.is_undirected(),
            progress_bar=False,
            computespectralgap = False
                                    )    
        _,_,edges_rewired_AFc_3,_ = sdrf_AFc(
            data,
            loops=nr_loops,
            remove_edges= False,
            tau=25000,
            is_undirected=data.is_undirected(),
            progress_bar=False,
            k = 3
                            )

        _,_,edges_rewired_AFc_4,_ = sdrf_AFc(
            data,
            loops=nr_loops,
            remove_edges= True,
            removal_bound=-20,
            tau=25000,
            is_undirected=data.is_undirected(),
            
            progress_bar=False,
            k = 4
                        )

        agreement_Bfc_no4cycle = calculate_agreement_percentage(edges_rewired_BFc_no4cycle, edges_rewired_BFc_w4cycle)
        agreement_JTc = calculate_agreement_percentage(edges_rewired_JTc, edges_rewired_BFc_w4cycle)
        agreement_JLc = calculate_agreement_percentage(edges_rewired_JLc, edges_rewired_BFc_w4cycle)
        agreement_AFc_3 = calculate_agreement_percentage(edges_rewired_AFc_3, edges_rewired_BFc_w4cycle)
        agreement_AFc_4 = calculate_agreement_percentage(edges_rewired_AFc_4, edges_rewired_BFc_w4cycle)
        
        
        total_BFc_no4cycle.append(agreement_Bfc_no4cycle)
        total_JTc.append(agreement_JTc)
        total_JLc.append(agreement_JLc)
        total_AFc_3.append(agreement_AFc_3)
        total_AFc_4.append(agreement_AFc_4)    
        
    return {'BFc_no4cycle':total_BFc_no4cycle,'BFc_mod':total_JTc,'JLc':total_JLc,'AFc_3':total_AFc_3,'AFc_4':total_AFc_4}


"""
Experiment details
"""

number_of_iterations = 30
dataset_names = ["Texas","Cornell","Wisconsin","Chameleon","Cora","Citeseer","Pubmed"]

nr_loops = {
    'Texas': {"min":71,"max":107},
    'Cornell':{"min":100,"max":151},
    'Wisconsin':{"min":108,"max":163},
    'Chameleon':{"min":665,"max":999},
    'Cora': {"min":80,"max":120},
    'Citeseer':{"min":67,"max":101},
    'Pubmed':{"min":92,"max":138}
}


"""
Running experiment
"""
all_counts_dictionary = {}

for dataset_name in dataset_names:
    dataset,data,G = load_data(dataset_name)

    dataset_lcc = lcc_dataset(dataset, to_undirected = True)
    data_lcc = dataset_lcc[0]

    all_counts = comparing_rewiring_edges(dataset_name,data_lcc,nr_loops[dataset_name],number_of_iterations)

    all_counts_dictionary[dataset_name] = all_counts

"""
Saving results
"""
all_counts_dataframe = pd.DataFrame(all_counts_dictionary)
all_counts_dataframe.to_csv(path_save + "rewiring_edges_agreement.csv")