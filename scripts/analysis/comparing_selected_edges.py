from experiment_utils.largestconnectedcomponent import lcc_dataset
from utils.load_datasets import load_data

import torch
from numba import cuda as numba_cuda

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("using device: ", device)

import numpy as np
from tqdm import tqdm
import pandas as pd

import random as random
import os 
import argparse


import json as json
import itertools

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


def load_sdrf_backend():
    if not torch.cuda.is_available() or not numba_cuda.is_available():
        raise RuntimeError(
            "No CUDA device available for numba/torch. "
            "This script uses CUDA kernels; run it on a GPU node (not a CPU/login node)."
        )

    try:
        visible_gpus = list(numba_cuda.gpus)
    except Exception as exc:
        raise RuntimeError(f"Numba cannot enumerate CUDA GPUs: {exc}") from exc

    if len(visible_gpus) == 0:
        raise RuntimeError(
            "CUDA appears available, but no GPU is visible to Numba in this job context. "
            "Check GPU allocation and CUDA_VISIBLE_DEVICES."
        )

    try:
        numba_cuda.select_device(0)
    except Exception as exc:
        raise RuntimeError(
            f"Numba failed to select visible CUDA device 0: {exc}. "
            "This often indicates a scheduler/device-index mismatch."
        ) from exc

    from experiment_utils.sdrf_cudaexperiment import sdrf_BFc, sdrf_JTc, sdrf_JLc, sdrf_AFc

    return sdrf_BFc, sdrf_JTc, sdrf_JLc, sdrf_AFc


def save_results(path_save, all_counts_dictionary, all_run_records, all_pairwise_records):
    os.makedirs(path_save, exist_ok=True)

    all_counts_dataframe = pd.DataFrame(all_counts_dictionary)
    all_counts_dataframe.to_csv(path_save + "rewiring_edges_agreement.csv")

    all_run_records_dataframe = pd.DataFrame(all_run_records)
    all_run_records_dataframe.to_csv(path_save + "rewiring_edges_agreement_long.csv", index=False)

    all_pairwise_records_dataframe = pd.DataFrame(all_pairwise_records)
    all_pairwise_records_dataframe.to_csv(path_save + "rewiring_edges_pairwise_method_agreement_long.csv", index=False)

    if not all_pairwise_records_dataframe.empty:
        pairwise_mean = (
            all_pairwise_records_dataframe
            .groupby(["dataset", "method_a", "method_b"], as_index=False)["agreement_pct"]
            .mean()
        )
        pairwise_mean.to_csv(path_save + "rewiring_edges_pairwise_method_agreement_mean.csv", index=False)

        for dataset in pairwise_mean["dataset"].unique():
            dataset_pairs = pairwise_mean[pairwise_mean["dataset"] == dataset]
            methods = sorted(set(dataset_pairs["method_a"]).union(set(dataset_pairs["method_b"])))
            matrix = pd.DataFrame(index=methods, columns=methods, dtype=float)

            for _, row in dataset_pairs.iterrows():
                method_a = row["method_a"]
                method_b = row["method_b"]
                value = row["agreement_pct"]
                matrix.loc[method_a, method_b] = value
                matrix.loc[method_b, method_a] = value

            matrix.to_csv(path_save + f"rewiring_edges_pairwise_matrix_{dataset}.csv")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare rewiring edge selections across curvature methods."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Run only selected datasets (default: all).",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Override number of iterations per dataset.",
    )
    return parser.parse_args()

def calculate_agreement_percentage(list_a, list_b):
    
    """
    Calculate the agreement percentage between the content of two lists of edges.

    Args:
        list_a (list): List of edges.
        list_b (list): List of edges.

    Returns:
        float: Agreement percentage between the two lists of edges.
    """
    
    if len(list_a) == 0 or len(list_b) == 0:
        return np.nan

    # If one method stops early, compare on the shared prefix length.
    shared_length = min(len(list_a), len(list_b))
    truncated_a = list_a[:shared_length]
    truncated_b = list_b[:shared_length]

    # Create Counter objects to count the occurrences of each edge
    count_a = Counter(truncated_a)
    count_b = Counter(truncated_b)

    # Calculate the common edge count (minimum of counts in both lists)
    common_edges_count = sum(min(count_a[edge], count_b[edge]) for edge in count_a)

    # Calculate the agreement percentage
    agreement_percentage = (common_edges_count / shared_length) * 100

    return agreement_percentage

def comparing_rewiring_edges(
    datasetname,
    data,
    nr_loops_min_max,
    iterations,
    sdrf_BFc,
    sdrf_JTc,
    sdrf_JLc,
    sdrf_AFc,
):
    """
    args:
        datasetname (str): name of the dataset
        data: the dataset
        nr_loops_min_max (dict): dictionary with the minimum and maximum number of loops
        iterations (int): number of iterations to run the experiment
        sdrf_BFc, sdrf_JTc, sdrf_JLc, sdrf_AFc: rewiring functions from selected backend
    
    Returns:
        dict: dictionary with the agreement percentages of the different curvature methods
    
    """
    
    total_BFc_no4cycle,total_BFc_mod,total_JLc,total_AFc_3,total_AFc_4 = [],[],[],[],[]
    run_records = []
    pairwise_records = []

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
            remove_edges= False,
            tau=25000,
            is_undirected=data.is_undirected(),
            
            progress_bar=False,
            k = 4
                        )

        agreement_Bfc_no4cycle = calculate_agreement_percentage(edges_rewired_BFc_no4cycle, edges_rewired_BFc_w4cycle)
        agreement_BFc_mod = calculate_agreement_percentage(edges_rewired_JTc, edges_rewired_BFc_w4cycle)
        agreement_JLc = calculate_agreement_percentage(edges_rewired_JLc, edges_rewired_BFc_w4cycle)
        agreement_AFc_3 = calculate_agreement_percentage(edges_rewired_AFc_3, edges_rewired_BFc_w4cycle)
        agreement_AFc_4 = calculate_agreement_percentage(edges_rewired_AFc_4, edges_rewired_BFc_w4cycle)

        method_edges = {
            "BFc_w4cycle": edges_rewired_BFc_w4cycle,
            "BFc_no4cycle": edges_rewired_BFc_no4cycle,
            "BFc_mod": edges_rewired_JTc,
            "JLc": edges_rewired_JLc,
            "AFc_3": edges_rewired_AFc_3,
            "AFc_4": edges_rewired_AFc_4,
        }

        method_names = list(method_edges.keys())
        for method_a, method_b in itertools.combinations_with_replacement(method_names, 2):
            agreement_pair = calculate_agreement_percentage(method_edges[method_a], method_edges[method_b])
            pairwise_records.append(
                {
                    "dataset": datasetname,
                    "run": k,
                    "nr_loops": nr_loops,
                    "method_a": method_a,
                    "method_b": method_b,
                    "agreement_pct": agreement_pair,
                }
            )
        
        
        total_BFc_no4cycle.append(agreement_Bfc_no4cycle)
        total_BFc_mod.append(agreement_BFc_mod)
        total_JLc.append(agreement_JLc)
        total_AFc_3.append(agreement_AFc_3)
        total_AFc_4.append(agreement_AFc_4)

        run_records.extend([
            {
                "dataset": datasetname,
                "run": k,
                "nr_loops": nr_loops,
                "method": "BFc_no4cycle",
                "agreement_pct": agreement_Bfc_no4cycle,
                "selected_edges": len(edges_rewired_BFc_no4cycle),
                "baseline_edges": len(edges_rewired_BFc_w4cycle)
            },
            {
                "dataset": datasetname,
                "run": k,
                "nr_loops": nr_loops,
                "method": "BFc_mod",
                "agreement_pct": agreement_BFc_mod,
                "selected_edges": len(edges_rewired_JTc),
                "baseline_edges": len(edges_rewired_BFc_w4cycle)
            },
            {
                "dataset": datasetname,
                "run": k,
                "nr_loops": nr_loops,
                "method": "JLc",
                "agreement_pct": agreement_JLc,
                "selected_edges": len(edges_rewired_JLc),
                "baseline_edges": len(edges_rewired_BFc_w4cycle)
            },
            {
                "dataset": datasetname,
                "run": k,
                "nr_loops": nr_loops,
                "method": "AFc_3",
                "agreement_pct": agreement_AFc_3,
                "selected_edges": len(edges_rewired_AFc_3),
                "baseline_edges": len(edges_rewired_BFc_w4cycle)
            },
            {
                "dataset": datasetname,
                "run": k,
                "nr_loops": nr_loops,
                "method": "AFc_4",
                "agreement_pct": agreement_AFc_4,
                "selected_edges": len(edges_rewired_AFc_4),
                "baseline_edges": len(edges_rewired_BFc_w4cycle)
            },
        ])
        
    return {
        'BFc_no4cycle': total_BFc_no4cycle,
        'BFc_mod': total_BFc_mod,
        'JLc': total_JLc,
        'AFc_3': total_AFc_3,
        'AFc_4': total_AFc_4,
        'run_records': run_records,
        'pairwise_records': pairwise_records
    }


"""
Experiment details
"""

number_of_iterations = 10
nr_loops = {
    'Texas': {"min":71,"max":107},
    'Cornell':{"min":100,"max":151},
    'Wisconsin':{"min":108,"max":163},
    'Chameleon':{"min":665,"max":999},
    'Squirrel':{"min":2193,"max":3291}, 
    'Actor': {"min":2599,"max":3899},
    'Cora': {"min":80,"max":120},
    'Citeseer':{"min":67,"max":101},
    'Pubmed':{"min":92,"max":138},
}


def main():
    args = parse_args()

    sdrf_BFc, sdrf_JTc, sdrf_JLc, sdrf_AFc = load_sdrf_backend()

    selected_datasets = list(nr_loops.keys()) if args.datasets is None else args.datasets
    invalid_datasets = [name for name in selected_datasets if name not in nr_loops]
    if invalid_datasets:
        raise ValueError(
            f"Unknown datasets: {invalid_datasets}. Valid options are: {list(nr_loops.keys())}"
        )

    iterations = number_of_iterations if args.iterations is None else args.iterations

    """
    Running experiment
    """
    all_counts_dictionary = {}
    all_run_records = []
    all_pairwise_records = []

    for dataset_name in selected_datasets:
        dataset, data, G = load_data(dataset_name)

        dataset_lcc = lcc_dataset(dataset, to_undirected=True)
        data_lcc = dataset_lcc[0]

        all_counts = comparing_rewiring_edges(
            dataset_name,
            data_lcc,
            nr_loops[dataset_name],
            iterations,
            sdrf_BFc,
            sdrf_JTc,
            sdrf_JLc,
            sdrf_AFc,
        )

        all_run_records.extend(all_counts['run_records'])
        all_pairwise_records.extend(all_counts['pairwise_records'])

        # Keep wide-format output backward-compatible.
        all_counts.pop('run_records', None)
        all_counts.pop('pairwise_records', None)

        all_counts_dictionary[dataset_name] = all_counts

        save_results(path_save, all_counts_dictionary, all_run_records, all_pairwise_records)
        print(f"Checkpoint saved after dataset: {dataset_name}")

    """
    Final save
    """
    save_results(path_save, all_counts_dictionary, all_run_records, all_pairwise_records)


if __name__ == "__main__":
    main()