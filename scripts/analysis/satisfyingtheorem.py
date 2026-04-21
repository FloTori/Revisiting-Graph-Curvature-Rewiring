from experiment_utils.largestconnectedcomponent import lcc_dataset
from utils.load_datasets import load_data, data_information
from experiment_utils.sdrf_cudaexperiment import sdrf_BFc
from experiment_utils.curvatures_cudaexperiment import BF_curvature_undirected

import torch_geometric
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path

"""
Run the theorem-satisfaction checks on the Texas dataset.
"""


def evaluate_conditions(theorem_terms):
    satisfy_conditions = []
    for sqrt_degree_xy, nr_triangles_xy, gamma_max_xy, delta_max_xy in theorem_terms:
        if gamma_max_xy != 0:
            condition1 = (delta_max_xy < 1 / sqrt_degree_xy) and (delta_max_xy < 1 / gamma_max_xy)
            if nr_triangles_xy != 0:
                condition2 = (delta_max_xy <= 1 / nr_triangles_xy) and (delta_max_xy < 1 / gamma_max_xy)
            else:
                condition2 = delta_max_xy < 1 / gamma_max_xy
        else:
            condition1 = delta_max_xy < 1 / sqrt_degree_xy
            if nr_triangles_xy != 0:
                condition2 = delta_max_xy <= 1 / nr_triangles_xy
            else:
                condition2 = True

        satisfy_conditions.append((condition1, condition2))

    return satisfy_conditions


def random_edge_baseline(data, loops, seed=0):
    graph = torch_geometric.utils.to_networkx(data).to_undirected()
    adjacency = torch.tensor(nx.adjacency_matrix(graph).todense(), dtype=torch.float).cuda()
    edge_index = data.edge_index.clone().cuda()
    node_count = adjacency.shape[0]

    curvature = torch.zeros(node_count, node_count).cuda()
    curvature, sqrt_degree, nr_triangles, gamma_max = BF_curvature_undirected(
        adjacency,
        edge_index,
        C=curvature,
        fcc=True,
    )
    delta_max = curvature + 2

    row = data.edge_index[0].cpu().numpy()
    col = data.edge_index[1].cpu().numpy()
    undirected_mask = row < col
    candidate_edges = np.column_stack((row[undirected_mask], col[undirected_mask]))

    if candidate_edges.shape[0] == 0:
        candidate_edges = np.column_stack((row, col))

    sample_count = min(loops, candidate_edges.shape[0])
    random_generator = np.random.default_rng(seed)
    selected_idx = random_generator.choice(candidate_edges.shape[0], size=sample_count, replace=False)
    selected_edges = candidate_edges[selected_idx]

    theorem_terms = []
    for x, y in selected_edges:
        theorem_terms.append(
            (
                sqrt_degree[x, y].item(),
                nr_triangles[x, y].item(),
                gamma_max[x, y].item(),
                delta_max[x, y].item(),
            )
        )

    satisfy_conditions = evaluate_conditions(theorem_terms)
    print(f"Triangles when edges satisfy condition 2b: {[term[1] for term, cond in zip(theorem_terms, satisfy_conditions) if cond[1]]}")
    print(f"Gamma max when edges satisfy condition 2b: {[term[2] for term, cond in zip(theorem_terms, satisfy_conditions) if cond[1]]}")
    print(f"Delta max when edges satisfy condition 2b: {[term[3] for term, cond in zip(theorem_terms, satisfy_conditions) if cond[1]]}")
    return satisfy_conditions, theorem_terms, sample_count


def print_condition_summary(name, satisfy_conditions):
    conditions_array = np.array(satisfy_conditions)

    print(f"{name}:")
    print("-- CONDITION 2 --")
    condition_2_satisfied = conditions_array[:, 0]
    print(
        f"  - Satisfied: {sum(condition_2_satisfied)} "
        f"({sum(condition_2_satisfied) / len(satisfy_conditions) * 100:.2f}%)"
    )
    print(
        f"  - Unsatisfied: {len(satisfy_conditions) - sum(condition_2_satisfied)} "
        f"({(len(satisfy_conditions) - sum(condition_2_satisfied)) / len(satisfy_conditions) * 100:.2f}%)"
    )

    print("-- CONDITION 2b --")
    condition_2b_satisfied = conditions_array[:, 1]
    print(
        f"  - Satisfied: {sum(condition_2b_satisfied)} "
        f"({sum(condition_2b_satisfied) / len(satisfy_conditions) * 100:.2f}%)"
    )
    print(
        f"  - Unsatisfied: {len(satisfy_conditions) - sum(condition_2b_satisfied)} "
        f"({(len(satisfy_conditions) - sum(condition_2b_satisfied)) / len(satisfy_conditions) * 100:.2f}%)"
    )


def plot_curvature_distributions(
    sdrf_theorem_terms,
    sdrf_satisfy_conditions,
    random_theorem_terms,
    random_satisfy_conditions,
    dataset_name,
):
    sdrf_curvature = np.array([delta_max_xy - 2 for _, _, _, delta_max_xy in sdrf_theorem_terms])
    random_curvature = np.array([delta_max_xy - 2 for _, _, _, delta_max_xy in random_theorem_terms])

    sdrf_condition_2b = np.array(sdrf_satisfy_conditions)[:, 1].astype(bool)
    random_condition_2b = np.array(random_satisfy_conditions)[:, 1].astype(bool)

    sdrf_curvature_satisfied = sdrf_curvature[sdrf_condition_2b]
    sdrf_curvature_unsatisfied = sdrf_curvature[~sdrf_condition_2b]
    random_curvature_satisfied = random_curvature[random_condition_2b]
    random_curvature_unsatisfied = random_curvature[~random_condition_2b]

    min_curvature = min(sdrf_curvature.min(), random_curvature.min())
    max_curvature = max(sdrf_curvature.max(), random_curvature.max())
    bins = np.linspace(min_curvature, max_curvature, 30)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    axes[0].hist(
        sdrf_curvature_satisfied,
        bins=bins,
        alpha=0.65,
        label=f"Condition 2b satisfied (n={len(sdrf_curvature_satisfied)})",
        color="tab:green",
    )
    axes[0].hist(
        sdrf_curvature_unsatisfied,
        bins=bins,
        alpha=0.65,
        label=f"Condition 2b not satisfied (n={len(sdrf_curvature_unsatisfied)})",
        color="tab:red",
    )
    axes[0].set_title("SDRF-selected edges")
    axes[0].set_xlabel("Balanced Forman curvature")
    axes[0].set_ylabel("Count")
    axes[0].legend(fontsize=8)

    axes[1].hist(
        random_curvature_satisfied,
        bins=bins,
        alpha=0.65,
        label=f"Condition 2b satisfied (n={len(random_curvature_satisfied)})",
        color="tab:green",
    )
    axes[1].hist(
        random_curvature_unsatisfied,
        bins=bins,
        alpha=0.65,
        label=f"Condition 2b not satisfied (n={len(random_curvature_unsatisfied)})",
        color="tab:red",
    )
    axes[1].set_title("Randomly selected edges")
    axes[1].set_xlabel("Balanced Forman curvature")
    axes[1].legend(fontsize=8)

    fig.suptitle(f"Curvature distributions split by condition 2b ({dataset_name})")
    plt.tight_layout()

    output_dir = Path("results_figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dataset_name}_curvature_distribution_condition2b_split.pdf"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved curvature distribution plot to: {output_path}")


def main():
    dataset_name = "Texas"

    dataset, _, _ = load_data(dataset_name)
    dataset_lcc = lcc_dataset(dataset, to_undirected=True)
    data_lcc = dataset_lcc[0]

    data_information(dataset_lcc, data_lcc)

    data_lcc.edge_index = torch_geometric.utils.to_undirected(data_lcc.edge_index.long())

    loops = 89
    print(f"Running {loops} rewiring loops on {dataset_name}...")

    _, _, _, satisfy_conditions, theorem_terms, _ = sdrf_BFc(
        data_lcc,
        loops=loops,
        remove_edges=False,
        removal_bound=0,
        tau=25000,
        int_node=False,
        is_undirected=True,
        fcc=True,
        computespectralgap=False,
        progress_bar=False,
    )

    print("\n SDRF EDGES")
    
    print(f"Triangles when edges satisfy condition 2b: {[term[1] for term, cond in zip(theorem_terms, satisfy_conditions) if cond[1]]}")
    print(f"Gamma max when edges satisfy condition 2b: {[term[2] for term, cond in zip(theorem_terms, satisfy_conditions) if cond[1]]}")
    print(f"Delta max when edges satisfy condition 2b: {[term[3] for term, cond in zip(theorem_terms, satisfy_conditions) if cond[1]]}")
    print(f"Collected {len(satisfy_conditions)} condition checks")

    print("\n RANDOM EDGES")
    random_satisfy_conditions, random_theorem_terms, random_sample_count = random_edge_baseline(
        data_lcc,
        loops,
        seed=42,
    )
    
    print(f"Random baseline sampled {random_sample_count} edges")

    print_condition_summary("SDRF-selected edges theorem-satisfaction checks", satisfy_conditions)
    print_condition_summary("Randomly selected edges theorem-satisfaction checks", random_satisfy_conditions)
    plot_curvature_distributions(
        theorem_terms,
        satisfy_conditions,
        random_theorem_terms,
        random_satisfy_conditions,
        dataset_name,
    )

if __name__ == "__main__":
    main()
