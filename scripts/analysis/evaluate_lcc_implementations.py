import csv
from pathlib import Path

import numpy as np
import torch_geometric
from scipy import sparse

from experiment_utils.largestconnectedcomponent import (
    get_largest_connected_component_networkx,
    get_largest_connected_component_pytorch,
    get_largest_connected_component_toppingetal,
)
from utils.load_datasets import load_data


DATASETS_TO_EVALUATE = [
    "Texas",
    "Cornell",
    "Wisconsin",
    "Chameleon",
    "Squirrel",
    "Cora",
    "Citeseer",
    "Pubmed",
]


def compute_pytorch_component_nodes(data, directed: bool, connectiontype: str) -> set[int]:
    """
    Reproduce the exact node-selection logic of get_largest_connected_component_pytorch,
    but return original node ids for direct comparison with other implementations.
    """
    adjacency = torch_geometric.utils.to_scipy_sparse_matrix(
        data.edge_index,
        num_nodes=data.num_nodes,
    )
    _, component = sparse.csgraph.connected_components(
        adjacency,
        directed=directed,
        connection=connectiontype,
    )

    _, counts = np.unique(component, return_counts=True)
    selected_component_label = counts.argsort()[-1:]
    subset_np = np.isin(component, selected_component_label)
    selected_nodes = np.where(subset_np)[0]
    return set(int(v) for v in selected_nodes.tolist())


def summarize_nodes(nodes: set[int], limit: int = 20) -> str:
    sorted_nodes = sorted(nodes)
    if len(sorted_nodes) <= limit:
        return str(sorted_nodes)
    return str(sorted_nodes[:limit]) + f" ... (+{len(sorted_nodes) - limit} more)"


def evaluate_dataset(dataset_name: str) -> list[dict]:
    dataset, data, graph = load_data(dataset_name)
    directed = not data.is_undirected()

    print("\n" + "=" * 100)
    print(f"Dataset: {dataset_name}")
    print(f"Nodes: {data.num_nodes} | Edges: {data.num_edges} | Directed: {directed}")

    topping_nodes = set(int(v) for v in get_largest_connected_component_toppingetal(data).tolist())

    rows = []

    if directed:
        nx_weak_nodes = set(
            get_largest_connected_component_networkx(
                connectiontype="weak",
                directed=True,
                G=graph,
            ).nodes()
        )
        nx_strong_nodes = set(
            get_largest_connected_component_networkx(
                connectiontype="strong",
                directed=True,
                G=graph,
            ).nodes()
        )

        # Call the original function for parity checks (size only, since returned subgraph is relabeled).
        pyt_weak_subgraph = get_largest_connected_component_pytorch(
            connectiontype="weak",
            directed=True,
            data=data,
            num_nodes=data.num_nodes,
        )
        pyt_strong_subgraph = get_largest_connected_component_pytorch(
            connectiontype="strong",
            directed=True,
            data=data,
            num_nodes=data.num_nodes,
        )

        pyt_weak_nodes = compute_pytorch_component_nodes(data, directed=True, connectiontype="weak")
        pyt_strong_nodes = compute_pytorch_component_nodes(data, directed=True, connectiontype="strong")

        print(f"Topping nodes (size={len(topping_nodes)}): {summarize_nodes(topping_nodes)}")
        print(f"NetworkX weak (size={len(nx_weak_nodes)}): {summarize_nodes(nx_weak_nodes)}")
        print(f"PyTorch weak (size={len(pyt_weak_nodes)}): {summarize_nodes(pyt_weak_nodes)}")
        print(f"NetworkX strong (size={len(nx_strong_nodes)}): {summarize_nodes(nx_strong_nodes)}")
        print(f"PyTorch strong (size={len(pyt_strong_nodes)}): {summarize_nodes(pyt_strong_nodes)}")

        print("Differences:")
        print(f"  - Topping vs NetworkX weak : {topping_nodes != nx_weak_nodes}")
        print(f"  - Topping vs PyTorch weak  : {topping_nodes != pyt_weak_nodes}")
        print(f"  - Topping vs NetworkX strong: {topping_nodes != nx_strong_nodes}")
        print(f"  - NetworkX weak vs PyTorch weak: {nx_weak_nodes != pyt_weak_nodes}")
        print(f"  - NetworkX strong vs PyTorch strong: {nx_strong_nodes != pyt_strong_nodes}")

        print("Parity checks with relabeled subgraph outputs:")
        print(f"  - weak size match: {pyt_weak_subgraph.num_nodes == len(pyt_weak_nodes)}")
        print(f"  - strong size match: {pyt_strong_subgraph.num_nodes == len(pyt_strong_nodes)}")

        rows.extend(
            [
                {
                    "dataset": dataset_name,
                    "directed": directed,
                    "implementation": "topping",
                    "connectiontype": "outgoing_traversal",
                    "component_size": len(topping_nodes),
                    "nodes": sorted(topping_nodes),
                },
                {
                    "dataset": dataset_name,
                    "directed": directed,
                    "implementation": "networkx",
                    "connectiontype": "weak",
                    "component_size": len(nx_weak_nodes),
                    "nodes": sorted(nx_weak_nodes),
                },
                {
                    "dataset": dataset_name,
                    "directed": directed,
                    "implementation": "pytorch_scipy",
                    "connectiontype": "weak",
                    "component_size": len(pyt_weak_nodes),
                    "nodes": sorted(pyt_weak_nodes),
                },
                {
                    "dataset": dataset_name,
                    "directed": directed,
                    "implementation": "networkx",
                    "connectiontype": "strong",
                    "component_size": len(nx_strong_nodes),
                    "nodes": sorted(nx_strong_nodes),
                },
                {
                    "dataset": dataset_name,
                    "directed": directed,
                    "implementation": "pytorch_scipy",
                    "connectiontype": "strong",
                    "component_size": len(pyt_strong_nodes),
                    "nodes": sorted(pyt_strong_nodes),
                },
            ]
        )

        return rows

    nx_nodes = set(
        get_largest_connected_component_networkx(
            connectiontype="weak",
            directed=False,
            G=graph,
        ).nodes()
    )

    pyt_subgraph = get_largest_connected_component_pytorch(
        connectiontype="weak",
        directed=False,
        data=data,
        num_nodes=data.num_nodes,
    )
    pyt_nodes = compute_pytorch_component_nodes(data, directed=False, connectiontype="weak")

    print(f"Topping nodes (size={len(topping_nodes)}): {summarize_nodes(topping_nodes)}")
    print(f"NetworkX nodes (size={len(nx_nodes)}): {summarize_nodes(nx_nodes)}")
    print(f"PyTorch nodes (size={len(pyt_nodes)}): {summarize_nodes(pyt_nodes)}")

    print("Differences:")
    print(f"  - Topping vs NetworkX: {topping_nodes != nx_nodes}")
    print(f"  - Topping vs PyTorch : {topping_nodes != pyt_nodes}")
    print(f"  - NetworkX vs PyTorch: {nx_nodes != pyt_nodes}")
    print("Parity check with relabeled subgraph output:")
    print(f"  - weak size match: {pyt_subgraph.num_nodes == len(pyt_nodes)}")

    rows.extend(
        [
            {
                "dataset": dataset_name,
                "directed": directed,
                "implementation": "topping",
                "connectiontype": "outgoing_traversal",
                "component_size": len(topping_nodes),
                "nodes": sorted(topping_nodes),
            },
            {
                "dataset": dataset_name,
                "directed": directed,
                "implementation": "networkx",
                "connectiontype": "weak",
                "component_size": len(nx_nodes),
                "nodes": sorted(nx_nodes),
            },
            {
                "dataset": dataset_name,
                "directed": directed,
                "implementation": "pytorch_scipy",
                "connectiontype": "weak",
                "component_size": len(pyt_nodes),
                "nodes": sorted(pyt_nodes),
            },
        ]
    )
    return rows


def save_summary(rows: list[dict], output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "dataset",
                "directed",
                "implementation",
                "connectiontype",
                "component_size",
                "nodes",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    all_rows = []
    skipped = []

    for dataset_name in DATASETS_TO_EVALUATE:
        try:
            all_rows.extend(evaluate_dataset(dataset_name))
        except Exception as exc:
            skipped.append((dataset_name, str(exc)))

    output_path = Path("results_data") / "lcc_implementation_components.csv"
    save_summary(all_rows, output_path)

    print("\n" + "=" * 100)
    print(f"Saved summary to: {output_path}")
    print(f"Evaluated datasets: {len(set(row['dataset'] for row in all_rows))}")
    if skipped:
        print("Skipped datasets:")
        for name, error in skipped:
            print(f"  - {name}: {error}")


if __name__ == "__main__":
    main()
