import numpy as np
import os
from torch_geometric.datasets import KarateClub,Planetoid,WebKB,Actor,WikipediaNetwork,Coauthor,Amazon
import torch_geometric as torch_geometric
import torch_geometric.transforms as T

from torch_geometric.datasets import TUDataset
DEFAULT_DATA_PATH = "data"

def get_dataset(
    name: str, data_dir=DEFAULT_DATA_PATH
):
    
    path = DEFAULT_DATA_PATH
    if name in ["Cora", "Citeseer", "Pubmed"]:
        dataset = Planetoid(path, name)
    elif name in ["Computers", "Photo"]:
        dataset = Amazon(path, name)
    elif name == "CoauthorCS":
        dataset = Coauthor(path, "CS")
    elif name in ["Cornell", "Texas", "Wisconsin"]:
        dataset = WebKB(path, name)
    elif name in ["Chameleon", "Squirrel"]:
        dataset = WikipediaNetwork(path, name, geom_gcn_preprocess=True)
    elif name == "Actor":
        dataset = Actor(path)
    elif name == "KarateClub":
        dataset = KarateClub()
    else:
        raise Exception(f"Unknown dataset: {name}") 

    return dataset


def get_dataset_graphs(name: str, data_dir=DEFAULT_DATA_PATH,make_undirected: bool = False):

    path = data_dir
    if name == "MUTAG":
        dataset = TUDataset(root=path, name=name)
    elif name == "ENZYMES":
        dataset = TUDataset(root=path, name=name)
    elif name == "PROTEINS":
        dataset = TUDataset(root=path, name=name)
    elif name == "IMDB-BINARY":
        dataset = TUDataset(root=path, name=name)
    else:
        raise Exception(f"Unknown dataset: {name}")
    
    return dataset

def load_data(name: str, make_undirected: bool = False):
    dataset = get_dataset(name)
    data = dataset[0]
    G = torch_geometric.utils.to_networkx(data)
    if data.is_undirected() or make_undirected:
        G = G.to_undirected() #This is for Networkx to represent it as a undirected Graph (Otherwise it would 'plot' i->j and j->i as two different edges)

    return dataset,data,G

def data_information(dataset,data):
    print()
    print(f'Dataset: {dataset}:')
    print('======================')

    print(f'Number of features: {dataset.num_features}')
    print(f'Number of classes: {dataset.num_classes}')
    print()
    
    # Gather some statistics about the graph.
    print(f'Number of nodes: {data.num_nodes}')
    print(f'Number of edges: {data.num_edges}')
    print(f'Average node degree: {data.num_edges / data.num_nodes:.2f}')
    print(f'Has isolated nodes: {data.has_isolated_nodes()}')
    print(f'Has self-loops: {data.has_self_loops()}')
    print(f'Is undirected: {data.is_undirected()}')