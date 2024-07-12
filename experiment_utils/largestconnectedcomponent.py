import networkx as nx
import numpy as np
from scipy import sparse

import torch
import torch_geometric
from torch_geometric.data import Data

def get_largest_connected_component_pytorch(connectiontype: str, directed ,data,num_nodes:int):
    """
    Pytorch (backend Scipy) implementation of lcc calculation
    
    Args:
    edge_index (torch tensor): edge index of the graph
    num nodes (int): number of nodes of the graph
    directed: (boolean): Wether input graph is directed or not.  If directed == False, connectiontype keyword is not referenced.
    connectiontype (str): Only relevant for directed graphs. Weak or Strong constrain for largest connected component

    Returns:
    Subgraph (Networkx Graph): Subgraph which corresponds to the largest connected component of the graph
    """

    adj = torch_geometric.utils.to_scipy_sparse_matrix(data.edge_index, num_nodes=num_nodes)
    num_components, component = sparse.csgraph.connected_components(adj, directed=directed, connection=connectiontype)
    #print("Total Number of Components: ",num_components)

    _, count = np.unique(component, return_counts=True)
    subset_np = np.in1d(component, count.argsort()[-1:])
    subset = torch.from_numpy(subset_np)
    subset = subset.to(data.edge_index.device, torch.bool)
    Subgraph = torch_geometric.utils.to_networkx(data.subgraph(subset))
    #print("Largest Connected Component size: ", len(Subgraph.nodes))
    return data.subgraph(subset)#Subgraph

def get_largest_connected_component_networkx(connectiontype: str,directed, G):
    """
    Network implementation of lcc calculation
    
    Args:
    G (networkx graph): Input Graph
    connectiontype (str or None): Only relevant for directed graphs. Weak or Strong constrain for largest connected component

    Returns:
    Subgraph (Networkx Graph): Subgraph which corresponds to the largest connected component of the graph
    """
    if not directed:
        return G.subgraph(max(nx.connected_components(G), key=len)).copy()
    elif directed and connectiontype == 'strong':
        return G.subgraph(max(nx.strongly_connected_components(G), key=len)).copy()
    elif directed and connectiontype == 'weak':
        return G.subgraph(max(nx.weakly_connected_components(G), key=len)).copy()
    
def get_component_toppingetal(data, start: int = 0) -> set:
    visited_nodes = set()
    queued_nodes = set([start])
    row, col = data.edge_index.numpy()
    while queued_nodes:
        current_node = queued_nodes.pop()
        visited_nodes.update([current_node])
        neighbors = col[np.where(row == current_node)[0]]
        neighbors = [
            n for n in neighbors if n not in visited_nodes and n not in queued_nodes
        ]
        queued_nodes.update(neighbors)
    return visited_nodes

def get_largest_connected_component_toppingetal(data):
    
    remaining_nodes = set(range(data.x.shape[0]))
    comps = []
    while remaining_nodes:
        start = min(remaining_nodes)
        comp = get_component_toppingetal(data, start)
        comps.append(comp)
        remaining_nodes = remaining_nodes.difference(comp)
    return np.array(list(comps[np.argmax(list(map(len, comps)))]))

def remap_edges(edges: list, mapper: dict) -> list:
    row = [e[0] for e in edges]
    col = [e[1] for e in edges]
    row = list(map(lambda x: mapper[x], row))
    col = list(map(lambda x: mapper[x], col))
    return [row, col]

def get_node_mapper(lcc: np.ndarray) -> dict:
    mapper = {}
    counter = 0
    for node in lcc:
        mapper[node] = counter
        counter += 1
    return mapper

def lcc_dataset(dataset,to_undirected = True):
    lcc = get_largest_connected_component_toppingetal(dataset[0])

    x_new = dataset.data.x[lcc]
    y_new = dataset.data.y[lcc]

    row, col = dataset.data.edge_index.numpy()
    edges = [[i, j] for i, j in zip(row, col) if i in lcc and j in lcc]
    edges = remap_edges(edges, get_node_mapper(lcc))

    if to_undirected:
        data = Data(
            x=x_new,
            edge_index=torch_geometric.utils.to_undirected(torch.LongTensor(edges)),
            y=y_new,
            train_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
            test_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
            val_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
             )
    else:
        data = Data(
            x=x_new,
            edge_index=torch.LongTensor(edges),
            y=y_new,
            train_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
            test_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
            val_mask=torch.zeros(y_new.size()[0], dtype=torch.bool),
             )
    dataset.data = data

    mapping = dict(
        zip(np.unique(dataset.data.y), range(len(np.unique(dataset.data.y))))
    )
    dataset.data.y = torch.LongTensor([mapping[u] for u in np.array(dataset.data.y)])

    return dataset