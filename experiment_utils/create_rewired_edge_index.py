import torch_geometric
from experiment_utils.sdrf_cuda import sdrf_BFc,sdrf_JTc,sdrf_JLc,sdrf_AFc
import torch


def create_rewired_edge_index(data,hyperparameters,intermediate_node,remove_edges,curvaturetype: str ):

    """
    Function to create a rewired edge index based on the curvature type

    Parameters:
    data (torch_geometric.data object): The graph data object
    hyperparameters (dict):             The hyperparameters for the curvature (loops,tau,C+) & for GNN training (lr,weight_decay,dropout,hidden_channels)
    intermediate_node (bool):           Whether to use an intermediate node when rewiring (i - k and k - j)
    remove_edges (bool):                Whether to remove edges when rewiring
    curvaturetype (str):                The curvature type to use (BFc, BFc_3, BFc_mod, JLc, AFc_3, AFc_4)
    
    Returns:
    G_rewired (nx.Graph): The rewired graph

    edge_index_rewired (torch.tensor):   The edge index of the rewired graph
    """
    if curvaturetype not in ["BFc","BFc_3","BFc_mod","JLc","AFc_3","AFc_4"]:
        raise NotImplementedError(
            f"{curvaturetype} not implemented.")
    
    if curvaturetype == "BFc":
        G_rewired,_ = sdrf_BFc(
            data,
            loops=hyperparameters["loops"],
            remove_edges=remove_edges,
            removal_bound=hyperparameters["C+"],
            tau=hyperparameters["tau"],
            int_node = intermediate_node,
            is_undirected=data.is_undirected(),
            fcc = True,
            progress_bar= False
                        )
        edge_index_rewired = torch_geometric.utils.to_undirected(torch.tensor(list(G_rewired.edges)).t())
    elif curvaturetype == "BFc_3":
        G_rewired,_ = sdrf_BFc(
            data,
            loops=hyperparameters["loops"],
            remove_edges=remove_edges,
            removal_bound=hyperparameters["C+"],
            tau=hyperparameters["tau"],
            int_node = intermediate_node,
            is_undirected=data.is_undirected(),
            fcc = False,
            progress_bar= False
                        )
        edge_index_rewired = torch_geometric.utils.to_undirected(torch.tensor(list(G_rewired.edges)).t())
    elif curvaturetype == "BFc_mod":
        G_rewired,_ = sdrf_JTc(
            data,
            loops=hyperparameters["loops"],
            remove_edges=remove_edges,
            removal_bound=hyperparameters["C+"],
            tau=hyperparameters["tau"],
            is_undirected=data.is_undirected(),
            progress_bar= False
                        )
        edge_index_rewired = torch_geometric.utils.to_undirected(torch.tensor(list(G_rewired.edges)).t())
    elif curvaturetype == "JLc":
        G_rewired,_ = sdrf_JLc(
            data,
            loops=hyperparameters["loops"],
            remove_edges=remove_edges,
            removal_bound=hyperparameters["C+"],
            tau=hyperparameters["tau"], 
            is_undirected=data.is_undirected(),
            progress_bar = False
                        )
        edge_index_rewired = torch_geometric.utils.to_undirected(torch.tensor(list(G_rewired.edges)).t())
    elif curvaturetype == "AFc_3":
        G_rewired,_ = sdrf_AFc(
            data,
            loops=hyperparameters["loops"],
            remove_edges=remove_edges,
            removal_bound=hyperparameters["C+"],
            tau=hyperparameters["tau"],
            is_undirected=data.is_undirected(),
            k = 3.,
            progress_bar= False
                        )
        edge_index_rewired = torch_geometric.utils.to_undirected(torch.tensor(list(G_rewired.edges)).t())
    elif curvaturetype == "AFc_4":
        G_rewired,_ = sdrf_AFc(
            data,
            loops=hyperparameters["loops"],
            remove_edges=remove_edges,
            removal_bound=hyperparameters["C+"],
            tau=hyperparameters["tau"],
            is_undirected=data.is_undirected(),
            k = 4,
            progress_bar= False
                        )
        edge_index_rewired = torch_geometric.utils.to_undirected(torch.tensor(list(G_rewired.edges)).t())
    
    return G_rewired,edge_index_rewired 