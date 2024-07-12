import numpy as np
import networkx as nx
from tqdm import tqdm

from scipy import sparse

import torch

from .curvatures import bfc_scipy
import torch_geometric

from joblib import Parallel, delayed


def softmax(a, tau=1):
    exp_a = np.exp(a * tau)
    return exp_a / exp_a.sum()

def rewiring_worker(cand_batch,C_og,A_sub,ind_x,ind_y,cons_fc):
    #candidates_xy_batch = candidates_xy[batches[count-1]:batches[count]]

    idxes_cand = A_sub.shape[0]*np.array([*range(len(cand_batch))])
    indices_cand = np.tile(idxes_cand,(2,1))+np.array(cand_batch).T


    sparse_id = sparse.eye(len(cand_batch))
    sparse_A_sub = sparse.dok_array(A_sub)

    sparse_A_submulti = sparse.dok_array(sparse.kron(sparse_id,sparse_A_sub))
    sparse_A_submulti[indices_cand[0],indices_cand[1]]=sparse_A_submulti[indices_cand[1],indices_cand[0]] = 1

    G_submulti = nx.from_scipy_sparse_array(sparse_A_submulti)
    G_submulti = G_submulti.to_undirected()

    edge_indx = np.array([ind_x,ind_y])
    edge_tss = np.tile(edge_indx,(len(cand_batch),1))
    edge_multi = edge_tss +np.tile(idxes_cand,(2,1)).T


    C_multisparse = bfc_scipy(G_submulti,cons_fc,False,edge_multi)
    C_post_rew = torch.tensor(C_multisparse.tocsr()[edge_multi.T[0],edge_multi.T[1]])
    C_og_tensor = torch.full((1,len(cand_batch)),C_og)
    improvements = (C_post_rew -C_og_tensor)[0]
    
    return improvements

def sdrf_scipy(
    data,
    loops=10,
    remove_edges=False,
    removal_bound=0.5,
    tau=1,
    consider_four_loops = False,
    batching = False,
    int_node = False,
    is_undirected=False,
    num_cores = 1
):
    N = data.num_nodes#.x.shape[0]

    G_in = torch_geometric.utils.to_networkx(data)
    if is_undirected:
        G_in = G_in.to_undirected()

    count_edge_removal = 0
    for x in tqdm(range(loops)):
        count_new_node = len(G_in.nodes)
        can_add = True
        C,Csparse = bfc_scipy(G_in,consider_four_loops)
        C = np.array(C,dtype = np.float64)

        ix_min = C.argmin()

        x = ix_min // N
        y = ix_min % N


        if is_undirected:
            x_neighbors = list(G_in.neighbors(x)) + [x] # !! We're adding x to the set of neighbours
            y_neighbors = list(G_in.neighbors(y)) + [y]
        else:
            x_neighbors = list(G_in.successors(x)) + [x]
            y_neighbors = list(G_in.predecessors(y)) + [y]

        candidates = []

        #D = torch.zeros(len(x_neighbors), len(y_neighbors)).cuda()
        for i in x_neighbors:
            for j in y_neighbors:
                if (i != j) and (not G_in.has_edge(i, j)):
                    candidates.append((i, j))

        Ball_x = set(x_neighbors)
        Ball_y = set(y_neighbors)
        TotalBall_xy = list(Ball_x.union(Ball_y))
        G_xysubgraph =  G_in.subgraph(TotalBall_xy).copy()

        A_xy =torch.tensor(nx.adjacency_matrix(G_xysubgraph).todense())

        G_xysubgraph_nodes =list(G_xysubgraph.nodes)

        indx = G_xysubgraph_nodes.index(x)
        indy = G_xysubgraph_nodes.index(y)
        cand_xy1 = np.array(candidates)[:,0]
        cand_xy2 = np.array(candidates)[:,1]

        ind_cand_xy1 = np.array([np.where(np.array(G_xysubgraph_nodes) == elem)[0].item() for elem in cand_xy1])
        ind_cand_xy2 = np.array([np.where(np.array(G_xysubgraph_nodes) == elem)[0].item() for elem in cand_xy2])

        candidates_xy_array =  np.vstack((ind_cand_xy1,ind_cand_xy2)).T
        candidates_xy = [*map(tuple, candidates_xy_array)]
        if len(candidates_xy):
            if batching and len(candidates_xy) > 200:
                batch_size = int(len(candidates_xy)/(num_cores))
                if batch_size ==0:
                    batch_size = num_cores
                batches = np.concatenate([np.arange(0,len(candidates_xy),batch_size,dtype = int), np.array([len(candidates_xy)])])
                
                candidates_arranged_batches = [candidates_xy[batches[count-1]:batches[count]] for count in range(1,len(batches))]
        
        
                results = Parallel(n_jobs=1)(delayed(rewiring_worker)(i,C[x,y],A_xy,indx,indy,consider_four_loops) for i in candidates_arranged_batches)
            
                improvementstotal = torch.hstack(results)
            else:
                improvementstotal = rewiring_worker(candidates_xy,C[x,y],A_xy,indx,indy,consider_four_loops)
                
            k, l = candidates[np.random.choice(range(len(candidates)), p=softmax(improvementstotal, tau=tau))] ##For directed graph: Makes sense: k is selected uit of "i" and "l" out of j

            if int_node:
                G_in.add_node(count_new_node)
                G_in.add_edge(k,count_new_node)
                G_in.add_edge(count_new_node, l)
            else:
                G_in.add_edge(k, l)
            del(G_xysubgraph)

        else:
            can_add = False
            if not remove_edges:
                break

        if remove_edges:
            ix_max = C.argmax()
            xmax = ix_max // N
            ymax = ix_max % N
            if C[xmax, ymax] > removal_bound:
                G_in.remove_edge(xmax, ymax)

                #if is_undirected:
                #    A[x, y] = A[y, x] = 0
                #else:
                #    A[x, y] = 0
                count_edge_removal += 1
            else:
                if can_add is False:
                    break
    C,_ = bfc_scipy(G_in,consider_four_loops)
    print(G_in)
    return G_in,C,count_edge_removal