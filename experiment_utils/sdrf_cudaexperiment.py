import numpy as np
import networkx as nx
import math
from numba import cuda
from tqdm import tqdm 

import torch

from .curvatures_cudaexperiment import BF_curvature_undirected,JT_curvature,JL_curvature,AF_curvature
import torch_geometric

from scipy.sparse.linalg import eigs
from scipy.sparse import coo_array,csr_array
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csgraph

NUMBA_CUDA_LOW_OCCUPANCY_WARNINGS=0

def softmax(a, tau=1):
    #exp_a = np.exp(a * tau)
    #softmax_2 = exp_a / exp_a.sum()

    m = torch.nn.Softmax(dim = 0)
    softmax =m(torch.Tensor(a*tau))
    return softmax.numpy()

@cuda.jit(
    "void(float32[:,:], float32[:,:], int32[:,:], int32[:,:], float32, float32, int32, float32[:,:], int32, int32, int32[:], int32[:], int32, int32,boolean)"
)

def _BFc_post_rewiring_undirected( A, A2,edge_index,indices_neigh, d_in_x, d_out_y, N, D, x, y, i_neighbors, j_neighbors, dim_i, dim_j,fcc = True
):
    I, J = cuda.grid(2)

    if (I < dim_i) and (J < dim_j):
        i = i_neighbors[I]
        j = j_neighbors[J]
    
        if (i == j) or (A[i, j] != 0):
            D[I, J] = -1000
            return

        A_i_j = A[i, j]
        A_i_j += 1
        
        if i == x:
            d_in_x += 1
        elif j == y:
            d_out_y += 1


        if d_in_x > d_out_y:
            d_max = d_in_x
            d_min = d_out_y
        else:
            d_max = d_out_y
            d_min = d_in_x
        
        if d_min ==1:#d_in_x * d_out_y == 0:
            D[I, J] = 0
            return
        
        A2_x_y = A2[x, y]
        # Difference in triangles term
        if (x == i) and (A[j, y] != 0):
            A2_x_y += 1.
        elif (y == j) and (A[x, i] != 0):
            A2_x_y += 1.

        # Difference in four-cycles term
        ind1_x,ind2_x = indices_neigh[x,0], indices_neigh[x,1]
        neighs_x = edge_index[1,ind1_x:ind2_x]

        ind1_y,ind2_y = indices_neigh[y,0],indices_neigh[y,1]
        neighs_y = edge_index[1,ind1_y:ind2_y]

        D[I, J] = (
                (2 / d_max)
                + (2 / d_min)
                - 2 
                + (2 / d_max + 1 / d_min) * A2_x_y * A[x, y]
            )
        
        if fcc:

            sharp_xy = 0
            lambda_xy = 0


            A_x_j = A[x,j] + 0
            if i == x and y !=j: 
                A_x_j += 1

            for k_count in range(len(neighs_x)):     
                k = neighs_x[k_count]
        
                ind1_k = indices_neigh[k,0]
                ind2_k = indices_neigh[k,1]
                neighs_k = edge_index[1,ind1_k:ind2_k]
            
                if k != i and k != j and y !=i and y!=j:
                    A2_k_y = A2[k, y]
                elif k ==i and y !=j:
                    A2_k_y = A2[k, y] + A[j,y]
                elif k ==j and y !=i:
                    A2_k_y = A2[k, y] + A[i,y]
                elif k!=j and y==i:
                    A2_k_y = A2[k, y] + A[k,j]
                elif k!=i and y==j:
                    A2_k_y = A2[k, y] + A[k,i]
                elif (k ==i and y ==j) or (k ==j and y == i):
                    A2_k_y = A2[k, y] + +1*A[k,k] + 1*A[y,y]

                A_k_y = A[k,y] + 0
                A_x_k = A[x,k] + 0
            
                if  (k == i and j ==y) or (k == y and j == i):
                    A_k_y +=1
                if (i == x and k == j) or (x==j and k == i):
                    A_x_k +=1
            
                if A_x_k*(1-A_k_y) !=0 and k!=y:

                    had = 0
                    for l_count in range(len(neighs_k)): #This doesn't sum over j since we haven't adapted the edge index yet
                        l = neighs_k[l_count]
                        A_k_l = A[k,l] + 0
                        A_x_l = A[x,l] + 0
                        A_y_l = A[y,l] + 0 
                        if (k == i and l == j) or (k == j and l == i):
                            A_k_l +=1
                        if (x == i and l == j) or (x == j and l == i):
                            A_x_l +=1
                        if (y == i and l == j) or (y == j and l == i):
                            A_y_l +=1
                        had += A_k_l*A_x_l*A_y_l

                    TMP =A_x_k*(1-A_k_y)*(A2_k_y -had- 1)

                    if TMP > 0:
                        sharp_xy += 1
                        if TMP > lambda_xy: 
                            lambda_xy = TMP
        
            for w_count in range(len(neighs_y)):
                w = neighs_y[w_count]
                ind1_w = indices_neigh[w,0]
                ind2_w = indices_neigh[w,1]
                neighs_w = edge_index[1,ind1_w:ind2_w]

                if w != i and w != j and x !=i and x!=j:
                    A2_w_x = A2[w, x]
                elif w ==i and x !=j:
                    A2_w_x = A2[w, x] + A[j,x]
                elif w ==j and x !=i:
                    A2_w_x = A2[w, x] + A[i,x]
                elif w!=j and x==i:
                    A2_w_x = A2[w, x] + A[w,j]
                elif w!=i and x==j:
                    A2_w_x = A2[w, x] + A[w,i]
                elif (w ==i and x ==j) or (w ==j and x == i):
                    A2_w_x = A2[w, x] +1*A[w,w] + 1*A[x,x]

                A_x_w = A[x,w] + 0
                if  w ==j and x ==i:
                    A_x_w +=1
            
                A_y_w = A[y,w] + 0
                A_w_x = A[w,x] + 0
            
                if  (w == i and j ==y) or (w == y and j == i):
                        A_y_w +=1
                if (i == x and w == j) or (x==j and w == i):
                        A_w_x +=1
                    
                if A_y_w*(1-A_w_x) !=0 and w != x:
                    had = 0
                    for l_count in range(len(neighs_w)): # If w ==j (SHOULD NEVER HAPPEN), this doesn't sum over i since we haven't adapted the edge index yet
                        l = neighs_w[l_count] 
                        A_w_l = A[w,l] + 0
                        A_x_l = A[x,l] + 0
                        A_y_l = A[y,l] + 0 
                        if (w == i and l == j) or (w == j and l == i):
                            A_w_l +=1
                        if (x == i and l == j) or (x == j and l == i):
                            A_x_l +=1
                        if (y == i and l == j) or (y == j and l == i):
                            A_y_l +=1
 
                        had += A_w_l*A_x_l*A_y_l

                
                    TMP = A_y_w*(1-A_x_w)*(A2_w_x -had- 1)

                    if TMP > 0:
                        sharp_xy +=  1
                        if TMP > lambda_xy:
                            lambda_xy = TMP

        
            if lambda_xy > 0:
                D[I, J] += sharp_xy / (d_max * lambda_xy)


def BFc_post_rewiring_undirected(A, x, y,edge_index, i_neighbors, j_neighbors, D=None,is_undirected = False,fcc = True):

    N = A.shape[0]
    A2 = torch.matmul(A, A)
    d_in = A.sum(axis = 0)#A[:, x].sum()
    d_out = A.sum(axis = 1)#A[y].sum()
    if D is None:
        D = torch.zeros(len(i_neighbors), len(j_neighbors)).cuda()

    ind1 = 0 
    ind2 = 0
    index_tuples = []
    for k in range(N):
        ind2 += int(d_in[k].item())
        index_tuples.append((ind1,ind2))
        ind1 = ind2 
    index_tuples = torch.tensor(index_tuples).cuda()

    d_in = d_in[x]
    d_out = d_out[y]
    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(D.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(D.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)
    if is_undirected:
        _BFc_post_rewiring_undirected[blockspergrid, threadsperblock](
            A,
            A2,
            edge_index,
            index_tuples,
            d_in,
            d_out,
            N,
            D,
            x,
            y,
            np.array(i_neighbors),
            np.array(j_neighbors),
            D.shape[0],
            D.shape[1],
            fcc
        )
    else:
        print("Not implemented for directed graphs")
        return
    return D


def sdrf_BFc(
    data,
    loops=10,
    remove_edges=False,
    removal_bound=0.5,
    tau=1,
    int_node = False,
    is_undirected=False,
    fcc = True,
    computespectralgap = False,
    progress_bar = True
):
    N = data.num_nodes
    G_in = torch_geometric.utils.to_networkx(data)
    
    if is_undirected:
        G_in = G_in.to_undirected()

    
    count_edge_removal = 0

    A = torch.tensor(nx.adjacency_matrix(G_in).todense(), dtype = torch.float)
    A = A.cuda()

    

    edge_index = data.edge_index.clone()
    edge_index = edge_index.cuda()
    N = A.shape[0]

    C = torch.zeros(N, N).cuda()

    edges_rewired= []
    spectral_gap = []
    #edges_satisfy_condition = []

    
    #triangle_delta_true_gamma_max = []
    sqrt_degree_nr_triangles_gamma_max_delta_max = []
    G_and_edge_rewired = []
    satisfy_conditions = []
    for idx in tqdm(range(loops), disable=not progress_bar):

        count_new_node = len(G_in.nodes)
        can_add = True
        if is_undirected:
            #_,delta_bound,A_squared = BF_curvature_undirected(A,edge_index ,C=C,fcc = fcc)   
            _,sqrt_degree,nr_triangles,gamma_max = BF_curvature_undirected(A,edge_index ,C=C,fcc = fcc)   
        else:
            print("Not implemented for directed graphs")
            return  


        ix_min = C.argmin()

        """ 
        Computing delta condition
        """
        A_upper = np.triu(A.cpu())
        edge_index_upper = np.where(A_upper)
        #C_upper = C.cpu()[edge_index_upper]

        #sqrt_degree_upper = sqrt_degree.cpu()[edge_index_upper]
        #nr_triangles_upper = nr_triangles.cpu()[edge_index_upper]
        #gamma_max_upper = gamma_max.cpu()[edge_index_upper]
        delta_max = (C +2)


        #condition = (delta_true_upper <= delta_bound_upper) & (C_upper < -0.2)
        #edges_satisfy_condition.append(torch.sum(condition).item()/len(edge_index_upper[0]))
        
        #nr_triangles.append(A_squared.cpu()[edge_index_upper])
        #delta_true.append(delta_true_upper)
        
        
        """
        Starting rewiring
        """
        x =  torch.div(ix_min,N,rounding_mode='trunc')
        y = ix_min % N

        x = x.item()
        y = y.item()

        #edges_selected = torch.zeros(N,N)
        #edges_selected[x,y] = 1
        #edges_selected_list.append(edges_selected[edge_index_upper])
        #edges_rewired.append((x,y,(C[x,y].item() + 2 <= delta_bound[x,y].item()) & (delta_bound[x,y].item() != 0)))
        edges_rewired.append((x,y))
        sqrt_degree_xy = sqrt_degree[x,y].item()
        nr_triangles_xy = nr_triangles[x,y].item()
        gamma_max_xy = gamma_max[x,y].item()
        delta_max_xy = delta_max[x,y].item()
        sqrt_degree_nr_triangles_gamma_max_delta_max.append((sqrt_degree_xy, nr_triangles_xy,gamma_max_xy,delta_max_xy))

        
        if gamma_max[x,y]!=0:
            condition1 = (delta_max_xy < 1/sqrt_degree_xy) & (delta_max_xy < 1/gamma_max_xy)
            if nr_triangles_xy !=0:
                condition2 = (delta_max_xy <= 1/nr_triangles_xy) & (delta_max_xy< 1/gamma_max_xy)
            else:
                condition2 =  (delta_max_xy< 1/gamma_max_xy)
        else: 
            condition1 = (delta_max_xy < 1/sqrt_degree_xy) 
            if nr_triangles_xy !=0:
                condition2 = (delta_max_xy <= 1/nr_triangles_xy) 
            else:
                condition2 =  True #Then there is no bound on delta

        satisfy_conditions.append((condition1,condition2))
        #triangle_delta_true_gamma_max.append((A_squared[x,y].item(),(C[x,y] +2).item(),delta_bound[x,y].item()))

        if is_undirected:
            x_neighbors = list(G_in.neighbors(x)) + [x] # !! We're adding x to the set of neighbours
            y_neighbors = list(G_in.neighbors(y)) + [y]
        else:
            x_neighbors = list(G_in.successors(x)) + [x]
            y_neighbors = list(G_in.predecessors(y)) + [y]

        candidates = []


        for i in x_neighbors:
            for j in y_neighbors:
                if (i != j) and (not G_in.has_edge(i, j)):
                    candidates.append((i, j))

        if len(candidates):
            D = BFc_post_rewiring_undirected(A,x,y,edge_index,x_neighbors,y_neighbors,D=None,is_undirected=is_undirected,fcc = fcc)
            improvements = []
            for (i, j) in candidates:
                improvements.append(
                    (D-C[x,y])[x_neighbors.index(i), y_neighbors.index(j)].item()
                )
            k, l = candidates[np.random.choice(range(len(candidates)), p=softmax(np.array(improvements), tau=tau))] ##For directed graph: Makes sense: k is selected uit of "i" and "l" out of j        

            if int_node:
                G_in.add_node(count_new_node)
                G_in.add_edge(k,count_new_node)
                G_in.add_edge(count_new_node, l)   
            else:
                G_in.add_edge(k, l)
                if is_undirected:
                    A[k, l] = A[l, k] = 1.
                    edge_index=A.to_sparse().indices()
                else:
                    A[k, l] = 1.
                    edge_index=A.to_sparse().indices()

        else:
            can_add = False
            if not remove_edges:
                break

        if remove_edges:
            ix_max = C.argmax()
            xmax = torch.div(ix_max,N,rounding_mode='trunc').item()
            ymax = (ix_max % N).item()
            if C[xmax, ymax] > removal_bound:
                G_in.remove_edge(xmax, ymax)

                if is_undirected:
                    A[xmax, ymax] = A[ymax, xmax] = 0.
                    edge_index=A.to_sparse().indices()
                else:
                    A[xmax, ymax] = 0.
                    edge_index=A.to_sparse().indices()
                count_edge_removal += 1
                
            else:
                if can_add is False:
                    break
        
        """
        Storing all intermediate graps and edge_indices
        """
        G_and_edge_rewired.append((G_in,edge_index.cpu()))

        """
        Computing spectral Gap
        """
        if computespectralgap:
            A_cpu = A.cpu()
            sparse_A_cpu = csr_array(A_cpu)
            sparse_D_cpu = csr_array(np.diag(A_cpu.sum(axis=0)))
            
            sparse_L_cpu = sparse_D_cpu - sparse_A_cpu

            eigvalues = np.real(eigs(sparse_L_cpu,k = 2, which='SM', return_eigenvectors=False))
            spectral_gap.append(eigvalues)

    
    return G_and_edge_rewired,count_edge_removal,spectral_gap,satisfy_conditions,sqrt_degree_nr_triangles_gamma_max_delta_max,edges_rewired

@cuda.jit(
    "void(float32[:,:], float32[:,:], float32, float32, int32, float32[:,:], int32, int32, int32[:], int32[:], int32, int32)"
)
def _JTc_post_delta(
    A, A2, d_in_x, d_out_y, N, D, x, y, i_neighbors, j_neighbors, dim_i, dim_j
):
    I, J = cuda.grid(2)

    if (I < dim_i) and (J < dim_j):
        i = i_neighbors[I]
        j = j_neighbors[J]

        if (i == j) or (A[i, j] != 0):
            D[I, J] = -1000
            return

        # Difference in degree terms
        if j == x:
            d_in_x += 1
        elif i == y:
            d_out_y += 1

        if d_in_x * d_out_y == 0:
            D[I, J] = 0
            return

        if d_in_x > d_out_y:
            d_max = d_in_x
            d_min = d_out_y
        else:
            d_max = d_out_y
            d_min = d_in_x

        # Difference in triangles term
        A2_x_y = A2[x, y]
        if (x == i) and (A[j, y] != 0):
            A2_x_y += A[j, y]
        elif (y == j) and (A[x, i] != 0):
            A2_x_y += A[x, i]

        # Difference in four-cycles term
        sharp_ij = 0
        lambda_ij = 0
        for z in range(N):
            A_z_y = A[z, y] + 0
            A_x_z = A[x, z] + 0
            A2_z_y = A2[z, y] + 0
            A2_x_z = A2[x, z] + 0

            if (z == i) and (y == j):
                A_z_y += 1
            if (x == i) and (z == j):
                A_x_z += 1
            if (z == i) and (A[j, y] != 0):
                A2_z_y += A[j, y]
            if (x == i) and (A[j, z] != 0):
                A2_x_z += A[j, z]
            if (y == j) and (A[z, i] != 0):
                A2_z_y += A[z, i]
            if (z == j) and (A[x, i] != 0):
                A2_x_z += A[x, i]

            TMP = A_z_y * (A2_x_z - A_x_z) * A[x, y]
            if TMP > 0:
                sharp_ij += 1
                if TMP > lambda_ij:
                    lambda_ij = TMP

            TMP = A_x_z * (A2_z_y - A_z_y) * A[x, y]
            if TMP > 0:
                sharp_ij += 1
                if TMP > lambda_ij:
                    lambda_ij = TMP

        D[I, J] = (
            (2 / d_max) + (2 / d_min) - 2 + (2 / d_max + 1 / d_min) * A2_x_y * A[x, y]
        )
        if lambda_ij > 0:
            D[I, J] += sharp_ij / (d_max * lambda_ij)


def JTc_post_delta(A, x, y, i_neighbors, j_neighbors, D=None):
    N = A.shape[0]
    A2 = torch.matmul(A, A)
    d_in = A[:, x].sum()
    d_out = A[y].sum()
    if D is None:
        D = torch.zeros(len(i_neighbors), len(j_neighbors)).cuda()

    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(D.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(D.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    _JTc_post_delta[blockspergrid, threadsperblock](
        A,
        A2,
        d_in,
        d_out,
        N,
        D,
        x,
        y,
        np.array(i_neighbors),
        np.array(j_neighbors),
        D.shape[0],
        D.shape[1],
    )
    return D


def sdrf_JTc(
    data,
    loops=10,
    remove_edges=True,
    removal_bound=0.5,
    tau=1,
    is_undirected=False,
    computespectralgap = False,
    progress_bar = True
):
    edge_index = data.edge_index
    if is_undirected:
        edge_index = torch_geometric.utils.to_undirected(edge_index)
    A = torch_geometric.utils.to_dense_adj(torch_geometric.utils.remove_self_loops(edge_index)[0])[0]
    N = A.shape[0]
    G = torch_geometric.utils.to_networkx(data)
    if is_undirected:
        G = G.to_undirected()
    A = A.cuda()
    C = torch.zeros(N, N).cuda()

    edges_rewired = []
    count_edge_removal = 0
    spectral_gap = []
    triangle_delta_true = []
    
    for idx in tqdm(range(loops), disable=not progress_bar):
        can_add = True
        _,delta_bound,A_squared = JT_curvature(A, C=C)
        ix_min = C.argmin().item()
        x = ix_min // N
        y = ix_min % N
        edges_rewired.append((x,y))

        """
        Computing delta condition
        """

        triangle_delta_true.append((A_squared[x,y].item(),(C[x,y] +2).item()))

        if is_undirected:
            x_neighbors = list(G.neighbors(x)) + [x]
            y_neighbors = list(G.neighbors(y)) + [y]
        else:
            x_neighbors = list(G.successors(x)) + [x]
            y_neighbors = list(G.predecessors(y)) + [y]
        candidates = []
        for i in x_neighbors:
            for j in y_neighbors:
                if (i != j) and (not G.has_edge(i, j)):
                    candidates.append((i, j))

        if len(candidates):
            D = JTc_post_delta(A, x, y, x_neighbors, y_neighbors)
            improvements = []
            for (i, j) in candidates:
                improvements.append(
                    (D - C[x, y])[x_neighbors.index(i), y_neighbors.index(j)].item()
                )

            k, l = candidates[
                np.random.choice(
                    range(len(candidates)), p=softmax(np.array(improvements), tau=tau)
                )
            ]
            G.add_edge(k, l)
            if is_undirected:
                A[k, l] = A[l, k] = 1
            else:
                A[k, l] = 1
        else:
            can_add = False
            if not remove_edges:
                break

        if remove_edges:
            ix_max = C.argmax().item()
            x = ix_max // N
            y = ix_max % N
            if C[x, y] > removal_bound:
                G.remove_edge(x, y)
                if is_undirected:
                    A[x, y] = A[y, x] = 0
                else:
                    A[x, y] = 0
            else:
                if can_add is False:
                    break
                """
        Computing spectral Gap
        """
        if computespectralgap:
            A_cpu = A.cpu()
            sparse_A_cpu = csr_array(A_cpu)
            sparse_D_cpu = csr_array(np.diag(A_cpu.sum(axis=0)))
            
            sparse_L_cpu = sparse_D_cpu - sparse_A_cpu

            eigvalues = np.real(eigs(sparse_L_cpu,k = 4, which='SM', return_eigenvectors=False))
            spectral_gap.append(eigvalues)

        
    return G,count_edge_removal,edges_rewired,spectral_gap,triangle_delta_true           


@cuda.jit(
    "void(float32[:,:], float32[:,:], float32, float32, int32, float32[:,:], int32, int32, int32[:], int32[:], int32, int32)"
)
def _JLc_post_delta(
    A, A2, d_in_x, d_out_y, N, D, x, y, i_neighbors, j_neighbors, dim_i, dim_j
):
    I, J = cuda.grid(2)

    if (I < dim_i) and (J < dim_j):
        i = i_neighbors[I]
        j = j_neighbors[J]

        if (i == j) or (A[i, j] != 0):
            D[I, J] = -1000
            return

        # Difference in degree terms
        if j == x:
            d_in_x += 1.
        elif i == y:
            d_out_y += 1.

        if d_in_x * d_out_y == 0:
            D[I, J] = 0
            return

        if d_in_x > d_out_y:
            d_max = d_in_x
            d_min = d_out_y
        else:
            d_max = d_out_y
            d_min = d_in_x

        # Difference in triangles term
        A2_x_y = A2[x, y]
        if (x == i) and (A[j, y] != 0):
            A2_x_y += A[j, y]
        elif (y == j) and (A[x, i] != 0):
            A2_x_y += A[x, i]

        if (1 - 1/d_in_x - 1/d_out_y - A2_x_y/d_min) > 0.:
            term1 = 1 - 1/d_in_x - 1/d_out_y - A2_x_y/d_min
        else:
            term1 = 0 

        if (1. - 1./d_in_x - 1./d_out_y - A2_x_y/d_max) > 0.:
            term2 = 1 - 1/d_in_x - 1/d_out_y - A2_x_y/d_max
        else:
            term2 = 0

        D[I, J] = (- term1 - term2 + A2_x_y/d_max )
        #D[I, J] = (-term1 + A2_x_y/d_max )

def JLc_post_delta(A, x, y, i_neighbors, j_neighbors, D=None):
    N = A.shape[0]
    A2 = torch.matmul(A, A)
    d_in = A[:, x].sum()
    d_out = A[y].sum()
    if D is None:
        D = torch.zeros(len(i_neighbors), len(j_neighbors)).cuda()

    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(D.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(D.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    _JLc_post_delta[blockspergrid, threadsperblock](
        A,
        A2,
        d_in,
        d_out,
        N,
        D,
        x,
        y,
        np.array(i_neighbors),
        np.array(j_neighbors),
        D.shape[0],
        D.shape[1],
    )
    return D


def sdrf_JLc(
    data,
    loops=10,
    remove_edges=True,
    removal_bound=0.5,
    tau=1,
    is_undirected=False,
    computespectralgap = False,
    progress_bar = True
):
    edge_index = data.edge_index
    if is_undirected:
        edge_index = torch_geometric.utils.to_undirected(edge_index)
    A = torch_geometric.utils.to_dense_adj(torch_geometric.utils.remove_self_loops(edge_index)[0])[0]
    N = A.shape[0]
    G = torch_geometric.utils.to_networkx(data)
    if is_undirected:
        G = G.to_undirected()
    A = A.cuda()
    C = torch.zeros(N, N).cuda()

    count_edge_removal = 0
    edges_rewired = []
    spectral_gap = []

    for idx in tqdm(range(loops), disable=not progress_bar):
        can_add = True
        _,A_squared = JL_curvature(A, C=C)
        ix_min = C.argmin().item()
        x = ix_min // N
        y = ix_min % N
        
        edges_rewired.append((x,y))



        if is_undirected:
            x_neighbors = list(G.neighbors(x)) + [x]
            y_neighbors = list(G.neighbors(y)) + [y]
        else:
            x_neighbors = list(G.successors(x)) + [x]
            y_neighbors = list(G.predecessors(y)) + [y]
        candidates = []
        for i in x_neighbors:
            if (not G.has_edge(i, y)):
                candidates.append((i, y))
        for j in y_neighbors:
            if (not G.has_edge(x, j)):
                candidates.append((x, j))

        if len(candidates):
            D = JLc_post_delta(A, x, y, x_neighbors, y_neighbors)
            improvements = []
            for (i, j) in candidates:
                improvements.append(
                    (D - C[x, y])[x_neighbors.index(i), y_neighbors.index(j)].item()
                )

            k, l = candidates[
                np.random.choice(
                    range(len(candidates)), p=softmax(np.array(improvements), tau=tau)
                )
            ]
            G.add_edge(k, l)
            if is_undirected:
                A[k, l] = A[l, k] = 1
            else:
                A[k, l] = 1
        else:
            can_add = False
            if not remove_edges:
                break

        if remove_edges:
            ix_max = C.argmax().item()
            x = ix_max // N
            y = ix_max % N
            if C[x, y] > removal_bound:
                G.remove_edge(x, y)
                if is_undirected:
                    A[x, y] = A[y, x] = 0
                else:
                    A[x, y] = 0
            else:
                if can_add is False:
                    break
        if computespectralgap:
            A_cpu = A.cpu()
            sparse_A_cpu = csr_array(A_cpu)
            sparse_D_cpu = csr_array(np.diag(A_cpu.sum(axis=0)))
                
            sparse_L_cpu = sparse_D_cpu - sparse_A_cpu

            eigvalues = np.real(eigs(sparse_L_cpu,k = 5, which='SM', return_eigenvectors=False))
            spectral_gap.append(eigvalues)

    return G,count_edge_removal,edges_rewired,spectral_gap

@cuda.jit(
    "void(float32[:,:], float32[:,:], float32[:,:], float32, float32, float32, int32, float32[:,:], int32, int32, int32[:], int32[:], int32, int32)"
)
def _AFc_post_delta(
    A, A2,A3,k, d_in_x, d_out_y, N, D, x, y, i_neighbors, j_neighbors, dim_i, dim_j
):
    I, J = cuda.grid(2)

    if (I < dim_i) and (J < dim_j):
        i = i_neighbors[I]
        j = j_neighbors[J]

        if (i == j) or (A[i, j] != 0):
            D[I, J] = -1000
            return

        # Difference in degree terms
        if j == x:
            d_in_x += 1.
        elif i == y:
            d_out_y += 1.

        if d_in_x * d_out_y == 0:
            D[I, J] = 0
            return

        # Difference in triangles term
        A2_x_y = A2[x, y]
        if (x == i) and (A[j, y] != 0):
            A2_x_y += A[j, y]
        elif (y == j) and (A[x, i] != 0):
            A2_x_y += A[x, i]
        
        if k ==4:
            A3_x_y = A3[x,y]
            if (x!=i) and (y!=j):
                A3_x_y += 1

        if k == 3:
            D[I, J] = 4- d_in_x - d_out_y + 3*A2_x_y
        elif k ==4:
            D[I, J] = 4- d_in_x - d_out_y + 3*A2_x_y + 2*(A3_x_y - d_in_x - d_out_y +1)

def AFc_post_delta(A,k, x, y, i_neighbors, j_neighbors, D=None):
    N = A.shape[0]
    A2 = torch.matmul(A, A)
    A3 =  torch.matmul(A2, A)
    d_in = A[:, x].sum()
    d_out = A[y].sum()
    if D is None:
        D = torch.zeros(len(i_neighbors), len(j_neighbors)).cuda()

    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(D.shape[0] / threadsperblock[0])
    blockspergrid_y = math.ceil(D.shape[1] / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    _AFc_post_delta[blockspergrid, threadsperblock](
        A,
        A2,
        A3,
        k,
        d_in,
        d_out,
        N,
        D,
        x,
        y,
        np.array(i_neighbors),
        np.array(j_neighbors),
        D.shape[0],
        D.shape[1],
    )
    return D


def sdrf_AFc(
    data,
    loops=10,
    remove_edges=True,
    removal_bound=0.5,
    tau=1,
    is_undirected=False,
    k = 3,
    computespectralgap = False,
    progress_bar = True
    
):
    edge_index = data.edge_index
    if is_undirected:
        edge_index = torch_geometric.utils.to_undirected(edge_index)

    A = torch_geometric.utils.to_dense_adj(torch_geometric.utils.remove_self_loops(edge_index)[0])[0]
    A = A.cuda()

    G = torch_geometric.utils.to_networkx(data)
    if is_undirected:
        G = G.to_undirected()

    N = A.shape[0]
    C = torch.zeros(N, N).cuda()

    count_edge_removal = 0
    edges_rewired = []
    spectral_gap = []

    for idx in tqdm(range(loops), disable=not progress_bar):
        can_add = True
        _,A_squared = AF_curvature(A,k, C=C)
        ix_min = C.argmin().item()
        x = ix_min // N
        y = ix_min % N

        edges_rewired.append((x,y))

        if is_undirected:
            x_neighbors = list(G.neighbors(x)) + [x]
            y_neighbors = list(G.neighbors(y)) + [y]
        else:
            x_neighbors = list(G.successors(x)) + [x]
            y_neighbors = list(G.predecessors(y)) + [y]
        candidates = []
        # Only adding triangles as candidates
        if k ==3:

            for i in x_neighbors:
                if (not G.has_edge(i, y)) and i!=x:
                    candidates.append((i, y))
            for j in y_neighbors:
                if (not G.has_edge(x, j)) and j!=y:
                    candidates.append((x, j))
        elif k ==4:
            for i in x_neighbors:
                for j in y_neighbors:
                    if (i != j) and (not G.has_edge(i, j)):
                        candidates.append((i, j))
        if len(candidates):
            D = AFc_post_delta(A,k, x, y, x_neighbors, y_neighbors)
            improvements = []
            for (i, j) in candidates:
                improvements.append(
                    (D - C[x, y])[x_neighbors.index(i), y_neighbors.index(j)].item()
                )

            idk, idl = candidates[
                np.random.choice(
                    range(len(candidates)), p=softmax(np.array(improvements), tau=tau)
                )
            ]
            G.add_edge(idk, idl)
            if is_undirected:
                A[idk, idl] = A[idl, idk] = 1
            else:
                A[idk, idl] = 1
        else:
            can_add = False
            if not remove_edges:
                break

        if remove_edges:
            A_upper = np.triu(A.cpu())
            edge_index_upper = np.where(A_upper)
            maxindex = C[edge_index_upper].argmax()

            x,y = np.array(edge_index_upper)[:,maxindex]

            #ix_max = C.argmax().item()
            #x = ix_max // N
            #y = ix_max % N
            if C[x, y] > removal_bound:
                G.remove_edge(x, y)
                if is_undirected:
                    A[x, y] = A[y, x] = 0
                else:
                    A[x, y] = 0
                count_edge_removal +=1
            else:
                if can_add is False:
                    print("Exiting loop at iteration: ", idx)
                    break
        if computespectralgap:
            A_cpu = A.cpu()
            sparse_A_cpu = csr_array(A_cpu)
            sparse_D_cpu = csr_array(np.diag(A_cpu.sum(axis=0)))
                
            sparse_L_cpu = sparse_D_cpu - sparse_A_cpu

            eigvalues = np.real(eigs(sparse_L_cpu,k = 6, which='SM', return_eigenvectors=False))
            spectral_gap.append(eigvalues)
            #Dcomputed = D[x_neighbors.index(k), y_neighbors.index(l)].item()
            #Cnew = balanced_forman_curvature_undirected_personal(A,edge_index ,fcc = fcc)
            #print(Cnew[x,y])
            #print(Dcomputed)
    return G,count_edge_removal,edges_rewired,spectral_gap
