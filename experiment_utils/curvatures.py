import networkx as nx
from tqdm import tqdm

import torch
import numpy as np
from scipy import sparse
from scipy import stats
from scipy.sparse import coo_matrix, csr_matrix, vstack, csc_matrix, coo_array


def bfc_naive(G: nx.Graph, v1: int, v2: int) -> float:
    """
    Balanced Forman curvature computation for a given edge in a graph.
    :param G: (undirected) graph under consideration.
    :param v1: first endpoint of the edge under consideration.
    :param v2: second endpoint of the edge under consideration.
    :return: Balanced Forman curvature for the edge under consideration.
    """
    deg1 = G.degree[v1]
    deg2 = G.degree[v2]
    deg_min = min(deg1, deg2)
    if deg_min == 1:
        return 0
    deg_max = max(deg1, deg2)

    S1_1 = set(G[v1])
    S1_2 = set(G[v2])

    triangles = S1_1.intersection(S1_2)
    
    squares_1 = set(
        k for k in S1_1.difference(S1_2) if k != v2 and set(G[k]).intersection(S1_2).difference(S1_1.union({v1})))
    squares_2 = set(
        k for k in S1_2.difference(S1_1) if k != v1 and set(G[k]).intersection(S1_1).difference(S1_2.union({v2})))
    if len(squares_1) == 0 or len(squares_2) == 0:
        return 2 / G.degree[v1] + 2 / G.degree[v2] - 2 + 2 * len(triangles) / deg_max + len(
            triangles) / deg_min

    A = nx.adjacency_matrix(G)#.todense()

    gamma = max(max([(A[[k],:] @ (A[[v2],:] - A[[v1],:].multiply(A[[v2],:])).T)[0, 0] - 1 for k in squares_1]),
                max([(A[[k],:] @ (A[[v1],:] - A[[v2],:].multiply(A[[v1],:])).T)[0, 0] - 1 for k in squares_2]))

    return 2 / deg1 + 2 / deg2 - 2 + 2 * len(triangles) / deg_max + len(
        triangles) / deg_min + (1 / (gamma * deg_max)) * (len(squares_1) + len(squares_2))


def bfc_nx(G: nx.Graph) -> nx.Graph:
    """
    Compute Balanced Forman curvature for the entire graph.
    :param G: (undirected) graph under consideration.
    :return: input graph with Balanced Forman curvature assigned for each edge.
    """
    N = len(G.nodes)
    Balanced_Forman_Curvature = torch.zeros(N, N,dtype = torch.float64)

    for v1, v2 in tqdm(G.edges):
        Balanced_Forman_Curvature[v1,v2] = bfc_naive(G, v1, v2)
    return Balanced_Forman_Curvature

#ONLINE SCIPY IMPLEMENTATION
#IMPORTANT: The columns of B should be ascendingly sorted corresponding to their neighbors in the rows

def dot_multiply(A,B,C, stepsize=10000, sum=False):
    """ Stepwise calculation of A.dot(B).multiply(C) for sparse matrices A,B,C
    Args:
    A (scipy sparse csr-matrix): Sparse matrix A of size nxn
    B (scipy sparse csc-matrix): Sparse matrix B of size nxn
    C (scipy sparse csr-matrix): Sparse matrix C of size nxn
    stepsize (int): Parameter to execute a memory intensive matrix multiplication stepwise
    sum (bool): If True we sum over the rows of A.dot(B)
    Returns: Sparse csr-matrix A.dot(B).multiply(C) and if True also sum over the rows of A.dot(B) """
    AB=A[0:stepsize, :].dot(B)
    if sum:
        AB_sum=coo_matrix(AB.sum(axis=1))
    ABC=AB.multiply(C[0:stepsize, :])
    for k in tqdm(range(stepsize, A.shape[0], stepsize)):
        AB=A[k:k+stepsize, :].dot(B) #overwrite AB to save memory
        if sum:
            AB_sum=vstack([AB_sum, coo_matrix(AB.sum(axis=1))])
        #last entry doesn't cause overflow even if it exceeds size of the matrix
        ABC=vstack([ABC, AB.multiply(C[k:k+stepsize, :])])
    if sum:
        return ABC, AB_sum
    else:
        return ABC

def balanced_forman_curvature_scipy(B,D_in,D_out,stepsize=None, four_cycle_contribution=True,directed=False,*args):
    """Online calculation of balanced Forman curvature
    Note: to save memory we only calculate the upper triangular part of the matrix
    Args:
    B (scipy sparse coo matrix): adjacency matrix of the undirected graph
    stepsize(int): parameter to execute a memory intensive matrix multiplication stepwise
    four_cycle_contribution(bool): whether to include the four cycle contribution or only consider triangles in the calculation
    Returns:
    Ricci (scipy sparse coo matrix): upper triangular part of the
    (simplified) balanced Forman curvature matrix"""
    #Create a deep copy of B and initialise all values of Ricci as 0
    Ricci=B.copy()
    Ricci.data[:]=0
    Ricci=Ricci.astype(float) #change datatype to float, otherwise entries are stored as rounded integers


    #CSR version of B to extract rows effficiently
    B_row=csr_matrix(B)

    #CSC version of B to extract columns efficiently
    B_col=csc_matrix(B)

    if stepsize:
        TRI=dot_multiply(B_row, B_col, B_row, stepsize=stepsize)
    else:
        B_squared=B.dot(B)
        #Only detect triangles when there is an edge
        TRI=B_squared.multiply(B)

    #Add zero values to TRI so the indices of TRI and B match, we also have to sort the column indices for this to work
    TRI=TRI+B
    TRI=csr_matrix(TRI)
    TRI.sort_indices()
    TRI=coo_matrix(TRI)
    TRI.data-=1


    #Check that columns are ascendingly sorted corresponding to their neighbors in the rows
    #It has to correspond to the right entries in TRI
    row=B.row
    col=B.col
    n=B.data.size

    #Optional: Give the edges which you want to compute:
    if args:
        edges_to_compute = args[0]
    else:
        edges_to_compute = [*range(n)]
    for idxk in edges_to_compute:
        if args:
            i = idxk[0]
            j = idxk[1]
            k = np.where((row == i) & (col == j))[0]
            #print(k)
        else:
            k = idxk
            i=row[k]
            j=col[k]
        if i< j or directed:
            d_i=D_in[i]
            d_j=D_out[j]
            maximum=max(d_i, d_j)
            minimum=min(d_i, d_j)
            if minimum>1:
                Ricci.data[k]=2/d_i+2/d_j-2 +2 *TRI.data[k]/maximum+TRI.data[k]/minimum

                #Inspired by original code and remark 10 in paper
                #We note that given k ∈ N_i\N_j the term (A_k ·(A_j −A_i ⊙A_j))−1 yields the number of nodes w
                #forming a 4-cycle of the form i ∼ k ∼ w ∼ j ∼ i with no diagonals inside.
                #Check if there is a four_cycle at i~j
                if four_cycle_contribution:
                    N_i=B_row.getrow(i)
                    N_j=B_row.getrow(j)
                    N_ij=N_i.multiply(N_j)
                    N_i_minus_N_j=N_i-N_ij #contains j
                    N_j_minus_N_i=N_j-N_ij #contains i


                    indices_i=coo_array(N_i_minus_N_j).col
                    indices_i=indices_i[~np.isin(indices_i, j)]
                    indices_j=coo_array(N_j_minus_N_i).col
                    indices_j=indices_j[~np.isin(indices_j, i)]

                    #Corresponds to calculating above expression for k ∈ N_i\N_j not equal to j
                    NFC_k=N_j_minus_N_i.dot(B_col[:,indices_i]).data-1

                    #Corresponds to calculating above expression for w ∈ N_j\N_i not equal to i
                    NFC_w=N_i_minus_N_j.dot(B_col[:,indices_j]).data-1

                    NFC=np.concatenate((NFC_k, NFC_w))
                    FC=(NFC>0).sum()

                    if FC !=0:
                        gamma_max=NFC.max()
                        Ricci.data[k]+= 1/(gamma_max * maximum) * FC
    return Ricci

def bfc_scipy(G_input,fcc,directed = False,*args):
   #G_input = G.copy()
   #G_input.remove_edges_from(nx.selfloop_edges(G_input))

   A = nx.adjacency_matrix(G_input)
   D_in = np.array(A.sum(axis=0)).flatten()
   D_out = np.array(A.sum(axis=1)).flatten() 

   AdjMatrixSparseScipy = sparse.coo_matrix(A)
   if args:
       if directed:
            Csparse =balanced_forman_curvature_scipy(AdjMatrixSparseScipy,D_in,D_out,None,fcc,directed,args[0])
            return Csparse
       else:
            Csparse =balanced_forman_curvature_scipy(AdjMatrixSparseScipy,D_in,D_out,None,fcc,directed,args[0])
            return Csparse 
   else:
       if directed:
            Csparse =balanced_forman_curvature_scipy(AdjMatrixSparseScipy,D_in,D_out,None,fcc,directed)
            C = torch.tensor(Csparse.todense() + np.transpose(Csparse.todense()),dtype = torch.float64)
            return C,Csparse
       else:
            Csparse =balanced_forman_curvature_scipy(AdjMatrixSparseScipy,D_in,D_out,None,fcc,directed)
            C = torch.tensor(Csparse.todense() + np.transpose(Csparse.todense()),dtype = torch.float64)
            return C,Csparse


def _balanced_forman_curvature_JTc(A, A2, d_in, d_out, N, C):
    for i in tqdm(range(N)):
        for j in range(N):
            if A[i, j] == 0:
                C[i, j] = 0
                continue

            if d_in[i] > d_out[j]:
                d_max = d_in[i]
                d_min = d_out[j]
            else:
                d_max = d_out[j]
                d_min = d_in[i]

            if d_max * d_min == 0:
                C[i, j] = 0
                continue

            sharp_ij = 0
            lambda_ij = 0
            for k in range(N):
                TMP = A[k, j] * (A2[i, k] - A[i, k]) * A[i, j]
                if TMP > 0:
                    sharp_ij += 1
                    if TMP > lambda_ij:
                        lambda_ij = TMP

                TMP = A[i, k] * (A2[k, j] - A[k, j]) * A[i, j]
                if TMP > 0:
                    sharp_ij += 1
                    if TMP > lambda_ij:
                        lambda_ij = TMP

            C[i, j] = (
                (2 / d_max)
                + (2 / d_min)
                - 2
                + (2 / d_max + 1 / d_min) * A2[i, j] * A[i, j]
            )
            if lambda_ij > 0:
                C[i, j] += sharp_ij / (d_max * lambda_ij)
    return C

def balanced_forman_curvature_toppingetal(A, C=None):
    N = A.shape[0]
    A2 = np.matmul(A, A)
    d_in = A.sum(axis=0)
    d_out = A.sum(axis=1)
    if C is None:
        C = torch.zeros((N, N),dtype = torch.float64)

    C = _balanced_forman_curvature_JTc(A, A2, d_in, d_out, N, C)
    return C