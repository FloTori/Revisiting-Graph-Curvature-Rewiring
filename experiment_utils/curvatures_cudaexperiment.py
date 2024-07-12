
from numba import cuda
import math
import torch

@cuda.jit(
    "void(float32[:,:], float32[:,:], int32[:,:], int32[:,:], float32[:], float32[:], int32, float32[:,:], float32[:,:], float32[:,:],boolean)"
)

def _BF_curvature_undirected(A, A2,edge_index,indices_neigh,d_in, d_out, N, C,gamma_max,sqrt_degree,fcc = True):
    i, j = cuda.grid(2)

    if (i < N) and (j < N):
        if A[i, j] == 0:
            C[i, j] = 0
            return

        if d_in[i] > d_out[j]:
            d_max = d_in[i]
            d_min = d_out[j]
        else:
            d_max = d_out[j]
            d_min = d_in[i]
        sqrt_degree[i,j] = math.sqrt(d_max)
        if d_min == 1:
           C[i, j] = 0
           return
        
        C[i, j] = ((2 / d_max) + (2 / d_min) - 2
                    + (2 / d_max + 1 / d_min) * A2[i, j] * A[i, j]
                  )
        if fcc:
            ind1_i,ind2_i = indices_neigh[i,0], indices_neigh[i,1]
            neighs_i = edge_index[1,ind1_i:ind2_i]

            ind1_j,ind2_j = indices_neigh[j,0],indices_neigh[j,1]
            neighs_j = edge_index[1,ind1_j:ind2_j]

        
            sharp_ij = 0
            lambda_ij = 0
            for k_count in range(len(neighs_i)):
                k = neighs_i[k_count]
          
                ind1_k = indices_neigh[k,0]
                ind2_k = indices_neigh[k,1]
                neighs_k = edge_index[1,ind1_k:ind2_k]
                if A[k,i]*(1-A[k,j]) !=0 and k != j: #Only have k in S(i)\S(j)

                    had = 0
                    for l_count in range(len(neighs_k)):
                        l = neighs_k[l_count]    
                        had += A[k,l]*A[i,l]*A[j,l]

                    TMP =A[k,i]*(1-A[k,j])*(A2[k,j] -had- 1)

                    if TMP > 0:
                        sharp_ij += 1
                        if TMP > lambda_ij:
                            lambda_ij = TMP

            for k_count in range(len(neighs_j)):
                k = neighs_j[k_count]
          
                ind1_k,ind2_k = indices_neigh[k,0],indices_neigh[k,1]
                neighs_k = edge_index[1,ind1_k:ind2_k]
          
                if A[j,k]*(1-A[k,i]) !=0 and k != i: #Only have k in S(j)\S(i)
                    had = 0
        
                    for l_count in range(len(neighs_k)):
                        l = neighs_k[l_count]    
                        had += A[k,l]*A[i,l]*A[j,l]
        
                    TMP = A[j,k]*(1-A[k,i])*(A2[k,i] -had- 1)
        
                    if TMP > 0:
                        sharp_ij += 1
                        if TMP > lambda_ij:
                            lambda_ij = TMP
                    
            if lambda_ij > 0:
                C[i, j] += sharp_ij / (d_max * lambda_ij) 

                #delta_upper[i,j] = min(1/lambda_ij,1/math.sqrt(d_max))
                gamma_max[i,j] = lambda_ij
            

def BF_curvature_undirected(A,edge_index, C=None,fcc = True):
    N = A.shape[0]
    threadsperblock = (16,16)#,10)
    blockspergrid_x = math.ceil(N / threadsperblock[0])
    blockspergrid_y = math.ceil(N / threadsperblock[1])

    blockspergrid_2d = (blockspergrid_x, blockspergrid_y)
    
    A2 = torch.matmul(A, A)

    d_in = A.sum(axis=0)
    d_out = A.sum(axis=1)

    ind1 = 0 
    ind2 = 0
    index_tuples = []
    for k in range(N):#test:
        ind2 += int(d_in[k].item())
        index_tuples.append((ind1,ind2))
        ind1 = ind2             
    index_tuples = torch.tensor(index_tuples).cuda()
    
    gamma_max = torch.zeros(N,N).cuda()
    nr_triangles = A2
    sqrt_degree = torch.zeros(N,N).cuda()
    #delta_upper = torch.zeros(N,N).cuda()
    if C is None:
        C = torch.zeros(N, N).cuda()

    _BF_curvature_undirected[blockspergrid_2d, threadsperblock](A, A2,edge_index,index_tuples,d_in, d_out, N, C,gamma_max,sqrt_degree,fcc)
    return C,sqrt_degree,nr_triangles,gamma_max

@cuda.jit(
    "void(float32[:,:], float32[:,:], float32[:], float32[:], int32, float32[:,:], float32[:,:])"
)
def _JT_curvature(A, A2, d_in, d_out, N, C,delta_upper):
    i, j = cuda.grid(2)

    if (i < N) and (j < N):
        if A[i, j] == 0:
            C[i, j] = 0
            return

        if d_in[i] > d_out[j]:
            d_max = d_in[i]
            d_min = d_out[j]
        else:
            d_max = d_out[j]
            d_min = d_in[i]
        delta_upper[i,j] = 2000#1/math.sqrt(d_max)
        if d_max * d_min == 0:
            C[i, j] = 0
            return

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
            (2 / d_max) + (2 / d_min) - 2 + (2 / d_max + 1 / d_min) * A2[i, j] * A[i, j]
        )
        if lambda_ij > 0:
            C[i, j] += sharp_ij / (d_max * lambda_ij)
            #delta_upper[i,j] = min(1/lambda_ij,1/math.sqrt(d_max))
            delta_upper[i,j] = 1/lambda_ij

def JT_curvature(A, C=None):
    N = A.shape[0]
    A2 = torch.matmul(A, A)
    d_in = A.sum(axis=0)
    d_out = A.sum(axis=1)
    if C is None:
        C = torch.zeros(N, N).cuda()
    delta_upper = torch.zeros(N,N).cuda()

    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(N / threadsperblock[0])
    blockspergrid_y = math.ceil(N / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    _JT_curvature[blockspergrid, threadsperblock](A, A2, d_in, d_out, N, C,delta_upper)
    return C,delta_upper,A2

@cuda.jit(
    "void(float32[:,:], float32[:,:], float32[:], float32[:], int32, float32[:,:])"
)
def _JL_curvature(A, A2, d_in, d_out, N, C):
    i, j = cuda.grid(2)

    if (i < N) and (j < N):
        if A[i, j] == 0:
            C[i, j] = 0
            return

        if d_in[i] > d_out[j]:
            d_max = d_in[i]
            d_min = d_out[j]
        else:
            d_max = d_out[j]
            d_min = d_in[i]


        if ( 1. - 1./d_in[i] - 1./d_out[j] - A2[i, j]/d_min) > 0.:
            term1 =  1. - 1./d_in[i] - 1./d_out[j] - A2[i, j]/d_min
        else: 
            term1 = 0 

        if ( 1. - 1./d_in[i] - 1./d_out[j] - A2[i, j]/d_max) > 0.:
            term2 =  1 - 1./d_in[i] - 1./d_out[j] - A2[i, j]/d_max
        else:
            term2 = 0

        C[i, j] = (-term1 - term2 + A2[i,j]/d_max)

def JL_curvature(A, C=None):
    N = A.shape[0]
    A2 = torch.matmul(A, A)
    d_in = A.sum(axis=0)
    d_out = A.sum(axis=1)
    if C is None:
        C = torch.zeros(N, N).cuda()
    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(N / threadsperblock[0])
    blockspergrid_y = math.ceil(N / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    _JL_curvature[blockspergrid, threadsperblock](A, A2, d_in, d_out, N, C)
    return C,A2


@cuda.jit(
    "void(float32[:,:], float32[:,:], float32[:,:], float32[:], float32[:], int32, float32[:,:],float32)"
)
def _AF_curvature(A, A2,A3, d_in, d_out, N, C,k):
    i, j = cuda.grid(2)

    if (i < N) and (j < N):
        if A[i, j] == 0:
            C[i, j] = 0 
            return

        if k == 3.:
            C[i, j] = 4 - d_in[i] -d_out[j] + 3*A2[i,j]
            return
        elif k == 4.:
            C[i,j] = 4 - d_in[i] -d_out[j] + 3*A2[i,j] + 2*(A3[i,j] - d_in[i] - d_out[j] + 1)
            return

            

def AF_curvature(A,k, C=None):
    N = A.shape[0]
    A2 = torch.matmul(A, A)
    A3 = torch.matmul(A2,A)
    d_in = A.sum(axis=0)
    d_out = A.sum(axis=1)
    if C is None:
        C = torch.zeros(N, N).cuda()

    threadsperblock = (16, 16)
    blockspergrid_x = math.ceil(N / threadsperblock[0])
    blockspergrid_y = math.ceil(N / threadsperblock[1])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    _AF_curvature[blockspergrid, threadsperblock](A, A2,A3 ,d_in, d_out, N, C,k)
    return C,A2

