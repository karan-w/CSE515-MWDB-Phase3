from networkx.classes.function import number_of_nodes
from networkx.generators.small import desargues_graph
from networkx.linalg import graphmatrix
from scipy.sparse import csr_matrix
import networkx as nx
import numpy as np

from numpy import array, zeros, diag, diagflat, dot
import concurrent.futures

from utils.graph import similarity_graph

# https://www.digitalocean.com/community/tutorials/how-to-use-threadpoolexecutor-in-python-3

class ASCOS_PP:
    def __init__(self, similarity_graph):
        self.similarity_graph = similarity_graph
        self.P = None
        self.Q = None

    # Input - adjacency matrix of the graph
    def compute_P(self):
        sparse_adjacency_matrix = graphmatrix.adjacency_matrix(self.similarity_graph.graph)
        self.dense_adjacency_matrix = sparse_adjacency_matrix.toarray() # numpy array dense representation of the adjacency matrix - n * n
        self.P = self.dense_adjacency_matrix / self.dense_adjacency_matrix.sum(axis=1)
        return

    def compute_Q(self):
        x = 1 - np.exp(np.negative(self.dense_adjacency_matrix)) # computed from A
        self.Q = self.P * x

    def compute_S(self, c):
        n = self.similarity_graph.number_of_nodes
        identity_matrix = np.identity(n) # n * n identity matrix 
        A = identity_matrix - (c * self.Q.transpose()) # n * n

        # Since we want column vectors, take transpose and iterate through them for parallelism
        identity_matrix_transpose = identity_matrix.transpose() 
        
        S = np.zeros((n, n)) # n * n - initialization -> S1, S2, S3, .. , Sn
        S_transpose = S.transpose() # all cols are now rows

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for i in range(0, n):
                b = (1 - c) * identity_matrix_transpose[i] # n * 1
                x = S_transpose[i].transpose() # to pass it as column vector
                data = {
                    'A': A,
                    'b': b,
                    'N': 25,
                    'x': x,
                } 
                # S0, ..., Sn-1
                futures.append(executor.submit(jacobi_iterative_method, data=data)) # Is the result serially presented as well?

            for index, future in enumerate(futures):
                S_transpose[index] = future.result().transpose()

        S = S_transpose.transpose()
        self.S = S
        
""" 
Solves the equation Ax=b via the Jacobi iterative method. 
"""
def jacobi_iterative_method(data):
    A = data['A']
    b = data['b']
    x = data['x']

    N = data['N'] # number_of_iterations
    
    # Create an initial guess if needed                                                                                                                                                 
    if x is None:
        x = zeros(len(A[0]))

    # Create a vector of the diagonal elements of A                                                                                                                                                
    # and subtract them from A                                                                                                                                                                     
    D = diag(A)
    R = A - diagflat(D)

    # Iterate for N times                                                                                                                                                                          
    for i in range(N):
        x = (b - dot(R,x)) / D
    return x # column vector