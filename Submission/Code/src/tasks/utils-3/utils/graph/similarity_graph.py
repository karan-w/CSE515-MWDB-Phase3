import itertools
import networkx as nx
from networkx.classes.function import number_of_nodes
import numpy as np
import matplotlib.pyplot as plt
import os

class SimilarityGraph:
    def __init__(self):
        pass

    def create(self, number_of_nodes, subject_subject_similarity_matrix, n):
        self.number_of_nodes = number_of_nodes
        self.subject_subject_similarity_matrix = subject_subject_similarity_matrix
        # https://networkx.org/documentation/stable/tutorial.html - Tutorial on how to use this graph
        graph = nx.DiGraph() 
        graph.add_nodes_from(range(number_of_nodes)) # 0 indexed - 0, 1, ... number_of_nodes - 1
        
        edges = [] # edges are coming from the subject-subject similarity matrix
        
        for subject_id, subject_similarity_scores in enumerate(subject_subject_similarity_matrix):
            subject_similarity_scores[subject_id] = 0 # Set the similarity score with itself to 0 to eliminate self loops
            top_n_similar_subject_indices = np.argsort(subject_similarity_scores)[-n:]
            n_similar_edges = [(subject_id, neighbour_subject_index, subject_subject_similarity_matrix[subject_id][neighbour_subject_index]) for neighbour_subject_index in top_n_similar_subject_indices]
            edges.append(n_similar_edges)

        # edges is a list of list, we need to flatten it first
        edges = list(itertools.chain(*edges))
        
        graph.add_weighted_edges_from(edges) #[(v1, v2, e1), .. ]

        self.graph = graph

    def draw(self):
        nx.draw(self.graph, with_labels=True, font_weight='bold')
        plt.show()
    
    def save(self, output_folder_path):
        nx.draw(self.graph, with_labels=True, font_weight='bold')
        similarity_graph_filepath = os.path.join(output_folder_path, "similarity_graph.png")
        plt.savefig(similarity_graph_filepath)    