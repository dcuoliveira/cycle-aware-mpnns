import networkx as nx
import numpy as np
from utils.find_neighborhood import find_neighborhood
from utils.find_neighborhood import neighborhood_difference
from neighborhood.Neighborhood import Neighborhood

class GraphDecomposition:
    # Initializaes N_{node}
    def __init__(self, G,r = 0):
        """
        Class to save decomposition of the graph using primitive cycles of length at most r + 2
        
        Args:
            G: input graph
        """
        
        self.Graph = G
        self.r = r
        # recover all N_{i}
        self.nodes_neigh = dict()
        # recover all N_{i/j}
        self.cavity_neigh = dict() 
        # init node and edge neighboorhoods
        self.init_node_neighborhoods() # node neighborhood
        self.init_edge_neighborhoods() # cavity neighborhood

    # Obtain the local neighborhood of N_{u} for all u
    def init_node_neighborhoods(self):
        for u in self.Graph.nodes:
            self.nodes_neigh[u] = Neighborhood(u,find_neighborhood(u,self.Graph,self.r))
    #  Obtain the local neighborhood of N_{u\v} for all u,v
    def init_edge_neighborhoods(self):
        for u in self.Graph.nodes:
            self.cavity_neigh[u] = dict()
        # cavity[v][u] = G_{v \leftarrow u}
        for u in self.Graph.nodes:
            for v in self.nodes_neigh[u].get_neighbors():
                    self.cavity_neigh[v][u] = neighborhood_difference(self.nodes_neigh[v],self.nodes_neigh[u])

    def to_edgeList(self,cavity_node_idx):
        edge_list = list()

        for u in self.Graph.nodes:
            for v in self.nodes_neigh[u].get_neighbors():
                edge_list.append((u,cavity_node_idx[(v,u)]))
                for w in self.cavity_neigh[u][v].get_neighbors():
                    edge_list.append((cavity_node_idx[(v,u)],cavity_node_idx[(w,v)]))
        
        #
        return edge_list
    
    # return neighbors of a node
    def get_neighborhood(self,i,j = None):
        if j is None:
            return self.nodes_neigh[i]
        else:
            return self.cavity_neigh[i][j]
