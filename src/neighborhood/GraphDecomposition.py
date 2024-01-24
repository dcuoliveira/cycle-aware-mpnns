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
        self.node_neigh = self.init_node_neighborhoods()
        # recover all N_{i/j}
        self.edge_neigh = self.init_edge_neighborhoods()

    # Obtain the local neighborhood of N_{u} for all u
    def init_node_neighborhoods(self):
        nodes_neigh = dict()
        for u in self.Graph.nodes:
            nodes_neigh[u] = Neighborhood(u,find_neighborhood(u,self.Graph,self.r))
            print(nodes_neigh[u].edges)
        #
        return nodes_neigh

    #  Obtain the local neighborhood of N_{u\v} for all u,v
    def init_edge_neighborhoods(self):
        edge_neigh = dict()
        for u in self.Graph.nodes:
            edge_neigh[u] = dict()
        #
        for u in self.Graph.nodes:
            for v in self.node_neigh[u].get_nodes():
                if v != u:
                    edge_neigh[u][v] = neighborhood_difference(self.node_neigh[v],self.node_neigh[u])
        #
        return edge_neigh