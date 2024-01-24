import networkx as nx
import numpy as np
import neighborhood.GraphDecomposition

class Neighborhood:
    # Initializaes N_{node}
    def __init__(self, node, edges):
        """
        Class to save the local neighborhoof of "node" given the list of edges
        
        Args:
            node: node to which the local neighborhood pertains
            edges: edges corresponding to the local neighborhood
        
        Returns:
            None
        """
        
        self.node = node
        self.edges = set(edges)
        self.Graph = nx.from_edgelist(edges)
        # we could also intialize for the message passing method here
        self.init_neighborhood()
    
    # initialization of the current node
    def init_neighborhood(self):
        return None
    
    # function to update the internal values of the node, given the other values
    # edge_neigh: the Graph decomposition
    def update_value(self,Graph_decomp):
        return None
    
    # return the nodes of this local neighborhood
    def get_nodes(self):
        return self.Graph.nodes