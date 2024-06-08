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
        # separate edges, edges directly connected to the node and edges that are not connected among them
        # also we create a set with the nodes in the neighborhood
        self.direct_neigh_edges = set()
        self.other_neigh_edges = set()
        self.neigh_nodes = set() 
        self.node_idx = dict() # index of each node in the neighborhood (some order)
        self.A = None
        self.V = None
        # we could also intialize for the message passing method here
        self.init_neighborhood()
        # init the adjacency matrix and the vector (v) -> to select the rows
        self.init_matrices()
    
    # initialization of the current node
    def init_neighborhood(self):
        # nodes in the local neighborhood
        for u,v in self.edges:
            if (u == self.node) or (v == self.node):
                self.direct_neigh_edges.add((u,v))
            else:
                self.other_neigh_edges.add((u,v))
            if (v != self.node):
                self.neigh_nodes.add(v)
            elif (u != self.node):
                self.neigh_nodes.add(u)
        # creating indices
        idx = 0
        for node_id in self.neigh_nodes:
            if node_id not in self.node_idx.keys():
                self.node_idx[node_id] = idx
                idx = idx + 1
    # function to initialize adjacency matrix and vector v
    def init_matrices(self):
        n = len(self.neigh_nodes)
        self.v = np.zeros(shape = n)
        self.A = np.zeros(shape = (n,n))
        for u,v in self.edges:
            if (u == self.node):
                self.v[self.node_idx[v]] = 1
            elif (v == self.node):
                self.v[self.node_idx[u]] = 1
            else:
                self.A[self.node_idx[u],self.node_idx[v]] = 1
                self.A[self.node_idx[v],self.node_idx[u]] = 1

    # function to update the internal values of the node, given the other values
    # edge_neigh: the Graph decomposition
    def update_value(self,Graph_decomp):
        return None
    
    # return the nodes of this local neighborhood
    def get_nodes(self):
        return self.Graph.nodes
    
    # return the neighbors of a vertex, considering the local neighborhood
    def get_neighbors(self):
        return self.neigh_nodes
    
    # return the adjacency matrix
    def get_adj(self):
        return self.A
    
    # return the vector of inmediate neighbors
    def get_v(self):
        return self.v