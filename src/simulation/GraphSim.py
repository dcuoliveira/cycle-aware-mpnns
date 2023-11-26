import networkx as nx
import numpy as np

class GraphSim:
    def __init__(self, graph_name: str, seed: int = 2294):
        """
        Random Graph simulation class.
        
        Args:
            graph_name: name of the graph to be generated.
            seed: random seed.
        
        Returns:
            None
        """
        
        self.graph_name = graph_name
        self.seed = seed

    def simulate_erdos(self, num_nodes: int, prob: float):
        """
        Simulate a random Erdős-Rényi graph.
        
        Args:
            num_nodes: number of nodes.
            prob: probability of edge creation.
            
        Returns:
            networkx graph object.
        """

        return nx.erdos_renyi_graph(n=num_nodes, p=prob, seed=self.seed)

    def simulate_k_regular(self, num_nodes: int, d: int):
        """
        Simulate a random k-regular graph.
        
        Args:
            num_nodes: number of nodes.
            d: degree of each node.
            
        Returns:
            networkx graph object.
        """
        
        return nx.random_regular_graph(d=d, n=num_nodes, seed=self.seed)

    def simulate_geometric(self, num_nodes: int, radius: float):
        """
        Simulate a random geometric graph.
        
        Args:
            n: number of nodes.
            num_nodes: radius for edge creation.
            
        Returns:
            networkx graph object.
        """

        return nx.random_geometric_graph(n=num_nodes, radius=radius, seed=self.seed)

    def simulate_barabasi_albert(self, num_nodes: int, m: int):
        """
        Simulate a random Barabási-Albert preferential attachment graph.
        
        Args:
            num_nodes: number of nodes.
            m: number of edges to attach from a new node to existing nodes.
            
        Returns:
            networkx graph object.
        """

        return nx.barabasi_albert_graph(n=num_nodes, m=m, seed=self.seed)

    def simulate_watts_strogatz(self, num_nodes: int, k: int, p: float):
        """
        Simulate a random Watts-Strogatz small-world graph.
        
        Args:
            num_nodes: number of nodes.
            k: each node is joined with its k nearest neighbors in a ring topology.
            p: probability of rewiring each edge.
            
        Returns:
            networkx graph object.
        """

        return nx.watts_strogatz_graph(n=num_nodes, k=k, p=p, seed=self.seed)
    
    def simulate_locally_tree_like_graph(self, num_nodes: int, avg_d: float):
        """
        Generate a locally tree-like graph structure using the networkx library.

        :param num_nodes: Number of nodes in the graph.
        :param avg_d: Average degree (number of edges per node) in the graph.

        :return G: A networkx graph object representing the locally tree-like graph.
        """
        
        # Create an empty graph
        G = nx.Graph()

        # Add nodes to the graph
        for i in range(num_nodes):
            G.add_node(i)

        # Add edges to create a locally tree-like structure
        for node in G.nodes():
            # Connect each node to a number of other nodes equal to the desired average degree
            # while avoiding self-loops and duplicate edges.
            while G.degree(node) < avg_d:
                target = np.random.choice(num_nodes)
                if target != node and not G.has_edge(node, target):
                    G.add_edge(node, target)

        return G
