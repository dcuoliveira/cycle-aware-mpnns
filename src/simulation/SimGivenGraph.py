from scipy.stats import multivariate_normal
import numpy as np

class SimGivenGraph:
    def __init__(self) -> None:
        pass

    def generate_covariance_matrix(self, graph, connected_cov=0.5):
        """
        Generate a covariance matrix for a multivariate Gaussian distribution
        based on a given graph structure.

        :param graph: A networkx graph object.
        :param connected_cov: The covariance value to use for connected nodes in the graph.
        :return: A covariance matrix as a numpy array.
        """

        num_nodes = len(graph.nodes)
        covariance_matrix = np.zeros((num_nodes, num_nodes))

        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    covariance_matrix[i, j] = 1  # Variance
                elif graph.has_edge(i, j):
                    covariance_matrix[i, j] = connected_cov

        return covariance_matrix

    def sample_from_graph_structured_gaussian(self, graph, n, connected_cov=0.5):
        """
        Sample n observations from a multivariate Gaussian distribution that
        respects the given graph structure.

        :param graph: A networkx graph object.
        :param n: The number of observations to sample.
        :param connected_cov: The covariance value to use for connected nodes in the graph.
        :return: A numpy array of sampled observations.
        """

        covariance_matrix = self.generate_covariance_matrix(graph, connected_cov)
        mean_vector = np.zeros(len(graph.nodes))  # Mean vector (can be adjusted as needed)

        # Sample from the multivariate Gaussian distribution
        samples = multivariate_normal.rvs(mean=mean_vector, cov=covariance_matrix, size=n)

        return samples
    
DEBUG = True

if __name__ == "__main__":
    if DEBUG:
        from simulation.GraphSim import GraphSim

        gs = GraphSim("test", seed=2294)
        graph = gs.simulate_locally_tree_like_graph(n=10, avg_degree=2)

        n_samples = 100
        sgg = SimGivenGraph()
        samples = sgg.sample_from_graph_structured_gaussian(graph=graph, n=n_samples)
        samples