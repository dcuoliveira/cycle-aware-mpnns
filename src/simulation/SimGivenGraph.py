from scipy.stats import multivariate_normal
import numpy as np
import networkx as nx

class SimGivenGraph:
    def __init__(self) -> None:
        pass

    def generate_covariance_matrix(self, graph, min_eigenvalue=0.01):
        """
        Generate a positive semidefinite covariance matrix for a multivariate Gaussian distribution
        based on a given graph structure, ensuring the matrix is symmetric and positive semidefinite.

        :param graph: A networkx graph object.
        :param min_eigenvalue: Minimum eigenvalue to ensure positive semidefiniteness.
        :return: A covariance matrix as a numpy array.
        """

        # Create an adjacency matrix from the graph
        adjacency_matrix = nx.to_numpy_array(graph)

        # Perform spectral decomposition
        eigenvalues, eigenvectors = np.linalg.eigh(adjacency_matrix)

        # Adjust negative eigenvalues to ensure positive semidefiniteness
        eigenvalues[eigenvalues < min_eigenvalue] = min_eigenvalue

        # Reconstruct the covariance matrix
        covariance_matrix = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T

        return covariance_matrix

    def sample_from_graph_structured_gaussian(self, graph, n, connected_cov=0.5, min_eigenvalue=0.01):
        """
        Sample n observations from a multivariate Gaussian distribution that
        respects the given graph structure and ensures the covariance matrix is positive semidefinite.

        :param graph: A networkx graph object.
        :param n: The number of observations to sample.
        :param connected_cov: The covariance value to use for connected nodes in the graph.
        :param min_eigenvalue: Minimum eigenvalue to ensure positive semidefiniteness.
        :return: A numpy array of sampled observations.
        """
        
        covariance_matrix = self.generate_covariance_matrix(graph, connected_cov, min_eigenvalue)
        mean_vector = np.zeros(len(graph.nodes))  # Mean vector (can be adjusted as needed)

        # Sample from the multivariate Gaussian distribution
        samples = multivariate_normal.rvs(mean=mean_vector, cov=covariance_matrix, size=n)

        return samples
    
DEBUG = False

if __name__ == "__main__":
    if DEBUG:

        import sys
        import os
        sys.path.append(os.path.join(os.getcwd(), "src"))

        from simulation.GraphSim import GraphSim

        num_nodes = 10
        avg_d = 2
        seed = 2294
        n_samples = 100

        gs = GraphSim(f"tree-like-avg{avg_d}", seed=seed)
        graph = gs.simulate_locally_tree_like_graph(num_nodes=num_nodes, avg_d=avg_d)

        sgg = SimGivenGraph()
        samples = sgg.sample_from_graph_structured_gaussian(graph=graph, n=n_samples)