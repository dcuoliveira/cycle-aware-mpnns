import os
import argparse

from simulation.GraphSim import GraphSim
from simulation.SimGivenGraph import SimGivenGraph
from utils.conn_data import save_pickle

parser = argparse.ArgumentParser()

# General parameters
parser.add_argument('--graph_name', type=str, help='Graph name to be generated.', default="locally_tree_like")
parser.add_argument('--num_nodes', type=int, help='Number of nodes.', default=20)
parser.add_argument('--seed', type=int, help='Random seed.', default=2294)
parser.add_argument('--simulations', type=int, help='Number of simulations.', default=1)
parser.add_argument('--source_path', type=str, help='Source path for saving output.', default=os.path.dirname(__file__))
parser.add_argument('--num_samples', type=int, help='Number of samples to be generated from the graph.', default=100)

# Erdos Renyi graph parameters
parser.add_argument('--prob', type=float, help='Probability of edge creation (for Erdos Renyi graph).', default=0.5)

# k-regular graph parameters
parser.add_argument('--k', type=int, help='Degree of each node (for k-regular graph).', default=3)

# Geometric graph parameters
parser.add_argument('--radius', type=float, help='Radius for edge creation (for geometric graph).', default=0.1)

# Barabasi Albert graph parameters
parser.add_argument('--m', type=int, help='Number of edges to attach from a new node to existing nodes (for Barabasi Albert graph).', default=1)

# Watts Strogatz graph parameters
parser.add_argument('--k_ws', type=int, help='Each node is connected to k nearest neighbors in ring topology (for Watts Strogatz graph).', default=4)
parser.add_argument('--p_ws', type=float, help='Probability of rewiring each edge (for Watts Strogatz graph).', default=0.1)

# Locally Tree-like graph parameters
parser.add_argument('--avg_d', type=int, help='Average degree (number of edges per node) in the graph.', default=3)

if __name__ == "__main__":
    args = parser.parse_args()

    gs = GraphSim(graph_name=args.graph_name, seed=args.seed)
    sgg = SimGivenGraph()

    # Check if path exists
    output_path = f"{args.source_path}/data/outputs/{args.graph_name}"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    for i in range(args.simulations):
        if args.graph_name == "erdos_renyi":
            # gen graph
            graph_info = gs.simulate_erdos(num_nodes=args.num_nodes, prob=args.prob)

            # simulate sample from graph
            samples = sgg.sample_from_graph_structured_gaussian(graph=graph_info, n=args.num_samples)
            graph_info.samples = samples

            save_pickle(path=f"{output_path}/graph_info_{args.num_nodes}_{args.prob}_sim_{i}.pkl", obj=graph_info)
        elif args.graph_name == "k_regular":
            # gen graph
            graph_info = gs.simulate_k_regular(num_nodes=args.num_nodes, k=args.k)

            # simulate sample from graph
            samples = sgg.sample_from_graph_structured_gaussian(graph=graph_info, n=args.num_samples)
            graph_info.samples = samples

            save_pickle(path=f"{output_path}/graph_info_{args.num_nodes}_{args.k}_sim_{i}.pkl", obj=graph_info)
        elif args.graph_name == "geometric":
            # gen graph
            graph_info = gs.simulate_geometric(num_nodes=args.num_nodes, radius=args.radius)

            # simulate sample from graph
            samples = sgg.sample_from_graph_structured_gaussian(graph=graph_info, n=args.num_samples)
            graph_info.samples = samples

            save_pickle(path=f"{output_path}/graph_info_{args.num_nodes}_{args.radius}_sim_{i}.pkl", obj=graph_info)
        elif args.graph_name == "barabasi_albert":
            # gen graph
            graph_info = gs.simulate_barabasi_albert(num_nodes=args.num_nodes, m=args.m)

            # simulate sample from graph
            samples = sgg.sample_from_graph_structured_gaussian(graph=graph_info, n=args.num_samples)
            graph_info.samples = samples

            save_pickle(path=f"{output_path}/graph_info_{args.num_nodes}_{args.m}_sim_{i}.pkl", obj=graph_info)
        elif args.graph_name == "watts_strogatz":
            # gen graph
            graph_info = gs.simulate_watts_strogatz(num_nodes=args.num_nodes, k=args.k_ws, p=args.p_ws)

            # simulate sample from graph
            samples = sgg.sample_from_graph_structured_gaussian(graph=graph_info, n=args.num_samples)

            # simulate sample from graph
            samples = sgg.sample_from_graph_structured_gaussian(graph=graph_info, n=args.num_samples)
            graph_info.samples = samples

            save_pickle(path=f"{output_path}/graph_info_{args.num_nodes}_{args.k_ws}_{args.p_ws}_sim_{i}.pkl", obj=graph_info)
        elif args.graph_name == "locally_tree_like":
            # gen graph
            graph_info = gs.simulate_locally_tree_like_graph(num_nodes=args.num_nodes, avg_d=args.avg_d)

            # simulate sample from graph
            samples = sgg.sample_from_graph_structured_gaussian(graph=graph_info, n=args.num_samples)
            graph_info.samples = samples

            save_pickle(path=f"{output_path}/graph_info_{args.num_nodes}_{args.avg_d}_sim_{i}.pkl", obj=graph_info)
        else:
            raise ValueError("Invalid graph name")
