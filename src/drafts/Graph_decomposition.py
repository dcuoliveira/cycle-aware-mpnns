from neighborhood.GraphDecomposition import GraphDecomposition
from simulation.GraphSim import GraphSim


Graph_simulator = GraphSim("test")
Graph = Graph_simulator.simulate_watts_strogatz(10, 4, p = 0)

print(Graph)
# obtain graph Decomposition
Graph_decomp = GraphDecomposition(Graph,0)

for node in Graph.nodes:
    print(Graph_decomp.node_neigh[node].edges)
#print(Graph_decomp.edge_neigh)
for node in Graph.nodes:
    for v in Graph_decomp.node_neigh[node].get_nodes():
        if v != node:
            print("Neighborhood of " + str(v) + "\\" + str(node))
            print(Graph_decomp.edge_neigh[node][v].edges)