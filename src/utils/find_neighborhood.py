import networkx as nx
import numpy as np
from neighborhood.Neighborhood import Neighborhood

# conda activate campnns
# find the local neighbooorhood of node i considering primitive cycles up to size r
def find_neighborhood(i, G, r):
    n = G.number_of_nodes()
    list_edges = set()
    # initialize queue
    Q = [i]
    # initial node initialization
    father = {i: -1}
    visited = {i: True}
    level = {i:0}
    # always add all nodes at distance 1
    for j in G[i]:
        list_edges.add((i,j))
        list_edges.add((j,i))
        level[j] = level[i] + 1
        father[j] = i
        Q.append(j)
    # apply BFS
    while len(Q) != 0:
        u = Q.pop(0)
        if (level[u] << 1) > r + 2:
            break
        #
        visited[u] = True
        #
        for v in G[u]:
            if v in father:
                if (v == father[u]) or (v not in visited):
                    continue
                if (level[u] + level[v] + 1) <= r + 2:
                    list_edges.add((u,v))
                    list_edges.add((v,u))
                    list_edges.add((father[u],u))
                    list_edges.add((u,father[u]))
            else:
                level[v] = level[u] + 1
                father[v] = u
                Q.append(v)
    edges = []
    for edge in list_edges:
        edges.append(edge)
    #
    return edges

# Obtain the local neighborhood of N_{u\v} given the local neighborhoods of N_{u} and N_{v}
def neighborhood_difference(Graph_u,Graph_v):
    edges = list(Graph_u.edges.difference(Graph_v.edges))
    nodes = set([u for u,_ in edges] + [v for _,v in edges])
    G_uv = Neighborhood(nodes,edges)
    return G_uv