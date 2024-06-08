import os

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader, Dataset
from neighborhood.GraphDecomposition import GraphDecomposition
import networkx as nx

# this class will be used specially to load the graph, node features, and build the local neighborhood

class GraphLoader(Dataset):

    def __init__(self,
                 name,
                 path,
                 mode,
                 num_layers,
                 r = 0,
                 self_loop=False,
                 normalize_adj=False,
                 transductive=False):
        """
        Parameters
        ----------
        name : str
            Name of the graph's ID, edges, label; and edges of the graph.
        path : str
            Path to the graph dataset.
        mode : str
            train / val / test.
        num_layers : int
            Depth of the model.
        r: int
            Size of the primitive cycles to consider
        self_loop : Boolean
            Whether to add self loops, default: False.
        normalize_adj : Boolean
            Whether to use symmetric normalization on the adjacency matrix, default: False.
        transductive : Boolean
            Whether to use all node features while training, as in a transductive setting, default: False.
        """
        super(GraphLoader, self).__init__()

        self.name = name # name of the dataset
        self.path = path # path of the dataset
        self.mode = mode
        self.num_layers = num_layers
        self.r = r
        self.self_loop = self_loop
        self.normalize_adj = normalize_adj
        self.transductive = transductive
        # for now this indices are defined as for the CORA dataset/ could change for others
        #self.idx = {
        #    'train' : np.array(range(140)),
        #    'val' : np.array(range(200, 500)),
        #    'test' : np.array(range(500, 1500))
        #}
        # for content
        self.idx = {
            'train' : np.array(range(3)),
            'val' : np.array(range(3, 5)),
            'test' : np.array(range(5, 8))
        }
        

        print('--------------------------------')
        print('Reading {} dataset from {}'.format(name,path))
        citations = np.loadtxt(os.path.join(path,name + '.cites'), dtype=np.int64)
        content = np.loadtxt(os.path.join(path,name + '.content'), dtype=str)
        print('Finished reading data.')

        print('Setting up data structures.')
        if transductive:
            idx = np.arange(content.shape[0])
        else:
            if mode == 'train':
                idx = self.idx['train']
            elif mode == 'val':
                idx = np.hstack((self.idx['train'], self.idx['val']))
            elif mode == 'test':
                idx = np.hstack((self.idx['train'], self.idx['test']))

        #### RECOVER VERTEX INFORMATION
        # node features (X) and target labels (y)
        features, labels = content[idx, 1:-1].astype(np.float32), content[idx, -1]
        d = {j : i for (i,j) in enumerate(sorted(set(labels)))}
        labels = np.array([d[l] for l in labels])

        # get article codes as vertices # ID 
        vertices = np.array(content[idx, 0], dtype=np.int64)

        # map article codes to vertices numbers
        d = {j : i for (i,j) in enumerate(vertices)}

        ### VERTEX DATA RECOVERED

        ### RECOVER EDGE DATA
        # get edges from article codes
        edges = np.array([e for e in citations if e[0] in d.keys() and e[1] in d.keys()])

        # map edges on article codes to edges on vertices
        edges = np.array([d[v] for v in edges.flatten()]).reshape(edges.shape)

        # graph
        u, v = edges[:, 0], edges[:, 1]
        G = nx.from_edgelist(edges)
        G_decomp = GraphDecomposition(G,r = self.r)
        # given the graph decomposition, now get the feature vector for each of them, for each cavity node/ exactly the same label

        cavity_idx = len(d)
        cavity_node_idx = dict()
        cavity_node_idx_rev = dict()
        cavity_feat_label_idx = list()
        for u in G.nodes:
            for v in G_decomp.nodes_neigh[u].get_neighbors():
                cavity_node_idx[(v,u)] = cavity_idx
                cavity_node_idx_rev[cavity_idx] = (v,u)
                cavity_feat_label_idx.append(v)
                cavity_idx = cavity_idx + 1
                for w in G_decomp.cavity_neigh[u][v].get_neighbors():
                    cavity_node_idx[(w,v)] = cavity_idx
                    cavity_node_idx_rev[cavity_idx] = (w,v)
                    cavity_feat_label_idx.append(w)
                    cavity_idx = cavity_idx + 1

        #
        cavity_labels = labels[cavity_feat_label_idx]
        cavity_features = features[cavity_feat_label_idx,:]
        # save information
        self.cavity_node_idx = cavity_node_idx
        self.cavity_node_idx_rev = cavity_node_idx_rev 
        labels = np.concatenate([labels,cavity_labels])
        features = np.vstack([features,cavity_features])
        # get adjacency matrix
        decomp_edges = G_decomp.to_edgeList(cavity_node_idx)
        u = [x for x,_ in decomp_edges]
        v = [y for _,y in decomp_edges]
        #
        m = len(u)
        n = cavity_idx
        # adjacency matrix
        adj = sp.coo_matrix((np.ones(m), (u, v)),
                            shape=(n, n),
                            dtype=np.float32)
        #adj += adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        if self_loop:
            adj += sp.eye(n)
        if normalize_adj:
            degrees = np.power(np.array(np.sum(adj, axis=1)), -0.5).flatten()
            degrees = sp.diags(degrees)
            adj = (degrees.dot(adj.dot(degrees)))
        print('Finished setting up data structures.')
        print('--------------------------------')
        ### RECOVER EDGE DATA
        self.G_decomp = G_decomp
        self.features = features
        self.labels = labels
        self.G = G
        self.adj = adj.tolil()

    def __len__(self):
        return len(self.idx[self.mode])

    def __getitem__(self, idx):
        if self.transductive:
            idx += int(self.idx[self.mode][0])
        else:
            if self.mode != 'train':
                idx += len(self.idx['train'])
        node_layers, mappings = self._form_computation_graph(idx)
        rows = self.adj.rows[node_layers[0]]
        features = self.features[node_layers[0], :]
        labels = self.labels[node_layers[-1]]
        features = torch.FloatTensor(features)
        labels = torch.LongTensor(labels)

        return features, node_layers, mappings, rows, labels

    def collate_wrapper(self, batch):
        """
        Parameters
        ----------
        batch : list
            A list of examples from this dataset.

        Returns
        -------
        features : torch.FloatTensor
            An (n' x input_dim) tensor of input node features.
        node_layers : list of numpy array
            node_layers[i] is an array of the nodes in the ith layer of the
            computation graph.
        mappings : list of dictionary
            mappings[i] is a dictionary mapping node v (labelled 0 to |V|-1)
            in node_layers[i] to its position in node_layers[i]. For example,
            if node_layers[i] = [2,5], then mappings[i][2] = 0 and
            mappings[i][5] = 1.
        rows : list
        labels : torch.LongTensor
            An (n') length tensor of node labels.
        """
        idx = [node_layers[-1][0] for node_layers in [sample[1] for sample in batch]]

        node_layers, mappings = self._form_computation_graph(idx)

        # get rows of the adjacency matrix associated with the nodes on the last leaf (first node_layers) of the computation graph
        rows = self.adj.rows[node_layers[0]]

        # get features associated with the nodes on the last leaf (0th node_layers) of the computation graph
        features = self.features[node_layers[0], :]

        # get labels associated with the nodes on the first leaf (-1th node_layers) of the computation graph
        labels = self.labels[node_layers[-1]]

        # get the Neighborhoods of each one
        #local_neighborhood = 

        # change format
        features = torch.FloatTensor(features)
        labels = torch.LongTensor(labels)

        return features, node_layers, mappings, rows, labels

    def get_dims(self):
        """
        Returns
        -------
        dimension of input features, dimension of output features
        """
        return self.features.shape[1], len(set(self.labels))

    def _form_computation_graph(self, idx):
        """
        Parameters
        ----------
        idx : int
            Index of the node for which the forward pass needs to be computed.

        Returns
        -------
        node_layers : list of numpy array
            node_layers[i] is an array of the nodes in the ith layer of the
            computation graph.
        mappings : list of dictionary
            mappings[i] is a dictionary mapping node v (labelled 0 to |V|-1)
            in node_layers[i] to its position in node_layers[i]. For example,
            if node_layers[i] = [2,5], then mappings[i][2] = 0 and
            mappings[i][5] = 1.
        """
        _list, _set = list, set
        rows = self.adj.rows

        if type(idx) is int:
            node_layers = [np.array([idx], dtype=np.int64)]
        elif type(idx) is list:
            node_layers = [np.array(idx, dtype=np.int64)]

        # num_layers equals the k-hop neighbors
        # lines 2 to 7 of algorithm 2 in Hamilton, Ying, and Leskovec (2018)
        for _ in range(self.num_layers):
            # get nodes in previous layer (prev)
            prev = node_layers[-1]
            arr = [node for node in prev]

            # get neighbors of nodes in previous layer (prev -> arr)
            arr.extend([v for node in arr for v in rows[node]])
            arr = np.array(_list(_set(arr)), dtype=np.int64)

            # add nodes to node_layers
            node_layers.append(arr)
        node_layers.reverse()

        mappings = [{j : i for (i,j) in enumerate(arr)} for arr in node_layers]

        return node_layers, mappings