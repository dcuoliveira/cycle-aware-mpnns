import os

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import DataLoader, Dataset
from neighborhood.GraphDecomposition import GraphDecomposition
import networkx as nx

class CoraLoader(Dataset):

    def __init__(self,
                 path,
                 mode,
                 num_layers,
                 self_loop=False,
                 normalize_adj=False,
                 transductive=False,
                 r = 0):
        """
        Parameters
        ----------
        r    : int
            Integer that indicates the size of primitive cycle to consider
        path : str
            Path to the cora dataset with cora.cites and cora.content files.
        mode : str
            train / val / test.
        num_layers : int
            Depth of the model.
        self_loop : Boolean
            Whether to add self loops, default: False.
        normalize_adj : Boolean
            Whether to use symmetric normalization on the adjacency matrix, default: False.
        transductive : Boolean
            Whether to use all node features while training, as in a transductive setting, default: False.
        """
        super(CoraLoader, self).__init__()

        self.path = path
        self.mode = mode
        self.num_layers = num_layers
        self.r = r
        self.self_loop = self_loop
        self.normalize_adj = normalize_adj
        self.transductive = transductive
        self.idx = {
            'train' : np.array(range(140)),
            'val' : np.array(range(200, 500)),
            'test' : np.array(range(500, 1500))
        }

        print('--------------------------------')
        print('Reading cora dataset from {}'.format(path))
        citations = np.loadtxt(os.path.join(path, 'cora.cites'), dtype=np.int64)
        content = np.loadtxt(os.path.join(path, 'cora.content'), dtype=str)
        print('Finished reading data.')
        self.obtain_graph_decomposition(content,citations)
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
        # adjacency matrix
        #u, v = edges[:, 0], edges[:, 1]
        #adj = sp.coo_matrix((np.ones(m), (u, v)),
        #                    shape=(n, n),
        #                    dtype=np.float32)
        #adj += adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        #if self_loop:
        #    adj += sp.eye(n)
        #if normalize_adj:
        #    degrees = np.power(np.array(np.sum(adj, axis=1)), -0.5).flatten()
        #    degrees = sp.diags(degrees)
        #    adj = (degrees.dot(adj.dot(degrees)))
        #print('Finished setting up data structures.')
        #print('--------------------------------')

        #self.features = features
        #self.labels = labels
        #self.adj = adj.tolil()

    # obtain decomposition of the whole graph
    def obtain_graph_decomposition(self, content, citations):
        # node features (X) and target labels (y)
        features, labels = content[:, 1:-1].astype(np.float32), content[:, -1]
        d = {j : i for (i,j) in enumerate(sorted(set(labels)))}
        labels = np.array([d[l] for l in labels])

        # get article codes as vertices # ID 
        vertices = np.array(content[:, 0], dtype=np.int64)

        # map article codes to vertices numbers
        d = {j : i for (i,j) in enumerate(vertices)}
        # get edges from article codes
        edges = np.array([e for e in citations if e[0] in d.keys() and e[1] in d.keys()])
        # map edges on article codes to edges on vertices
        edges = np.array([d[v] for v in edges.flatten()]).reshape(edges.shape)
        # compute graph decomposition
        G = nx.from_edgelist(edges)
        # compute graph decomposition
        self.G_decomp = GraphDecomposition(G,r = self.r)
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.idx[self.mode])

    def __getitem__(self, idx):
        if self.transductive:
            idx += int(self.idx[self.mode][0])
        else:
            if self.mode != 'train':
                idx += len(self.idx['train'])
        node_layers, ancestors_layers, ancestor_mapping, all_considered_nodes = self._form_computation_cavity_graph(idx)
        all_nodes_idx = {j:i for i,j in enumerate(all_considered_nodes)}
        features = self.features[all_considered_nodes, :]
        labels = self.labels[node_layers[-1]]
        features = torch.FloatTensor(features)
        labels = torch.LongTensor(labels)

        return features, node_layers, ancestors_layers, ancestor_mapping, all_considered_nodes, all_nodes_idx, labels

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

        # original
        #node_layers, mappings = self._form_computation_graph(idx)
        ## new
        node_layers, ancestors_layers, ancestor_mapping,all_considered_nodes = self._form_computation_cavity_graph(idx)

        # get rows of the adjacency matrix associated with the nodes on the last leaf (first node_layers) of the computation graph
        #rows = self.adj.rows[node_layers[0]]
        all_nodes_idx = {j:i for i,j in enumerate(all_considered_nodes)}
        # get features associated with the nodes on the last leaf (0th node_layers) of the computation graph
        features = self.features[all_considered_nodes, :]

        # get labels associated with the nodes on the first leaf (-1th node_layers) of the computation graph
        labels = self.labels[node_layers[-1]]

        # change format
        features = torch.FloatTensor(features)
        labels = torch.LongTensor(labels)

        return features, node_layers,ancestors_layers, ancestor_mapping, all_considered_nodes, all_nodes_idx, labels

    def get_dims(self):
        """
        Returns
        -------
        dimension of input features, dimension of output features
        """
        return self.features.shape[1], len(set(self.labels))
    
    # computation graph taking into account the primitive cycles
    def _form_computation_cavity_graph(self, idx):
        """
        Parameters
        ----------
        idx : int
            Index of the node for which the forward pass needs to be computed.
        ancestor: int
            Index of the ancestor of node 'idx', if value is None then it has no ancestor
        Returns
        -------
        node_layers : list of numpy array
            node_layers[i] is an array of the nodes in the ith layer of the
            computation graph.
        ancertors_layers : list of numpy array
            ancestors_layers[i] is an array of the ancestors of the nodes in the ith layer of the
            computation graph, that is, on the node_layers.
        ancestor_mapping : list of dictionary
            ancestor_mapping is a dictionary where ancestor_mapping[i] is a mapping of the unique ancestors
            of the nodes in the ith layer of the computation graph to their position in the array of unique ancestors.
        all_considered_nodes : list
            list of all nodes considered for the computation of the forward pass.
        
        """
        _list, _set = list, set
        #rows = self.adj.rows

        if type(idx) is int:
            ancestors_layers = [np.array([idx],dtype=np.int64)]
            node_layers = [np.array([idx], dtype=np.int64)]
        elif type(idx) is list:
            ancestors_layers = [np.array(idx, dtype=np.int64)]
            node_layers = [np.array(idx, dtype=np.int64)]

        # num_layers equals the k-hop neighbors
        # lines 2 to 7 of algorithm 2 in Hamilton, Ying, and Leskovec (2018)
        all_considered_nodes = [] # saves all nodes considered for the execution
        for _ in range(self.num_layers):
            # get nodes and ancestors in previous layer (prev)
            prev_ancestor = ancestors_layers[-1]
            prev = node_layers[-1]
            # recover
            ancestor_arr = [node for node in prev_ancestor] 
            arr = [node for node in prev]
            
            new_neighbors = list()
            new_neighbors_ancestors = list()
            for node, father in zip(arr, ancestor_arr):
                neighborhood = self.G_decomp.get_neighborhood(node)
                if father != node:
                    neighborhood = self.G_decomp.get_neighborhood(node , father)#[node][father]
                
                for v in neighborhood.get_neighbors():
                    new_neighbors.append(v)
                    new_neighbors_ancestors.append(node)
            #arr = new_neighbors
            #ancestor_arr = new_neighbors_ancestors
            ### previous
            # get neighbors of nodes in previous layer (prev -> arr)
            #arr.extend([v for node in arr for v in rows[node]])
            #arr = np.array(_list(_set(arr)), dtype=np.int64)
            arr = np.array(new_neighbors, dtype=np.int64)
            ancestor_arr = np.array(new_neighbors_ancestors, dtype=np.int64)
            # add nodes to node_layers
            node_layers.append(arr)
            # add ancestor to ancestor_layers
            ancestors_layers.append(ancestor_arr)
        
        node_layers.reverse()
        ancestors_layers.reverse()
        all_considered_nodes = [j for sub in node_layers for j in sub]
        all_considered_nodes = list(np.unique(all_considered_nodes))
        #mappings = [{j : i for (i,j) in enumerate(arr)} for arr in node_layers]
        ancestor_mapping =  [{j : i for (i,j) in enumerate(list(np.unique(arr,return_index=True)[0]))} for arr in ancestors_layers]

        return node_layers, ancestors_layers, ancestor_mapping, all_considered_nodes
    
    # previous method for computation graph
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