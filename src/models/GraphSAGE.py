import numpy as np
import torch
import torch.nn as nn

from aggregators.LSTMAggregator import LSTMAggregator
from aggregators.MaxPoolAggregator import MaxPoolAggregator
from aggregators.MeanPoolAggregator import MeanPoolAggregator
from aggregators.MeanAggregator import MeanAggregator


class GraphSAGE(nn.Module):

    def __init__(self,
                 input_dim,
                 hidden_dims,
                 output_dim,
                 agg_class=MaxPoolAggregator,
                 dropout=0.5,
                 num_samples=25,
                 device='cpu'):
        """
        Parameters
        ----------
        input_dim : int
            Dimension of input node features.MaxPoolAggregator
        hidden_dims : list of ints
            Dimension of hidden layers. Must be non empty.
        output_dim : int
            Dimension of output node features.
        agg_class : An aggregator class.
            Aggregator. One of the aggregator classes imported at the top of
            this module. Default: MaxPoolAggregator.
        dropout : float
            Dropout rate. Default: 0.5.
        num_samples : int
            Number of neighbors to sample while aggregating. Default: 25.
        device : str
            'cpu' or 'cuda:0'. Default: 'cpu'.
        """
        super(GraphSAGE, self).__init__()

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.agg_class = agg_class
        self.num_samples = num_samples
        self.device = device
        self.num_layers = len(hidden_dims) + 1

        self.aggregators = nn.ModuleList([agg_class(input_dim, input_dim, device)])
        self.aggregators.extend([agg_class(dim, dim, device) for dim in hidden_dims])


        c = 3 if agg_class == LSTMAggregator else 2
        self.fcs = nn.ModuleList([nn.Linear(c*input_dim, hidden_dims[0])])
        self.fcs.extend([nn.Linear(c*hidden_dims[i-1], hidden_dims[i]) for i in range(1, len(hidden_dims))])
        self.fcs.extend([nn.Linear(c*hidden_dims[-1], output_dim)])

        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for hidden_dim in hidden_dims])

        self.dropout = nn.Dropout(dropout)

        self.relu = nn.ReLU()
    # WARNING, we also need to pass the graph decomposition here, if we want to use the adjacency of the local neighborhoods
    def forward(self,features, node_layers,ancestor_layers, ancestor_mappings,all_nodes_idx): 
        """
        Parameters
        ----------
        features : torch.Tensor
            An (n' x input_dim) tensor of input node features.
        node_layers : list of numpy array
            node_layers[i] is an array of the nodes in the ith layer of the
            computation graph.
        mappings : list of dictionary
            mappings[i] is a dictionary mapping node v (labelled 0 to |V|-1)
            in node_layers[i] to its position in node_layers[i]. For example,
            if node_layers[i] = [2,5], then mappings[i][2] = 0 and
            mappings[i][5] = 1.
        rows : numpy array
            rows[i] is an array of neighbors of node i.

        Returns
        -------
        out : torch.Tensor
            An (len(node_layers[-1]) x output_dim) tensor of output node features.
        """
        # ancestor mapping saves all nodes
        all_node_layers = [list(node_layers[self.num_layers])]
        all_ancestor_layers = [list(ancestor_layers[self.num_layers])]
        for k in range(self.num_layers - 1,-1,-1):
            new_arr1 = all_node_layers[-1] + list(node_layers[k])
            new_arr2 = all_ancestor_layers[-1] + list(ancestor_layers[k])
            all_node_layers.append(new_arr1)
            all_ancestor_layers.append(new_arr2)
        all_node_layers.reverse()
        all_ancestor_layers.reverse()
        #
        out = features
        for k in range(self.num_layers):
            # define nodes in the k-th hop of the computational graph to be processed
            curr_nodes = node_layers[k]
            ancestors = ancestor_layers[k]
            # create a idx
            uall_nodes = list(np.unique(all_node_layers[k],return_index=True)[0])
            uall_ancestors = list(np.unique(all_ancestor_layers[k],return_index=True)[0])
            # indices of all nodes and ancestors from level k to 0
            uall_nodes_idx = {j:i for i,j in enumerate(uall_nodes)}
            uall_ancestors_idx = {j:i for i,j in enumerate(uall_ancestors)}

            curr_ancestors_idx = [uall_nodes_idx[v] for v in ancestors]
            curr_nodes_idx = [uall_nodes_idx[v] for v in curr_nodes]
            # define mappings from node ids to node indices in tall_nodes_idxhe k-th hop of the computational graph
            #mapping = node_mappings[k] # -> ancestor mapping
            # aggregate - line 11 of Algorithm 2 in Hamilton, Ying, and Leskovec (2017)
            # forward(self, features, nodes, ancestors, nodes_mapping, ancestor_mapping)
            aggregate = self.aggregators[k](out, # source tensor
                                            curr_ancestors_idx,
                                            curr_nodes_idx, # indices of elements to aggregate in source tensor
                                            ancestors, # sorted mapping of nodes to indices
                                            uall_ancestors_idx # new indices for the nodes
                                            )
            print(aggregate.shape)
            print(out.shape)
            print(len(uall_ancestors_idx))
            # Here we are basically removing some indices 
            # concat - line 12 of Algorithm 2 in Hamilton, Ying, and Leskovec (2017)
            update_new_anestors = torch.tensor([uall_nodes_idx[v] for v in uall_ancestors])
            print(update_new_anestors.shape)
            out = torch.cat((torch.index_select(out, 0, update_new_anestors), aggregate), dim=1)#torch.cat((out[uancestors, :], aggregate), dim=1)
            out = self.fcs[k](out)

            # normalize - line 13 of Algorithm 2 in Hamilton, Ying, and Leskovec (2017)
            if k+1 < self.num_layers:
                out = self.relu(out)
                out = self.bns[k](out)
                out = self.dropout(out)
                out = out.div(out.norm(dim=1, keepdim=True)+1e-6)

        return out