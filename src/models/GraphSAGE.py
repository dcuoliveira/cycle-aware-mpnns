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
            Dimension of input node features.
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

    def forward(self, features, node_layers, mappings, rows):
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
        out = features
        for k in range(self.num_layers):

            # define nodes in the k-th hop of the computational graph to be processed
            nodes = node_layers[k+1]

            # define mappings from node ids to node indices in the k-th hop of the computational graph
            mapping = mappings[k]
            init_mapped_nodes = np.array([mappings[0][v] for v in nodes], dtype=np.int64)

            # get adjacency matrix rows associated with the nodes in the k-th hop of the computational graph
            cur_rows = rows[init_mapped_nodes]

            # aggregate - line 11 of Algorithm 2 in Hamilton, Ying, and Leskovec (2017)
            aggregate = self.aggregators[k](out, # source tensor
                                            nodes, # indices of elements to aggregate in source tensor
                                            mapping, # sorted mapping of nodes to indices
                                            cur_rows, # QUESTION: what is this?
                                            self.num_samples # dimension in which to aggregate
                                            )
            cur_mapped_nodes = np.array([mapping[v] for v in nodes], dtype=np.int64)

            # concat - line 12 of Algorithm 2 in Hamilton, Ying, and Leskovec (2017)
            out = torch.cat((out[cur_mapped_nodes, :], aggregate), dim=1)
            out = self.fcs[k](out)

            # normalize - line 13 of Algorithm 2 in Hamilton, Ying, and Leskovec (2017)
            if k+1 < self.num_layers:
                out = self.relu(out)
                out = self.bns[k](out)
                out = self.dropout(out)
                out = out.div(out.norm(dim=1, keepdim=True)+1e-6)

        return out