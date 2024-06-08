import numpy as np
import torch
import torch.nn as nn

class Aggregator(nn.Module):

    def __init__(self, input_dim=None, output_dim=None, device='cpu'):
        """
        Parameters
        ----------
        input_dim : int or None.
            Dimension of input node features. Used for defining fully
            connected layer in pooling aggregators. Default: None.
        output_dim : int or None
            Dimension of output node features. Used for defining fully
            connected layer in pooling aggregators. Currently only works when
            input_dim = output_dim. Default: None.
        """
        super(Aggregator, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.device = device

    def forward(self, features, nodes, mapping, rows, num_samples=25):
        """
        Parameters
        ----------
        features : torch.Tensor
            An (n' x input_dim) tensor of input node features.
        nodes : numpy array
            nodes is a numpy array of nodes in the current layer of the computation graph.
        mapping : dict
            mapping is a dictionary mapping node v (labelled 0 to |V|-1) to
            its position in the layer of nodes in the computationn graph
            before nodes. For example, if the layer before nodes is [2,5],
            then mapping[2] = 0 and mapping[5] = 1.
        rows : numpy array
            rows[i] is an array of neighbors of node i which is present in nodes.
        num_samples : int
            Number of neighbors to sample while aggregating. Default: 25.

        Returns
        -------
        out : torch.Tensor
            An (len(nodes) x output_dim) tensor of output node features.
            Currently only works when output_dim = input_dim.
        """
        _choice, _len, _min = np.random.choice, len, min
        mapped_rows = [np.array([mapping[v] for v in row], dtype=np.int64) for row in rows]
        if num_samples == -1:
            sampled_rows = mapped_rows
        else:
            sampled_rows = [_choice(row, _min(_len(row), num_samples), _len(row) < num_samples) for row in mapped_rows]

        n = _len(nodes)
        if self.__class__.__name__ == 'LSTMAggregator':
            out = torch.zeros(n, 2*self.output_dim).to(self.device)
        else:
            out = torch.zeros(n, self.output_dim).to(self.device)
        for i in range(n):
            if _len(sampled_rows[i]) != 0:
                out[i, :] = self._aggregate(features[sampled_rows[i], :])

        return out

    # out, # source tensor curr_ancestors_idx,curr_nodes_idx, ancestors, uall_ancestors_idx
    def forward(self, features,curr_ancestors_idx, curr_nodes_idx,ancestors,uall_ancestors_idx):
        # all_ancestors is already a dictionary
        _choice, _len, _min = np.random.choice, len, min
        n = _len(uall_ancestors_idx) # number of unique ancestors up to the root
        # for each ancestor the nodes has
        uancestors = list(np.unique(ancestors, return_index=True)[0])
        #
        ancestors = np.array(ancestors)
        if self.__class__.__name__ == 'LSTMAggregator':
            out = torch.zeros(n, 2*self.output_dim).to(self.device)
        else:
            out = torch.zeros(n, self.output_dim).to(self.device)
        #
        for id_ancestor in uancestors:
            agg_ids = (ancestors == id_ancestor)
            curr_nodes_idx_aux = np.array(curr_nodes_idx)
            sel_rows = curr_nodes_idx_aux[agg_ids]
            print(features.shape)
            print(sel_rows)
            sel_features = torch.index_select(features, 0, torch.tensor(sel_rows))
            out[uall_ancestors_idx[id_ancestor],] = self._aggregate(features=sel_features)
            #
            
        #print(ancestors)
        #print(nodes)
        return out


    def _aggregate(self, features):
        """
        Parameters
        ----------

        Returns
        -------
        """
        raise NotImplementedError