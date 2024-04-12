import torch
import torch.nn as nn
import numpy as np

from aggregators.Aggregator import Aggregator

class LSTMAggregator(Aggregator):

    def __init__(self, input_dim, output_dim, device='cpu'):
        """
        Parameters
        ----------
        input_dim : int
            Dimension of input node features. Used for defining LSTM layer.
        output_dim : int
            Dimension of output node features. Used for defining LSTM layer. Currently only works when output_dim = input_dim.

        """
        # super(LSTMAggregator, self).__init__(input_dim, output_dim, device)
        super().__init__(input_dim, output_dim, device)

        self.name = "LSTMAggregator"
        self.lstm = nn.LSTM(input_dim, output_dim, bidirectional=True, batch_first=True)

    def _aggregate(self, features):
        """
        Parameters
        ----------
        features : torch.Tensor
            Input features.

        Returns
        -------
        Aggregated feature.
        """
        perm = np.random.permutation(np.arange(features.shape[0]))
        features = features[perm, :]
        features = features.unsqueeze(0)

        out, _ = self.lstm(features)
        out = out.squeeze(0)
        out = torch.sum(out, dim=0)

        return out