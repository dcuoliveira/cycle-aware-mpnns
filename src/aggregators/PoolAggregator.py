import torch.nn as nn

from aggregators.Aggregator import Aggregator

class PoolAggregator(Aggregator):

    def __init__(self, input_dim, output_dim, device='cpu'):
        """
        Parameters
        ----------
        input_dim : int
            Dimension of input node features. Used for defining fully connected layer.
        output_dim : int
            Dimension of output node features. Used for defining fully connected layer. Currently only works when output_dim = input_dim.
        """
        super(PoolAggregator, self).__init__(input_dim, output_dim, device)

        self.name = "PoolAggregator"
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()

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
        out = self.relu(self.fc1(features))
        return self._pool_fn(out)

    def _pool_fn(self, features):
        """
        Parameters
        ----------

        Returns
        -------
        """
        raise NotImplementedError