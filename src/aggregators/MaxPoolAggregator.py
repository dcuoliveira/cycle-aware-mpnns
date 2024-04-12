import torch

from aggregators.PoolAggregator import PoolAggregator

class MaxPoolAggregator(PoolAggregator):

    def __init__(self):
        self.name = "MaxPoolAggregator"

    def _pool_fn(self, features):
        """
        Parameters
        ----------
        features : torch.Tensor
            Input features.

        Returns
        -------
        Aggregated feature.
        """
        return torch.max(features, dim=0)[0]