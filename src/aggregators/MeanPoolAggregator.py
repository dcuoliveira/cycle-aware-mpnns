import torch

from aggregators.PoolAggregator import PoolAggregator

class MeanPoolAggregator(PoolAggregator):

    def __init__(self):
        self.name = "MeanPoolAggregator"

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
        return torch.mean(features, dim=0)