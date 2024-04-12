import torch

from aggregators.Aggregator import Aggregator

class MeanAggregator(Aggregator):

    def __init__(self):
        self.name = "MeanAggregator"

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
        return torch.mean(features, dim=0)