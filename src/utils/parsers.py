import sys
import torch.nn as nn

from aggregators.LSTMAggregator import LSTMAggregator
from aggregators.MaxPoolAggregator import MaxPoolAggregator
from aggregators.MeanPoolAggregator import MeanPoolAggregator
from aggregators.MeanAggregator import MeanAggregator

def get_agg_class(agg_class):
    """
    Parameters
    ----------
    agg_class : str
        Name of the aggregator class.

    Returns
    -------
    layers.Aggregator
        Aggregator class.
    """

    if agg_class == 'LSTMAggregator':
        return LSTMAggregator
    elif agg_class == 'MaxPoolAggregator':
        return MaxPoolAggregator
    elif agg_class == 'MeanPoolAggregator':
        return MeanPoolAggregator
    elif agg_class == 'MeanAggregator':
        return MeanAggregator
    else:
        raise ValueError('Invalid aggregator class: {}'.format(agg_class))

def get_criterion(task):
    """
    Parameters
    ----------
    task : str
        Name of the task.

    Returns
    -------
    criterion : torch.nn.modules._Loss
        Loss function for the task.
    """
    if task == 'node_classification':
        criterion = nn.CrossEntropyLoss()

    return criterion