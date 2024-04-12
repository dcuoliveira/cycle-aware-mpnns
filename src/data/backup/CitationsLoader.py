import sys
import os
import shutil
import torch
from torch_geometric.datasets import Planetoid
import torch_geometric.utils as pyg_utils

# sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from utils.conn_data import load_pickle, save_pickle
from utils.SupressPrint import SuppressPrint

INPUTS_PATH = os.path.join(os.path.dirname(__file__), "inputs")

class CitationsLoader(object):
    """
    Class to load the citation dataset from Planetoid.

    Args:
        name (str): Name of the dataset to be loaded. It can be one of the following: "cora", "citeseer" or "pubmed".
        download (bool): If True, the dataset will be downloaded from the internet. If False, the dataset will be loaded from the local directory.

    Returns:
        None

    """
    
    def __init__(self, name: str, download: bool):
        super().__init__()
    
        self.name = name
        self.download = download
        self._read_data(name=name, download=download)

    def _read_data(self, name: str, download: bool):
        if download:
            with SuppressPrint():
                dataset = Planetoid(root=os.path.join(INPUTS_PATH), name=name)
        else:
            dataset = load_pickle(os.path.join(INPUTS_PATH, self.example_name, "graph_info.pickle"))

        # graph as torch dataset
        self.data = dataset[0]

        # networkx graph object
        self.G = pyg_utils.to_networkx(self.data, to_undirected=True)

        # adjacency matrix
        edge_index = self.data.edge_index
        self.n_nodes = self.data.num_nodes
        torch_Adj = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.size(1)), (self.n_nodes, self.n_nodes))
        self.Adj = torch_Adj.to_dense()

        # define output data 
        results = {
            "data": self.data,
            "G": self.G,
            "Adj": self.Adj
        }

        # delete the original dataset
        output_path = os.path.join(INPUTS_PATH, name)
        if download:
            shutil.rmtree(output_path)

        # create directory for the dataset
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # save the output data
        save_pickle(path=f"{output_path}/graph_data.pkl", obj=results)

DEBUG = False

if __name__ == "__main__":
    if DEBUG:
        loader = CitationsLoader(name="cora", download=True)