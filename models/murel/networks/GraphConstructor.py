import torch_geometric.transforms as T
from torch_geometric.data import Data


class GraphConstructor:
    def __init__(self):
        pass

    @staticmethod
    def getKNNConstructor(k=6, force_undirected=True):
        return T.KNNGraph(k=k, force_undirected=force_undirected)
