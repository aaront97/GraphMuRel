import torch_geometric.transfroms as T
from torch_geometric.data import Data


class GraphConstructor:
    def __init__(self, construction_type='knn'):
        if construction_type == 'knn':
            self.constructor = T.KNNGraph(k=6, force_undirected=True)

    def construct_graph(self, boxes, object_feat):
        data = Data(x=object_feat, pos=boxes)
        graph = self.constructor(data)
        return graph
