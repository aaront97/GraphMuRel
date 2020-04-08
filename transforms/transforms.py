import torch
import collections
from torch_geometric.data import Batch


class Compose:
    def __init__(self, transform_list):
        self.transform_list = transform_list

    def __call__(self, batch):
        for trfm in self.transform_list:
            batch = trfm(batch)
        return batch


class BatchGraph:
    def __init__(self):
        pass

    def __call__(self, batch):
        if isinstance(batch, collections.Mapping):
            if 'graph' in batch:
                batch['graph'] = Batch.from_data_list(batch['graph'])
        return batch

# class PadQuestions:
#     def __init__(self):
#         pass
#
#     def __call__(self, batch):
#         if isinstance(batch, collections.Mapping):
#             max_dim = max([int(item) for item in batch['question_lengths']])
#             res = []
#             for ids in batch['question_ids']:
#                 padded = ids.new_full((max_dim, ), 0)
#                 padded[:ids.size(0)] = ids
#                 res.append(padded)
#             batch['question_ids'] = res
#             return batch
#         else:
#             return batch


class Pad1DTensors:
    def __init__(self, dict_keys):
        self.dict_keys = dict_keys

    def __call__(self, batch):
        if isinstance(batch, collections.Mapping):
            for key in self.dict_keys:
                max_dim = max([x.size(0) for x in batch[key]])
                res = []
                for item in batch[key]:
                    padded = item.new_full((max_dim, ), 0)
                    padded[:item.size(0)] = item
                    res.append(padded)
                batch[key] = res
        return batch


class ConvertBatchListToDict:
    def __init__(self):
        pass

    def __call__(self, batch):
        return self.convertBLtoD(batch)

    def convertBLtoD(self, batch):
        if isinstance(batch[0], collections.Mapping):
            return {key: [d[key] for d in batch] for key in batch[0]}
        else:
            return batch


class StackTensors:
    def __init__(self):
        pass

    def __call__(self, batch):
        for key in batch:
            if isinstance(batch[key], collections.Sequence) and torch.is_tensor(batch[key][0]):
                batch[key] = torch.stack(batch[key])
        return batch


