import torch
import collections
class Compose:
    def __init__(self, transform_list):
        self.transform_list = transform_list
    def __call__(self, batch):
        for trfm in self.transform_list:
            batch = trfm(batch)
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

class CreateBatchItem:
    def __init__(self):
        pass
    def __call__(self, batch):
        return self._createBatchItem(batch)
    
    def _createBatchItem(self, batch):
        out = {}
        for key, value in batch.items():
            if torch.is_tensor(value[0]):
                out[key] = torch.stack(value, dim=0)
            else:
                out[key] = value
        return out

class PrepareBaselineBatch:
    def __init__(self):
        pass
    def __call__(self, batch):
        return (batch['concat_vector'].detach(), torch.squeeze(batch['answer_id']).detach())

class PrepareBaselineTestBatch:
    def __init__(self):
        pass
    def __call__(self, batch):
        return self._createBatchTestItem(batch)
    
    def _createBatchTestItem(self, batch):
        return (batch['concat_vector'].detach(), batch['question_id'])