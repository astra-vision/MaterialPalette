import math

from easydict import EasyDict
from torch.utils.data import Dataset, default_collate


class EmptyDataset(Dataset):
    def __init__(self, length):
        self.length = length
    def __getitem__(self, _):
        return None
    def __len__(self):
        return self.length

class MultiLoader:
    """Iterator wrapper to iterate over multiple dataloaders at the same time."""
    def __init__(self, a, b):
        # a = self._repeat(a, b)
        self.loaders = [a,b]

    def __iter__(self):
        return zip(*self.loaders)

    def __len__(self):
        return min(map(len, self.loaders))

    def _repeat(self, a, b):
        if len(a) < len(b):
            k = math.ceil(len(b)/len(a))
            return RepeatLoader(a, k)
        return a

class RepeatLoader:
    def __init__(self, loader, k):
        self.loader = loader
        self.k = k

    def __iter__(self):
        for _ in range(self.k):
            for x in self.loader:
                yield x

    def __len__(self):
        return self.k*len(self.loader)

def collate_fn(data):
    return data if None in data else EasyDict(default_collate(data))