import numpy as np
from torch.utils.data import Dataset
from .process_dataset import *
import random

class COFSDataset(Dataset):
    def __init__(self, data, index, length, config):
        self.data = data
        self.index = index
        self.length = length
        self.permutation_num = config['data']['permutation_num']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        index = self.index[idx]
        length = self.length[idx]

        return sample, index, length


        # order = permute(int(length[0]))
        # random.shuffle(order)
        # order_len = min(self.permutation_num, len(order))
        # order = order[:order_len]






