import numpy as np
from torch.utils.data import Dataset
from .process_dataset import *
import random

class COFSDataset(Dataset):
    def __init__(self, seq, layout, length):
        self.seq = seq
        self.layout = layout
        self.length = length

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        seq = self.seq[idx]
        layout = self.layout[idx]
        length = self.length[idx]

        return seq, layout, length









