from torch.utils.data import Dataset

class COFSDataset(Dataset):
    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        index = self.index[idx]

        return sample, index









