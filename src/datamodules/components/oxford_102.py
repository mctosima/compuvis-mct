import torch
from torch.utils.data import Dataset


class Oxford102(Dataset):
    def __init__(self, data_dir: str = "data/oxford-102-flower-pytorch/flower_data/"):

        super().__init__()

        self.data_dir = data_dir
        self.train_dir = data_dir + "/train"
        self.test_dir = data_dir + "/test"
        self.valid_dir = data_dir + "/valid"

    def __len__(self):
        return len(self.list)
