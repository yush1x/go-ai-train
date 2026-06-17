import torch
from torch.utils.data import Dataset
import h5py

class TrainDataset(Dataset):
    def __init__(self, h5_path):
        self.h5_path = h5_path
        self.f = None

        with h5py.File(self.h5_path, "r") as f:
            self.length = f["state"].shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.f is None:
            self.f = h5py.File(self.h5_path, "r")

        state = torch.from_numpy(self.f["state"][idx]).float()
        policy = torch.tensor(self.f["policy"][idx]).long()
        value = torch.tensor(self.f["value"][idx]).float()

        return state, policy, value
