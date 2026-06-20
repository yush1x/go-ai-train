from bisect import bisect_right
from pathlib import Path

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


class SelfPlayDataset(Dataset):
    def __init__(self, h5_path):
        self.h5_paths = self._resolve_h5_paths(h5_path)
        self.files = [None] * len(self.h5_paths)
        self.cumulative_lengths = []

        total_length = 0
        for path in self.h5_paths:
            with h5py.File(path, "r") as f:
                total_length += f["features"].shape[0]
                self.cumulative_lengths.append(total_length)

        if total_length == 0:
            raise ValueError("selfplay dataset is empty")

    def __len__(self):
        return self.cumulative_lengths[-1]

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self):
            raise IndexError(idx)

        file_idx = bisect_right(self.cumulative_lengths, idx)
        previous_length = 0 if file_idx == 0 else self.cumulative_lengths[file_idx - 1]
        local_idx = idx - previous_length

        if self.files[file_idx] is None:
            self.files[file_idx] = h5py.File(self.h5_paths[file_idx], "r")

        f = self.files[file_idx]

        features = torch.from_numpy(f["features"][local_idx]).float()
        policy = torch.from_numpy(f["policy"][local_idx]).float()
        value = torch.tensor(f["value"][local_idx]).float()
        score = torch.tensor(f["score"][local_idx]).float()
        ownership = torch.from_numpy(f["ownership"][local_idx]).long()

        return features, policy, value, score, ownership

    @staticmethod
    def _resolve_h5_paths(h5_path):
        if isinstance(h5_path, (str, Path)):
            path = Path(h5_path)
            if path.is_dir():
                paths = sorted(path.glob("*.h5"))
            else:
                paths = [path]
        else:
            paths = [Path(path) for path in h5_path]

        if not paths:
            raise ValueError("no selfplay h5 files found")

        return paths
