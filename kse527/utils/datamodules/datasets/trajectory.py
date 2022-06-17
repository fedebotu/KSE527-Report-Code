import numpy as np
import torch
from torch.utils.data.dataset import Dataset


class TrajectoryDataset(Dataset):

    def __init__(self, xs, us, targets, t_span, H=0):
        super().__init__()
        self.xs = torch.Tensor(xs)  # [#.traj, T, state dim]
        self.us = torch.Tensor(us)  # [#.traj, T, action dim]
        self.targets = torch.Tensor(targets) # [#.traj rollout, T, state dim]
        self.len = len(xs)

        if t_span.shape[0] != self.len: # cast to batch
            t_span = t_span[None]
        self.t_span = torch.Tensor(t_span).expand(self.len, -1)

    def __getitem__(self, idx):
        x0 = self.xs[idx, :]
        u0 = self.us[idx, :]
        target = self.targets[idx, :]
        t_span = self.t_span[idx, :]
        return x0, u0, target, t_span

    def __len__(self):
        return self.len
