
import torch
import torch.nn as nn
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, TensorDataset
from typing import Any, List, Union, Optional
import os
import numpy as np

from kse527.utils.datamodules.datasets.trajectory import TrajectoryDataset

class TrajectoryDataModule(LightningDataModule):

    def __init__(self, path='data/', splits=None, batch_size=1024, shuffle=True, pin_memory=False, drop_last=False, num_workers=4):
        super().__init__()
        if splits:
            self.train_data, self.val_data, self.test_data = splits
        else: 
            self.path=path
            self.prepare_data()
        self.batch_size = batch_size
        self.shuffle, self.pin_memory, self.drop_last, self.num_workers= shuffle, pin_memory, drop_last, num_workers
        self.setup()

    def prepare_data(self):
        self.train_data = np.load(os.path.join(self.path, 'train.npz'))
        self.val_data   = np.load(os.path.join(self.path, 'val.npz'))
        self.test_data  = np.load(os.path.join(self.path, 'test.npz'))

    def setup(self, stage: Optional[str] = None) -> None:
        td, vd, tstd = self.train_data, self.val_data, self.test_data 
        self.train_dataset = TrajectoryDataset(td['x0'], td['u0'], td['target'], td['t_span'])
        self.val_dataset   = TrajectoryDataset(vd['x0'], vd['u0'], vd['target'], vd['t_span'])
        self.test_dataset  = TrajectoryDataset(tstd['x0'], tstd['u0'], tstd['target'], tstd['t_span'])

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """ Train dataloader """
        return self._data_loader(self.train_dataset, shuffle=self.shuffle)

    def val_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ Val dataloader """
        return self._data_loader(self.val_dataset)

    def test_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ Test dataloader """
        return self._data_loader(self.test_dataset)

    def _data_loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory
        )

    def normalize_fn(self):
        pass


if __name__ == '__main__':

    dataset = TrajectoryDataModule('data/pendulum/', num_workers=2)
    x, u, y, t = next(iter(dataset.train_dataloader()))

    print(f"Input shape: {x.shape}\nOutput shape: {y.shape}")
    print("Train Dataloader lenght: {}".format(len(dataset.train_dataloader())))