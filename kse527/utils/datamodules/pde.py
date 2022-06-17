from pathlib import Path
from typing import Any, List, Union, Optional

import scipy.io
import numpy as np
import mat73
import os
from os import makedirs
from os.path import exists, join
import gdown
from pathlib import Path
import zipfile

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset
from pytorch_lightning import LightningDataModule

import kse527
from kse527.utils.datamodules.utils import UnitGaussianNormalizer

# Get default path
default_path = Path(os.path.dirname(kse527.__file__)).parent


class DownloadManager:

    def __init__(self, root_dir=default_path, download=True,
                id="1SsC1Fy1ijHzNm0AYsF44K8w1QNOaGLg3",
                filename='lasagne.zip'):
        self.id, self.filename = id, filename
        self.root_dir = root_dir
        self.output_path = join(root_dir, self.filename)
        download_exists = not self.download_exists()
        self.to_download = download and download_exists

    def download(self):
        makedirs(self.root_dir, exist_ok=True)
        gdown.download(id=self.id, output=self.output_path, quiet=False, resume=True)

    def download_exists(self):
        return exists(self.output_path)

    def extract(self):
        with zipfile.ZipFile(self.output_path, 'r') as zip_ref:
            zip_ref.extractall(self.root_dir)


class NavierStokes(LightningDataModule):

    def __init__(self, data_dir=default_path, ntrain=1000, ntest=200, subsampling_rate=1, viscosity=1e-3,
                 batch_size=32, target_time=40, shuffle=False, pin_memory=False, drop_last=False,
                 normalize=False):
        super().__init__()
        self.data_dir = Path(data_dir).expanduser()
        assert viscosity in [1e-3, 1e-4, 1e-5], f"Viscosity setting: {viscosity} not available."
        self.download(data_dir, viscosity)
        self.viscosity = viscosity
        self.set_file_names()
        self.viscosity = viscosity
        self.ntrain = ntrain
        self.ntest = ntest
        self.subsampling_rate = subsampling_rate
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.target_time = target_time
        self.normalize = normalize

    def set_file_names(self):
        if self.viscosity == 1e-3:
            self.file_name = f'ns_V1e-3_N5000_T50.mat'
            self.sol_file_name = f'ns_V1e-3_N5000_T50_sol'
            self.t_span_file_name = f'ns_V1e-3_N5000_T50_t_span'
            self.ic_file_name = f'ns_V1e-3_N5000_T50_ic'
        elif self.viscosity == 1e-4:
            self.file_name = f'ns_V1e-4_N10000_T30.mat'
            self.sol_file_name = f'ns_V1e-4_N10000_T30_sol'
            self.t_span_file_name = f'ns_V1e-4_N10000_T30_t_span'
            self.ic_file_name = f'ns_V1e-4_N10000_T30_ic'   
        elif self.viscosity == 1e-5:
            self.file_name = f'NavierStokes_V1e-5_N1200_T20.mat'
            self.sol_file_name = f'ns_V1e-5_N1200_T20_sol'
            self.t_span_file_name = f'ns_V1e-5_N1200_T20_t_span'
            self.ic_file_name = f'ns_V1e-5_N1200_T20_ic'           

    def download(self, root_dir, viscosity):
        if viscosity == 1e-3:
            id = '1r3idxpsHa21ijhlu3QQ1hVuXcqnBTO7d'
            filename = 'ns1e-3.zip'
        elif viscosity == 1e-4:
            id = '1RmDQQ-lNdAceLXrTGY_5ErvtINIXnpl3'
            filename = 'ns1e-4.zip'
        elif viscosity == 1e-5:
            id = '1lVgpWMjv9Z6LEv3eZQ_Qgj54lYeqnGl5'
            filename = 'ns1e-5.zip'

        download_manager = DownloadManager(root_dir=root_dir, filename=filename, download=True)
        if download_manager.to_download:
            download_manager.id = id
            download_manager.download()
            download_manager.extract()

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'test' and hasattr(self, 'dataset_test'):
            return

        if not (self.data_dir / self.sol_file_name).is_file(): # preprocess .mat file only the first time
            if self.viscosity in [1e-3, 1e-4]:
                data = mat73.loadmat(self.data_dir / self.file_name)
            else:
               data = scipy.io.loadmat(self.data_dir / self.file_name)

            rate = self.subsampling_rate
            t_span = np.array(data['t'], dtype=int).flatten()
            x = data['a'][:, ::rate, ::rate]   
            sol = data['u'][:, ::rate, ::rate]   

            np.save(self.data_dir / self.t_span_file_name, t_span)
            np.save(self.data_dir / self.ic_file_name, x)
            np.save(self.data_dir / self.sol_file_name, sol)

        else:
            t_span = np.load(self.data_dir / self.t_span_file_name)
            x = np.load(self.data_dir / self.ic_file_name)
            sol = np.load(self.data_dir / self.sol_file_name)  

        # downsample
        x = x[:, ::rate, ::rate]   
        sol = sol[:, ::rate, ::rate]  

        # select solution at desired time (`target_time`) as `y`
        sol_idx = np.where(t_span == self.target_time)[0][0]

        y = sol[..., sol_idx]
        x, y = torch.tensor(x, dtype=torch.float).squeeze(), torch.tensor(y, dtype=torch.float).squeeze()

        x_train, x_test = x[:self.ntrain], x[-self.ntest:]
        y_train, y_test = y[:self.ntrain], y[-self.ntest:]

        if self.normalize:
            x_normalizer = UnitGaussianNormalizer(x_train)
            x_train = x_normalizer.encode(x_train)
            x_test = x_normalizer.encode(x_test)

        self.y_normalizer = UnitGaussianNormalizer(y_train)


        self.dataset_train = TensorDataset(x_train, y_train)
        self.dataset_test = TensorDataset(x_test, y_test)
        self.dataset_val = self.dataset_test

    def train_dataloader(self, *args: Any, **kwargs: Any) -> DataLoader:
        """ The train dataloader """
        return self._data_loader(self.dataset_train, shuffle=self.shuffle)

    def val_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The val dataloader """
        return self._data_loader(self.dataset_val)

    def test_dataloader(self, *args: Any, **kwargs: Any) -> Union[DataLoader, List[DataLoader]]:
        """ The test dataloader """
        return self._data_loader(self.dataset_test)

    def _data_loader(self, dataset: Dataset, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            drop_last=self.drop_last,
            pin_memory=self.pin_memory
        )

if __name__ == '__main__':
    dmodule = NavierStokes('data', viscosity=1e-5, target_time=20)
    dmodule.setup()
    # import pdb; pdb.set_trace()
    x, y = next(iter(dmodule.train_dataloader()))
    print(x.shape, y.shape)
    x, y = next(iter(dmodule.test_dataloader()))
    print(x.shape, y.shape)