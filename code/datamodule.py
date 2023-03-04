# -*- coding:utf-8 -*-

from typing import Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from skimage.restoration import denoise_wavelet
from sklearn.preprocessing import MinMaxScaler, Normalizer
from torch.utils.data import DataLoader, TensorDataset


def data_split(df: pd.DataFrame, split_size: float = 0.2) -> Tuple[pd.DataFrame]:
    split_line1 = int(df.shape[0] * (1 - split_size))
    train_val, test = df.iloc[:split_line1], df.iloc[split_line1:]

    split_line2 = int(train_val.shape[0] * (1 - split_size))
    train, val = train_val.iloc[:split_line2], train_val.iloc[split_line2:]

    print(train.shape, val.shape, test.shape)

    return train, val, test


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    df.set_index("Date", inplace=True)
    df = df.iloc[:, 1:]
    df["Close"] = denoise_wavelet(
        df.iloc[:, 0],
        wavelet="haar",
        method="VisuShrink",
        mode="soft",
        rescale_sigma=True,
    )
    return df


def DatasetCreation(dataset, time_steps: int = 1):
    """
    define a function that gives a dataset and a time step, which then returns the input and output data
    """
    DataX, DataY = [], []
    for i in range(len(dataset) - time_steps - 1):
        a = dataset[i : i + time_steps]
        DataX.append(a)
        DataY.append(dataset[i + time_steps, 0])
    return np.array(DataX), np.array(DataY)


class TSDataModule(pl.LightningDataModule):
    def __init__(
        self,
        path: str,
        split_size: float = 0.2,
        num_time_steps: int = 12,
        batch_size: int = 32,
        num_cpus: int = 3,
        shuffle: bool = True,
    ):
        super().__init__()
        self.path = path
        self.batch_size = batch_size
        self.split_size = split_size
        self.num_time_steps = num_time_steps
        self.scaler = MinMaxScaler()
        self.num_cpus = num_cpus
        self.shuffle = shuffle

    def setup(self, stage: str):
        df = load_dataset(self.path)
        train, val, test = data_split(df, split_size=self.split_size)
        train_norm = self.scaler.fit_transform(train)
        val_norm = self.scaler.transform(val)
        # test_norm = self.scaler.transform(test)

        self.train_x, self.train_y = DatasetCreation(
            train_norm, time_steps=self.num_time_steps
        )
        self.val_x, self.val_y = DatasetCreation(
            val_norm, time_steps=self.num_time_steps
        )

    def train_dataloader(self):
        return DataLoader(
            TensorDataset(torch.Tensor(self.train_x), torch.Tensor(self.train_y)),
            batch_size=self.batch_size,
            num_workers=self.num_cpus,
            shuffle=self.shuffle,
        )

    def val_dataloader(self):
        return DataLoader(
            TensorDataset(torch.Tensor(self.val_x), torch.Tensor(self.val_y)),
            batch_size=self.batch_size,
            num_workers=self.num_cpus,
        )

    # def test_dataloader(self):
    #     return DataLoader(self.mnist_test, batch_size=self.batch_size)


if __name__ == "__main__":
    path = "data/input_data.csv"
    num_time_steps = 12
    batch_size = 32
    data_module = TSDataModule(
        path, num_time_steps=num_time_steps, batch_size=batch_size
    )
    data_module.setup("train")
    train_loader = data_module.train_dataloader()
    for x, y in train_loader:
        break
    print(x.shape, y.shape)
