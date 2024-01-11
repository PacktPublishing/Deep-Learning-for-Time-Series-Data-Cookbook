import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader

data_directory = 'assets/datasets/Car'

train = pd.read_table(f'{data_directory}/Car_TRAIN.tsv', header=None)
test = pd.read_table(f'{data_directory}/Car_TEST.tsv', header=None)


class TSCDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


class TSCDataModule(LightningDataModule):
    def __init__(self, train_df, test_df, batch_size=1):
        super().__init__()
        self.train_df = train_df
        self.test_df = test_df
        self.batch_size = batch_size

        self.scaler = MinMaxScaler()
        self.encoder = OneHotEncoder(categories='auto', sparse=False)

        self.train = None
        self.validation = None
        self.test = None

    def setup(self, stage=None):
        y_train = self.encoder.fit_transform(np.expand_dims(train.iloc[:, 0], axis=-1))
        y_test = self.encoder.transform(np.expand_dims(test.iloc[:, 0], axis=-1))

        X_train = train.iloc[:, 1:]
        X_test = test.iloc[:, 1:]

        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        X_train, X_val, y_train, y_val = \
            train_test_split(X_train, y_train,
                             test_size=0.2,
                             stratify=y_train)

        X_train = torch.from_numpy(X_train).unsqueeze(1)
        X_val = torch.from_numpy(X_val).unsqueeze(1)
        X_test = torch.from_numpy(X_test).unsqueeze(1)

        y_train = torch.from_numpy(y_train)
        y_val = torch.from_numpy(y_val)
        y_test = torch.from_numpy(y_test)

        self.train = TSCDataset(X_train, y_train)
        self.validation = TSCDataset(X_val, y_val)
        self.test = TSCDataset(X_test, y_test)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.validation, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)


datamodule = TSCDataModule(train_df=train, test_df=test)

datamodule.setup()

x, y = next(iter(datamodule.train_dataloader()))
