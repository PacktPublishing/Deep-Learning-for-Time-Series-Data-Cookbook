import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import lightning.pytorch as pl
from torch.utils.data import Dataset, DataLoader

data_directory = "assets/datasets/Car"

train = pd.read_table(f"{data_directory}/Car_TRAIN.tsv", header=None)
test = pd.read_table(f"{data_directory}/Car_TEST.tsv", header=None)


class TSCDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


class TSCDataModule(pl.LightningDataModule):
    def __init__(self, train_df, test_df, batch_size=1):
        super().__init__()
        self.train_df = train_df
        self.test_df = test_df
        self.batch_size = batch_size

        self.scaler = MinMaxScaler()
        self.encoder = OneHotEncoder(categories="auto", sparse_output=False)

        self.train = None
        self.validation = None
        self.test = None

    def setup(self, stage=None):
        y_train = self.encoder.fit_transform(
            self.train_df.iloc[:, 0].values.reshape(-1, 1)
        )
        y_test = self.encoder.transform(self.test_df.iloc[:, 0].values.reshape(-1, 1))

        X_train = train.iloc[:, 1:]
        X_test = test.iloc[:, 1:]

        X_train = self.scaler.fit_transform(X_train)
        X_test = self.scaler.transform(X_test)

        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, stratify=y_train
        )

        X_train, X_val, X_test = [
            torch.tensor(arr, dtype=torch.float).unsqueeze(1)
            for arr in [X_train, X_val, X_test]
        ]
        y_train, y_val, y_test = [
            torch.tensor(arr, dtype=torch.long) for arr in [y_train, y_val, y_test]
        ]

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
