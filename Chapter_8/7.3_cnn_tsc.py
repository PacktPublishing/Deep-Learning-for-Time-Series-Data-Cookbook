import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

import torch
from torch import nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from lightning.pytorch.callbacks import EarlyStopping
from torch.utils.data import Dataset, DataLoader
import lightning.pytorch as pl

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


class TSCDataModule(pl.LightningDataModule):
    def __init__(self, train_df, test_df, batch_size):
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
                             test_size=0.25,
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


class ConvolutionalTSC(nn.Module):
    def __init__(self, input_dim, output_dim=1):
        super(ConvolutionalTSC, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=input_dim,
                               out_channels=64,
                               kernel_size=3,
                               stride=1,
                               padding=1)

        self.conv2 = nn.Conv1d(in_channels=64,
                               out_channels=32,
                               kernel_size=3,
                               stride=1,
                               padding=1)

        self.conv3 = nn.Conv1d(in_channels=32,
                               out_channels=16,
                               kernel_size=3,
                               stride=1,
                               padding=1)

        self.maxp = nn.MaxPool1d(kernel_size=3)

        self.fc1 = nn.Linear(in_features=336, out_features=32)
        self.fc2 = nn.Linear(in_features=32, out_features=output_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.maxp(x)
        x = F.relu(self.conv2(x))
        x = self.maxp(x)
        x = F.relu(self.conv3(x))
        x = self.maxp(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class TSCCnnModel(pl.LightningModule):
    def __init__(self, output_dim):
        super().__init__()
        self.network = ConvolutionalTSC(
            input_dim=1,
            output_dim=output_dim,
        )

    def forward(self, x):
        x = x.type(torch.FloatTensor)
        return self.network(x)

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_pred = self.forward(x)
        loss = F.cross_entropy(y_pred, y.type(torch.FloatTensor))

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y.type(torch.FloatTensor))

        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch

        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y.type(torch.FloatTensor))

        self.log('test_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)


datamodule = TSCDataModule(train_df=train,
                           test_df=test,
                           batch_size=8)

model = TSCCnnModel(output_dim=4)

early_stop_callback = EarlyStopping(monitor="val_loss",
                                    min_delta=1e-4,
                                    patience=10,
                                    verbose=False,
                                    mode="min")

trainer = pl.Trainer(
    max_epochs=30,
    accelerator='cpu',
    log_every_n_steps=2,
    enable_model_summary=True,
    callbacks=[early_stop_callback],
)

trainer.fit(model, datamodule)
