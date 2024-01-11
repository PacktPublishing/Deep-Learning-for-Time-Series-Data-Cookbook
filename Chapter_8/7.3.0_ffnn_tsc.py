# https://www.kaggle.com/code/purplejester/pytorch-deep-time-series-classification
# https://www.youtube.com/watch?v=PCgrgHgy26c&ab_channel=VenelinValkov

# https://jovian.com/ningboming/time-series-classification-cnn
# https://github.com/okrasolar/pytorch-timeseries/tree/master/data

# https://github.com/philipdarke/torchtime

# https://towardsdatascience.com/xai-for-forecasting-basis-expansion-17a16655b6e4
#
#     https: // nixtla.github.io / neuralforecast / tsdataset.html
#
# https://www.sktime.net/en/stable/api_reference/auto_generated/sktime.classification.deep_learning.InceptionTimeClassifier.html
# https://towardsdatascience.com/timesnet-the-latest-advance-in-time-series-forecasting-745b69068c9c
#
# https://github.com/okrasolar/pytorch-timeseries/tree/master


# !!!!! https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/04-inception-resnet-densenet.html


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


class MyTSCDataset(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)


class MyTSCDataModule(pl.LightningDataModule):
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

        X_train = torch.from_numpy(X_train)  # .unsqueeze(1)
        X_val = torch.from_numpy(X_val)  # .unsqueeze(1)
        X_test = torch.from_numpy(X_test)  # .unsqueeze(1)

        y_train = torch.from_numpy(y_train)
        y_val = torch.from_numpy(y_val)
        y_test = torch.from_numpy(y_test)

        self.train = MyTSCDataset(X_train, y_train)
        self.validation = MyTSCDataset(X_val, y_val)
        self.test = MyTSCDataset(X_test, y_test)

    def train_dataloader(self):
        return DataLoader(self.train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.validation, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test, batch_size=self.batch_size)


class FeedForwardNet(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, output_size)
        )

    def forward(self, X):
        return self.net(X)


class TimeSeriesFFNNModel(pl.LightningModule):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.network = FeedForwardNet(
            input_size=input_dim,
            output_size=output_dim,
        )

    def forward(self, x):
        network_input = x.squeeze(-1)

        network_input = network_input.reshape(network_input.shape[0], -1)
        network_input = network_input.type(torch.FloatTensor)

        prediction = self.network(network_input)

        return prediction

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_pred = self.forward(x)

        loss = F.cross_entropy(y_pred, y.type(torch.FloatTensor))

        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_pred = self.forward(x)

        loss = F.cross_entropy(y_pred, y.type(torch.FloatTensor))

        self.log('val_loss', loss)

        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch

        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y)

        self.log('test_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)


model = TimeSeriesFFNNModel(input_dim=train.shape[1] - 1, output_dim=4)

datamodule = MyTSCDataModule(train_df=train, test_df=test, batch_size=8)

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
