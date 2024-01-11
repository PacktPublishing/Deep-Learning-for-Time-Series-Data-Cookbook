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
from torchmetrics.classification.accuracy import Accuracy

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


class Conv1dSamePadding(nn.Conv1d):
    """Represents the "Same" padding functionality from Tensorflow.
    See: https://github.com/pytorch/pytorch/issues/3867
    Note that the padding argument in the initializer doesn't do anything now
    """

    def forward(self, input):
        return conv1d_same_padding(input, self.weight, self.bias, self.stride,
                                   self.dilation, self.groups)


def conv1d_same_padding(input, weight, bias, stride, dilation, groups):
    # stride and dilation are expected to be tuples.
    kernel, dilation, stride = weight.size(2), dilation[0], stride[0]
    l_out = l_in = input.size(2)
    padding = (((l_out - 1) * stride) - l_in + (dilation * (kernel - 1)) + 1)
    if padding % 2 != 0:
        input = F.pad(input, [0, 1])

    return F.conv1d(input=input, weight=weight, bias=bias, stride=stride,
                    padding=padding // 2,
                    dilation=dilation, groups=groups)


class ConvBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 stride: int) -> None:
        super().__init__()

        self.layers = nn.Sequential(
            Conv1dSamePadding(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=kernel_size,
                              stride=stride),
            nn.BatchNorm1d(num_features=out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)


class ResidualNeuralNetworkModel(nn.Module):
    def __init__(self, in_channels: int, mid_channels: int = 64,
                 num_pred_classes: int = 1) -> None:
        super().__init__()

        self.input_args = {
            'in_channels': in_channels,
            'num_pred_classes': num_pred_classes
        }

        self.layers = nn.Sequential(*[
            ResNNBlock(in_channels=in_channels, out_channels=mid_channels),
            ResNNBlock(in_channels=mid_channels, out_channels=mid_channels * 2),
            ResNNBlock(in_channels=mid_channels * 2, out_channels=mid_channels * 2),

        ])

        self.fc = nn.Linear(mid_channels * 2, num_pred_classes)

    def forward(self, x):
        x = self.layers(x)
        out = self.fc(x.mean(dim=-1))
        # out = torch.sigmoid(out)

        return out


class ResNNBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        channels = [in_channels, out_channels, out_channels, out_channels]
        kernel_sizes = [8, 5, 3]

        self.layers = nn.Sequential(*[
            ConvBlock(in_channels=channels[i], out_channels=channels[i + 1],
                      kernel_size=kernel_sizes[i], stride=1) for i in range(len(kernel_sizes))
        ])

        self.match_channels = False
        if in_channels != out_channels:
            self.match_channels = True
            self.residual = nn.Sequential(*[
                Conv1dSamePadding(in_channels=in_channels, out_channels=out_channels,
                                  kernel_size=1, stride=1),
                nn.BatchNorm1d(num_features=out_channels)
            ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore

        if self.match_channels:
            return self.layers(x) + self.residual(x)
        return self.layers(x)


class TSCResNet(pl.LightningModule):
    def __init__(self, output_dim):
        super().__init__()

        self.resnet = \
            ResidualNeuralNetworkModel(in_channels=1,
                                       num_pred_classes=output_dim)

    def forward(self, x):
        out = self.resnet.forward(x)

        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = x.type(torch.FloatTensor)
        y = y.type(torch.FloatTensor)

        y_pred = self.forward(x)

        loss = F.cross_entropy(y_pred, y)
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.type(torch.FloatTensor)
        y = y.type(torch.FloatTensor)

        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y)

        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        x = x.type(torch.FloatTensor)
        y = y.type(torch.FloatTensor)

        y_pred = self(x)
        loss = F.cross_entropy(y_pred, y)

        acc = Accuracy(task='multiclass', num_classes=4)
        acc_score = acc(y_pred, y)

        self.log('acc_score', acc_score)
        self.log('test_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


model = TSCResNet(output_dim=4)

datamodule = TSCDataModule(train_df=train, test_df=test, batch_size=8)

early_stop_callback = EarlyStopping(monitor="val_loss",
                                    min_delta=1e-4,
                                    patience=20,
                                    verbose=False,
                                    mode="min")

trainer = pl.Trainer(
    max_epochs=100,
    accelerator='cpu',
    log_every_n_steps=2,
    enable_model_summary=True,
    callbacks=[early_stop_callback],
)

trainer.fit(model, datamodule)
trainer.test(model, datamodule)
