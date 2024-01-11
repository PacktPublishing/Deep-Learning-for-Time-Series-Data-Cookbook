from typing import List

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch import nn
import torch
import torch.nn.functional as F
from pytorch_forecasting import TimeSeriesDataSet
from lightning.pytorch.callbacks import EarlyStopping
import lightning.pytorch as pl

N_LAGS = 7
HORIZON = 4
TARGET = ['Incoming Solar', 'Air Temp', 'Vapor Pressure']

mvtseries = pd.read_csv('assets/daily_multivariate_timeseries.csv',
                        parse_dates=['datetime'],
                        index_col='datetime')


class MultivariateSeriesDataModule(pl.LightningDataModule):
    def __init__(self,
                 data: pd.DataFrame,
                 n_lags: int,
                 horizon: int,
                 target_variables: List[str],
                 test_size: float = 0.2,
                 batch_size: int = 32):
        super().__init__()

        self.data = data
        self.var_names = self.data.columns.tolist()
        self.batch_size = batch_size
        self.test_size = test_size
        self.n_lags = n_lags
        self.horizon = horizon
        self.target_variables = target_variables

        self.training = None
        self.validation = None
        self.test = None
        self.predict_set = None

    def setup(self, stage=None):
        self.data['time_index'] = np.arange(self.data.shape[0])
        self.data['group_id'] = 0

        unique_times = self.data['time_index'].sort_values().unique()

        tr_ind, ts_ind = \
            train_test_split(unique_times,
                             test_size=self.test_size,
                             shuffle=False)

        tr_ind, vl_ind = \
            train_test_split(tr_ind,
                             test_size=0.1,
                             shuffle=False)

        training_df = self.data.loc[self.data['time_index'].isin(tr_ind), :]
        validation_df = self.data.loc[self.data['time_index'].isin(vl_ind), :]
        test_df = self.data.loc[self.data['time_index'].isin(ts_ind), :]

        self.training = TimeSeriesDataSet(
            data=training_df,
            time_idx='time_index',
            target=self.target_variables,
            group_ids=['group_id'],
            max_encoder_length=self.n_lags,
            max_prediction_length=self.horizon,
            time_varying_unknown_reals=self.var_names,
            scalers={k: MinMaxScaler() for k in self.var_names
                     if k not in self.target_variables}
        )

        self.validation = TimeSeriesDataSet.from_dataset(self.training, validation_df)
        self.test = TimeSeriesDataSet.from_dataset(self.training, test_df)
        self.predict_set = TimeSeriesDataSet.from_dataset(self.training, self.data, predict=True)

    def train_dataloader(self):
        return self.training.to_dataloader(batch_size=self.batch_size, shuffle=False, num_workers=0)

    def val_dataloader(self):
        return self.validation.to_dataloader(batch_size=self.batch_size, shuffle=False, num_workers=0)

    def test_dataloader(self):
        return self.test.to_dataloader(batch_size=self.batch_size, shuffle=False, num_workers=0)

    def predict_dataloader(self):
        return self.predict_set.to_dataloader(batch_size=1, shuffle=False, num_workers=0)


class MultiOutputLSTM(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, num_layers, horizon, n_vars):
        super().__init__()
        self.n_vars = n_vars
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = int(self.n_vars * self.horizon)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, self.output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.hidden_dim).to(self.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])

        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x['encoder_cont'])

        y_pred = y_pred.unsqueeze(-1).view(-1, self.horizon, self.n_vars)
        y_pred = [y_pred[:, :, i] for i in range(self.n_vars)]

        loss = [F.mse_loss(y_pred[i], y[0][i]) for i in range(self.n_vars)]

        loss = torch.mean(torch.stack(loss))

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x['encoder_cont'])

        y_pred = y_pred.unsqueeze(-1).view(-1, self.horizon, self.n_vars)
        y_pred = [y_pred[:, :, i] for i in range(self.n_vars)]

        loss = [F.mse_loss(y_pred[i], y[0][i]) for i in range(self.n_vars)]

        loss = torch.mean(torch.stack(loss))

        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x['encoder_cont'])

        y_pred = y_pred.unsqueeze(-1).view(-1, self.horizon, self.n_vars)
        y_pred = [y_pred[:, :, i] for i in range(self.n_vars)]

        loss = [F.mse_loss(y_pred[i], y[0][i]) for i in range(self.n_vars)]

        loss = torch.mean(torch.stack(loss))

        self.log('test_loss', loss)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch

        y_pred = self(x['encoder_cont'])

        y_pred = y_pred.unsqueeze(-1).view(-1, self.horizon, self.n_vars)
        y_pred = [y_pred[:, :, i] for i in range(self.n_vars)]

        return y_pred

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


model = MultiOutputLSTM(input_dim=9,
                        hidden_dim=32,
                        num_layers=1,
                        horizon=HORIZON,
                        n_vars=len(TARGET))

datamodule = MultivariateSeriesDataModule(data=mvtseries,
                                          n_lags=N_LAGS,
                                          horizon=HORIZON,
                                          target_variables=TARGET)

early_stop_callback = EarlyStopping(monitor="val_loss",
                                    min_delta=1e-4,
                                    patience=10,
                                    verbose=False,
                                    mode="min")

trainer = pl.Trainer(max_epochs=20, callbacks=[early_stop_callback])
trainer.fit(model, datamodule)

trainer.test(model=model, datamodule=datamodule)
forecasts = trainer.predict(model=model, datamodule=datamodule)
