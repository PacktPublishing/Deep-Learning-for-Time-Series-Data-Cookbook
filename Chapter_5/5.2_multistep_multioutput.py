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

mvtseries = pd.read_csv(
    "assets/daily_multivariate_timeseries.csv",
    parse_dates=["datetime"],
    index_col="datetime",
)

N_LAGS = 14
HORIZON = 7
TARGET = ["Incoming Solar", "Air Temp", "Vapor Pressure"]
n_vars = mvtseries.shape[1]


class MultivariateSeriesDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data: pd.DataFrame,
        target_variables: List[str],
        n_lags: int,
        horizon: int,
        test_size: float = 0.2,
        batch_size: int = 16,
    ):
        super().__init__()
        self.data = data
        self.batch_size = batch_size
        self.test_size = test_size
        self.n_lags = n_lags
        self.horizon = horizon
        self.target_variables = target_variables
        self.target_scaler = {k: MinMaxScaler() for k in target_variables}
        self.feature_names = [
            col for col in data.columns if col not in self.target_variables
        ]

        self.training = None
        self.validation = None
        self.test = None
        self.predict_set = None

        self.setup()

    def preprocess_data(self):
        self.data["time_index"] = np.arange(len(self.data))
        self.data["group_id"] = 0

    def split_data(self):
        time_indices = self.data["time_index"].values
        train_indices, test_indices = train_test_split(
            time_indices, test_size=self.test_size, shuffle=False
        )
        train_indices, val_indices = train_test_split(
            train_indices, test_size=0.1, shuffle=False
        )
        return train_indices, val_indices, test_indices

    def scale_target(self, df, indices):
        for c in self.target_variables:
            scaled_values = self.target_scaler[c].transform(df.loc[indices, [c]])
            df.loc[indices, c] = scaled_values

    def setup(self, stage=None):
        self.preprocess_data()
        train_indices, val_indices, test_indices = self.split_data()

        train_df = self.data.loc[self.data["time_index"].isin(train_indices)]
        val_df = self.data.loc[self.data["time_index"].isin(val_indices)]
        test_df = self.data.loc[self.data["time_index"].isin(test_indices)]

        for c in self.target_variables:
            self.target_scaler[c].fit(train_df[[c]])

        self.scale_target(train_df, train_df.index)
        self.scale_target(val_df, val_df.index)
        self.scale_target(test_df, test_df.index)

        self.training = TimeSeriesDataSet(
            train_df,
            time_idx="time_index",
            target=self.target_variables,
            group_ids=["group_id"],
            max_encoder_length=self.n_lags,
            max_prediction_length=self.horizon,
            time_varying_unknown_reals=self.feature_names + self.target_variables,
            scalers={name: MinMaxScaler() for name in self.feature_names},
        )
        self.validation = TimeSeriesDataSet.from_dataset(self.training, val_df)
        self.test = TimeSeriesDataSet.from_dataset(self.training, test_df)
        self.predict_set = TimeSeriesDataSet.from_dataset(
            self.training, self.data, predict=True
        )

    def train_dataloader(self):
        return self.training.to_dataloader(batch_size=self.batch_size, shuffle=False)

    def val_dataloader(self):
        return self.validation.to_dataloader(batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return self.test.to_dataloader(batch_size=self.batch_size, shuffle=False)

    def predict_dataloader(self):
        return self.predict_set.to_dataloader(batch_size=1, shuffle=False)


class MultiOutputLSTM(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, num_layers, horizon, n_output):
        super().__init__()
        self.n_output = n_output
        self.horizon = horizon
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.output_dim = int(self.n_output * self.horizon)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, self.output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.hidden_dim).to(
            self.device
        )
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.hidden_dim).to(
            self.device
        )
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])

        return out

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x["encoder_cont"])

        y_pred = y_pred.unsqueeze(-1).view(-1, self.horizon, self.n_output)
        y_pred = [y_pred[:, :, i] for i in range(self.n_output)]

        loss = [F.mse_loss(y_pred[i], y[0][i]) for i in range(self.n_output)]

        loss = torch.mean(torch.stack(loss))

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x["encoder_cont"])

        y_pred = y_pred.unsqueeze(-1).view(-1, self.horizon, self.n_output)
        y_pred = [y_pred[:, :, i] for i in range(self.n_output)]

        loss = [F.mse_loss(y_pred[i], y[0][i]) for i in range(self.n_output)]

        loss = torch.mean(torch.stack(loss))

        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x["encoder_cont"])

        y_pred = y_pred.unsqueeze(-1).view(-1, self.horizon, self.n_output)
        y_pred = [y_pred[:, :, i] for i in range(self.n_output)]

        loss = [F.mse_loss(y_pred[i], y[0][i]) for i in range(self.n_output)]

        loss = torch.mean(torch.stack(loss))

        self.log("test_loss", loss)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch

        y_pred = self(x["encoder_cont"])

        y_pred = y_pred.unsqueeze(-1).view(-1, self.horizon, self.n_output)
        y_pred = [y_pred[:, :, i] for i in range(self.n_output)]

        return y_pred

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


model = MultiOutputLSTM(
    input_dim=n_vars, hidden_dim=32, num_layers=2, horizon=HORIZON, n_output=len(TARGET)
)

datamodule = MultivariateSeriesDataModule(
    data=mvtseries,
    n_lags=N_LAGS,
    horizon=HORIZON,
    test_size=0.3,
    batch_size=32,
    target_variables=TARGET,
)

early_stop_callback = EarlyStopping(
    monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min"
)

trainer = pl.Trainer(max_epochs=30, callbacks=[early_stop_callback])
trainer.fit(model, datamodule)

trainer.test(model, datamodule.test_dataloader())

forecasts = trainer.predict(model=model, datamodule=datamodule)
