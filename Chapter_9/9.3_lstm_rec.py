from datetime import datetime
import matplotlib

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch import nn
import torch
import torch.nn.functional as F
from pytorch_forecasting import TimeSeriesDataSet
from lightning.pytorch.callbacks import EarlyStopping
import lightning.pytorch as pl

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

N_LAGS = 192
N_VARIABLES = 1

dataset = pd.read_csv("assets/datasets/taxi/taxi_data.csv")
labels = pd.read_csv("assets/datasets/taxi/taxi_labels.csv")
dataset["ds"] = pd.Series([datetime.fromtimestamp(x) for x in dataset["timestamp"]])
dataset = dataset.drop("timestamp", axis=1)
dataset["unique_id"] = "NYT"
dataset = dataset.rename(columns={"value": "y"})

is_anomaly = []
for i, r in labels.iterrows():
    dt_start = datetime.fromtimestamp(r.start)
    dt_end = datetime.fromtimestamp(r.end)
    anomaly_in_period = [dt_start <= x <= dt_end for x in dataset["ds"]]

    is_anomaly.append(anomaly_in_period)

dataset["is_anomaly"] = pd.DataFrame(is_anomaly).any(axis=0).astype(int)
dataset["ds"] = pd.to_datetime(dataset["ds"])

series = dataset.set_index("ds")


class TaxiDataModule(pl.LightningDataModule):
    def __init__(self, data: pd.DataFrame, n_lags: int, batch_size: int):
        super().__init__()

        self.data = data
        self.batch_size = batch_size
        self.n_lags = n_lags

        self.train_df = None
        self.test_df = None
        self.training = None
        self.validation = None
        self.predict_set = None

    def setup(self, stage=None):
        self.data["timestep"] = np.arange(self.data.shape[0])

        unique_times = self.data["timestep"].sort_values().unique()

        tr_ind, ts_ind = train_test_split(unique_times, test_size=0.6, shuffle=False)

        tr_ind, vl_ind = train_test_split(tr_ind, test_size=0.1, shuffle=False)

        self.train_df = self.data.loc[self.data["timestep"].isin(tr_ind), :]
        self.test_df = self.data.loc[self.data["timestep"].isin(ts_ind), :]
        validation_df = self.data.loc[self.data["timestep"].isin(vl_ind), :]

        self.training = TimeSeriesDataSet(
            data=self.train_df,
            time_idx="timestep",
            target="y",
            group_ids=["unique_id"],
            max_encoder_length=self.n_lags,
            max_prediction_length=1,
            time_varying_unknown_reals=["y"],
        )

        self.validation = TimeSeriesDataSet.from_dataset(self.training, validation_df)
        self.test = TimeSeriesDataSet.from_dataset(self.training, self.test_df)
        self.predict_set = TimeSeriesDataSet.from_dataset(
            self.training, self.data, predict=True
        )

    def train_dataloader(self):
        return self.training.to_dataloader(batch_size=self.batch_size, shuffle=False)

    def val_dataloader(self):
        return self.validation.to_dataloader(batch_size=self.batch_size, shuffle=False)

    def predict_dataloader(self):
        return self.predict_set.to_dataloader(batch_size=1, shuffle=False)


class Encoder(nn.Module):
    def __init__(self, context_len, n_variables, embedding_dim=64):
        super(Encoder, self).__init__()
        self.context_len, self.n_variables = context_len, n_variables
        self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim
        self.lstm1 = nn.LSTM(
            input_size=self.n_variables,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.lstm2 = nn.LSTM(
            input_size=self.hidden_dim,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True,
        )

    def forward(self, x):
        batch_size = x.shape[0]
        x, (_, _) = self.lstm1(x)
        x, (hidden_n, _) = self.lstm2(x)
        return hidden_n.reshape((batch_size, self.embedding_dim))


class Decoder(nn.Module):
    def __init__(self, context_len, n_variables=1, input_dim=64):
        super(Decoder, self).__init__()
        self.context_len, self.input_dim = context_len, input_dim
        self.hidden_dim, self.n_variables = 2 * input_dim, n_variables
        self.lstm1 = nn.LSTM(
            input_size=input_dim, hidden_size=input_dim, num_layers=1, batch_first=True
        )
        self.lstm2 = nn.LSTM(
            input_size=input_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.output_layer = nn.Linear(self.hidden_dim, self.n_variables)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.repeat(self.context_len, self.n_variables)
        x = x.reshape((batch_size, self.context_len, self.input_dim))
        x, (hidden_n, cell_n) = self.lstm1(x)
        x, (hidden_n, cell_n) = self.lstm2(x)
        x = x.reshape((batch_size, self.context_len, self.hidden_dim))

        return self.output_layer(x)


class AutoencoderLSTM(pl.LightningModule):
    def __init__(self, context_len, n_variables, embedding_dim):
        super().__init__()
        self.encoder = Encoder(context_len, n_variables, embedding_dim)
        self.decoder = Decoder(context_len, n_variables, embedding_dim)

    def forward(self, x):
        xh = self.encoder(x)
        xh = self.decoder(xh)
        return xh

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x["encoder_cont"])
        loss = F.mse_loss(y_pred, x["encoder_cont"])

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x["encoder_cont"])
        loss = F.mse_loss(y_pred, x["encoder_cont"])

        self.log("val_loss", loss)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x["encoder_cont"])
        loss = F.mse_loss(y_pred, x["encoder_cont"])
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)


datamodule = TaxiDataModule(
    data=dataset.drop("is_anomaly", axis=1), n_lags=N_LAGS, batch_size=64
)

model = AutoencoderLSTM(n_variables=1, context_len=N_LAGS, embedding_dim=2)

early_stop_callback = EarlyStopping(
    monitor="val_loss", min_delta=1e-4, patience=5, verbose=False, mode="min"
)

trainer = pl.Trainer(max_epochs=3, accelerator="cpu", callbacks=[early_stop_callback])
trainer.fit(model, datamodule)

dl = datamodule.test.to_dataloader(batch_size=1, shuffle=False)
preds = trainer.predict(model, dataloaders=dl)
preds = pd.Series(np.array([x.numpy() for x in preds]))

test = datamodule.test_df.head(len(preds))
is_anomaly = dataset.loc[test.index, :]["is_anomaly"].values

preds.index = test["ds"].values
preds_df = preds.reset_index()
preds_df.columns = ["ds", "Error"]
preds_df["is_anomaly"] = is_anomaly
preds_df = preds_df.set_index("ds")


#


def find_anomaly_periods(anomaly_series):
    change_points = anomaly_series.diff().fillna(0).abs()
    change_indices = change_points[change_points > 0].index
    return list(zip(change_indices[::2], change_indices[1::2]))


def plot_anomalies(ax, anomaly_periods):
    for start, end in anomaly_periods:
        ax.axvspan(start, end, color="red", alpha=0.3)


def setup_plot(ds, anomaly_periods, lab="True Values"):
    fig = plt.figure(figsize=(15, 10))

    # True values and anomalies
    ax1 = plt.subplot()
    ax1.plot(ds.index, ds["y"], label=lab, color="blue", linewidth=2)
    plot_anomalies(ax1, anomaly_periods)
    ax1.set_title(f"{lab} with Anomalies", fontsize=16)
    ax1.set_ylabel(lab, fontsize=14)
    ax1.legend(loc="upper left", fontsize=12)
    ax1.grid(True, which="major", linestyle="--", linewidth="0.5", color="grey")

    # Format x-axis
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


predicted_anomaly_periods = find_anomaly_periods(preds_df["is_anomaly"])
setup_plot(preds_df.rename(columns={"Error": "y"}), predicted_anomaly_periods, "Error")
