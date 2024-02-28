# needs pip install -U "ray[data,train,tune,serve]"

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_forecasting import TimeSeriesDataSet

# from pytorch_lightning import LightningModule, Trainer
import lightning.pytorch as pl
from gluonts.dataset.repository.datasets import get_dataset

from ray.train.lightning import RayTrainReportCallback
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.torch import TorchTrainer

dataset = get_dataset("nn5_daily_without_missing", regenerate=False)

N_LAGS = 7
HORIZON = 3


class LogTransformation:

    @staticmethod
    def transform(x):
        xt = np.sign(x) * np.log(np.abs(x) + 1)

        return xt

    @staticmethod
    def inverse_transform(xt):
        x = np.sign(xt) * (np.exp(np.abs(xt)) - 1)

        return x


class LocalScaler:

    def __init__(self):
        self.scalers = {}

    def fit(self, df: pd.DataFrame):
        df = df.copy()
        df_g = df.groupby("group_id")
        for g, df_ in df_g:
            scl = StandardScaler()
            scl.fit(df_[["value"]])

            self.scalers[g] = scl

    def transform(self, df: pd.DataFrame):
        df = df.copy()
        df["value"] = LogTransformation.transform(df["value"])

        df_g = df.groupby("group_id")
        transf_df_l = []
        for g, df_ in df_g:
            df_[["value"]] = self.scalers[g].transform(df_[["value"]])

            transf_df_l.append(df_)

        transf_df = pd.concat(transf_df_l)
        transf_df = transf_df.sort_index()

        return transf_df

    def inverse_transform(self, df: pd.DataFrame, col_name=None):
        df = df.copy()
        if col_name is None:
            col_name = "value"

        df_g = df.groupby("group_id")
        itransf_df_l = []
        for g, df_ in df_g:
            df_[[col_name]] = self.scalers[g].inverse_transform(df_[[col_name]])

            itransf_df_l.append(df_)

        itransf_df = pd.concat(itransf_df_l)
        itransf_df = itransf_df.sort_index()
        itransf_df[col_name] = LogTransformation.inverse_transform(itransf_df[col_name])

        return itransf_df


class GlobalDataModule(pl.LightningDataModule):
    def __init__(
        self, data, n_lags: int, horizon: int, test_size: float, batch_size: int
    ):
        super().__init__()

        self.data = data
        self.batch_size = batch_size
        self.test_size = test_size
        self.n_lags = n_lags
        self.horizon = horizon

        self.training = None
        self.validation = None
        self.test = None
        self.predict_set = None

        self.target_scaler = LocalScaler()

    def setup(self, stage=None):
        data_list = list(self.data.train)

        data_list = [
            pd.Series(
                ts["target"],
                index=pd.date_range(
                    start=ts["start"].to_timestamp(),
                    freq=ts["start"].freq,
                    periods=len(ts["target"]),
                ),
            )
            for ts in data_list
        ]

        tseries_df = pd.concat(data_list, axis=1)
        tseries_df["time_index"] = np.arange(tseries_df.shape[0])

        ts_df = tseries_df.melt("time_index")
        ts_df = ts_df.rename(columns={"variable": "group_id"})

        unique_times = ts_df["time_index"].sort_values().unique()

        tr_ind, ts_ind = train_test_split(
            unique_times, test_size=self.test_size, shuffle=False
        )

        tr_ind, vl_ind = train_test_split(tr_ind, test_size=0.1, shuffle=False)

        training_df = ts_df.loc[ts_df["time_index"].isin(tr_ind), :]
        validation_df = ts_df.loc[ts_df["time_index"].isin(vl_ind), :]
        test_df = ts_df.loc[ts_df["time_index"].isin(ts_ind), :]

        self.target_scaler.fit(training_df)

        training_df = self.target_scaler.transform(training_df)
        validation_df = self.target_scaler.transform(validation_df)
        test_df = self.target_scaler.transform(test_df)

        self.training = TimeSeriesDataSet(
            data=training_df,
            time_idx="time_index",
            target="value",
            group_ids=["group_id"],
            max_encoder_length=self.n_lags,
            max_prediction_length=self.horizon,
            target_normalizer=None,
            time_varying_unknown_reals=["value"],
        )

        self.validation = TimeSeriesDataSet.from_dataset(self.training, validation_df)
        self.test = TimeSeriesDataSet.from_dataset(self.training, test_df)
        self.predict_set = TimeSeriesDataSet.from_dataset(
            self.training, ts_df, predict=True
        )

    def train_dataloader(self):
        return self.training.to_dataloader(batch_size=self.batch_size, shuffle=False)

    def val_dataloader(self):
        return self.validation.to_dataloader(batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return self.test.to_dataloader(batch_size=self.batch_size, shuffle=False)

    def predict_dataloader(self):
        return self.predict_set.to_dataloader(batch_size=1, shuffle=False)


class GlobalLSTM(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

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

        loss = F.mse_loss(y_pred, y[0])
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_pred = self(x["encoder_cont"])

        loss = F.mse_loss(y_pred, y[0])
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch

        y_pred = self(x["encoder_cont"])

        loss = F.mse_loss(y_pred, y[0])
        self.log("test_loss", loss)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch

        y_pred = self(x["encoder_cont"])

        return y_pred

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)


# Define a configuration for hyperparameter search space
search_space = {
    "hidden_dim": tune.choice([8, 16, 32]),
    "num_layers": tune.choice([1, 2]),
}


def train_tune(config_hyper):
    hidden_dim = config_hyper["hidden_dim"]
    num_layers = config_hyper["num_layers"]

    model = GlobalLSTM(
        input_dim=1, hidden_dim=hidden_dim, output_dim=HORIZON, num_layers=num_layers
    )

    data_module = GlobalDataModule(
        dataset, n_lags=N_LAGS, horizon=HORIZON, batch_size=128, test_size=0.3
    )
    trainer = pl.Trainer(callbacks=[RayTrainReportCallback()])

    trainer.fit(model, data_module)


scaling_config = ScalingConfig(
    num_workers=1, use_gpu=False, resources_per_worker={"CPU": 1, "GPU": 0}
)

run_config = RunConfig(
    checkpoint_config=CheckpointConfig(
        num_to_keep=1,
        checkpoint_score_attribute="val_loss",
        checkpoint_score_order="min",
    ),
)

ray_trainer = TorchTrainer(
    train_tune,
    scaling_config=scaling_config,
    run_config=run_config,
)

scheduler = ASHAScheduler(max_t=30, grace_period=1, reduction_factor=2)

tuner = tune.Tuner(
    ray_trainer,
    param_space={"train_loop_config": search_space},
    tune_config=tune.TuneConfig(
        metric="val_loss",
        mode="min",
        num_samples=2,
        scheduler=scheduler,
    ),
)

results = tuner.fit()

best_model_conf = results.get_best_result(metric="val_loss", mode="min")
