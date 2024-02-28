import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch import nn
import torch
import torch.nn.functional as F
from pytorch_forecasting import TimeSeriesDataSet
import lightning.pytorch as pl

N_LAGS = 7
HORIZON = 1

mvtseries = pd.read_csv(
    "assets/daily_multivariate_timeseries.csv",
    parse_dates=["datetime"],
    index_col="datetime",
)

n_vars = mvtseries.shape[1] - 1


class MultivariateSeriesDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data: pd.DataFrame,
        n_lags: int,
        horizon: int,
        test_size: float = 0.2,
        batch_size: int = 16,
    ):
        super().__init__()
        self.data = data
        self.feature_names = [col for col in data.columns if col != "Incoming Solar"]
        self.batch_size = batch_size
        self.test_size = test_size
        self.n_lags = n_lags
        self.horizon = horizon
        self.target_scaler = MinMaxScaler()
        self.training = None
        self.validation = None
        self.test = None
        self.predict_set = None
        self.setup()

    def preprocess_data(self):
        self.data["target"] = self.data["Incoming Solar"]
        self.data["time_index"] = np.arange(len(self.data))
        self.data["group_id"] = (
            0  # Assuming a single group for simplicity; adjust if needed
        )

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
        scaled_values = self.target_scaler.transform(df.loc[indices, ["target"]])
        df.loc[indices, "target"] = scaled_values

    def setup(self, stage=None):
        self.preprocess_data()
        train_indices, val_indices, test_indices = self.split_data()

        train_df = self.data.loc[self.data["time_index"].isin(train_indices)]
        val_df = self.data.loc[self.data["time_index"].isin(val_indices)]
        test_df = self.data.loc[self.data["time_index"].isin(test_indices)]

        # Scale the target variable
        self.target_scaler.fit(train_df[["target"]])
        self.scale_target(train_df, train_df.index)
        self.scale_target(val_df, val_df.index)
        self.scale_target(test_df, test_df.index)

        train_df = train_df.drop("Incoming Solar", axis=1)
        val_df = val_df.drop("Incoming Solar", axis=1)
        test_df = test_df.drop("Incoming Solar", axis=1)

        # Setup datasets
        self.training = TimeSeriesDataSet(
            train_df,
            time_idx="time_index",
            target="target",
            group_ids=["group_id"],
            max_encoder_length=self.n_lags,
            max_prediction_length=self.horizon,
            time_varying_unknown_reals=self.feature_names,
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


class MultivariateLSTM(pl.LightningModule):
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
        y_pred = y_pred.squeeze(1)
        loss = F.mse_loss(y_pred, y[0])
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x["encoder_cont"])

        y_pred = y_pred.squeeze(1)

        loss = F.mse_loss(y_pred, y[0])
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x["encoder_cont"])
        y_pred = y_pred.squeeze(1)
        loss = F.mse_loss(y_pred, y[0])
        self.log("test_loss", loss)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch

        y_pred = self(x["encoder_cont"])
        y_pred = y_pred.squeeze(1)
        # todo revert transform

        return y_pred

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.005)


datamodule = MultivariateSeriesDataModule(
    data=mvtseries, n_lags=N_LAGS, horizon=HORIZON, test_size=0.3, batch_size=32
)

model = MultivariateLSTM(input_dim=n_vars, hidden_dim=32, num_layers=1, output_dim=1)

trainer = pl.Trainer(max_epochs=30)
trainer.fit(model, datamodule)

trainer.test(model, datamodule.test_dataloader())
forecasts = trainer.predict(model=model, datamodule=datamodule)

# preds = trainer.predict(model, dataloaders=datamodule.train_dataloader())

dl = datamodule.training.to_dataloader(batch_size=1, shuffle=False)
preds = trainer.predict(model, dataloaders=dl)
preds = torch.concat(preds).numpy()
pd.Series(preds).plot()
