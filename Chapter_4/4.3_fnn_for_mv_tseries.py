import os
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F
from torch import nn
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.models import BaseModel
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
        self.target_scaler = StandardScaler()
        self.training = None
        self.validation = None
        self.test = None
        self.predict_set = None

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
            scalers={name: StandardScaler() for name in self.feature_names},
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


class FeedForwardNet(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, output_size),
        )

    def forward(self, X):
        # Flatten the input tensor from [batch_size, N_LAGS, n_vars] to [batch_size, N_LAGS*n_vars]
        X = X.view(X.size(0), -1)
        return self.net(X)


class FeedForwardModel(BaseModel):
    def __init__(self, input_dim: int, output_dim: int):
        self.save_hyperparameters()

        super().__init__()
        self.network = FeedForwardNet(
            input_size=input_dim,
            output_size=output_dim,
        )

        self.train_loss_history = []
        self.val_loss_history = []

        self.train_loss_sum = 0.0
        self.val_loss_sum = 0.0
        self.train_batch_count = 0
        self.val_batch_count = 0

    def forward(self, x):
        network_input = x["encoder_cont"].squeeze(-1)

        prediction = self.network(network_input)
        output = self.to_network_output(prediction=prediction)

        return output

    def on_train_epoch_end(self):
        # Compute the average loss and reset counters
        if self.train_batch_count > 0:
            avg_train_loss = self.train_loss_sum / self.train_batch_count
            self.train_loss_history.append(avg_train_loss)
            self.train_loss_sum = 0.0
            self.train_batch_count = 0

    def on_validation_epoch_end(self):
        # Compute the average loss and reset counters
        if self.val_batch_count > 0:
            avg_val_loss = self.val_loss_sum / self.val_batch_count
            self.val_loss_history.append(avg_val_loss)
            self.val_loss_sum = 0.0
            self.val_batch_count = 0

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x).prediction
        y_pred = y_pred.squeeze(1)

        y_actual = y[0].squeeze(1)

        loss = F.mse_loss(y_pred, y_actual)
        self.train_loss_sum += loss.item()
        self.train_batch_count += 1

        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x).prediction
        y_pred = y_pred.squeeze(1)

        y_actual = y[0].squeeze(1)

        loss = F.mse_loss(y_pred, y_actual)
        self.val_loss_sum += loss.item()
        self.val_batch_count += 1
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch

        y_pred = self(x).prediction
        y_pred = y_pred.squeeze(1)

        y_actual = y[0].squeeze(1)

        loss = F.mse_loss(y_pred, y_actual)
        self.log("test_loss", loss)

    def predict_step(self, batch, batch_idx):
        x, y = batch

        y_pred = self(x).prediction
        y_pred = y_pred.squeeze(1)

        return y_pred

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)


datamodule = MultivariateSeriesDataModule(
    data=mvtseries, n_lags=7, horizon=1, batch_size=32, test_size=0.3
)

datamodule.setup()

model = FeedForwardModel(input_dim=N_LAGS * n_vars, output_dim=1)

trainer = pl.Trainer(max_epochs=30)
trainer.fit(model, datamodule)

trainer.test(model=model, datamodule=datamodule)

forecasts = trainer.predict(model=model, datamodule=datamodule)

# Plot the average loss per epoch
plt.figure(figsize=(10, 6))
plt.plot(model.train_loss_history, label="Average Training Loss")
plt.plot(model.val_loss_history, label="Average Validation Loss")
plt.title("Average Training and Validation Losses Per Epoch")
plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.legend()
plt.grid(True)  # Optional: Add grid for better readability

# Save the figure to the 'plots' directory
plots_dir = "./assets/plots"
os.makedirs(plots_dir, exist_ok=True)
plot_path = os.path.join(plots_dir, "average_training_validation_loss_per_epoch.png")
plt.savefig(plot_path)
plt.close()

print(f"Plot saved to {plot_path}")
