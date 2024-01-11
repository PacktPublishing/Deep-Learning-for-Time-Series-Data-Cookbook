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

mvtseries = pd.read_csv('assets/daily_multivariate_timeseries.csv',
                        parse_dates=['datetime'],
                        index_col='datetime')

n_vars = mvtseries.shape[1]


class ExampleDataModule(pl.LightningDataModule):
    def __init__(self,
                 data: pd.DataFrame,
                 batch_size: int):
        super().__init__()

        self.data = data
        self.batch_size = batch_size

    def setup(self, stage=None):
        pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass

    def predict_dataloader(self):
        pass


class ExampleModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.network = ...

    def forward(self, x):
        pass

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        pass

    def configure_optimizers(self):
        pass


class MultivariateSeriesDataModule(pl.LightningDataModule):
    def __init__(self,
                 data: pd.DataFrame,
                 n_lags: int,
                 horizon: int,
                 test_size: float = 0.2,
                 batch_size: int = 32):
        super().__init__()

        self.data = data
        self.var_names = self.data.columns.tolist()
        self.batch_size = batch_size
        self.test_size = test_size
        self.n_lags = n_lags
        self.horizon = horizon

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
            time_idx="time_index",
            target="Incoming Solar",
            group_ids=['group_id'],
            max_encoder_length=self.n_lags,
            max_prediction_length=self.horizon,
            time_varying_unknown_reals=self.var_names,
            scalers={k: MinMaxScaler() for k in self.var_names
                     if k != 'Incoming Solar'}
        )

        self.validation = TimeSeriesDataSet.from_dataset(self.training, validation_df)
        self.test = TimeSeriesDataSet.from_dataset(self.training, test_df)
        self.predict_set = TimeSeriesDataSet.from_dataset(self.training, self.data, predict=True)

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
            nn.Linear(8, output_size)
        )

    def forward(self, X):
        return self.net(X)


class FeedForwardNetAlternative(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.l1 = nn.Linear(input_size, 16)
        self.relu_l1 = nn.ReLU()
        self.l2 = nn.Linear(16, 8)
        self.relu_l2 = nn.ReLU()
        self.l3 = nn.Linear(8, output_size)

    def forward(self, x):
        l1_output = self.l1(x)
        l1_actf_output = self.relu_l1(l1_output)
        l2_output = self.l2(l1_actf_output)
        l2_actf_output = self.relu_l2(l2_output)
        l3_output = self.l3(l2_actf_output)

        return l3_output


class FeedForwardModel(pl.LightningModule):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.network = FeedForwardNet(
            input_size=input_dim,
            output_size=output_dim,
        )

    def forward(self, x):
        network_input = x.squeeze(-1)

        network_input = network_input.reshape(network_input.shape[0], -1)

        prediction = self.network(network_input)

        return prediction

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_pred = self(x["encoder_cont"])
        y_pred = y_pred.squeeze(1)

        y_actual = y[0].squeeze(1)

        loss = F.mse_loss(y_pred, y_actual)

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x["encoder_cont"])
        y_pred = y_pred.squeeze(1)

        y_actual = y[0].squeeze(1)

        loss = F.mse_loss(y_pred, y_actual)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch

        y_pred = self(x["encoder_cont"])
        y_pred = y_pred.squeeze(1)

        y_actual = y[0].squeeze(1)

        loss = F.mse_loss(y_pred, y_actual)
        self.log('test_loss', loss)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch

        y_pred = self(x["encoder_cont"])
        y_pred = y_pred.squeeze(1)

        return y_pred

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)


datamodule = MultivariateSeriesDataModule(data=mvtseries,
                                          n_lags=7,
                                          horizon=1,
                                          batch_size=32,
                                          test_size=0.3)

model = FeedForwardModel(input_dim=N_LAGS * n_vars, output_dim=1)

trainer = pl.Trainer(max_epochs=30)
trainer.fit(model, datamodule)

trainer.test(model=model, datamodule=datamodule)

forecasts = trainer.predict(model=model, datamodule=datamodule)
