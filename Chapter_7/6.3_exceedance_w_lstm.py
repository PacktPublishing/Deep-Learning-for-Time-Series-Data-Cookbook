import numpy as np
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from lightning.pytorch.callbacks import EarlyStopping
import lightning.pytorch as pl
from torchmetrics import AUROC

N_LAGS = 14
HORIZON = 7

mvtseries = pd.read_csv('assets/daily_multivariate_timeseries.csv',
                        parse_dates=['datetime'],
                        index_col='datetime')


class ExceedanceDataModule(pl.LightningDataModule):
    def __init__(self, data: pd.DataFrame, test_size: float, batch_size: int):
        super().__init__()

        self.data = data
        self.var_names = self.data.columns.tolist()
        self.batch_size = batch_size
        self.test_size = test_size

        self.training = None
        self.validation = None
        self.test = None
        self.predict_set = None

    def setup(self, stage=None):
        self.data['target'] = (self.data['Incoming Solar'].diff() < -1500).astype(int)

        self.data['time_index'] = np.arange(self.data.shape[0])
        self.data['group_id'] = 0

        unique_times = self.data['time_index'].sort_values().unique()

        tr_ind, ts_ind = \
            train_test_split(unique_times,
                             test_size=self.test_size,
                             shuffle=False)

        tr_ind, vl_ind = train_test_split(tr_ind,
                                          test_size=0.1,
                                          shuffle=False)

        training_df = self.data.loc[self.data['time_index'].isin(tr_ind), :]
        validation_df = self.data.loc[self.data['time_index'].isin(vl_ind), :]
        test_df = self.data.loc[self.data['time_index'].isin(ts_ind), :]

        self.training = TimeSeriesDataSet(
            data=training_df,
            time_idx="time_index",
            target="target",
            group_ids=['group_id'],
            max_encoder_length=14,
            max_prediction_length=7,
            time_varying_unknown_reals=self.var_names + ['target'],
            scalers={k: MinMaxScaler()
                     for k in self.var_names
                     if k != 'target'}
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


class ExceedanceLSTM(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, self.hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, 1)

    def forward(self, x):
        h0 = torch.zeros(self.lstm.num_layers, x.size(0), self.hidden_dim).to(self.device)
        c0 = torch.zeros(self.lstm.num_layers, x.size(0), self.hidden_dim).to(self.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = torch.sigmoid(out)

        return out

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_bin = (y[0] > 0).any(axis=1).long().type(torch.FloatTensor)
        y_pred = self(x['encoder_cont'])

        loss = F.binary_cross_entropy(y_pred.squeeze(-1), y_bin)

        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch

        y_bin = (y[0] > 0).any(axis=1).long().type(torch.FloatTensor)
        y_pred = self(x['encoder_cont'])

        loss = F.binary_cross_entropy(y_pred.squeeze(-1), y_bin)

        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch

        y_bin = (y[0] > 0).any(axis=1).long().type(torch.FloatTensor)
        y_pred = self(x['encoder_cont'])

        loss = F.binary_cross_entropy(y_pred.squeeze(-1), y_bin)

        auroc = AUROC(task='binary')
        auc_score = auroc(y_pred, y_bin)

        self.log('test_bce', loss)
        self.log('test_auc', auc_score)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.01)




datamodule = ExceedanceDataModule(data=mvtseries,
                                  batch_size=64,
                                  test_size=0.3)

model = ExceedanceLSTM(input_dim=10, hidden_dim=32, num_layers=1)

early_stop_callback = EarlyStopping(monitor="val_loss",
                                    min_delta=1e-4,
                                    patience=10,
                                    verbose=False,
                                    mode="min")

trainer = pl.Trainer(
    max_epochs=100,
    accelerator="cpu",
    callbacks=[early_stop_callback]
)

trainer.fit(model, datamodule)
trainer.test(model, datamodule)
