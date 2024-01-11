import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from pytorch_lightning import LightningDataModule
from pytorch_forecasting import NBeats, TimeSeriesDataSet
from gluonts.dataset.repository.datasets import get_dataset
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping

dataset = get_dataset('nn5_daily_without_missing', regenerate=False)

# DATA MODULE

N_LAGS = 7
HORIZON = 7


class GlobalDataModule(LightningDataModule):
    def __init__(self,
                 data,
                 n_lags: int,
                 horizon: int,
                 test_size: float = 0.2,
                 batch_size: int = 32):
        super().__init__()

        self.data = data
        self.batch_size = batch_size
        self.test_size = test_size
        self.n_lags = n_lags
        self.horizon = horizon

        self.training = None
        self.validation = None
        self.test = None

    def setup(self, stage=None):
        data_list = list(self.data.train)

        data_list = [pd.Series(ds['target'], index=pd.date_range(start=ds['start'].to_timestamp(),
                                                                 freq=ds['start'].freq,
                                                                 periods=len(ds['target'])))
                     for ds in data_list]

        tseries_df = pd.concat(data_list, axis=1)
        tseries_df['time_index'] = np.arange(tseries_df.shape[0])

        tseries_long = tseries_df.melt('time_index')
        tseries_long = tseries_long.rename(columns={'variable': 'group_id'})

        unique_times = tseries_long['time_index'].sort_values().unique()

        train_index, test_index = train_test_split(unique_times,
                                                   test_size=self.test_size,
                                                   shuffle=False)

        train_index, validation_index = train_test_split(train_index,
                                                         test_size=0.1,
                                                         shuffle=False)

        training_df = tseries_long.loc[tseries_long['time_index'].isin(train_index), :]
        validation_df = tseries_long.loc[tseries_long['time_index'].isin(validation_index), :]
        test_df = tseries_long.loc[tseries_long['time_index'].isin(test_index), :]

        self.training = TimeSeriesDataSet(
            data=training_df,
            time_idx='time_index',
            target='value',
            group_ids=['group_id'],
            max_encoder_length=self.n_lags,
            max_prediction_length=self.horizon,
            time_varying_unknown_reals=['value'],
        )

        self.validation = TimeSeriesDataSet.from_dataset(self.training, validation_df)
        self.test = TimeSeriesDataSet.from_dataset(self.training, test_df)

    def train_dataloader(self):
        return self.training.to_dataloader(batch_size=self.batch_size, shuffle=False)

    def val_dataloader(self):
        return self.validation.to_dataloader(batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return self.test.to_dataloader(batch_size=self.batch_size, shuffle=False)

    def predict_dataloader(self):
        return self.test.to_dataloader(batch_size=self.batch_size, shuffle=False)


datamodule = GlobalDataModule(data=dataset,
                              n_lags=N_LAGS,
                              horizon=HORIZON)

datamodule.setup()

# SETTING UP MODEL

early_stop_callback = EarlyStopping(monitor="val_loss",
                                    min_delta=1e-4,
                                    patience=10,
                                    verbose=False,
                                    mode="min")

trainer = pl.Trainer(
    max_epochs=30,
    accelerator="auto",
    enable_model_summary=True,
    gradient_clip_val=0.01,
    callbacks=[early_stop_callback],
)

model = NBeats.from_dataset(
    dataset=datamodule.training,
    stack_types=['trend', 'seasonality'],
    num_blocks=[3, 3],  # The number of blocks per stack.
    num_block_layers=[4, 4],  # Number of fully connected layers
    widths=[256, 2048],
    sharing=[True],
    backcast_loss_ratio=1.0,
)

# model.size()/1e3

# FITTING

trainer.fit(
    model,
    train_dataloaders=datamodule.train_dataloader(),
    val_dataloaders=datamodule.val_dataloader(),
)

# EVALUATING

best_model_path = trainer.checkpoint_callback.best_model_path
best_model = NBeats.load_from_checkpoint(best_model_path)

actuals = torch.cat([y[0] for x, y in iter(datamodule.test.to_dataloader(batch_size=1, shuffle=False))])
predictions = best_model.predict(datamodule.test.to_dataloader(batch_size=1, shuffle=False))
# TypeError: can't convert mps:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first.
(actuals - predictions.cpu()).abs().mean()

# Forecasting
forecasts = best_model.predict(datamodule.predict_dataloader())

# INTERPRETATION

raw_predictions = best_model.predict(datamodule.val_dataloader(),
                                     mode="raw",
                                     return_x=True)

best_model.plot_interpretation(x=raw_predictions[1],
                               output=raw_predictions[0],
                               idx=0)
