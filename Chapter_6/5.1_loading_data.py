from pprint import pprint

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_lightning import LightningDataModule
from gluonts.dataset.repository.datasets import get_dataset

dataset = get_dataset('nn5_daily_without_missing', regenerate=False)

print(len(list(dataset.train)))
print(len(list(dataset.train)[0]['target']))

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

        data_list = [pd.Series(ts['target'],
                               index=pd.date_range(start=ts['start'].to_timestamp(),
                                                   freq=ts['start'].freq,
                                                   periods=len(ts['target'])))
                     for ts in data_list]

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

x, y = next(iter(datamodule.train_dataloader()))

pprint(x)
pprint(y)
