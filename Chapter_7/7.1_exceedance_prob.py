from pprint import pprint

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pytorch_forecasting import TimeSeriesDataSet
import lightning.pytorch as pl

mvtseries = pd.read_csv('assets/daily_multivariate_timeseries.csv',
                        parse_dates=['datetime'],
                        index_col='datetime')

# mvtseries['Incoming Solar'].diff().plot()
mvtseries['target'] = (mvtseries['Incoming Solar'].diff() < -2000).astype(int)


class ExceedanceDataModule(pl.LightningDataModule):
    def __init__(self,
                 data: pd.DataFrame,
                 test_size: float = 0.2,
                 batch_size: int = 1):
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
        self.data['target'] = (self.data['Incoming Solar'].diff() < -2000).astype(int)

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
            time_varying_unknown_reals=self.var_names,
            scalers={k: StandardScaler()
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

    def predict_dataloader(self):
        return self.predict_set.to_dataloader(batch_size=1, shuffle=False)


datamodule = ExceedanceDataModule(data=mvtseries)

datamodule.setup()

x, y = next(iter(datamodule.train_dataloader()))

pprint(x)
pprint(y[0])

#


df = mvtseries['Incoming Solar'].diff().reset_index()

plt.rcParams['figure.figsize'] = [12, 6]
sns.set_theme(style='darkgrid')

sns.lineplot(x='datetime',
             y='Incoming Solar',
             data=df)
plt.axhline(y=-2000, color='r', linestyle='-')
plt.axhline(y=2000, color='r', linestyle='-')
plt.show()
plt.ylabel('Solar Radiation 1st differences')
plt.xlabel('')
plt.title('Solar radiation daily variations')
plt.show()

plt.savefig('assets/time_series_plot_diff.png')



