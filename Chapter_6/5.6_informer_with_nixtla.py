from gluonts.dataset.repository.datasets import get_dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler

from neuralforecast.core import NeuralForecast
from neuralforecast.models import Informer

dataset = get_dataset('nn5_daily_without_missing', regenerate=False)

N_LAGS = 7
HORIZON = 7

data_list = list(dataset.train)

data_list = [pd.Series(ds['target'],
                       index=pd.date_range(start=ds['start'].to_timestamp(),
                                           freq=ds['start'].freq,
                                           periods=len(ds['target'])))
             for ds in data_list]

tseries_df = pd.concat(data_list, axis=1)
tseries_df[tseries_df.columns] = StandardScaler().fit_transform(tseries_df)
tseries_df = tseries_df.reset_index()

df = tseries_df.melt('index')
df.columns = ['ds', 'unique_id', 'y']
df['ds'] = pd.to_datetime(df['ds'])

n_time = len(df.ds.unique())
val_size = int(.2 * n_time)

model = [Informer(h=HORIZON,  # Forecasting horizon
                  input_size=N_LAGS,  # Input size
                  max_steps=100,  # Number of training iterations
                  val_check_steps=5,  # Compute validation loss every 100 steps
                  early_stop_patience_steps=3),  # Stop training if validation loss does not improve
         ]

nf = NeuralForecast(
    models=model,
    freq='D')

nf.fit(df=df, val_size=val_size)

forecasts = nf.predict()
forecasts.head()
