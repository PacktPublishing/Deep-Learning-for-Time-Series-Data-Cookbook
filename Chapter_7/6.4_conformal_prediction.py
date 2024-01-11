import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from statsforecast import StatsForecast
from statsforecast.models import ARIMA
from statsforecast.utils import ConformalIntervals

HORIZON = 7

# Load your data
dataset = pd.read_csv(
    "assets/daily_multivariate_timeseries.csv",
    parse_dates=["datetime"],
)

series = dataset[['Incoming Solar']].reset_index()
series['id'] = 'Solar'
series = series.rename(columns={'index': 'ds', 'Incoming Solar': 'y', 'id': 'unique_id'})

train, test = train_test_split(series, test_size=HORIZON)

intervals = ConformalIntervals(h=HORIZON)

models = [
    ARIMA(order=(2, 0, 2),
          season_length=365,
          prediction_intervals=intervals),
]

sf = StatsForecast(
    df=train,
    models=models,
    freq='D',
)

forecasts = sf.forecast(h=HORIZON, level=[95])
model = 'ARIMA'

trainl = train.tail(90)

fig, ax = plt.subplots(1, 1, figsize=(20, 7))
plt.plot(np.arange(0, len(trainl['y'])), trainl['y'])
plt.plot(np.arange(len(trainl['y']), len(trainl['y']) + HORIZON), forecasts[model], label=model)
plt.plot(np.arange(len(trainl['y']), len(trainl['y']) + HORIZON), forecasts[f'{model}-lo-95'], color='r', label='lo-95')
plt.plot(np.arange(len(trainl['y']), len(trainl['y']) + HORIZON), forecasts[f'{model}-hi-95'], color='r', label='hi-95')
plt.legend()
