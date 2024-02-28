import pandas as pd
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

import matplotlib

matplotlib.use('TkAgg')

data = pd.read_csv('assets/datasets/time_series_solar.csv',
                   parse_dates=['Datetime'],
                   index_col='Datetime')

series = data['Incoming Solar']

series_daily = series.resample('D').sum()

acf_scores = acf(x=series_daily, nlags=365)
pacf_scores = pacf(x=series_daily, nlags=365)

acf_plot_ = plot_acf(series_daily, lags=365)
pacf_plot_ = plot_pacf(series_daily, lags=365)
plot = plot_acf(series, lags=48)
plot = plot_acf(series, lags=365 * 24 * 2)
plot = plot_pacf(series, lags=72)
