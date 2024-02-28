import pandas as pd
import numpy as np
import matplotlib

matplotlib.use('TkAgg')

data = pd.read_csv('assets/datasets/time_series_smf1.csv',
                   parse_dates=['datetime'],
                   index_col='datetime')

stat_by_variable = {
    'Incoming Solar': 'sum',
    'Wind Dir': 'mean',
    'Snow Depth': 'sum',
    'Wind Speed': 'mean',
    'Dewpoint': 'mean',
    'Precipitation': 'sum',
    'Vapor Pressure': 'mean',
    'Relative Humidity': 'mean',
    'Air Temp': 'max',
}

data_daily = data.resample('D').agg(stat_by_variable)

data_daily.tail(365).plot(figsize=(15, 6))

data_logscale = np.sign(data_daily) * np.log(np.abs(data_daily) + 1)

data_logscale.plot(figsize=(15, 8))
data_logscale.head(365 * 2).plot(figsize=(15, 6))
