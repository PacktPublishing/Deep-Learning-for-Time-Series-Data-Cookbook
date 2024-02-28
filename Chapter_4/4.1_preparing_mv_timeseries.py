import numpy as np
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet

# loading a multivariate time series
mvtseries = pd.read_csv('assets/datasets/time_series_smf1.csv',
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

mvtseries = mvtseries.resample('D').agg(stat_by_variable)
mvtseries = mvtseries.ffill()

mvtseries.to_csv('assets/daily_multivariate_timeseries.csv')

TARGET = 'Incoming Solar'
N_LAGS = 3
HORIZON = 1

input_data = []
output_data = []
for i in range(N_LAGS, mvtseries.shape[0] - HORIZON + 1):
    input_data.append(mvtseries.iloc[i - N_LAGS:i].values)
    output_data.append(mvtseries.iloc[i:(i + HORIZON)][TARGET])

input_data, output_data = np.array(input_data), np.array(output_data)

# USING TIMESERIESDATASET


mvtseries.T.head(5)

mvtseries['time_index'] = np.arange(mvtseries.shape[0])
mvtseries['group_id'] = 0

# create the dataset from the pandas dataframe
dataset = TimeSeriesDataSet(
    data=mvtseries,
    group_ids=["group_id"],
    target="Incoming Solar",
    time_idx="time_index",
    max_encoder_length=7,
    max_prediction_length=1,
    time_varying_unknown_reals=['Incoming Solar',
                                'Wind Dir',
                                'Snow Depth',
                                'Wind Speed',
                                'Dewpoint',
                                'Precipitation',
                                'Vapor Pressure',
                                'Relative Humidity',
                                'Air Temp'],
)

# convert the dataset to a dataloader
data_loader = dataset.to_dataloader(batch_size=1, shuffle=False)

x, y = next(iter(data_loader))

x['encoder_cont']
y
