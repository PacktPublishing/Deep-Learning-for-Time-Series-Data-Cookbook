import pandas as pd
from pmdarima.arima import ndiffs, nsdiffs
import numpy as np
import statsmodels.stats.api as sms
from statsmodels.formula.api import ols
from scipy import stats

data = pd.read_csv('assets/datasets/time_series_solar.csv',
                   parse_dates=['Datetime'],
                   index_col='Datetime')

series = data['Incoming Solar']

series_daily = series.resample('D').sum()

ndiffs(x=series_daily, test='kpss')
ndiffs(x=series_daily, test='adf')

# series_changes = series_log.head(365).diff()[1:]
series_changes = series_daily.diff()[1:]
plt = series_changes.plot(title='Changes in solar radiation in consecutive days')

plt.figure.savefig('assets/uts_daily_changes.png')

nsdiffs(x=series_changes, test='ch', m=365)
nsdiffs(x=series_changes, test='ocsb', m=365)
