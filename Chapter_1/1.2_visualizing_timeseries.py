import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

data = pd.read_csv('assets/datasets/time_series_solar.csv',
                   parse_dates=['Datetime'],
                   index_col='Datetime')

series = data['Incoming Solar']

series.plot(figsize=(12, 6), title='Solar radiation time series')

series_df = series.reset_index()

plt.rcParams['figure.figsize'] = [12, 6]
sns.set_theme(style='darkgrid')

sns.lineplot(x='Datetime',
             y='Incoming Solar',
             data=series_df)

plt.ylabel('Solar Radiation')
plt.xlabel('')
plt.title('Solar radiation time series')
plt.show()

plt.savefig('assets/time_series_plot.png')
