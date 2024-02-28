import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# calculate the correlation matrix
corr_matrix = data_daily.corr(method='pearson')

# plot the heatmap
sns.heatmap(data=corr_matrix,
            cmap=sns.diverging_palette(230, 20, as_cmap=True),
            xticklabels=data_daily.columns,
            yticklabels=data_daily.columns,
            center=0,
            square=True,
            linewidths=.5,
            cbar_kws={"shrink": .5})

plt.xticks(rotation=30)

plt.savefig('assets/corr_heatmap.png')
