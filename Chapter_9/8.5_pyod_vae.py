# https://pyod.readthedocs.io/en/latest/pyod.models.html#pyod.models.vae.VAE


from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec

from sklearn.preprocessing import StandardScaler
from pyod.models.vae import VAE
from adtk.visualization import plot
from tensorflow.keras.losses import mean_squared_error

N_LAGS = 144

# READING DATASET
# https://github.com/numenta/NAB
dataset = pd.read_csv('assets/datasets/taxi/taxi_data.csv')
labels = pd.read_csv('assets/datasets/taxi/taxi_labels.csv')
dataset['ds'] = pd.Series([datetime.fromtimestamp(x) for x in dataset['timestamp']])
dataset = dataset.drop('timestamp', axis=1)
dataset['unique_id'] = 'NYT'
dataset = dataset.rename(columns={'value': 'y'})

is_anomaly = []
for i, r in labels.iterrows():
    dt_start = datetime.fromtimestamp(r.start)
    dt_end = datetime.fromtimestamp(r.end)
    anomaly_in_period = [dt_start <= x <= dt_end for x in dataset['ds']]

    is_anomaly.append(anomaly_in_period)

dataset['is_anomaly'] = pd.DataFrame(is_anomaly).any(axis=0).astype(int)
dataset['ds'] = pd.to_datetime(dataset['ds'])


# plot(series['y'], anomaly=series['is_anomaly'])

# TRANSFORMING THE DATASET
series = dataset['y']

input_data = []
for i in range(N_LAGS, series.shape[0]):
    input_data.append(series.iloc[i - N_LAGS:i].values)

input_data = np.array(input_data)
input_data_n = StandardScaler().fit_transform(input_data)
input_data_n = pd.DataFrame(input_data_n)

# MODELING

model = VAE(encoder_neurons=[144, 4],
            decoder_neurons=[4, 144],
            latent_dim=2,  # You can change the latent dimensionality
            hidden_activation='relu',
            output_activation='sigmoid',
            loss=mean_squared_error,
            optimizer='adam',
            epochs=20,
            batch_size=32,
            dropout_rate=0.2,
            l2_regularizer=0.1,
            validation_size=0.1,
            preprocessing=True,
            verbose=1)

model.fit(input_data_n)

anomaly_scores = model.decision_scores_

plt.hist(anomaly_scores, bins='auto')
plt.title("Histogram for Model Anomaly Scores")
plt.show()

# CHECKING FOR ANOMALIES

predictions = model.predict(input_data_n)

probs = model.predict_proba(input_data_n)[:, 1]
probabilities = pd.Series(probs, index=series.tail(len(probs)).index)

ds = dataset.tail(-144).copy()
ds['Predicted Probability'] = probabilities
ds = ds.set_index('ds')


def find_anomaly_periods(anomaly_series):
    """
    Identify the start and end of each anomaly period.

    Parameters:
    anomaly_series (pd.Series): A series indicating where anomalies occur.

    Returns:
    list of tuples: Start and end indices of each anomaly period.
    """
    change_points = anomaly_series.diff().fillna(0).abs()
    change_indices = change_points[change_points > 0].index
    return list(zip(change_indices[::2], change_indices[1::2]))


def plot_anomalies(ax, anomaly_periods):
    """
    Highlight anomalies on a matplotlib axis.

    Parameters:
    ax (matplotlib.axes.Axes): The matplotlib axis to plot on.
    anomaly_periods (list of tuples): Anomaly periods to highlight.
    """
    for start, end in anomaly_periods:
        ax.axvspan(start, end, color='red', alpha=0.3)


def setup_plot(ds, anomaly_periods):
    """
    Setup the plot with two subplots.

    Parameters:
    ds (pd.DataFrame): The dataset containing the data to plot.
    anomaly_periods (list of tuples): Anomaly periods to highlight.
    """
    fig = plt.figure(figsize=(15, 10))
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1])

    # True values and anomalies
    ax1 = plt.subplot(gs[0])
    ax1.plot(ds.index, ds['y'], label='True Values', color='blue', linewidth=2)
    plot_anomalies(ax1, anomaly_periods)
    ax1.set_title('True Values with Anomalies', fontsize=16)
    ax1.set_ylabel('True Values', fontsize=14)
    ax1.legend(loc='upper left', fontsize=12)
    ax1.grid(True, which='major', linestyle='--', linewidth='0.5', color='grey')

    # Predicted probability of anomaly
    ax2 = plt.subplot(gs[1], sharex=ax1)
    ax2.plot(ds.index, ds['Predicted Probability'], label='Predicted Probability of Anomaly', color='green', linewidth=2)
    plot_anomalies(ax2, anomaly_periods)
    ax2.set_title('Predicted Probability of Anomaly', fontsize=16)
    ax2.set_ylabel('Probability', fontsize=14)
    ax2.set_xlabel('Date', fontsize=14)
    ax2.legend(loc='upper left', fontsize=12)
    ax2.grid(True, which='major', linestyle='--', linewidth='0.5', color='grey')

    # Format x-axis
    plt.setp(ax1.get_xticklabels(), visible=False)
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax2.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax2.xaxis.get_major_locator()))
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

anomaly_periods = find_anomaly_periods(ds['is_anomaly'])
setup_plot(ds, anomaly_periods)