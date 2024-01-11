from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from pyod.models.auto_encoder_torch import AutoEncoder
from adtk.visualization import plot

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

model = AutoEncoder(hidden_neurons=[144, 4, 4, 144],
                    hidden_activation='relu',
                    epochs=20,
                    batch_norm=True,
                    learning_rate=0.001,
                    batch_size=32,
                    dropout_rate=0.2)

model.fit(input_data_n)

# CHECKING ANOMALY SCORES

anomaly_scores = model.decision_scores_

plt.hist(anomaly_scores, bins='auto')
plt.title("Histogram for Model Anomaly Scores")
plt.show()

# CHECKING FOR ANOMALIES

predictions = model.predict(input_data_n)

probs = model.predict_proba(input_data_n)[:, 1]
probabilities = pd.Series(probs, index=series.tail(len(probs)).index)

ds = dataset.tail(-144)
ds['Predicted Probability'] = probabilities
ds = ds.set_index('ds')

plot(ds['Predicted Probability'], anomaly=ds['is_anomaly'])
