from pprint import pprint
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

from datetime import datetime
import pandas as pd

from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS

from adtk.visualization import plot

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

series = dataset.set_index('ds')

# series.to_csv('/Users/vcerq/Dropbox/8d2_series.csv')
plot(series['y'], anomaly=series['is_anomaly'])

horizon = 1
n_lags = 144

models = [NHITS(h=horizon,
                input_size=n_lags,
                max_steps=30,
                n_freq_downsample=[2, 1, 1],
                mlp_units=3 * [[128, 128]],
                accelerator='cpu')]

nf = NeuralForecast(models=models, freq='30T')
nf.fit(df=dataset.drop('is_anomaly', axis=1), val_size=n_lags)

insample = nf.predict_insample()
insample = insample.tail(-n_lags)

error = (insample['NHITS'] - insample['y']).abs()

preds = pd.DataFrame({
    'Error': error.values,
    'ds': dataset['ds'].tail(-n_lags),
    'is_anomaly': dataset['is_anomaly'].tail(-n_lags),
})

preds = preds.set_index('ds')
# preds.to_csv('/Users/vcerq/Dropbox/8d2_preds.csv')
plot(preds['Error'], anomaly=preds['is_anomaly'], anomaly_color="orange")
