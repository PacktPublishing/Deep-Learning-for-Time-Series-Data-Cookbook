import matplotlib.pyplot as plt

import pandas as pd
from sklearn.preprocessing import StandardScaler
from gluonts.dataset.repository.datasets import get_dataset

from neuralforecast.core import NeuralForecast
from neuralforecast.models import Informer, VanillaTransformer
from neuralforecast.losses.numpy import mae

dataset = get_dataset("nn5_daily_without_missing", regenerate=False)

N_LAGS = 7
HORIZON = 7

data_list = list(dataset.train)

data_list = [
    pd.Series(
        ds["target"],
        index=pd.date_range(
            start=ds["start"].to_timestamp(),
            freq=ds["start"].freq,
            periods=len(ds["target"]),
        ),
    )
    for ds in data_list
]

tseries_df = pd.concat(data_list, axis=1)
tseries_df[tseries_df.columns] = StandardScaler().fit_transform(tseries_df)
tseries_df = tseries_df.reset_index()

df = tseries_df.melt("index")
df.columns = ["ds", "unique_id", "y"]
df["ds"] = pd.to_datetime(df["ds"])

n_time = len(df.ds.unique())
val_size = int(0.1 * n_time)
test_size = int(0.1 * n_time)

models = [
    Informer(
        h=HORIZON,
        input_size=N_LAGS,
        max_steps=1000,
        val_check_steps=10,
        early_stop_patience_steps=15,
    ),
    VanillaTransformer(
        h=HORIZON,
        input_size=N_LAGS,
        max_steps=1000,
        val_check_steps=10,
        early_stop_patience_steps=15,
    ),
]

nf = NeuralForecast(models=models, freq="D")

cv = nf.cross_validation(df=df, val_size=val_size, test_size=test_size, n_windows=None)

cv.head()

Y_plot = cv[cv["unique_id"] == 2]
cutoffs = cv["cutoff"].unique()[::HORIZON]
Y_plot = Y_plot[cv["cutoff"].isin(cutoffs)]

plt.figure(figsize=(20, 5))
plt.plot(Y_plot["ds"], Y_plot["y"], label="True")
plt.plot(Y_plot["ds"], Y_plot["Informer"], label="Informer")
plt.plot(Y_plot["ds"], Y_plot["VanillaTransformer"], label="VanillaTransformer")
plt.xlabel("Date")
plt.ylabel("Value")
plt.grid()
plt.legend()
plt.show()

mae_informer = mae(cv["y"], cv["Informer"])
mae_transformer = mae(cv["y"], cv["VanillaTransformer"])
