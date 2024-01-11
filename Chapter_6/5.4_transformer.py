from gluonts.dataset.repository.datasets import get_dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from neuralforecast.core import NeuralForecast
from neuralforecast.models import VanillaTransformer

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

model = [
    VanillaTransformer(
        h=HORIZON,  # Forecasting horizon
        input_size=N_LAGS,  # Input size
        max_steps=100,  # Number of training iterations
        val_check_steps=5,  # Compute validation loss every 100 steps
        early_stop_patience_steps=3,
    ),  # Stop training if validation loss does not improve
]

nf = NeuralForecast(models=model, freq="D")

Y_df = df[df["unique_id"] == 0]


# Leave out the last HORIZON points for hold-out testing
Y_train_df = Y_df.iloc[:-2*HORIZON]  # leaving HORIZON for validation and another HORIZON for testing
Y_val_df = Y_df.iloc[-2*HORIZON:-HORIZON]  # Use this for validation
Y_test_df = Y_df.iloc[-HORIZON:]  # The true values for the test set

training_df = pd.concat([Y_train_df, Y_val_df])

# Fit the model on the training data without the hold-out test set
nf.fit(df=training_df, val_size=HORIZON)

forecasts = nf.predict()

# Filter the data for the first time series
Y_df = df[df["unique_id"] == 0]
Y_hat_df = forecasts[forecasts.index == 0].reset_index()

# Merge the true values and the forecasts to make plotting easier
Y_hat_df = Y_test_df.merge(Y_hat_df, how="outer", on=["unique_id", "ds"])
plot_df = pd.concat([Y_train_df, Y_val_df, Y_hat_df]).set_index("ds")
plot_df = plot_df.iloc[-150:]

fig, ax = plt.subplots(1, 1, figsize=(20, 7))
plot_df[["y", "VanillaTransformer"]].plot(ax=ax, linewidth=2)

ax.set_title("First Time Series Forecast", fontsize=22)
ax.set_ylabel("Value", fontsize=20)
ax.set_xlabel("Timestamp [t]", fontsize=20)
ax.legend(prop={"size": 15})
ax.grid()

plt.show()
