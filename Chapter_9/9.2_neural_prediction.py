from datetime import datetime
import matplotlib

import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


def find_anomaly_periods(anomaly_series):
    change_points = anomaly_series.diff().fillna(0).abs()
    change_indices = change_points[change_points > 0].index
    return list(zip(change_indices[::2], change_indices[1::2]))


def plot_anomalies(ax, anomaly_periods):
    for start, end in anomaly_periods:
        ax.axvspan(start, end, color="red", alpha=0.3)


def setup_plot(ds, anomaly_periods, lab="True Values"):
    fig = plt.figure(figsize=(15, 10))

    # True values and anomalies
    ax1 = plt.subplot()
    ax1.plot(ds.index, ds["y"], label=lab, color="blue", linewidth=2)
    plot_anomalies(ax1, anomaly_periods)
    ax1.set_title(f"{lab} with marked anomalous samples", fontsize=16)
    ax1.set_ylabel(lab, fontsize=14)
    ax1.legend(loc="upper left", fontsize=12)
    ax1.grid(True, which="major", linestyle="--", linewidth="0.5", color="grey")

    # Format x-axis
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


dataset = pd.read_csv("assets/datasets/taxi/taxi_data.csv")
labels = pd.read_csv("assets/datasets/taxi/taxi_labels.csv")
dataset["ds"] = pd.Series([datetime.fromtimestamp(x) for x in dataset["timestamp"]])
dataset = dataset.drop("timestamp", axis=1)
dataset["unique_id"] = "NYT"
dataset = dataset.rename(columns={"value": "y"})

is_anomaly = []
for i, r in labels.iterrows():
    dt_start = datetime.fromtimestamp(r.start)
    dt_end = datetime.fromtimestamp(r.end)
    anomaly_in_period = [dt_start <= x <= dt_end for x in dataset["ds"]]

    is_anomaly.append(anomaly_in_period)

dataset["is_anomaly"] = pd.DataFrame(is_anomaly).any(axis=0).astype(int)
dataset["ds"] = pd.to_datetime(dataset["ds"])

series = dataset.set_index("ds")

anomaly_periods = find_anomaly_periods(series["is_anomaly"])
setup_plot(series, anomaly_periods, "Number of trips")

horizon = 1
n_lags = 144

models = [
    NHITS(
        h=horizon,
        input_size=n_lags,
        max_steps=30,
        n_freq_downsample=[2, 1, 1],
        mlp_units=3 * [[128, 128]],
        accelerator="cpu",
    )
]

nf = NeuralForecast(models=models, freq="30T")
nf.fit(df=dataset.drop("is_anomaly", axis=1), val_size=n_lags)

insample = nf.predict_insample()
insample = insample.tail(-n_lags)

error = (insample["NHITS"] - insample["y"]).abs()

preds = pd.DataFrame(
    {
        "Error": error.values,
        "ds": dataset["ds"].tail(-n_lags),
        "is_anomaly": dataset["is_anomaly"].tail(-n_lags),
    }
)

preds = preds.set_index("ds")

predicted_anomaly_periods = find_anomaly_periods(preds["is_anomaly"])
setup_plot(preds.rename(columns={"Error": "y"}), predicted_anomaly_periods, "Error")
