import matplotlib

from datasetsforecast.m3 import M3
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

import matplotlib.pyplot as plt

matplotlib.use("TkAgg")

dataset, *_ = M3.load("./assets/data", "Quarterly")
series = "Q1"
q1 = dataset.query(f'unique_id=="{series}"')

models = [AutoARIMA(season_length=4)]

sf = StatsForecast(
    df=q1,
    models=models,
    freq="Q",
    n_jobs=1,
)

forecasts = sf.forecast(h=8, level=[99], fitted=True).reset_index()

insample_forecasts = sf.forecast_fitted_values().reset_index()

anomaly_series = (insample_forecasts["y"] >= insample_forecasts["AutoARIMA-hi-99"]) | (
    insample_forecasts["y"] <= insample_forecasts["AutoARIMA-lo-99"]
)


def find_anomaly_periods(anomaly_series):
    change_points = anomaly_series.diff().fillna(0).abs()
    change_indices = change_points[change_points > 0].index
    return list(zip(change_indices[::2], change_indices[1::2]))


def plot_anomalies(ax, anomaly_periods):
    for start, end in anomaly_periods:
        ax.axvspan(start, end, color="red", alpha=0.3)


def setup_plot(ds, anomaly_periods):
    fig = plt.figure(figsize=(15, 10))

    # True values and anomalies
    ax1 = plt.subplot()
    ax1.plot(ds.index, ds["y"], label="True Values", color="blue", linewidth=2)
    plot_anomalies(ax1, anomaly_periods)
    ax1.set_title("True Values with Anomalies", fontsize=16)
    ax1.set_ylabel("True Values", fontsize=14)
    ax1.legend(loc="upper left", fontsize=12)
    ax1.grid(True, which="major", linestyle="--", linewidth="0.5", color="grey")

    # Format x-axis
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


anomaly_periods = find_anomaly_periods(anomaly_series)
setup_plot(insample_forecasts, anomaly_periods)
