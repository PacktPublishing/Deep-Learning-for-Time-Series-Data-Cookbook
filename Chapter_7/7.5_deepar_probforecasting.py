from neuralforecast.models import DeepAR
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import DistributionLoss, HuberMQLoss
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")

dataset = pd.read_csv(
    "assets/daily_multivariate_timeseries.csv",
    parse_dates=["datetime"],
)

time_column = "datetime"
series_columns = [col for col in dataset.columns if col != time_column]
target_column = "Incoming Solar"  # Specify the solar column for differencing

# Prepare a long format dataframe where each row represents a single observation for a given series at a given time
long_format_dataset = pd.DataFrame()

for series in series_columns:
    temp_df = pd.DataFrame()
    temp_df["ds"] = dataset[time_column]
    temp_df["unique_id"] = series
    # Apply differencing only to the solar column
    if series == target_column:
        temp_df["y"] = dataset[series].diff()
        # Drop the first row where the difference is NaN
        temp_df = temp_df.dropna().reset_index(drop=True)
    else:
        temp_df["y"] = dataset[series]

    long_format_dataset = pd.concat([long_format_dataset, temp_df], axis=0)

# Reset index after concatenation
long_format_dataset = long_format_dataset.reset_index(drop=True)

# Splitting the dataset into train and test sets
# Ensure the split keeps the time series structure intact
split_date = long_format_dataset["ds"].max() - pd.Timedelta(days=14)
train = long_format_dataset[long_format_dataset["ds"] <= split_date]
test = long_format_dataset[long_format_dataset["ds"] > split_date]

# model

nf = NeuralForecast(
    models=[
        DeepAR(
            h=14,
            input_size=21,  # Adjust input size to include features
            lstm_n_layers=1,
            trajectory_samples=100,
            loss=DistributionLoss(
                distribution="Normal", level=[80, 90], return_params=False
            ),
            learning_rate=0.005,
            max_steps=500,
            val_check_steps=10,
            early_stop_patience_steps=-1,
            scaler_type="minmax",
            enable_progress_bar=True,
        ),
    ],
    freq="D",
)


nf.fit(df=train, val_size=14)
Y_hat_df = nf.predict()

Y_hat_df.reset_index(inplace=True)
Y_hat_solar = Y_hat_df[Y_hat_df["unique_id"] == target_column]
test_solar = test[test["unique_id"] == target_column]

# Ensure alignment between Y_hat_solar and test_solar based on 'ds'
Y_hat_solar = Y_hat_solar.sort_values("ds").reset_index(drop=True)
test_solar = test_solar.sort_values("ds").reset_index(drop=True)

plot_df_solar = pd.concat(
    [
        pd.concat(
            (
                train[train["unique_id"] == target_column].tail(35),
                test[test["unique_id"] == target_column],
            )
        ),
        Y_hat_solar,
    ]
).reset_index(drop=True)


plt.figure(figsize=(10, 6))
plt.plot(plot_df_solar["ds"], plot_df_solar["y"], c="black", label="True Solar")
plt.plot(
    plot_df_solar["ds"], plot_df_solar["DeepAR"], c="purple", label="Predicted Mean"
)
plt.plot(
    plot_df_solar["ds"],
    plot_df_solar["DeepAR-median"],
    c="blue",
    label="Predicted Median",
)
plt.fill_between(
    x=plot_df_solar["ds"][-14:],
    y1=plot_df_solar["DeepAR-lo-90"][-14:].values,
    y2=plot_df_solar["DeepAR-hi-90"][-14:].values,
    alpha=0.4,
    label="Confidence Interval 90%",
)
plt.fill_between(
    x=plot_df_solar["ds"][-14:],
    y1=plot_df_solar["DeepAR-lo-80"][-14:].values,
    y2=plot_df_solar["DeepAR-hi-80"][-14:].values,
    alpha=0.2,
    label="Confidence Interval 80%",
)
plt.legend()
plt.grid()
plt.title("Incoming Solar Prediction")
plt.xlabel("Date")
plt.ylabel("Solar Power")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
