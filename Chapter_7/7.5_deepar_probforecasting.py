from neuralforecast.models import DeepAR
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import DistributionLoss, HuberMQLoss
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np


matplotlib.use("TkAgg")


def load_and_prepare_data(file_path, time_column, series_column, aggregation_freq):
    """Load the time series data and prepare it for modeling."""
    dataset = pd.read_csv(file_path, parse_dates=[time_column])
    dataset.set_index(time_column, inplace=True)

    # Selecting target series and resampling to weekly frequency
    target_series = dataset[series_column].resample(aggregation_freq).mean()

    return target_series


def add_time_features(dataframe, date_column):
    """Add time-related features to the DataFrame."""
    dataframe["week_of_year"] = (
        dataframe[date_column].dt.isocalendar().week.astype(float)
    )
    dataframe["month"] = dataframe[date_column].dt.month.astype(float)

    # Fourier features for week of year
    dataframe["sin_week"] = np.sin(2 * np.pi * dataframe["week_of_year"] / 52)
    dataframe["cos_week"] = np.cos(2 * np.pi * dataframe["week_of_year"] / 52)
    dataframe["sin_2week"] = np.sin(4 * np.pi * dataframe["week_of_year"] / 52)
    dataframe["cos_2week"] = np.cos(4 * np.pi * dataframe["week_of_year"] / 52)

    # Cyclic encoding for month
    dataframe["sin_month"] = np.sin(2 * np.pi * dataframe["month"] / 12)
    dataframe["cos_month"] = np.cos(2 * np.pi * dataframe["month"] / 12)

    return dataframe


def scale_features(dataframe, feature_columns):
    """Scale features."""
    scaler = MinMaxScaler()
    dataframe[feature_columns] = scaler.fit_transform(dataframe[feature_columns])

    return dataframe, scaler


def split_data(dataframe, date_column, split_time):
    """Split the data into training and test sets."""
    train = dataframe[dataframe[date_column] <= split_time]
    test = dataframe[dataframe[date_column] > split_time]

    return train, test


FILE_PATH = "assets/daily_multivariate_timeseries.csv"
TIME_COLUMN = "datetime"
TARGET_COLUMN = "Incoming Solar"
AGGREGATION_FREQ = "W"

weekly_data = load_and_prepare_data(
    FILE_PATH, TIME_COLUMN, TARGET_COLUMN, AGGREGATION_FREQ
)
weekly_data = weekly_data.reset_index().rename(columns={TARGET_COLUMN: "y"})

# Add time-related features
weekly_data = add_time_features(weekly_data, TIME_COLUMN)

# Scale features before splitting to prevent data leakage
numerical_features = [
    "y",
    "week_of_year",
    "sin_week",
    "cos_week",
    "sin_2week",
    "cos_2week",
    "sin_month",
    "cos_month",
]
features_to_scale = ["y", "week_of_year"]
weekly_data, scaler = scale_features(weekly_data, features_to_scale)

weekly_data["ds"] = weekly_data[TIME_COLUMN]
weekly_data.drop(["datetime"], axis=1, inplace=True)
weekly_data["unique_id"] = "Incoming Solar"

SPLIT_TIME = weekly_data["ds"].max() - pd.Timedelta(weeks=52)
train, test = split_data(weekly_data, "ds", SPLIT_TIME)

nf = NeuralForecast(
    models=[
        DeepAR(
            h=52,
            input_size=52,
            lstm_n_layers=3,
            lstm_hidden_size=128,
            trajectory_samples=100,
            loss=DistributionLoss(
                distribution="Normal", level=[80, 90], return_params=False
            ),
            futr_exog_list=[
                "week_of_year",
                "sin_week",
                "cos_week",
                "sin_2week",
                "cos_2week",
                "sin_month",
                "cos_month",
            ],
            learning_rate=0.001,
            max_steps=1000,
            val_check_steps=10,
            start_padding_enabled=True,
            early_stop_patience_steps=30,
            scaler_type="identity",
            enable_progress_bar=True,
        ),
    ],
    freq="W",
)


nf.fit(df=train, val_size=52)
Y_hat_df = nf.predict(
    futr_df=test[
        [
            "ds",
            "unique_id",
            "week_of_year",
            "sin_week",
            "cos_week",
            "sin_2week",
            "cos_2week",
            "sin_month",
            "cos_month",
        ]
    ]
)

Y_hat_df.reset_index(inplace=True)
Y_hat_solar = Y_hat_df[Y_hat_df["unique_id"] == TARGET_COLUMN]
test_solar = test[test["unique_id"] == TARGET_COLUMN]

# Ensure alignment between Y_hat_solar and test_solar based on 'ds'
Y_hat_solar = Y_hat_solar.sort_values("ds").reset_index(drop=True)
test_solar = test_solar.sort_values("ds").reset_index(drop=True)

plot_df_solar = pd.concat(
    [
        pd.concat(
            (
                train[train["unique_id"] == TARGET_COLUMN].tail(700),
                test[test["unique_id"] == TARGET_COLUMN],
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
    x=plot_df_solar["ds"],
    y1=plot_df_solar["DeepAR-lo-90"].values,
    y2=plot_df_solar["DeepAR-hi-90"].values,
    alpha=0.4,
    label="Confidence Interval 90%",
)
plt.fill_between(
    x=plot_df_solar["ds"],
    y1=plot_df_solar["DeepAR-lo-80"].values,
    y2=plot_df_solar["DeepAR-hi-80"].values,
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
