import seaborn as sns
import numpy as np
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_lightning.callbacks import Callback
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pytorch_lightning import LightningDataModule
from gluonts.dataset.common import ListDataset
from gluonts.torch.model.deepar import DeepAREstimator
from gluonts.evaluation import make_evaluation_predictions
import matplotlib.pyplot as plt

# Load and preprocess the data
mvtseries = pd.read_csv(
    "assets/daily_multivariate_timeseries.csv",
    parse_dates=["datetime"],
    index_col="datetime",
)


class ContinuousDataModule(LightningDataModule):
    def __init__(
        self, data: pd.DataFrame, test_size: float = 0.2, batch_size: int = 16
    ):
        super().__init__()
        self.data = data
        self.feature_names = [col for col in data.columns if col != "Incoming Solar"]
        self.batch_size = batch_size
        self.test_size = test_size
        self.target_scaler = StandardScaler()
        self.train_indices = None
        self.val_indices = None
        self.test_indices = None

    def preprocess_data(self):
        self.data["target"] = self.data["Incoming Solar"]
        self.data["time_index"] = np.arange(len(self.data))
        self.data[
            "group_id"
        ] = 0  # Assuming a single group for simplicity; adjust if needed

    def split_data(self):
        time_indices = self.data["time_index"].values
        train_indices, test_indices = train_test_split(
            time_indices, test_size=self.test_size, shuffle=False
        )
        train_indices, val_indices = train_test_split(
            train_indices, test_size=0.1, shuffle=False
        )

        # Store indices as attributes
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.test_indices = test_indices

        return train_indices, val_indices, test_indices

    def scale_target(self, df, indices):
        scaled_values = self.target_scaler.transform(df.loc[indices, ["target"]])
        df.loc[indices, "target"] = scaled_values

    def setup(self, stage=None):
        self.preprocess_data()
        train_indices, val_indices, test_indices = self.split_data()

        train_df = self.data.loc[self.data["time_index"].isin(train_indices)]
        val_df = self.data.loc[self.data["time_index"].isin(val_indices)]
        test_df = self.data.loc[self.data["time_index"].isin(test_indices)]

        # Scale the target variable
        self.target_scaler.fit(train_df[["target"]])
        self.scale_target(train_df, train_df.index)
        self.scale_target(val_df, val_df.index)
        self.scale_target(test_df, test_df.index)

        # Setup datasets
        self.training = TimeSeriesDataSet(
            train_df,
            time_idx="time_index",
            target="target",
            group_ids=["group_id"],
            max_encoder_length=14,
            max_prediction_length=7,
            time_varying_unknown_reals=self.feature_names,
            scalers={name: StandardScaler() for name in self.feature_names},
        )
        self.validation = TimeSeriesDataSet.from_dataset(self.training, val_df)
        self.test = TimeSeriesDataSet.from_dataset(self.training, test_df)
        self.predict_set = TimeSeriesDataSet.from_dataset(
            self.training, self.data, predict=True
        )

    def train_dataloader(self):
        return self.training.to_dataloader(batch_size=self.batch_size, shuffle=False)

    def val_dataloader(self):
        return self.validation.to_dataloader(batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        return self.test.to_dataloader(batch_size=self.batch_size, shuffle=False)

    def prepare_deepar_dataset(self):
        # Ensure data is preprocessed and split
        self.setup()

        # Extract additional features
        additional_features = self.data[self.feature_names].values

        # Prepare train and test datasets
        train_df = self.data.loc[self.data["time_index"].isin(self.train_indices)]
        test_df = self.data.loc[self.data["time_index"].isin(self.test_indices)]

        train_ds = ListDataset(
            [{'target': train_df['target'].values,
              'start': train_df.index[0],
              'feat_dynamic_real': additional_features[self.train_indices]}],
            freq='D'
        )
        test_ds = ListDataset(
            [{'target': test_df['target'].values,
              'start': test_df.index[0],
              'feat_dynamic_real': additional_features[self.test_indices]}],
            freq='D'
        )

        return train_ds, test_ds


datamodule = ContinuousDataModule(data=mvtseries)
train_ds, test_ds = datamodule.prepare_deepar_dataset()

deepar_estimator = DeepAREstimator(
    prediction_length=7,
    context_length=14,
    trainer_kwargs={"max_epochs": 100},  # Adjust epochs and learning rate
    freq='D'
)

predictor = deepar_estimator.train(training_data=train_ds)

forecast_it, ts_it = make_evaluation_predictions(
    dataset=test_ds,  # Test dataset
    predictor=predictor,  # Predictor from DeepAR
    num_samples=100,  # Number of sample paths for generating the forecast
)

forecasts = list(forecast_it)
tss = list(ts_it)


def plot_forecasts(tss, forecasts, past_length=50, num_plots=1):
    for idx in range(num_plots):
        ts_entry = tss[idx]
        forecast_entry = forecasts[idx]

        plt.figure(figsize=(10, 6))

        # Convert Period to timestamp for plotting
        historical_dates = ts_entry.index.to_timestamp()[-past_length:]
        forecast_dates = forecast_entry.index.to_timestamp()

        # Plotting historical data
        plt.plot(historical_dates, ts_entry[-past_length:], label='Historical Data', color='b')

        # Plotting forecast data
        plt.plot(forecast_dates, forecast_entry.mean, label='Forecast Mean', color='g')

        # Plotting confidence intervals if available
        if hasattr(forecast_entry, 'quantile'):
            plt.fill_between(
                forecast_dates,
                forecast_entry.quantile(0.1),
                forecast_entry.quantile(0.9),
                color='g',
                alpha=0.3,
                label='80% Confidence Interval'
            )

        if hasattr(forecast_entry, 'quantile'):
            plt.fill_between(
                forecast_dates,
                forecast_entry.quantile(0.025),
                forecast_entry.quantile(0.975),
                color='g',
                alpha=0.3,
                label='95% Confidence Interval'
            )

        plt.title(f"Forecast using DeepAR")
        plt.xlabel("Date")
        plt.ylabel("Value")
        plt.legend()
        plt.grid(True)
        plt.show()

plot_forecasts(tss, forecasts, past_length=50, num_plots=1)



