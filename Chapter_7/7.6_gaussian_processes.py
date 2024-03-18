import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel, PeriodicKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import lightning.pytorch as pl
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt


mvtseries = pd.read_csv(
    "assets/daily_multivariate_timeseries.csv",
    parse_dates=["datetime"],
    index_col="datetime",
)


class ContinuousDataModule(pl.LightningDataModule):
    def __init__(
        self, data: pd.DataFrame, test_size: float = 0.2, batch_size: int = 16
    ):
        super().__init__()
        self.data = data
        self.feature_names = [col for col in data.columns if col != "Incoming Solar"]
        self.feature_names.insert(0, "time_index")
        self.batch_size = batch_size
        self.test_size = test_size
        self.target_scaler = StandardScaler()
        self.feature_scaler = StandardScaler()
        self.train_indices = None
        self.val_indices = None
        self.test_indices = None

    def preprocess_data(self):
        self.data["target"] = self.data["Incoming Solar"]
        self.data["time_index"] = np.arange(len(self.data))
        self.data["group_id"] = (
            0  # Assuming a single group for simplicity; adjust if needed
        )

    def split_data(self):
        time_indices = self.data["time_index"].values
        train_indices, test_indices = train_test_split(
            time_indices, test_size=self.test_size, shuffle=False
        )

        self.train_indices = train_indices
        self.test_indices = test_indices

    def setup(self, stage=None):
        self.preprocess_data()
        self.split_data()

        # Scale features
        self.feature_scaler.fit(self.data[self.feature_names])
        scaled_features = self.feature_scaler.transform(self.data[self.feature_names])

        # Scale target
        self.target_scaler.fit(self.data[["target"]])
        scaled_target = self.target_scaler.transform(self.data[["target"]])

        # Convert scaled data to PyTorch tensors
        self.train_x = torch.tensor(
            scaled_features[self.train_indices], dtype=torch.float32
        )
        self.train_y = torch.tensor(
            scaled_target[self.train_indices].squeeze(), dtype=torch.float32
        )

        self.test_x = torch.tensor(
            scaled_features[self.test_indices], dtype=torch.float32
        )
        self.test_y = torch.tensor(
            scaled_target[self.test_indices].squeeze(), dtype=torch.float32
        )

        # Store original test features for plotting
        self.original_x = torch.tensor(
            scaled_features[
                np.concatenate((self.train_indices, self.test_indices), axis=0)
            ],
            dtype=torch.float32,
        )

        self.original_y = torch.tensor(
            scaled_target[
                np.concatenate((self.train_indices, self.test_indices), axis=0)
            ],
            dtype=torch.float32,
        )

    def train_dataloader(self):
        # Returns a PyTorch DataLoader for training data
        train_dataset = torch.utils.data.TensorDataset(self.train_x, self.train_y)
        return torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

    def test_dataloader(self):
        # Returns a PyTorch DataLoader for test data
        test_dataset = torch.utils.data.TensorDataset(self.test_x, self.test_y)
        return torch.utils.data.DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False
        )


class GPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel()) + ScaleKernel(PeriodicKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


datamodule = ContinuousDataModule(data=mvtseries)
datamodule.setup()

likelihood = GaussianLikelihood()
model = GPModel(datamodule.train_x[:, 0], datamodule.train_y, likelihood)

model.train()
likelihood.train()

optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# "Loss" for GPs - the marginal log likelihood
mll = ExactMarginalLogLikelihood(likelihood, model)

training_iter = 100
for i in range(training_iter):
    optimizer.zero_grad()
    output = model(datamodule.train_x[:, 0])
    loss = -mll(output, datamodule.train_y)
    loss.backward()
    optimizer.step()

# Make predictions
model.eval()
likelihood.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    # Test points are regularly spaced along [0,1]
    # Make predictions by feeding model through likelihood
    observed_pred = likelihood(model(datamodule.original_x[:, 0]))

# Extract mean and confidence interval
mean = observed_pred.mean
lower, upper = observed_pred.confidence_region()

# Assuming the first column of test_x is the time or sequence component
time_feature = datamodule.original_x[:, 0].numpy()

plt.figure(figsize=(10, 6))

# If 'time_feature' is not already sorted, you might need to sort it along with 'mean' and confidence intervals
sorted_indices = np.argsort(time_feature)
time_feature_sorted = time_feature[sorted_indices]
mean_sorted = mean.numpy()[sorted_indices]
lower_sorted = lower.numpy()[sorted_indices]
upper_sorted = upper.numpy()[sorted_indices]

# Define the end of training data
train_end = max(datamodule.train_indices)

# Segment the data
historical_x = time_feature_sorted[: train_end + 1]
future_x = time_feature_sorted[train_end + 1 :]

historical_y = datamodule.original_y.numpy()[sorted_indices][: train_end + 1]
future_y = datamodule.original_y.numpy()[sorted_indices][train_end + 1 :]

fitted_values = mean.numpy()[: train_end + 1]
predicted_values = mean.numpy()[train_end + 1 :]

fitted_lower = lower.numpy()[: train_end + 1]
fitted_upper = upper.numpy()[: train_end + 1]

predicted_lower = lower.numpy()[train_end + 1 :]
predicted_upper = upper.numpy()[train_end + 1 :]

plt.figure(figsize=(12, 7))

# Plot historical data
plt.plot(historical_x, historical_y, "k*", label="Historical Data")

# Plot actual future data
plt.plot(
    future_x,
    future_y,
    color=(0, 0.2, 0),
    marker="*",
    linestyle="None",
    label="Actual Future Data",
)

# Plot fitted values and uncertainty for historical data
plt.plot(historical_x, fitted_values, "b", label="Fitted Values for Historical Data")
plt.fill_between(historical_x, fitted_lower, fitted_upper, color="b", alpha=0.2)

# Plot predicted values and uncertainty for future data
plt.plot(future_x, predicted_values, "g", label="Predicted Future Values")
plt.fill_between(future_x, predicted_lower, predicted_upper, color="g", alpha=0.2)

plt.title("Gaussian Process Forecast with Historical and Predictive Data")
plt.xlabel("Time Feature")
plt.ylabel("Target Variable")
plt.legend()
plt.show()
