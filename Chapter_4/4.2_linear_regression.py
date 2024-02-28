import numpy as np
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn

N_LAGS = 7
HORIZON = 1
BATCH_SIZE = 10

mvtseries = pd.read_csv(
    "assets/daily_multivariate_timeseries.csv",
    parse_dates=["datetime"],
    index_col="datetime",
)

num_vars = mvtseries.shape[1] + 1


def create_training_set(
        data: pd.DataFrame,
        n_lags: int,
        horizon: int,
        test_size: float = 0.2,
        batch_size: int = 16):
    data["target"] = data["Incoming Solar"]
    data["time_index"] = np.arange(len(data))
    data["group_id"] = 0  # Assuming a single group for simplicity

    time_indices = data["time_index"].values

    train_indices, _ = train_test_split(
        time_indices,
        test_size=test_size,
        shuffle=False)

    train_indices, _ = train_test_split(train_indices,
                                        test_size=0.1,
                                        shuffle=False)

    train_df = data.loc[data["time_index"].isin(train_indices)]
    train_df_mod = train_df.copy()

    target_scaler = StandardScaler()
    target_scaler.fit(train_df_mod[["target"]])
    train_df_mod["target"] = target_scaler.transform(train_df_mod[["target"]])
    train_df_mod = train_df_mod.drop("Incoming Solar", axis=1)

    feature_names = [
        col for col in data.columns
        if col != "target" and col != "Incoming Solar"
    ]

    training_dataset = TimeSeriesDataSet(
        train_df_mod,
        time_idx="time_index",
        target="target",
        group_ids=["group_id"],
        max_encoder_length=n_lags,
        max_prediction_length=horizon,
        time_varying_unknown_reals=feature_names,
        scalers={name: StandardScaler()
                 for name in feature_names},
    )

    loader = training_dataset.to_dataloader(batch_size=batch_size,
                                            shuffle=False)

    return loader


class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, X):
        X = X.view(X.size(0), -1)
        return self.linear(X)


data_loader = create_training_set(
    data=mvtseries,
    n_lags=N_LAGS,
    horizon=HORIZON,
    batch_size=BATCH_SIZE,
    test_size=0.3
)

model = LinearRegressionModel(N_LAGS * num_vars, HORIZON)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 3) Training loop
num_epochs = 10
for epoch in range(num_epochs):

    for batch in data_loader:
        x, y = batch

        X = x["encoder_cont"].squeeze(-1)
        y_pred = model(X)
        y_pred = y_pred.squeeze(1)

        y_actual = y[0].squeeze(1)

        loss = criterion(y_pred, y_actual)

        loss.backward()
        optimizer.step()

        optimizer.zero_grad()

    print(f"epoch: {epoch + 1}, loss = {loss.item():.4f}")
