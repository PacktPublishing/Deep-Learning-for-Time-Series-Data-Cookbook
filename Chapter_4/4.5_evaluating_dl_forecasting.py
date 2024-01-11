import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if len(data.shape) == 1 else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [("var%d(t-%d)" % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [("var%d(t)" % (j + 1)) for j in range(n_vars)]
        else:
            names += [("var%d(t+%d)" % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


df = pd.read_csv(
    "../assets/datasets/time_series_solar.csv",
    parse_dates=["Datetime"],
    index_col="Datetime",
)

# Resample the data to daily frequency
df = df.resample('D').sum()

values = df["Incoming Solar"].values

data = series_to_supervised(values, 3)
print(data)

# ---

scaler = MinMaxScaler(feature_range=(-1, 1))
train, test = train_test_split(data, test_size=0.2, shuffle=False)
train = scaler.fit_transform(train)
test = scaler.transform(test)

X_train, y_train = train[:, :-1], train[:, -1]
X_test, y_test = test[:, :-1], test[:, -1]

X_train = torch.from_numpy(X_train).type(torch.Tensor)
X_test = torch.from_numpy(X_test).type(torch.Tensor)
y_train = torch.from_numpy(y_train).type(torch.Tensor).view(-1)
y_test = torch.from_numpy(y_test).type(torch.Tensor).view(-1)


class FeedForwardNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(FeedForwardNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.activation = nn.Tanh()

    def forward(self, x):
        out = self.activation(self.fc1(x))
        out = self.fc2(out)
        return out


input_dim = X_train.shape[1]
hidden_dim = 32
model = FeedForwardNN(input_dim, hidden_dim)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

epochs = 200

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    out = model(X_train).reshape(-1,)
    loss = loss_fn(out, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, Loss: {loss.item()}")


model.eval()
y_pred = model(X_test).reshape(-1,)
test_loss = loss_fn(y_pred, y_test)
print(f"Test Loss: {test_loss.item()}")

# ------------------------- Evaluation

from sklearn.metrics import mean_squared_error
from sktime.performance_metrics.forecasting import MeanAbsoluteScaledError, MeanAbsolutePercentageError
import numpy as np


y_pred = model(X_test).detach().numpy()
y_true = y_test.detach().numpy()
y_train = y_train.detach().numpy()

rmse_sklearn = np.sqrt(mean_squared_error(y_true, y_pred))
print(f"RMSE: {rmse_sklearn}")

mape_sktime = MeanAbsolutePercentageError(symmetric=False)
mape = mape_sktime(y_true, y_pred)
print(f"MAPE: {mape}")

smape_sktime = MeanAbsolutePercentageError(symmetric=True)
smape = smape_sktime(y_true, y_pred)
print(f"SMAPE: {smape}")

mase_sktime = MeanAbsoluteScaledError()
mase = mase_sktime(y_true, y_pred, y_train=y_train)
print(f"MASE: {mase}")
