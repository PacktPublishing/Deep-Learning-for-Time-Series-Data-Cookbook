import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sktime.transformations.series.date import DateTimeFeatures
from sklearn.preprocessing import OneHotEncoder
from sktime.transformations.series.fourier import FourierFeatures
import matplotlib

matplotlib.use('TkAgg')

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


series = pd.read_csv(
    "assets/datasets/time_series_solar.csv",
    parse_dates=["Datetime"],
    index_col="Datetime",
)['Incoming Solar']

# Resample the data to daily frequency
series = series.resample('D').sum()

# train test split
train, test = train_test_split(series, test_size=0.2, shuffle=False)

scaler = MinMaxScaler(feature_range=(-1, 1))

train_norm = scaler.fit_transform(train.values.reshape(-1, 1)).flatten()
train_norm = pd.Series(train_norm, index=train.index)

test_norm = scaler.transform(test.values.reshape(-1, 1)).flatten()
test_norm = pd.Series(test_norm, index=test.index)

train_df = series_to_supervised(train_norm, 3)
test_df = series_to_supervised(test_norm, 3)

# ---

X_train, y_train = train_df.values[:, :-1], train_df.values[:, -1]
X_test, y_test = test_df.values[:, :-1], test_df.values[:, -1]

### Seasonal dummies


date_features = DateTimeFeatures(ts_freq='D',
                                 keep_original_columns=False,
                                 feature_scope='efficient')

train_dates = date_features.fit_transform(train_df.iloc[:, -1])
train_dates = train_dates[['month_of_year', 'day_of_week']]

encoder = OneHotEncoder(drop='first', sparse_output=False)
encoded_feats = encoder.fit_transform(train_dates)

train_dummies = pd.DataFrame(encoded_feats,
                             columns=encoder.get_feature_names_out(),
                             dtype=int)

train_dummies.index = train_dates.index

# on test data

test_dates = date_features.transform(test_df.iloc[:, -1])
test_dates = test_dates[['month_of_year', 'day_of_week']]

test_encoded_feats = encoder.transform(test_dates)

test_dummies = pd.DataFrame(test_encoded_feats,
                            columns=encoder.get_feature_names_out(),
                            dtype=int)

# Same for Fourier series


fourier = FourierFeatures(sp_list=[365.25],
                          fourier_terms_list=[2],
                          keep_original_columns=False)

train_fourier = fourier.fit_transform(train_df.iloc[:, -1])
test_fourier = fourier.transform(test_df.iloc[:, -1])

# train_fourier.head(366).plot()

### appending
X_train = np.hstack([X_train, train_dummies, train_fourier])
X_test = np.hstack([X_test, test_dummies, test_fourier])

# Modeling as before

X_train = torch.from_numpy(X_train).type(torch.Tensor)
X_test = torch.from_numpy(X_test).type(torch.Tensor)
y_train = torch.from_numpy(y_train).type(torch.Tensor).view(-1)
y_test = torch.from_numpy(y_test).type(torch.Tensor).view(-1)

X_train = X_train.view([X_train.shape[0], X_train.shape[1], 1])
X_test = X_test.view([X_test.shape[0], X_test.shape[1], 1])


class LSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).requires_grad_()
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out


model = LSTM(input_dim=1, hidden_dim=32, output_dim=1, num_layers=2)

loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

epochs = 200

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()

    out = model(X_train).reshape(-1, )
    loss = loss_fn(out, y_train)
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch: {epoch}, Loss: {loss.item()}")

model.eval()
y_pred = model(X_test).reshape(-1, )

y_pred_np = y_pred.detach().numpy().reshape(-1, 1)
y_pred_orig = scaler.inverse_transform(y_pred_np).flatten()
