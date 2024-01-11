import numpy as np
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
import torch
from torch import nn

N_LAGS = 7
HORIZON = 1
BATCH_SIZE = 1

mvtseries = pd.read_csv('assets/daily_multivariate_timeseries.csv',
                        parse_dates=['datetime'],
                        index_col='datetime')

num_vars = mvtseries.shape[1]

# TimeSeriesDataSet
mvtseries['time_index'] = np.arange(mvtseries.shape[0])
mvtseries['group_id'] = 0

dataset = TimeSeriesDataSet(
    data=mvtseries,
    group_ids=["group_id"],
    target="Incoming Solar",
    time_idx="time_index",
    max_encoder_length=7,
    max_prediction_length=1,
    time_varying_unknown_reals=['Incoming Solar', 'Wind Dir', 'Snow Depth', 'Wind Speed', 'Dewpoint',
                                'Precipitation', 'Vapor Pressure', 'Relative Humidity', 'Air Temp'],
)

# Data Loader
data_loader = dataset.to_dataloader(batch_size=BATCH_SIZE, shuffle=False)


class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out = self.linear(x)
        return out


model = LinearRegressionModel(N_LAGS * num_vars, 1)

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# 3) Training loop
num_epochs = 100
for epoch in range(num_epochs):

    for batch in data_loader:
        x, y = batch

        lags = x['encoder_cont']
        lags = lags.view(-1)

        # Forward pass and loss
        y_predicted = model(lags)
        loss = criterion(y_predicted, y[0])

        loss.backward()
        optimizer.step()

        optimizer.zero_grad()

    print(f'epoch: {epoch + 1}, loss = {loss.item():.4f}')
