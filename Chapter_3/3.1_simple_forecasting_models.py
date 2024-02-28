import pandas as pd

series = pd.read_csv(
    "assets/datasets/time_series_solar.csv",
    parse_dates=["Datetime"],
    index_col="Datetime",
)['Incoming Solar']

# Naive forecast
series.shift(1)

# Seasonal naive forecast
m = 12  # Replace 'm' with the length of your season
series.shift(m)

# Mean forecast
series.expanding().mean()
