import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

series = pd.read_csv(
    "assets/datasets/time_series_solar.csv",
    parse_dates=["Datetime"],
    index_col="Datetime",
)['Incoming Solar']

# fit the ARIMA model
# the order tuple (p, d, q) represents the order of the AR, I, and MA parts, respectively
model = ARIMA(series, order=(1, 1, 1), freq="H")
model_fit = model.fit()

# make prediction
yhat = model_fit.predict(start=0, end=5, typ="levels")
print(yhat)

# --- Auto-arima

print("\n\n------------------\n\nAuto-arima\n\n")
from pmdarima import auto_arima

model = auto_arima(
    series, trace=True, error_action="ignore", suppress_warnings=True
)
model_fit = model.fit(series)

forecast = model_fit.predict(n_periods=6)
print(forecast)
