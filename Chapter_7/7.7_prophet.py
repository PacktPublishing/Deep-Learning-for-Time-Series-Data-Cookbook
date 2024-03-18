import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load your data
mvtseries = pd.read_csv(
    "assets/daily_multivariate_timeseries.csv",
    parse_dates=["datetime"],
)

# Preprocess the data
mvtseries["ds"] = mvtseries["datetime"]
mvtseries["y"] = mvtseries["Incoming Solar"]

plt.plot(mvtseries["y"])
plt.show()

# Split the data into train and test sets
train_data, test_data = train_test_split(mvtseries, test_size=0.2, shuffle=False)

# Initialize and fit the Prophet model
model = Prophet()
model.fit(train_data[["ds", "y"]])

# Create a dataframe for future predictions
future = model.make_future_dataframe(periods=len(test_data))
forecast = model.predict(future)

# Plot the forecast
fig = model.plot(forecast)
plt.show()

# If you want to plot the components (trend, yearly seasonality, and weekly seasonality)
fig2 = model.plot_components(forecast)
plt.show()
