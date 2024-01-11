# https://nixtla.github.io/statsforecast/docs/tutorials/anomalydetection.html
from datasetsforecast.m3 import M3
import matplotlib

print(matplotlib.__version__)
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA

dataset, *_ = M3.load('./data', 'Quarterly')
series = "Q1"
q1 = dataset.query(f'unique_id=="{series}"')
# q1.iloc[20,2]=20000

models = [AutoARIMA(season_length=4)]

sf = StatsForecast(
    df=q1,
    models=models,
    freq='Q',
    n_jobs=1,
)

forecasts = sf.forecast(h=8, level=[99], fitted=True).reset_index()

insample_forecasts = sf.forecast_fitted_values().reset_index()

is_an = (insample_forecasts['y'] >= insample_forecasts['AutoARIMA-hi-99']) | (
        insample_forecasts['y'] <= insample_forecasts['AutoARIMA-lo-99'])
anomalies = insample_forecasts.loc[is_an]
# anomalies.head()

# StatsForecast.plot(insample_forecasts, plot_random=False, plot_anomalies=True)

# StatsForecast.plot(insample_forecasts,
#                    unique_ids=['Q2'],
#                    plot_anomalies=True)
# plt.rcParams["figure.figsize"] = (14, 6)
# plot = StatsForecast.plot(insample_forecasts,
#                          unique_ids=[series],
#                          plot_anomalies=True)
# plot.show()

# plot=StatsForecast.plot(insample_forecasts, plot_random=False, plot_anomalies=True)

# ax = plot.get_axes()[0]
# ax.set_xlabel('')

# plot.savefig('q1_anomalies.png')

from adtk.visualization import plot

insample_forecasts['is_anomaly'] = is_an.astype(int)
insample_forecasts = insample_forecasts.set_index('ds')
plot(insample_forecasts['y'], anomaly=insample_forecasts['is_anomaly'], anomaly_color="red")

