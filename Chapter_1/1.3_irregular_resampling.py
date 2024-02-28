import pandas as pd
import numpy as np
import seaborn as sns

n_sales = 1000
start = pd.Timestamp('2023-01-01 09:00')
end = pd.Timestamp('2023-04-01')
n_days = (end - start).days + 1

irregular_series = pd.to_timedelta(np.random.rand(n_sales) * n_days,
                                   unit='D') + start

# print(pd.Series(irregular_series.sort_values()[:6]).reset_index().to_latex())

series_sales = pd.Series(0, index=irregular_series).resample('D').count()
series_sales.name = 'Sales'
series_sales.index.name = 'Date'

sns.lineplot(x='Date',
             y='Sales',
             data=series_sales.reset_index()). \
    set(title='Daily sales count')
