import numpy as np
import pandas as pd
from gluonts.dataset.repository.datasets import get_dataset

dataset = get_dataset("nn5_daily_without_missing", regenerate=False)

data_list = list(dataset.train)

data_list = [
    pd.Series(
        ts["target"],
        index=pd.date_range(
            start=ts["start"].to_timestamp(),
            freq=ts["start"].freq,
            periods=len(ts["target"]),
        ),
    )
    for ts in data_list
]

tseries_df = pd.concat(data_list, axis=1)
tseries_df.columns = [f"Time Series id: {i}" for i in tseries_df.columns]

tseries_df.round(2).head()

tseries_df["time_index"] = np.arange(tseries_df.shape[0])

tseries_long = tseries_df.reset_index().melt(["time_index", "index"])
tseries_long = tseries_long.rename(columns={"variable": "group_id"})
