import pandas as pd

data = pd.read_csv('assets/datasets/time_series_smf1.csv',
                   parse_dates=['datetime'],
                   index_col='datetime')

# print(data.head().to_latex(escape=False,header=['\\rotatebox{90}{' + c + '}' for c in data.columns]))

class LogTransformation:

    @staticmethod
    def transform(x):
        xt = np.sign(x) * np.log(np.abs(x) + 1)

        return xt

    @staticmethod
    def inverse_transform(xt):
        x = np.sign(xt) * (np.exp(np.abs(xt)) - 1)

        return x


data_log = LogTransformation.transform(data)

mv_plot = data_log.tail(1000).plot(figsize=(15, 8),
                                   title='Multivariate time series',
                                   xlabel='',
                                   ylabel='Value')
# mv_plot.legend(['A simple line'])
mv_plot.legend(fancybox=True, framealpha=1)
mv_plot.figure.savefig('assets/mts_plot.png')
