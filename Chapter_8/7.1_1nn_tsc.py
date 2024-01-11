import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier

data_directory = 'assets/datasets/Car'

train = pd.read_table(f'{data_directory}/Car_TRAIN.tsv', header=None)
test = pd.read_table(f'{data_directory}/Car_TEST.tsv', header=None)

scaler = MinMaxScaler()

y_train = train.iloc[:, 0]
y_test = test.iloc[:, 0]
X_train = train.iloc[:, 1:]
X_test = test.iloc[:, 1:]
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

class_dist = y_train.value_counts(normalize=True)
class_dist.plot.bar()

classifier = KNeighborsTimeSeriesClassifier()

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)
