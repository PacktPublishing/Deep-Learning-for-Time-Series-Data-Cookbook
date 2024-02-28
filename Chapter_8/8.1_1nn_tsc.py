import matplotlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier
from sklearn.metrics import confusion_matrix


matplotlib.use("TkAgg")

data_directory = "assets/datasets/Car"

train = pd.read_table(f"{data_directory}/Car_TRAIN.tsv", header=None)
test = pd.read_table(f"{data_directory}/Car_TEST.tsv", header=None)

scaler = MinMaxScaler()

y_train = train.iloc[:, 0]
y_test = test.iloc[:, 0]
X_train = train.iloc[:, 1:]
X_test = test.iloc[:, 1:]
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

class_dist = y_train.value_counts(normalize=True)
class_dist.plot.bar()
plt.show()

classifier = KNeighborsTimeSeriesClassifier()

classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)


def plot_res(y_test, y_pred):
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.reset_index(drop=True), label="True Labels", marker="o")
    plt.plot(y_pred, label="Predictions", marker="x", linestyle="--")
    plt.title("Predictions vs True Labels")
    plt.xlabel("Sample Index")
    plt.ylabel("Label")
    plt.legend()
    plt.show()

    # Compute the confusion matrix
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(10, 8))
    sns.set(style="whitegrid")

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=np.unique(y_test),
        yticklabels=np.unique(y_test),
    )

    plt.title("Confusion Matrix of Predictions", fontsize=16)
    plt.xlabel("Predicted Labels", fontsize=14)
    plt.ylabel("True Labels", fontsize=14)
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.show()


plot_res(y_test, y_pred)
