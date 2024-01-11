import pandas as pd
from sktime.datasets import load_italy_power_demand



# pip install 'sktime[dl]'
# pip install keras-self-attention


X_train, y_train = \
    load_italy_power_demand(split="train", return_type="numpy3D")

X_test, y_test = load_italy_power_demand(split="test",
                                         return_type="numpy3D")

# FCN
from sktime.classification.deep_learning.fcn import FCNClassifier

fcn = FCNClassifier(n_epochs=200,
                    loss='categorical_crossentropy',
                    activation='sigmoid',
                    batch_size=4)

fcn.fit(X_train, y_train)

fcn_pred = fcn.predict(X_test)

# CNN
from sktime.classification.deep_learning.cnn import CNNClassifier

cnn = CNNClassifier(n_epochs=200,
                    loss='categorical_crossentropy',
                    activation='sigmoid',
                    kernel_size=7,
                    batch_size=4)

cnn.fit(X_train, y_train)

cnn_pred = cnn.predict(X_test)

# Inception
from sktime.classification.deep_learning import InceptionTimeClassifier

inception = InceptionTimeClassifier(n_epochs=200,
                                    loss='categorical_crossentropy',
                                    use_residual=True,
                                    use_bottleneck=True,
                                    batch_size=4)

inception.fit(X_train, y_train)

inception_pred = inception.predict(X_test)

# tapNet

from sktime.classification.deep_learning.tapnet import TapNetClassifier

tapnet = TapNetClassifier(n_epochs=200,
                          loss='categorical_crossentropy',
                          batch_size=4)

tapnet.fit(X_train, y_train)

tapnet_pred = tapnet.predict(X_test)

# tapNet

from sktime.classification.deep_learning.lstmfcn import LSTMFCNClassifier

lstmfcn = LSTMFCNClassifier(n_epochs=200,
                            attention=True,
                            batch_size=4)

lstmfcn.fit(X_train, y_train)

lstmfcn_pred = lstmfcn.predict(X_test)

from sklearn.metrics import accuracy_score

perf = {
    'FCN': accuracy_score(y_test, fcn_pred),
    'CNN': accuracy_score(y_test, cnn_pred),
    'InceptionTime': accuracy_score(y_test, inception_pred),
    'TapNet': accuracy_score(y_test, tapnet_pred),
    'LSTMFCN': accuracy_score(y_test, lstmfcn_pred),
}

perf = pd.Series(perf)
perf.plot.barh()
