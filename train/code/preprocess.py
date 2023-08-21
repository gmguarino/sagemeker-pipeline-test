
import pandas as pd
import numpy as np
import os

from sklearn.datasets import make_classification


if __name__=="__main__":

    X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=2, n_repeated=0, n_classes=2)
    data = np.concatenate([X, y[:, np.newaxis]], axis=1)
    data = pd.DataFrame(data, columns=["X_" + str(i) for i in range(20)] + ["y"])

    train = data.sample(frac=0.8).copy()
    test = data.drop(train.index).copy()

    output_train_path = os.path.join("/opt/ml/processing/train", "train.parquet")
    print("Save train data to {}".format(output_train_path))
    train.to_parquet(output_train_path)

    output_test_path = os.path.join("/opt/ml/processing/test", "test.parquet")
    print("Save test data to {}".format(output_test_path))
    test.to_parquet(output_test_path)
