import pandas as pd
import numpy as np
import argparse

from sklearn.datasets import make_classification

from sklearn.linear_model import LogisticRegression

import joblib
import os


def get_data_train():
    df_train = pd.read_parquet(os.path.join(os.environ['SM_CHANNEL_TRAIN'], "train.parquet")).sample(frac=1)
    X = df_train.drop("y", axis=1).values
    y = df_train.y.values
    return X, y


def get_data_test():
    df_test = pd.read_parquet(os.path.join(os.environ['SM_CHANNEL_TEST'], "test.parquet")).sample(frac=1)
    X = df_test.drop("y", axis=1).values
    y = df_test.y.values
    return X, y


def get_model():
    return LogisticRegression()
    
    
def main():
    parser = argparse.ArgumentParser()

    # Sagemaker specific arguments. Defaults are set in the environment variables.
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    args = parser.parse_args()
    
    print(os.environ)
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=2, n_redundant=2, n_repeated=0, n_classes=2)
    data = np.concatenate([X, y[:, np.newaxis]], axis=1)
    data = pd.DataFrame(data, columns=["X_" + str(i) for i in range(20)] + ["y"])

    train = data.sample(frac=0.8).copy()
    test = data.drop(train.index).copy()
    
    # X, y = get_data_train()
    X = train.drop("y", axis=1).values
    y = train.y.values
    
    model = get_model()
    model.fit(X, y)
    
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))

        
if __name__=="__main__":
    main()
