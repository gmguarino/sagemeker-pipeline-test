import pandas as pd
import joblib
import pathlib
import json

from sklearn.metrics import f1_score


if __name__ == "__main__":
    data = pd.read_parquet("/opt/ml/processing/test/test.parquet")

    X = data.drop("y", axis=1).values
    y = data.y.values

    model = joblib.load("/opt/ml/model/model.joblib")
    preds = model.predict(X)
    score = f1_score(y, preds)
    result_dict = {"f1_score": score}
    output_dir = "/opt/ml/processing/evaluation"
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    evaluation_path = f"{output_dir}/evaluation.json"
    with open(evaluation_path, "w") as f:
        f.write(json.dumps(result_dict))

