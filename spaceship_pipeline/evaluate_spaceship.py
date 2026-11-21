import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

validation = pd.read_csv("prepared/validation.csv")
X = validation.drop(columns="Transported")
y = validation["Transported"]

model = joblib.load("spaceship_model.joblib")
predicted = model.predict(X)
positive_class_index = list(model.classes_).index(True)
probability = model.predict_proba(X)[:, positive_class_index]

metrics = {
    "accuracy": accuracy_score(y, predicted),
    "roc_auc": roc_auc_score(y, probability),
}
Path("metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")

fpr, tpr, _ = roc_curve(y, probability)
pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv("roc.csv", index=False)
pd.DataFrame({"actual": y, "predicted": predicted}).to_csv(
    "confusion.csv", index=False
)