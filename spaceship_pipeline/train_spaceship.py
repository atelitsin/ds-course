from pathlib import Path

import joblib
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestClassifier

train = pd.read_csv("prepared/train.csv")
params = yaml.safe_load(Path("params.yaml").read_text())["train"]

model = RandomForestClassifier(
    n_estimators=params["n_estimators"], random_state=42
)
model.fit(train.drop(columns="Transported"), train["Transported"])
joblib.dump(model, "spaceship_model.joblib")