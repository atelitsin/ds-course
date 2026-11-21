from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv("kaggle/input/spaceship-titanic/train.csv")
prepared = data.select_dtypes(include="number").fillna(0)
prepared["Transported"] = data["Transported"]

train, validation = train_test_split(
    prepared, test_size=0.2, random_state=42, stratify=prepared["Transported"]
)

Path("prepared").mkdir(exist_ok=True)
train.to_csv("prepared/train.csv", index=False)
validation.to_csv("prepared/validation.csv", index=False)