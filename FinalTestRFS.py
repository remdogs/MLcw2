import pandas as pd
import numpy as np
import joblib
import json
import sys

if len(sys.argv) < 2:
    print("Usage: python FinalTestRFS.py <path_to_test_dataset.xls>")
    sys.exit(1)

test_path = sys.argv[1]
df = pd.read_excel(test_path)
df = df.replace(999, np.nan)
df = df.fillna(df.median(numeric_only=True))

with open("models/rfs_features.json") as f:
    feats = json.load(f)

model = joblib.load("models/final_rfs_model.joblib")
X = df[feats]
pred = model.predict(X)

out = pd.DataFrame({"ID": df["ID"], "RFS": pred})
out.to_csv("RFSPrediction.csv", index=False)
print("Saved RFSPrediction.csv")