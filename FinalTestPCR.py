import pandas as pd
import numpy as np
import joblib
import json
import sys

if len(sys.argv) < 2:
    print("Usage: python FinalTestPCR.py <path_to_test_dataset.xls>")
    sys.exit(1)

test_path = sys.argv[1]
df = pd.read_excel(test_path)
df = df.replace(999, np.nan)
df = df.fillna(df.median(numeric_only=True))

with open("models/pcr_features.json") as f:
    feats = json.load(f)

model = joblib.load("models/final_pcr_model.joblib")
X = df[feats]
pred = model.predict(X)

out = pd.DataFrame({"ID": df["ID"], "PCR": pred})
out.to_csv("PCRPrediction.csv", index=False)
print("Saved PCRPrediction.csv")