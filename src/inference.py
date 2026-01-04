import joblib
import pandas as pd
from config import MODEL_PATH

model = joblib.load(MODEL_PATH)

def predict(data: dict):
    df = pd.DataFrame([data])
    prob = model.predict_proba(df)[0][1]
    pred = int(prob >= 0.5)
    return {"prediction": pred, "probability": float(prob)}
