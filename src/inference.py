import joblib
import pandas as pd
from src.config import MODEL_PATH
import os


_model = None

def load_model():
    global _model
    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Train the model first."
            )
        _model = joblib.load(MODEL_PATH)
    return _model


def predict(data: dict):
    model = load_model()
    df = pd.DataFrame([data])
    prob = model.predict_proba(df)[0][1]
    pred = int(prob >= 0.5)
    return {"prediction": pred,
