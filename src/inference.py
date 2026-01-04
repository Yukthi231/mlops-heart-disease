import os
import joblib
import pandas as pd
from src.config import MODEL_PATH

_model = None


def load_model():
    """
    Lazily load the ML model. Avoids loading at import time,
    which is important for CI environments where the model
    file may not yet exist.
    """
    global _model

    if _model is None:
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(
                f"Model not found at {MODEL_PATH}. Train the model first."
            )
        _model = joblib.load(MODEL_PATH)

    return _model


def predict(data: dict):
    """
    Run inference on a single JSON-like record.
    """
    model = load_model()

    df = pd.DataFrame([data])

    prob = model.predict_proba(df)[0][1]
    pred = int(prob >= 0.5)

    return {
        "prediction": pred,
        "probability": float(prob)
    }
