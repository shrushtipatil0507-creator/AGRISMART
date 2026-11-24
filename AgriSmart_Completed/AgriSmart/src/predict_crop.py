import math
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "crop_recommendation_model.joblib"
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]


def load_crop_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Crop model not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


def predict_crop(values):
    if values is None or len(values) != len(FEATURES):
        raise ValueError("Exactly 7 input values are required.")
    try:
        numeric_values = [float(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise ValueError("All input values must be numeric.") from exc
    if not all(math.isfinite(value) for value in numeric_values):
        raise ValueError("Input values must be finite numbers.")

    input_data = pd.DataFrame([numeric_values], columns=FEATURES)
    model = load_crop_model()
    return str(model.predict(input_data)[0])
