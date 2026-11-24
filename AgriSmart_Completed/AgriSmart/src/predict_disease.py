from functools import lru_cache
import json
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "plant_disease" / "agris_mobilenetv2.keras"
CLASS_PATH = ROOT / "models" / "plant_disease" / "class_names.json"
IMG_SIZE = (224, 224)
CONFIDENCE_THRESHOLD = 0.60


@lru_cache(maxsize=1)
def load_disease_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required for plant disease detection. Install requirements.txt.") from exc
    return tf.keras.models.load_model(MODEL_PATH)


@lru_cache(maxsize=1)
def load_class_names():
    if not CLASS_PATH.exists():
        raise FileNotFoundError(f"Class names file not found: {CLASS_PATH}")
    with open(CLASS_PATH, "r", encoding="utf-8") as file:
        names = json.load(file)
    if not isinstance(names, list) or not names:
        raise ValueError("Disease class names file must contain a non-empty list.")
    return names


def predict_disease(image):
    """Predict one of the model's 10 supported tomato leaf categories."""
    if image is None:
        raise ValueError("Please provide a leaf image.")
    try:
        if not isinstance(image, Image.Image):
            image = Image.open(image)
        image = image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("The uploaded file is not a valid image.") from exc

    image_array = np.asarray(image.resize(IMG_SIZE), dtype=np.float32)
    try:
        from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
    except ImportError as exc:
        raise RuntimeError("TensorFlow is required for plant disease detection. Install requirements.txt.") from exc
    image_array = preprocess_input(image_array)
    image_array = np.expand_dims(image_array, axis=0)

    model = load_disease_model()
    class_names = load_class_names()
    predictions = np.asarray(model.predict(image_array, verbose=0))[0]

    if predictions.ndim != 1 or len(predictions) != len(class_names):
        raise ValueError(
            f"Model returned {len(predictions)} scores for {len(class_names)} classes."
        )
    if not np.all(np.isfinite(predictions)):
        raise ValueError("Model returned invalid prediction scores.")

    predicted_index = int(np.argmax(predictions))
    confidence = float(predictions[predicted_index])
    predicted_class = class_names[predicted_index]
    uncertain = confidence < CONFIDENCE_THRESHOLD

    return {
        "disease": predicted_class,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "is_uncertain": uncertain,
        "all_predictions": predictions.astype(float).tolist(),
        "threshold": CONFIDENCE_THRESHOLD,
    }
