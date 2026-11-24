
"""
AgriSmart - Crop Recommendation Model Training
Trains and compares multiple machine learning algorithms.
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# 1. Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

# 2. Load training data
X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze("columns")

# 3. Define machine learning algorithms
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000))
    ]),

    "Decision Tree": Pipeline([
        ("classifier", DecisionTreeClassifier(random_state=42))
    ]),

    "Random Forest": Pipeline([
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ))
    ]),

    "Support Vector Machine": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC())
    ])
}


def train_models():
    """Train all selected models and save them."""

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    print("\nAgriSmart - Model Training")
    print("=" * 45)

    for model_name, model in models.items():

        print(f"\nTraining: {model_name}")

        model.fit(X_train, y_train)

        filename = (
            model_name.lower()
            .replace(" ", "_")
            + ".joblib"
        )

        model_path = MODEL_DIR / filename

        joblib.dump(model, model_path)

        print(f"Model saved: {model_path.name}")

    print("\nAll models trained and saved successfully!")


if __name__ == "__main__":
    train_models()
