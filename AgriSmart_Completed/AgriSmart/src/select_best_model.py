
"""
AgriSmart - Final Model Selection
Selects and saves the best evaluated model.
"""

from pathlib import Path
import shutil
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports" / "evaluation"

RESULTS_PATH = REPORT_DIR / "model_comparison.csv"

SOURCE_MODEL = MODEL_DIR / "random_forest.joblib"
FINAL_MODEL = MODEL_DIR / "crop_recommendation_model.joblib"


def select_best_model():

    results = pd.read_csv(RESULTS_PATH)

    best_model = results.sort_values(
        by="F1 Score",
        ascending=False
    ).iloc[0]

    if best_model["Model"] != "Random Forest":
        raise ValueError(
            "Expected Random Forest to be the best model. "
            "Please review the evaluation results."
        )

    if not SOURCE_MODEL.exists():
        raise FileNotFoundError("Trained Random Forest model not found.")

    shutil.copy2(SOURCE_MODEL, FINAL_MODEL)

    print("\nAgriSmart - Final Model Selection")
    print("=" * 40)
    print(f"Selected model: {best_model['Model']}")
    print(f"Test accuracy: {best_model['Accuracy']:.4%}")
    print(f"Macro F1-score: {best_model['F1 Score']:.4%}")
    print(f"Saved as: {FINAL_MODEL.name}")


if __name__ == "__main__":
    select_best_model()
