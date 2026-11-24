
"""
AgriSmart - Model Evaluation
Evaluates trained models using unseen test data.
"""

from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns


# 1. Project directories
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports" / "evaluation"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

# 2. Load test data
X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze("columns")

# 3. Identify trained models
model_files = sorted(MODEL_DIR.glob("*.joblib"))

if not model_files:
    raise FileNotFoundError("No trained models found in models/")

results = []


def evaluate_models():

    print("\nAgriSmart - Model Evaluation")
    print("=" * 55)

    for model_file in model_files:

        model_name = model_file.stem.replace("_", " ").title()

        model = joblib.load(model_file)

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        precision = precision_score(
            y_test, predictions, average="macro", zero_division=0
        )
        recall = recall_score(
            y_test, predictions, average="macro", zero_division=0
        )
        f1 = f1_score(
            y_test, predictions, average="macro", zero_division=0
        )

        results.append({
            "Model": model_name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })

        print(f"\nModel: {model_name}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Macro Precision: {precision:.4f}")
        print(f"Macro Recall: {recall:.4f}")
        print(f"Macro F1 Score: {f1:.4f}")

        print("\nClassification Report:")
        print(classification_report(
            y_test,
            predictions,
            zero_division=0
        ))

        # Save confusion matrix for each model
        matrix = confusion_matrix(
            y_test, predictions, labels=model.classes_
        )

        plt.figure(figsize=(14, 10))
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            xticklabels=model.classes_,
            yticklabels=model.classes_
        )

        plt.title(f"Confusion Matrix - {model_name}")
        plt.xlabel("Predicted Crop")
        plt.ylabel("Actual Crop")
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()

        plt.savefig(
            REPORT_DIR / f"{model_file.stem}_confusion_matrix.png"
        )
        plt.close()

    # 4. Save comparison table
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by="F1 Score", ascending=False
    )

    results_df.to_csv(
        REPORT_DIR / "model_comparison.csv",
        index=False
    )

    print("\nFINAL MODEL COMPARISON")
    print("=" * 55)
    print(results_df.to_string(index=False))

    print("\nEvaluation reports saved in reports/evaluation/")


if __name__ == "__main__":
    evaluate_models()
