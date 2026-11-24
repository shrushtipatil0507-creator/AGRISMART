
"""
AgriSmart - Explainable AI Module

Uses SHAP to explain the trained Random Forest
crop recommendation model.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt


# 1. Define project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT / "models" /
    "crop_recommendation_model.joblib"
)

TEST_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" /
    "X_test.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "reports" / "explainability"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def explain_model():

    # 2. Load the trained model and test data
    print("\nLoading AgriSmart model and test data...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError("Final trained model not found.")

    if not TEST_DATA_PATH.exists():
        raise FileNotFoundError("Test dataset not found.")

    model = joblib.load(MODEL_PATH)
    X_test = pd.read_csv(TEST_DATA_PATH)

    # 3. Create SHAP TreeExplainer
    print("Creating SHAP explainer...")

    explainer = shap.TreeExplainer(
    model.named_steps["classifier"]
)

    # 4. Calculate SHAP values
    print("Calculating SHAP values...")

    shap_result = explainer(X_test)

    values = np.asarray(shap_result.values)
    base_values = np.asarray(shap_result.base_values)

    # 5. Calculate global feature importance
    # For multiclass output, aggregate across all classes.

    if values.ndim == 3:
        global_importance = np.abs(values).mean(axis=(0, 2))
    else:
        global_importance = np.abs(values).mean(axis=0)

    importance_df = pd.DataFrame({
        "Feature": X_test.columns,
        "Mean_Absolute_SHAP_Value": global_importance
    })

    importance_df = importance_df.sort_values(
        by="Mean_Absolute_SHAP_Value",
        ascending=True
    )

    importance_df.to_csv(
        OUTPUT_DIR / "feature_importance.csv",
        index=False
    )

    # 6. Generate global feature importance chart
    plt.figure(figsize=(10, 6))

    plt.barh(
        importance_df["Feature"],
        importance_df["Mean_Absolute_SHAP_Value"]
    )

    plt.title("AgriSmart - Global SHAP Feature Importance")
    plt.xlabel("Mean Absolute SHAP Value")
    plt.ylabel("Input Features")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "global_feature_importance.png",
        dpi=300
    )

    plt.close()

    # 7. Explain one individual prediction
    sample_index = 0

    prediction = model.predict(
        X_test.iloc[[sample_index]]
    )[0]

    class_index = list(model.classes_).index(prediction)

    if values.ndim == 3:
        local_values = values[sample_index, :, class_index]
        local_base_value = base_values[sample_index, class_index]
    else:
        local_values = values[sample_index]
        local_base_value = base_values[sample_index]

    explanation = shap.Explanation(
        values=local_values,
        base_values=local_base_value,
        data=X_test.iloc[sample_index].values,
        feature_names=X_test.columns.tolist()
    )

    # 8. Generate individual explanation chart
    shap.plots.waterfall(
        explanation,
        show=False
    )

    plt.title(
        f"SHAP Explanation - Predicted Crop: {prediction}"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "individual_prediction_explanation.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # 9. Display results
    print("\nAgriSmart - Explainable AI Results")
    print("=" * 45)

    print(f"Predicted crop: {prediction}")

    print("\nGlobal Feature Importance:")
    print(
        importance_df.sort_values(
            by="Mean_Absolute_SHAP_Value",
            ascending=False
        ).to_string(index=False)
    )

    print("\nSHAP visualizations saved successfully!")
    print(f"Output directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    explain_model()
