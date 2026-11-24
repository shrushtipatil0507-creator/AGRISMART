# AgriSmart — Project Progress and End-to-End Flow

## Verified project assets in the supplied archive

- Crop Random Forest pipeline: `models/crop_recommendation_model.joblib`
- Crop model comparison: `reports/evaluation/model_comparison.csv`
- SHAP global and individual explanation images: `reports/explainability/`
- Tomato MobileNetV2 model: `models/plant_disease/agris_mobilenetv2.keras`
- Tomato class list: `models/plant_disease/class_names.json`
- Disease metrics and report: `models/plant_disease/test_metrics.json` and `reports/disease_detection/`
- Streamlit entry point: `app.py`
- Original tests: `tests/test_crop_prediction.py` and `tests/test_disease_prediction.py`

## Added modules

| Module | Main file | Purpose |
|---|---|---|
| Weather service | `src/weather_service.py` | City geocoding, current conditions, five-day forecast, transparent weather flags |
| Fertilizer rules | `src/advisory_rules.py` | Soil-test status-based guidance without unsupported fertilizer doses |
| Irrigation rules | `src/advisory_rules.py` | Crop-stage, field moisture and forecast-rainfall decision prompts |
| Integrated UI | `app.py` | One dashboard with six tabs |

## End-to-end user flow

1. User opens the Streamlit dashboard.
2. User may enter soil and climate values to obtain a crop class from the saved Random Forest model.
3. The dashboard attempts a SHAP explanation of that prediction.
4. User may upload a tomato leaf image; the MobileNetV2 model predicts among its ten known classes.
5. User enters a town/city to retrieve current weather and a five-day forecast.
6. Weather rules flag selected conditions for attention; they do not predict yield or diagnose crop disease.
7. User enters actual soil-test nutrient ratings to obtain cautious fertilizer next steps.
8. User enters observed soil moisture and forecast rainfall to obtain an irrigation decision prompt.
9. User verifies all advice against local agricultural guidance before acting.

## Important scope boundary

The archive does not include locally validated fertilizer dose tables, soil texture and water-holding data, field area, rooting depth, crop coefficient data, or crop-stage irrigation calibration. The advisory screens therefore do not calculate exact fertilizer quantities or irrigation volumes.

## Test status in the build environment

The following checks were run after code changes:
- Python compile check across project `.py` files.
- Crop prediction tests.
- Fertilizer advisory rule tests.
- Irrigation advisory rule tests.
- Weather risk-rule tests.

TensorFlow and Streamlit were unavailable in the isolated build environment, so disease-model inference and live dashboard launch must still be verified on the target Windows/Python environment. Weather API calls also require an internet connection.
