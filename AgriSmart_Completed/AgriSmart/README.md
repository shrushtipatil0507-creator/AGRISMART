# AgriSmart
## Explainable Crop Advisory & Plant Health Monitoring System

AgriSmart is an academic AI and Data Science prototype that combines tabular machine learning, explainable AI, tomato leaf image classification, weather information and conservative rule-based farm advisory screens.

> **Important:** AgriSmart is a decision-support prototype. It is not a substitute for a laboratory soil test, local KVK/Department of Agriculture recommendations, plant pathology diagnosis or field inspection.

## 1. Features

### Implemented machine-learning modules
1. **Crop recommendation:** Random Forest classifier using N, P, K, temperature, humidity, pH and rainfall.
   - Supplied evaluation artifact reports 99.55% held-out accuracy and 99.55% macro F1.
   - Model file: `models/crop_recommendation_model.joblib`.
2. **Explainable AI:** SHAP explanations for the crop classifier, including a saved global feature-importance image and an individual explanation artifact.
3. **Plant disease classification:** MobileNetV2 model for 10 tomato leaf categories.
   - Supplied test artifact reports 94.59% accuracy and 93.78% macro F1.
   - Model: `models/plant_disease/agris_mobilenetv2.keras`.
   - Labels: `models/plant_disease/class_names.json`.
4. **Streamlit dashboard:** Crop recommendation and leaf-image prediction.

### Added in this continuation
5. **Weather integration:** Open-Meteo geocoding and 5-day forecast by town/city; displays current conditions and forecast.
6. **Weather risk screening:** Transparent flags for forecast heavy rain, high temperature and current high humidity. These are screening heuristics, not crop-specific disease or yield predictions.
7. **Fertilizer advisory:** Takes the user's N/P/K soil-test ratings (Low/Medium/High/Not tested) and provides cautious next-step guidance. It does not invent a fertilizer quantity.
8. **Irrigation advisory:** Uses crop, crop stage, observed soil moisture and user-entered forecast rainfall to produce a conservative decision prompt. It does not calculate irrigation volume or an exact interval.
9. **Tests:** Unit tests for crop prediction validation, advisory rules and weather risk rules.

## 2. Project structure

```text
AgriSmart/
├── app.py
├── requirements.txt
├── requriments.txt              # legacy filename retained for compatibility
├── data/
│   ├── raw/Crop_recommendation.csv
│   └── processed/
├── models/
│   ├── crop_recommendation_model.joblib
│   └── plant_disease/
│       ├── agris_mobilenetv2.keras
│       ├── class_names.json
│       └── test_metrics.json
├── src/
│   ├── predict_crop.py
│   ├── predict_disease.py
│   ├── weather_service.py
│   └── advisory_rules.py
├── tests/
│   ├── test_crop_prediction.py
│   ├── test_disease_prediction.py
│   ├── test_advisory_rules.py
│   └── test_weather_service.py
└── reports/
    ├── eda/
    ├── evaluation/
    ├── explainability/
    └── disease_detection/
```

## 3. Setup on Windows / VS Code

Use the same Python version and dependency versions used to create the model artifacts where possible. The project README from the supplied archive recommends Python 3.12.

Open PowerShell in the `AgriSmart` project root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation for the current terminal session, use:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Run the dashboard:

```powershell
streamlit run app.py
```

Streamlit prints a local URL (normally `http://localhost:8501`). Open that URL in the browser.

## 4. Application flow

1. Open **Overview** to review the scope and model metrics recorded in the supplied evaluation artifacts.
2. Open **Crop Recommendation**, enter the seven values and submit. AgriSmart predicts one of the 22 crop classes and attempts to display a SHAP local explanation.
3. Open **Plant Health**, upload a tomato leaf image (JPG/JPEG/PNG) and select Detect. The classifier returns one of its 10 supported categories and a model score.
4. Open **Weather & Risk**, enter a town/city and retrieve current conditions and a five-day forecast. The app then displays configured screening flags.
5. Open **Fertilizer Advisory**, enter soil-test nutrient ratings. The module explains cautious next steps without estimating a fertilizer dose.
6. Open **Irrigation Advisory**, enter crop stage, observed soil moisture and expected rainfall to receive a conservative irrigation decision prompt.

## 5. Run tests

From the project root with the virtual environment activated:

```powershell
python -m unittest discover -s tests -v
```

The disease prediction test loads TensorFlow and the `.keras` model, so TensorFlow must be installed for the complete suite. Weather API network tests should be run with internet access; the unit tests for the risk rules do not require network access.

## 6. Weather service

The app uses:
- Open-Meteo Geocoding API to resolve the town/city.
- Open-Meteo Forecast API to retrieve current weather and five daily forecast values.

A town/city is entered by the user; the application does not request device GPS coordinates. The app displays the provider and forecast timezone. Weather requests use a timeout and show a readable error if the service is unavailable.

Check Open-Meteo's current API documentation, attribution requirements and usage terms before public or commercial deployment:
- https://open-meteo.com/en/docs
- https://open-meteo.com/en/docs/geocoding-api

## 7. Fertilizer and irrigation safety

### Fertilizer
The crop model's N/P/K input fields come from the crop recommendation dataset. The archive does not establish the lab units, local soil-test calibration or field area needed to calculate fertilizer quantities. Therefore, the fertilizer screen asks for soil-test categories rather than reusing those model inputs as a lab report.

India's Department of Fertilizers / Press Information Bureau summarised ICAR guidance on 24 March 2026: there is no single ideal NPK ratio for every region; nutrient needs vary by crop, soil health and agro-climatic conditions. Soil-test-based balanced and integrated nutrient management is recommended. The app consequently directs users to crop- and region-specific soil-test recommendations and the 4R principle rather than providing unsupported kg/acre doses.

### Irrigation
The irrigation screen uses the user's observed soil moisture and entered forecast rainfall. Its rules are deliberately simple and are described in the UI. It does not have soil texture, rooting depth, field area, crop coefficient, evapotranspiration calibration or measured soil moisture. It therefore does not calculate a volume or exact irrigation interval.

## 8. Model evaluation claims

The values shown in the dashboard and this README are transcribed from the artifacts included in the supplied project:
- `reports/evaluation/model_comparison.csv`
- `models/plant_disease/test_metrics.json`

They should be presented with the original dataset, split and test conditions. The current continuation did not retrain either model or independently reproduce the full evaluation. Do not claim that 99.55% or 94.59% is guaranteed performance on real farms or field photographs.

The disease model supports only its ten listed tomato classes. The PlantVillage-style benchmark images do not guarantee similar performance on field photographs, other crops or unseen diseases.

## 9. Known environment / verification notes

- The supplied crop model was serialized with scikit-learn 1.9.1. Loading it under a different scikit-learn version can produce an `InconsistentVersionWarning`. Use the matching training environment when possible; do not ignore compatibility warnings in a production system.
- TensorFlow and Streamlit were not installed in the isolated build environment used for the code checks in this continuation. The complete dashboard and `.keras` inference therefore require verification on the user's Windows environment after installing `requirements.txt`.
- Weather data requires an internet connection.
- The provisional 60% disease abstention threshold is not calibrated. It is a UI safeguard only.

## 10. References

1. Open-Meteo, *Weather Forecast API Documentation*: https://open-meteo.com/en/docs
2. Open-Meteo, *Geocoding API Documentation*: https://open-meteo.com/en/docs/geocoding-api
3. Press Information Bureau, Ministry of Chemicals and Fertilizers, *Government Promotes Balanced Use of Fertilizers and Sustainable Practices*, 24 March 2026: https://www.pib.gov.in/PressReleasePage.aspx?PRID=2244397&lang=1&reg=3
4. MobileNetV2: Sandler et al., *MobileNetV2: Inverted Residuals and Linear Bottlenecks*, CVPR 2018: https://arxiv.org/abs/1801.04381
5. SHAP: Lundberg and Lee, *A Unified Approach to Interpreting Model Predictions*, NeurIPS 2017: https://arxiv.org/abs/1705.07874

## Academic project

Developed as an Artificial Intelligence and Data Science engineering project.
