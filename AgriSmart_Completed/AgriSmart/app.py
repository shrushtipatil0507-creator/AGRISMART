from pathlib import Path

import pandas as pd
import streamlit as st
import joblib
import matplotlib.pyplot as plt
from PIL import Image

from src.predict_disease import predict_disease
from src.weather_service import (
    WeatherServiceError,
    geocode_location,
    get_weather_forecast,
    build_weather_risks,
)
from src.advisory_rules import fertilizer_advisory, irrigation_advisory

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "models" / "crop_recommendation_model.joblib"
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]

st.set_page_config(
    page_title="AgriSmart | Intelligent Crop Advisory",
    page_icon="🌱",
    layout="wide",
)

st.title("🌱 AgriSmart")
st.subheader("Explainable Crop Advisory & Plant Health Monitoring")
st.write(
    "An academic decision-support prototype combining crop classification, "
    "plant-leaf image classification, weather screening and cautious farm advisory rules."
)
st.info(
    "Prototype notice: outputs are decision-support information, not a substitute "
    "for soil testing, local agricultural recommendations or a qualified agronomist."
)


@st.cache_resource
def load_crop_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Crop model not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@st.cache_data(ttl=1800, show_spinner=False)
def cached_geocode(location: str):
    return geocode_location(location)


@st.cache_data(ttl=1800, show_spinner=False)
def cached_weather(latitude: float, longitude: float, timezone: str):
    return get_weather_forecast(latitude, longitude, timezone)


tabs = st.tabs([
    "🏠 Overview",
    "🌾 Crop Recommendation",
    "🍃 Plant Health",
    "🌦️ Weather & Risk",
    "🧪 Fertilizer Advisory",
    "💧 Irrigation Advisory",
])

with tabs[0]:
    st.header("Project overview")
    metric_cols = st.columns(3)
    metric_cols[0].metric("Crop model", "Random Forest", "99.55% reported test accuracy")
    metric_cols[1].metric("Leaf model", "MobileNetV2", "94.59% reported test accuracy")
    metric_cols[2].metric("Tomato classes", "10", "PlantVillage categories")
    st.markdown(
        """
        **Available modules**
        - Crop recommendation from N, P, K, temperature, humidity, soil pH and rainfall.
        - SHAP-based explanation of the crop model, when the installed SHAP version supports the saved model.
        - Tomato leaf category prediction from an uploaded JPG, JPEG or PNG image.
        - Current weather and a five-day forecast for a town/city.
        - Transparent weather screening flags for heavy rain, high temperature and high humidity.
        - Soil-test status prompts for fertilizer decisions and a rainfall/soil-moisture irrigation decision aid.
        """
    )
    st.caption(
        "The model metrics above are reported in the supplied project artifacts; "
        "they are not independently re-created by opening the dashboard."
    )

with tabs[1]:
    st.header("🌾 Crop Recommendation")
    st.write("Enter the seven model inputs. Use values in the same units and conventions as the training dataset.")
    with st.form("crop_form"):
        left, right = st.columns(2)
        with left:
            nitrogen = st.number_input("Nitrogen (N)", min_value=0.0, value=90.0)
            phosphorus = st.number_input("Phosphorus (P)", min_value=0.0, value=42.0)
            potassium = st.number_input("Potassium (K)", min_value=0.0, value=43.0)
            temperature = st.number_input("Temperature (°C)", value=20.8)
        with right:
            humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=82.0)
            ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5)
            rainfall = st.number_input("Rainfall (dataset units)", min_value=0.0, value=202.9)
        crop_submitted = st.form_submit_button("🌱 Recommend suitable crop", use_container_width=True)

    if crop_submitted:
        try:
            input_data = pd.DataFrame(
                [[nitrogen, phosphorus, potassium, temperature, humidity, ph, rainfall]],
                columns=FEATURES,
            )
            model = load_crop_model()
            prediction = str(model.predict(input_data)[0])
            st.success(f"Recommended crop: **{prediction.replace('_', ' ').title()}**")

            st.subheader("Why did the model recommend this crop?")
            try:
                import shap

                classifier = model.named_steps.get("classifier", model)
                explainer = shap.TreeExplainer(classifier)
                explanation = explainer(input_data)
                values = explanation.values
                if getattr(values, "ndim", 0) == 3:
                    class_index = list(classifier.classes_).index(prediction)
                    local_explanation = shap.Explanation(
                        values=values[0, :, class_index],
                        base_values=float(explanation.base_values.reshape(-1, len(classifier.classes_))[0, class_index]),
                        data=input_data.iloc[0].to_numpy(),
                        feature_names=FEATURES,
                    )
                elif getattr(values, "ndim", 0) == 2:
                    local_explanation = shap.Explanation(
                        values=values[0],
                        base_values=float(explanation.base_values.reshape(-1)[0]),
                        data=input_data.iloc[0].to_numpy(),
                        feature_names=FEATURES,
                    )
                else:
                    raise ValueError("Unsupported SHAP output shape for this model/version.")
                shap.plots.waterfall(local_explanation, show=False)
                fig = plt.gcf()
                st.pyplot(fig, clear_figure=True)
                plt.close(fig)
            except Exception as exc:
                st.warning("The crop prediction succeeded, but the live SHAP explanation could not be rendered.")
                st.caption(f"Technical detail: {exc}")

            global_plot = ROOT / "reports" / "explainability" / "global_feature_importance.png"
            if global_plot.exists():
                st.image(str(global_plot), caption="Saved global feature-importance visualization")
            st.caption(
                "SHAP explains the fitted model's behaviour; it does not prove agricultural causality "
                "or guarantee yield or cultivation success."
            )
        except Exception as exc:
            st.error(f"Crop prediction failed: {exc}")

with tabs[2]:
    st.header("🍃 Plant Disease Detection")
    st.write("Upload a tomato leaf image. The current model supports the 10 classes listed below.")
    class_file = ROOT / "models" / "plant_disease" / "class_names.json"
    if class_file.exists():
        import json
        class_names = json.loads(class_file.read_text(encoding="utf-8"))
        with st.expander("View supported categories"):
            st.write([name.replace("__", " ").replace("_", " ") for name in class_names])

    uploaded_image = st.file_uploader(
        "Upload tomato leaf image",
        type=["jpg", "jpeg", "png"],
        key="leaf_image",
    )
    if uploaded_image is not None:
        try:
            image = Image.open(uploaded_image).convert("RGB")
            st.image(image, caption="Uploaded leaf image", width=350)
            if st.button("🔍 Detect plant category", use_container_width=True):
                with st.spinner("Loading model and analyzing image..."):
                    result = predict_disease(image)
                display_name = result["predicted_class"].replace("__", " ").replace("_", " ")
                st.metric("Model score", f"{result['confidence'] * 100:.2f}%")
                if result["is_uncertain"]:
                    st.warning(f"Low-confidence output: {display_name}. Please try a clearer image.")
                elif display_name.lower().endswith("healthy"):
                    st.success(f"Predicted category: **{display_name}**")
                else:
                    st.warning(f"Predicted category: **{display_name}**")
                st.progress(float(max(0.0, min(1.0, result["confidence"]))))
                st.caption(
                    "The 60% abstention threshold is provisional and not scientifically calibrated. "
                    "A model score is not a verified probability of disease."
                )
                st.warning(
                    "This model is limited to its 10 tomato PlantVillage categories. "
                    "Field photographs, other crops and unsupported diseases may be misclassified. "
                    "Confirm treatment decisions with an agricultural expert."
                )
        except Exception as exc:
            st.error(f"Image analysis failed: {exc}")

with tabs[3]:
    st.header("🌦️ Weather & Agricultural Risk Screening")
    st.write("Enter a town/city. The application uses Open-Meteo geocoding and forecast services.")
    location = st.text_input("Town or city", value="", placeholder="e.g. Kalaburagi", key="weather_location")
    if st.button("Get weather forecast", key="get_weather", use_container_width=True):
        try:
            with st.spinner("Finding location and retrieving forecast..."):
                place = cached_geocode(location)
                weather = cached_weather(
                    place["latitude"], place["longitude"], place.get("timezone", "auto")
                )
            st.session_state["agris_weather"] = {"place": place, "weather": weather}
        except (WeatherServiceError, ValueError) as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error(f"Unexpected weather error: {exc}")

    saved = st.session_state.get("agris_weather")
    if saved:
        place, weather = saved["place"], saved["weather"]
        current, daily = weather["current"], weather["daily"]
        st.subheader(f"{place['name']}, {place.get('admin1', '')} {place.get('country', '')}")
        st.caption(f"Forecast timezone: {weather.get('timezone', place.get('timezone', 'local'))} · Current time: {current.get('time', 'not supplied')}")
        m1, m2, m3 = st.columns(3)
        m1.metric("Temperature", f"{current.get('temperature_2m', '—')} °C")
        m2.metric("Relative humidity", f"{current.get('relative_humidity_2m', '—')}%")
        m3.metric("Current precipitation", f"{current.get('precipitation', '—')} mm")
        forecast_df = pd.DataFrame({
            "Date": daily.get("time", []),
            "Min temperature (°C)": daily.get("temperature_2m_min", []),
            "Max temperature (°C)": daily.get("temperature_2m_max", []),
            "Rain (mm)": daily.get("precipitation_sum", []),
            "Rain probability (%)": daily.get("precipitation_probability_max", []),
        })
        st.subheader("Five-day forecast")
        st.dataframe(forecast_df, use_container_width=True, hide_index=True)
        st.subheader("Agricultural screening flags")
        for risk in build_weather_risks(weather):
            if risk["level"] == "Watch":
                st.warning(f"**{risk['title']}** — {risk['detail']}")
            else:
                st.info(f"**{risk['title']}** — {risk['detail']}")
        st.caption(
            "Weather risk thresholds are illustrative configurable prototype screening rules, not locally calibrated "
            "crop-specific thresholds. Forecasts can change. Provider: Open-Meteo."
        )
        st.caption("Open-Meteo documentation: https://open-meteo.com/en/docs")

with tabs[4]:
    st.header("🧪 Fertilizer Advisory")
    st.write(
        "This module uses soil-test categories supplied by the user. It deliberately does not calculate fertilizer "
        "doses from the crop model's NPK inputs because their lab units and local soil-test calibration are not established."
    )
    with st.form("fertilizer_form"):
        crop = st.text_input("Crop", value="Tomato")
        st.caption("Select the nutrient status exactly as shown on a soil health card or laboratory report.")
        n_status = st.selectbox("Nitrogen (N) status", ["Not tested", "Low", "Medium", "High"])
        p_status = st.selectbox("Phosphorus (P) status", ["Not tested", "Low", "Medium", "High"])
        k_status = st.selectbox("Potassium (K) status", ["Not tested", "Low", "Medium", "High"])
        fertilizer_submit = st.form_submit_button("Generate fertilizer guidance", use_container_width=True)
    if fertilizer_submit:
        try:
            advice = fertilizer_advisory(crop, {
                "Nitrogen (N)": n_status,
                "Phosphorus (P)": p_status,
                "Potassium (K)": k_status,
            })
            st.subheader(f"Guidance for {advice['crop']}")
            for item in advice["items"]:
                st.markdown(f"**{item['nutrient']} — {item['status']}**")
                st.write(item["guidance"])
            st.info(advice["principle"])
            st.caption(advice["source"])
            st.caption(
                "For an exact nutrient/fertilizer dose, use a current crop- and region-specific soil-test recommendation "
                "from a recognised soil testing laboratory or KVK."
            )
        except ValueError as exc:
            st.error(str(exc))

with tabs[5]:
    st.header("💧 Irrigation Advisory")
    st.write("A simple decision aid based on farmer-observed root-zone moisture and expected rainfall.")
    with st.form("irrigation_form"):
        irrigation_crop = st.text_input("Crop", value="Tomato")
        stage = st.selectbox(
            "Crop stage",
            ["Establishment", "Vegetative", "Flowering/Fruiting", "Maturity"],
        )
        moisture = st.selectbox(
            "Observed soil moisture",
            ["Unknown", "Dry", "Moist", "Wet / waterlogged"],
        )
        forecast_rain = st.number_input(
            "Expected rainfall over the next 24–48 hours (mm)",
            min_value=0.0,
            value=0.0,
            help="Enter the forecast for your field location. You can refer to the Weather & Risk tab. The 20 mm screen is illustrative, not a validated threshold.",
        )
        irrigation_submit = st.form_submit_button("Generate irrigation guidance", use_container_width=True)
    if irrigation_submit:
        try:
            advice = irrigation_advisory(irrigation_crop, stage, moisture, forecast_rain)
            st.subheader(f"Guidance for {advice['crop']} — {advice['stage']}")
            st.success(advice["recommendation"])
            st.write(f"**Reason:** {advice['reason']}")
            st.caption(advice["limitation"])
            st.warning(
                "This is not a measured irrigation schedule. Check root-zone moisture and follow local "
                "crop-stage/soil-specific recommendations before applying water."
            )
        except ValueError as exc:
            st.error(str(exc))

st.divider()
st.caption(
    "AgriSmart | Academic AI & Data Science prototype | Weather provider: Open-Meteo | "
    "Agronomic decisions require local validation."
)
