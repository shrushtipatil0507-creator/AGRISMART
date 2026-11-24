"""Conservative, explainable advisory rules for AgriSmart.

These are educational decision-support prompts, not fertilizer prescriptions
or calibrated irrigation schedules. Soil test interpretation and local crop
recommendations must come from a recognised soil-testing/agricultural service.
"""
from __future__ import annotations

NUTRIENTS = ("Nitrogen (N)", "Phosphorus (P)", "Potassium (K)")
# Demonstration-only rainfall screen; not a validated crop/soil irrigation threshold.
RAINFALL_SCREEN_MM = 20.0


def fertilizer_advisory(crop: str, nutrient_status: dict[str, str]) -> dict:
    """Summarise supplied lab ratings without inventing fertilizer doses."""
    if not crop or not crop.strip():
        raise ValueError("Select or enter a crop.")
    guidance = []
    for nutrient in NUTRIENTS:
        status = nutrient_status.get(nutrient)
        if status not in {"Low", "Medium", "High", "Not tested"}:
            raise ValueError(f"Invalid soil-test status for {nutrient}.")
        if status == "Low":
            message = "Discuss a crop- and region-specific nutrient correction with the soil-testing laboratory or local KVK."
        elif status == "Medium":
            message = "Follow the soil-test crop recommendation; avoid adding extra nutrient without a crop-stage basis."
        elif status == "High":
            message = "Avoid automatic additional application of this nutrient; confirm the crop-specific recommendation."
        else:
            message = "No status supplied. Obtain a representative soil test before deciding on fertilizer."
        guidance.append({"nutrient": nutrient, "status": status, "guidance": message})

    return {
        "crop": crop.strip(),
        "items": guidance,
        "principle": (
            "Use soil-test-based, crop-specific integrated nutrient management and the 4R principle "
            "(right source, rate, time and place). This prototype does not calculate kg/acre or fertilizer doses."
        ),
        "source": "ICAR guidance as summarised by India's Department of Fertilizers / Press Information Bureau (24 March 2026).",
    }


def irrigation_advisory(
    crop: str,
    crop_stage: str,
    soil_moisture: str,
    forecast_rain_mm: float,
) -> dict:
    """Return a cautious irrigation decision prompt using farmer observations."""
    if not crop or not crop.strip():
        raise ValueError("Enter the crop.")
    if crop_stage not in {"Establishment", "Vegetative", "Flowering/Fruiting", "Maturity"}:
        raise ValueError("Choose a valid crop stage.")
    if soil_moisture not in {"Dry", "Moist", "Wet / waterlogged", "Unknown"}:
        raise ValueError("Choose a valid soil moisture condition.")
    if forecast_rain_mm < 0:
        raise ValueError("Forecast rainfall cannot be negative.")

    if soil_moisture == "Wet / waterlogged":
        recommendation = "Do not add irrigation now. Check drainage and avoid traffic on saturated soil."
        reason = "The reported field condition is wet or waterlogged."
    elif forecast_rain_mm >= RAINFALL_SCREEN_MM:
        recommendation = "Reassess after the forecast rain. Defer routine irrigation unless crop-specific monitoring shows immediate need."
        reason = "The entered forecast indicates potentially meaningful rainfall; actual effective rainfall depends on soil and storm distribution."
    elif soil_moisture == "Dry":
        recommendation = "Irrigation may be needed. Confirm root-zone moisture and use the locally recommended method and crop-stage schedule."
        reason = "The field is reported dry and the entered forecast rainfall is below 20 mm."
    elif soil_moisture == "Moist":
        recommendation = "Avoid calendar-based irrigation. Recheck root-zone moisture and the next forecast before irrigating."
        reason = "The field is reported moist."
    else:
        recommendation = "Check soil moisture in the root zone before deciding. Do not irrigate based only on this dashboard."
        reason = "Soil moisture is unknown."

    return {
        "crop": crop.strip(),
        "stage": crop_stage,
        "recommendation": recommendation,
        "reason": reason,
        "forecast_rain_mm": float(forecast_rain_mm),
        "limitation": (
            "The 20 mm rainfall screen is illustrative, not locally calibrated. No irrigation volume or exact interval "
            "is calculated because soil texture, rooting depth, field area, crop coefficients and measured "
            "soil moisture are not available."
        ),
    }
