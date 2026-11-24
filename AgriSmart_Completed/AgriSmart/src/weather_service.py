"""Open-Meteo geocoding and forecast helpers for AgriSmart.

Uses the public Open-Meteo APIs. No API key is required for non-commercial
academic use; check the provider's current terms and attribution requirements
before public/commercial deployment.
"""
from __future__ import annotations

from typing import Any
import requests

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
TIMEOUT_SECONDS = 12
# Demonstration screening thresholds only; validate and tune with local agronomic experts.
HEAVY_RAIN_SCREEN_MM = 50.0
HIGH_MAX_TEMP_C = 36.0
HIGH_RELATIVE_HUMIDITY_PERCENT = 85.0


class WeatherServiceError(RuntimeError):
    """Raised when location or weather data cannot be retrieved."""


def geocode_location(location: str) -> dict[str, Any]:
    """Resolve a town/city name to coordinates using Open-Meteo."""
    if not isinstance(location, str) or not location.strip():
        raise ValueError("Enter a town or city name.")
    try:
        response = requests.get(
            GEOCODING_URL,
            params={"name": location.strip(), "count": 1, "language": "en", "format": "json"},
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise WeatherServiceError(f"Could not find the location service: {exc}") from exc

    results = payload.get("results") or []
    if not results:
        raise WeatherServiceError(f"No matching location found for '{location}'.")
    place = results[0]
    return {
        "name": place.get("name", location.strip()),
        "admin1": place.get("admin1", ""),
        "country": place.get("country", ""),
        "latitude": float(place["latitude"]),
        "longitude": float(place["longitude"]),
        "timezone": place.get("timezone", "auto"),
    }


def get_weather_forecast(latitude: float, longitude: float, timezone: str = "auto") -> dict[str, Any]:
    """Fetch current conditions and a 5-day daily forecast."""
    try:
        response = requests.get(
            FORECAST_URL,
            params={
                "latitude": latitude,
                "longitude": longitude,
                "timezone": timezone or "auto",
                "forecast_days": 5,
                "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code,wind_speed_10m",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max,et0_fao_evapotranspiration",
            },
            timeout=TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise WeatherServiceError(f"Weather service request failed: {exc}") from exc

    if "current" not in payload or "daily" not in payload:
        raise WeatherServiceError("Weather provider returned incomplete forecast data.")
    return payload


def build_weather_risks(weather: dict[str, Any]) -> list[dict[str, str]]:
    """Create transparent screening flags, not crop-specific disease forecasts."""
    current = weather.get("current", {})
    daily = weather.get("daily", {})
    risks: list[dict[str, str]] = []

    rain_values = [v for v in daily.get("precipitation_sum", []) if isinstance(v, (int, float))]
    probability_values = [v for v in daily.get("precipitation_probability_max", []) if isinstance(v, (int, float))]
    max_temp_values = [v for v in daily.get("temperature_2m_max", []) if isinstance(v, (int, float))]

    if rain_values and max(rain_values) >= HEAVY_RAIN_SCREEN_MM:
        risks.append({
            "level": "Watch",
            "title": "Heavy-rain screening flag",
            "detail": f"At least one forecast day shows {max(rain_values):.1f} mm precipitation. Check field drainage and avoid field operations on waterlogged soil.",
        })
    elif probability_values and max(probability_values) >= 70:
        risks.append({
            "level": "Watch",
            "title": "High rain probability",
            "detail": f"Forecast rain probability reaches {max(probability_values):.0f}%. Recheck the forecast before irrigation or spraying.",
        })

    if max_temp_values and max(max_temp_values) >= HIGH_MAX_TEMP_C:
        risks.append({
            "level": "Watch",
            "title": "High-temperature screening flag",
            "detail": f"Forecast maximum temperature reaches {max(max_temp_values):.1f} °C. Monitor crop water stress and follow local crop-stage guidance.",
        })

    humidity = current.get("relative_humidity_2m")
    if isinstance(humidity, (int, float)) and humidity >= HIGH_RELATIVE_HUMIDITY_PERCENT:
        risks.append({
            "level": "Information",
            "title": "High current humidity",
            "detail": f"Current relative humidity is {humidity:.0f}%. Humid conditions can favour some foliar diseases, but this is not a disease diagnosis.",
        })

    if not risks:
        risks.append({
            "level": "Information",
            "title": "No configured screening flag triggered",
            "detail": "The simple thresholds in this prototype did not flag the returned weather. This does not mean field conditions are risk-free.",
        })
    return risks
