import unittest
from src.weather_service import build_weather_risks


class TestWeatherRiskRules(unittest.TestCase):
    def test_rain_and_heat_flags(self):
        weather = {
            "current": {"relative_humidity_2m": 90},
            "daily": {
                "precipitation_sum": [2, 55],
                "precipitation_probability_max": [20, 80],
                "temperature_2m_max": [34, 38],
            },
        }
        risks = build_weather_risks(weather)
        titles = {item["title"] for item in risks}
        self.assertIn("Heavy-rain screening flag", titles)
        self.assertIn("High-temperature screening flag", titles)
        self.assertIn("High current humidity", titles)

    def test_empty_forecast_has_fallback(self):
        risks = build_weather_risks({"current": {}, "daily": {}})
        self.assertEqual(len(risks), 1)
        self.assertEqual(risks[0]["level"], "Information")


if __name__ == "__main__":
    unittest.main()
