import unittest
from src.advisory_rules import fertilizer_advisory, irrigation_advisory


class TestAdvisoryRules(unittest.TestCase):
    def test_fertilizer_returns_three_nutrients_without_dose(self):
        result = fertilizer_advisory("Tomato", {
            "Nitrogen (N)": "Low",
            "Phosphorus (P)": "Medium",
            "Potassium (K)": "High",
        })
        self.assertEqual(len(result["items"]), 3)
        self.assertIn("does not calculate", result["principle"])

    def test_fertilizer_rejects_invalid_status(self):
        with self.assertRaises(ValueError):
            fertilizer_advisory("Tomato", {"Nitrogen (N)": "Unknown"})

    def test_irrigation_defers_when_soil_is_waterlogged(self):
        result = irrigation_advisory("Tomato", "Vegetative", "Wet / waterlogged", 0)
        self.assertIn("Do not add irrigation", result["recommendation"])

    def test_irrigation_considers_forecast_rain(self):
        result = irrigation_advisory("Maize", "Vegetative", "Dry", 25)
        self.assertIn("Reassess after", result["recommendation"])

    def test_irrigation_rejects_negative_rainfall(self):
        with self.assertRaises(ValueError):
            irrigation_advisory("Maize", "Vegetative", "Dry", -1)


if __name__ == "__main__":
    unittest.main()
