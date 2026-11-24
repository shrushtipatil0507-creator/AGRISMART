import unittest
from pathlib import Path
from PIL import Image

from src.predict_disease import (
    predict_disease,
    load_class_names
)


class TestDiseasePrediction(unittest.TestCase):

    def test_prediction_with_sample_image(self):
        project_root = Path(__file__).resolve().parents[1]
        image_path = project_root / "sample_leaf.jpg"

        image = Image.open(image_path)
        result = predict_disease(image)

        self.assertIn(result["disease"], load_class_names())
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)
        self.assertEqual(len(result["all_predictions"]), 10)

    def test_prediction_with_no_image(self):
        with self.assertRaises(ValueError):
            predict_disease(None)


if __name__ == "__main__":
    unittest.main()