
import unittest

from src.predict_crop import predict_crop


class TestCropPrediction(unittest.TestCase):

    def test_crop_prediction_returns_result(self):
        sample_values = [90, 42, 43, 20.8, 82, 6.5, 202.9]
        result = predict_crop(sample_values)

        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_crop_prediction_is_valid(self):
        sample_values = [90, 42, 43, 20.8, 82, 6.5, 202.9]
        result = predict_crop(sample_values)

        self.assertEqual(result, "rice")

    def test_rejects_wrong_number_of_values(self):
        with self.assertRaises(ValueError):
            predict_crop([90, 42, 43])

    def test_rejects_non_numeric_values(self):
        with self.assertRaises(ValueError):
            predict_crop([90, 42, "invalid", 20.8, 82, 6.5, 202.9])

    def test_rejects_infinite_values(self):
        with self.assertRaises(ValueError):
            predict_crop([90, 42, 43, float("inf"), 82, 6.5, 202.9])


if __name__ == "__main__":
    unittest.main()
