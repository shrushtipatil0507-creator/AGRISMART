
from PIL import Image
from src.predict_disease import predict_disease

# Load a sample tomato leaf image
image_path = "sample_leaf.jpg"

image = Image.open(image_path)

result = predict_disease(image)

print("\nAgriSmart - Plant Disease Prediction")
print("------------------------------------")
print("Predicted Disease:", result["disease"])
print("Model Score:", round(result["confidence"] * 100, 2), "%")
