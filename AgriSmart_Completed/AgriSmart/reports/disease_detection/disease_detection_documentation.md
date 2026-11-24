# AgriSmart – Plant Disease Detection Module

## 1. Overview

The Plant Disease Detection module is an image-classification component of AgriSmart. It uses a deep learning model to classify uploaded tomato leaf images into one of ten supported categories.

The model is based on MobileNetV2 with transfer learning using ImageNet-pretrained weights.

## 2. Dataset

- Dataset: PlantVillage
- Source: Kaggle – emmarex/plantdisease
- Selected crop: Tomato
- Number of categories: 10
- Total images: 16,011

### Dataset split

| Split | Images | Percentage |
|---|---:|---:|
| Training | 11,207 | 70% |
| Validation | 2,402 | 15% |
| Testing | 2,402 | 15% |
| Total | 16,011 | 100% |

The dataset was divided using stratified sampling with random state 42.

## 3. Supported Categories

1. Tomato Bacterial Spot
2. Tomato Early Blight
3. Tomato Late Blight
4. Tomato Leaf Mold
5. Tomato Septoria Leaf Spot
6. Tomato Spider Mites
7. Tomato Target Spot
8. Tomato Yellow Leaf Curl Virus
9. Tomato Mosaic Virus
10. Tomato Healthy

## 4. Image Preprocessing

The following preprocessing operations were applied:

- Images decoded as RGB.
- Images resized to 224 × 224 pixels.
- MobileNetV2 preprocessing applied to normalize pixel values.
- Images grouped into batches of 32.
- Training data shuffled only where configured by the training pipeline.
- Class weights calculated to address class imbalance.

## 5. Model Architecture

**Model:** MobileNetV2

MobileNetV2 was selected because it is a computationally efficient convolutional neural network suitable for image classification.

### Architecture components

1. Pretrained MobileNetV2 feature extractor.
2. Global Average Pooling layer.
3. Dropout layer with rate 0.3.
4. Dense output layer with 10 neurons.
5. Softmax activation for multi-class classification.

### Training strategy

**Stage 1 – Transfer Learning**
- ImageNet-pretrained MobileNetV2 base.
- Base model frozen.
- Adam optimizer.
- Learning rate: 0.001.
- Maximum epochs: 10.

**Stage 2 – Fine-tuning**
- Selected final layers unfrozen.
- Batch normalization layers kept frozen.
- Adam optimizer.
- Learning rate: 0.00001.
- Maximum epochs: 5.

Early stopping and model checkpointing were used to retain the best-performing model.

## 6. Evaluation Results

The best fine-tuned model was evaluated on the held-out test set.

| Metric | Result |
|---|---:|
| Test Accuracy | 94.59% |
| Test Loss | 0.1680 |
| Macro Precision | 93.90% |
| Macro Recall | 93.83% |
| Macro F1-score | 93.78% |
| Weighted F1-score | 94.61% |

The detailed classification report and confusion matrix are available in this folder.

## 7. Class-wise Observations

The model performed particularly well on Tomato Yellow Leaf Curl Virus, Tomato Mosaic Virus, and Tomato Healthy categories.

Lower performance was observed for:
- Tomato Early Blight: F1-score 82.23%.
- Tomato Target Spot: F1-score 87.95%.

These results indicate that some disease categories are more challenging to distinguish than others.

## 8. Application Integration

The trained model was integrated into the AgriSmart Streamlit application.

The workflow is:

1. User uploads a tomato leaf image.
2. Image is converted to RGB.
3. Image is resized to 224 × 224.
4. MobileNetV2 preprocessing is applied.
5. The trained model predicts the category.
6. The application displays the predicted category and model confidence score.

## 9. Model Artifacts

The trained model and supporting files are stored in:

`models/plant_disease/`

Files:
- `agris_mobilenetv2.keras`
- `class_names.json`
- `classification_report.txt`
- `confusion_matrix.png`
- `test_metrics.json`

## 10. Limitations

- The current model supports tomato leaf categories only.
- PlantVillage images are largely collected under controlled conditions.
- Real-world field images may differ in lighting, backgrounds, camera quality and disease appearance.
- Test accuracy does not guarantee equivalent performance on field photographs.
- A model confidence score is not a calibrated probability of disease.
- The system is a prototype decision-support tool and does not replace professional agricultural diagnosis.

## 11. Future Improvements

- Validate using independently collected field images.
- Add more crop species and disease categories.
- Investigate confidence calibration and unknown-image rejection.
- Explore Grad-CAM for visual explanations.
- Optimize inference for deployment on resource-constrained devices.

## 12. Conclusion

The AgriSmart Plant Disease Detection module successfully integrates a fine-tuned MobileNetV2 image-classification model into a Streamlit application. It achieved 94.59% test accuracy on the held-out PlantVillage test set and supports ten tomato leaf categories.

The module demonstrates the practical use of transfer learning for agricultural image classification while acknowledging the limitations of controlled-dataset evaluation.