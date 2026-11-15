# Tomato Leaf Disease Classification: Handcrafted Features vs. CNN

## 1. Project Overview

This project investigates and compares two different approaches to image classification:

1. **Handcrafted Feature Extraction:** Extracting manually designed visual features using traditional image processing techniques.
2. **Convolutional Neural Networks (CNNs):** Automatically learning visual features and representations from images.

The project uses the same Tomato Leaf Disease dataset and test split as the previous [CNN-based classification project](https://github.com/7alasaleh/Tomato-Leaf-Disease-Classification-Using-CNN-Models).

The objective is to evaluate three handcrafted feature extraction methods using different Machine Learning classifiers, compare individual and combined feature representations, and assess their performance against the CNN models trained in the previous project.

The project focuses on classifying tomato leaf images into six categories, providing a comparative study of traditional Machine Learning and Deep Learning approaches.

## 2. Dataset Description

The dataset used in this project is:

**New Plant Diseases Dataset**

- Source: Kaggle
- Dataset: `vipoooool/new-plant-diseases-dataset`

Six tomato leaf categories were selected:

| Class | Number of Images |
|---|---:|
| Tomato___Bacterial_spot | 1,702 |
| Tomato___Early_blight | 1,920 |
| Tomato___Late_blight | 1,851 |
| Tomato___Leaf_Mold | 1,882 |
| Tomato___Septoria_leaf_spot | 1,745 |
| Tomato___healthy | 1,926 |
| **Total** | **11,026** |

### 2.1 Dataset Splitting

A group-aware stratified split was used to reduce potential data leakage between training, validation, and test sets.

The dataset was divided approximately as follows:

| Split | Number of Images |
|---|---:|
| Training | 7,727 |
| Validation | 1,668 |
| Test | 1,631 |
| **Total** | **11,026** |

The original split was preserved using `split_manifest.csv`.

Dataset integrity checks showed:

- Total leakage groups: 7,894.
- Groups appearing across multiple splits: 0.
- Missing image paths: 0.
- Unique image paths: 11,026.

This grouping strategy was used to reduce the risk of highly similar or related images appearing in different splits.

---

## 3. Image Preprocessing

The following preprocessing steps were applied:

1. Load images using OpenCV.
2. Convert images from BGR to RGB.
3. Resize images to 224 × 224 pixels.
4. Convert resized images to grayscale for handcrafted feature extraction.

The handcrafted methods were applied to grayscale images, while the previous CNN experiments used RGB images.

This difference reflects the nature of the two approaches.

---

## 4. Handcrafted Feature Extraction

Three feature extraction methods were implemented:

### 4.1 LBP — Local Binary Patterns

LBP is a texture descriptor that captures local patterns by comparing each pixel with its neighboring pixels.

**Parameters:**

- Number of Neighbors (P): 8
- Radius (R): 1
- Method: Uniform
- Histogram Bins: 10
- Histogram Normalization: Applied

**Feature dimensionality:** 10 features per image.

### 4.2 HOG — Histogram of Oriented Gradients

HOG extracts information about image shape and structure by describing the distribution of gradient orientations.

**Parameters:**

- Orientations: 9
- Pixels per Cell: (8, 8)
- Cells per Block: (2, 2)
- Block Normalization: L2-Hys

**Feature dimensionality:** 26,244 features per image.

### 4.3 GLCM — Gray-Level Co-occurrence Matrix

GLCM captures statistical texture information by analyzing the spatial relationships between neighboring grayscale pixel values.

**Parameters:**

- Quantization Levels: 32
- Distances: [1, 2]
- Angles: [0, π/4, π/2, 3π/4]
- Symmetric: True
- Normed: True

The following properties were extracted:

- Contrast
- Dissimilarity
- Homogeneity
- Energy
- Correlation

**Feature dimensionality:** 40 features per image.

### 4.4 Combined Features

The three feature sets were concatenated in the following order:

`LBP + HOG + GLCM`

Total dimensionality:

10 + 26,244 + 40 = **26,294 features per image**.

### 4.5 Feature Dimensions

| Feature Set | Training Samples | Validation Samples | Test Samples | Features |
|---|---:|---:|---:|---:|
| LBP | 7,727 | 1,668 | 1,631 | 10 |
| HOG | 7,727 | 1,668 | 1,631 | 26,244 |
| GLCM | 7,727 | 1,668 | 1,631 | 40 |
| Combined | 7,727 | 1,668 | 1,631 | 26,294 |

---

## 5. Classification Algorithms

Three machine learning classifiers were used.

### 5.1 Support Vector Machine (SVM)

Implementation: `LinearSVC`

Parameters:

- C = 1.0
- Max Iterations = 10,000
- StandardScaler applied through a Pipeline
- Random State = 42

### 5.2 Random Forest

Parameters:

- Number of Estimators = 200
- Max Features = sqrt
- Random State = 42
- n_jobs = -1

### 5.3 K-Nearest Neighbors (KNN)

Parameters:

- Number of Neighbors = 5
- Weights = distance
- StandardScaler applied through a Pipeline
- n_jobs = -1

The same baseline classifier configurations were used across the different feature sets.

---

## 6. Experimental Design

A total of 12 experiments were conducted.

| Feature Set | SVM | Random Forest | KNN |
|---|---|---|---|
| LBP | ✓ | ✓ | ✓ |
| HOG | ✓ | ✓ | ✓ |
| GLCM | ✓ | ✓ | ✓ |
| Combined | ✓ | ✓ | ✓ |

The validation set was used to evaluate and select the best-performing models.

Final evaluation was then performed on the test set without retraining the selected models.

---

## 7. Validation Results

The following table presents the validation performance of all 12 experiments.

| Feature Set | Classifier | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---|---:|---:|---:|---:|
| LBP | Random Forest | 60.73% | 59.61% | 61.14% | 59.37% |
| LBP | KNN | 55.52% | 55.22% | 55.50% | 54.87% |
| LBP | SVM | 51.38% | 49.74% | 51.49% | 46.79% |
| HOG | SVM | 44.84% | 44.59% | 44.91% | 44.70% |
| HOG | Random Forest | 55.28% | 55.05% | 55.16% | 53.41% |
| HOG | KNN | 32.19% | 51.81% | 31.46% | 27.72% |
| GLCM | SVM | 64.99% | 64.42% | 65.22% | 64.25% |
| GLCM | Random Forest | **71.88%** | **71.75%** | **72.09%** | **71.61%** |
| GLCM | KNN | 66.19% | 65.84% | 66.37% | 65.80% |
| Combined | SVM | 51.50% | 50.78% | 51.53% | 51.05% |
| Combined | Random Forest | 70.38% | 70.38% | 70.46% | 69.83% |
| Combined | KNN | 34.53% | 52.34% | 33.83% | 30.61% |

The best validation result was achieved by:

**GLCM + Random Forest — Macro F1: 71.61%**

---

## 8. Final Test Results — Handcrafted Features

All 12 handcrafted experiments were evaluated on the same test set containing 1,631 images.

| Feature Set | Classifier | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 |
|---|---|---:|---:|---:|---:|---:|
| GLCM | Random Forest | 72.35% | 72.42% | 72.64% | **72.27%** | 72.16% |
| Combined | Random Forest | **72.41%** | **72.65%** | 72.62% | 72.11% | 71.94% |
| GLCM | KNN | 68.49% | 68.30% | 68.74% | 68.31% | 68.18% |
| GLCM | SVM | 65.91% | 65.90% | 66.38% | 65.43% | 65.18% |
| LBP | Random Forest | 62.60% | 62.04% | 63.02% | 61.30% | 61.17% |
| HOG | Random Forest | 57.57% | 56.44% | 57.29% | 55.53% | 55.82% |
| LBP | KNN | 55.36% | 54.65% | 55.54% | 54.89% | 54.76% |
| Combined | SVM | 52.36% | 51.60% | 52.37% | 51.91% | 51.93% |
| LBP | SVM | 53.53% | 51.97% | 53.69% | 49.41% | 49.38% |
| HOG | SVM | 45.43% | 44.99% | 45.40% | 45.16% | 45.20% |
| Combined | KNN | 35.01% | 55.68% | 34.76% | 31.04% | 31.46% |
| HOG | KNN | 32.68% | 55.22% | 32.39% | 28.21% | 28.70% |

### Best Handcrafted Model

**GLCM + Random Forest**

| Metric | Test Result |
|---|---:|
| Accuracy | 72.35% |
| Macro Precision | 72.42% |
| Macro Recall | 72.64% |
| Macro F1 | **72.27%** |
| Weighted F1 | 72.16% |

Although Combined + Random Forest achieved the highest accuracy among handcrafted experiments, GLCM + Random Forest achieved the highest Macro F1.

Therefore, GLCM + Random Forest was considered the best overall handcrafted model based on macro-averaged classification performance.

---

## 9. Class-wise Evaluation

Additional evaluation was performed using classification reports and One-vs-Rest metrics.

The best Random Forest model from each feature set was selected for this analysis:

- LBP + Random Forest
- HOG + Random Forest
- GLCM + Random Forest
- Combined + Random Forest

### 9.1 GLCM + Random Forest

Selected class-wise recall results:

| Class | Recall |
|---|---:|
| Bacterial Spot | 81.64% |
| Early Blight | 59.25% |
| Late Blight | 64.98% |
| Leaf Mold | 65.96% |
| Septoria Leaf Spot | 67.33% |
| Healthy | 96.67% |

The model performed particularly well on healthy leaves. Early Blight was the most challenging class for this model.

### 9.2 Combined + Random Forest

Selected class-wise recall results:

| Class | Recall |
|---|---:|
| Bacterial Spot | 91.41% |
| Early Blight | 66.78% |
| Late Blight | 51.26% |
| Leaf Mold | 70.53% |
| Septoria Leaf Spot | 60.96% |
| Healthy | 94.81% |

Combined features achieved higher recall for Bacterial Spot and Leaf Mold compared with GLCM alone.

However, its recall for Late Blight and Septoria was lower.

### 9.3 LBP and HOG Observations

- LBP + Random Forest achieved a low Leaf Mold recall of 29.82%.
- HOG + Random Forest achieved a low Septoria recall of 21.91%.

These differences indicate that individual handcrafted descriptors capture different aspects of the images and do not perform equally across all disease categories.

---

## 10. CNN Results

The handcrafted experiments were compared with the previously trained CNN models.

CNN metrics were obtained from:

`all_models_test_metrics.csv`

All models were evaluated on 1,631 test samples.

| CNN Model | Accuracy | Macro Precision | Macro Recall | Macro F1 |
|---|---:|---:|---:|---:|
| SoftVoting Ensemble | **99.63%** | 99.63% | 99.64% | **99.64%** |
| EfficientNetB0 | 99.57% | 99.58% | 99.56% | 99.57% |
| ResNet18 | 99.33% | 99.33% | 99.32% | 99.32% |
| DenseNet121 | 99.20% | 99.21% | 99.20% | 99.20% |
| MobileNetV2 | 99.14% | 99.15% | 99.12% | 99.13% |
| AlexNet | 95.59% | 95.62% | 95.57% | 95.59% |

**SoftVoting Ensemble** is an ensemble model rather than an individual CNN architecture.

The best individual CNN architecture was **EfficientNetB0**.

---

## 11. CNN vs. Handcrafted Features — Final Comparison

The following table combines the final test results of all CNN and handcrafted models.

| Rank | Approach | Model | Accuracy | Macro F1 |
|---:|---|---|---:|---:|
| 1 | CNN Ensemble | SoftVoting Ensemble | 99.63% | 99.64% |
| 2 | CNN Individual | EfficientNetB0 | 99.57% | 99.57% |
| 3 | CNN Individual | ResNet18 | 99.33% | 99.32% |
| 4 | CNN Individual | DenseNet121 | 99.20% | 99.20% |
| 5 | CNN Individual | MobileNetV2 | 99.14% | 99.13% |
| 6 | CNN Individual | AlexNet | 95.59% | 95.59% |
| 7 | Handcrafted | GLCM + Random Forest | 72.35% | 72.27% |
| 8 | Handcrafted | Combined + Random Forest | 72.41% | 72.11% |
| 9 | Handcrafted | GLCM + KNN | 68.49% | 68.31% |
| 10 | Handcrafted | GLCM + SVM | 65.91% | 65.43% |
| 11 | Handcrafted | LBP + Random Forest | 62.60% | 61.30% |
| 12 | Handcrafted | HOG + Random Forest | 57.57% | 55.53% |
| 13 | Handcrafted | LBP + KNN | 55.36% | 54.89% |
| 14 | Handcrafted | Combined + SVM | 52.36% | 51.91% |
| 15 | Handcrafted | LBP + SVM | 53.53% | 49.41% |
| 16 | Handcrafted | HOG + SVM | 45.43% | 45.16% |
| 17 | Handcrafted | Combined + KNN | 35.01% | 31.04% |
| 18 | Handcrafted | HOG + KNN | 32.68% | 28.21% |

The CNN and handcrafted results were confirmed to use the same test split from `split_manifest.csv`, containing 1,631 images.

### Main Comparison

| Approach | Best Model | Accuracy | Macro F1 |
|---|---|---:|---:|
| CNN Ensemble | SoftVoting Ensemble | 99.63% | 99.64% |
| CNN Individual | EfficientNetB0 | 99.57% | 99.57% |
| Handcrafted | GLCM + Random Forest | 72.35% | 72.27% |

CNN models achieved substantially higher classification performance than the handcrafted methods evaluated in this project.

---

## 12. Discussion

The experimental results lead to several observations:

1. **CNN Performance:** CNN models significantly outperformed the tested handcrafted approaches, suggesting that automatically learned image representations were more effective for this classification task.

2. **GLCM Effectiveness:** GLCM was the strongest handcrafted feature extraction method according to Macro F1, despite using only 40 features per image.

3. **Feature Combination:** Combining LBP, HOG, and GLCM did not improve Macro F1 over GLCM alone when using Random Forest.

4. **Feature Dimensionality:** HOG generated a large feature vector, but its higher dimensionality did not guarantee better classification performance.

5. **Classifier Dependency:** Classification performance depended strongly on the selected classifier. Random Forest generally performed better than SVM and KNN for these experiments.

6. **Class-wise Differences:** Some categories were more difficult to classify than others. For example, GLCM + Random Forest performed well on healthy leaves but had lower recall for Early Blight.

7. **Efficiency:** GLCM required only 40 features per image and had a recorded test prediction time of approximately 0.087 seconds for Random Forest. Combined features required 26,294 features and had a recorded test prediction time of approximately 0.170 seconds for Random Forest.

These timing measurements are specific to the recorded execution environment and should not be interpreted as hardware-independent benchmarks.

---

## 13. Conclusion

This project compared traditional handcrafted feature extraction methods with CNN-based image classification for six tomato leaf categories.

Among the handcrafted approaches, GLCM combined with Random Forest achieved the best overall Macro F1 score of 72.27% on the test set.

The CNN models achieved substantially higher performance. EfficientNetB0, the best individual CNN architecture, achieved 99.57% Macro F1, while the SoftVoting Ensemble achieved 99.64%.

The results suggest that CNN-based approaches were considerably more effective at learning discriminative visual representations for the selected dataset.

However, GLCM remains a useful traditional approach because it provides a compact feature representation and reasonable classification performance without requiring deep neural network training.

---

## 14. Limitations

- The experiments cover only six selected tomato leaf categories.
- CNN models used RGB images, whereas handcrafted feature extraction used grayscale images.
- Therefore, the comparison evaluates two complete approaches with different image representations, rather than isolating the effect of the classifier alone.
- A comprehensive hyperparameter optimization study was not conducted for every classifier.
- The reported CNN results are from a final run, not averages and standard deviations over multiple random seeds.
- The CNN metrics CSV contains aggregate model metrics, not per-image predictions.
- Results should not be interpreted as evidence that the models are ready for independent real-world agricultural diagnosis.

---

## 15. Output Files

The following files were generated during the experiments in Kaggle.

### Result Files

- `handcrafted_test_results.csv`
- `classification_reports.csv`
- `one_vs_rest_metrics.csv`
- `one_vs_rest_metrics_selected.csv`
- `final_cnn_vs_handcrafted_comparison.csv`

### Visualization Files

- `handcrafted_validation_results.png`
- `validation_accuracy_f1.png`
- `validation_execution_time.png`
- `final_test_confusion_matrices.png`
- `final_accuracy_comparison.png`
- `final_macro_f1_comparison.png`
- `best_model_approach_comparison.png`

### Extracted Features

Feature arrays were saved in:

`/kaggle/working/handcrafted_features`

### Dataset Manifest

The original split information was stored in:

`split_manifest.csv`

---

## 16. Reproducibility

The experiments were conducted in Kaggle Notebooks using Python.

Main libraries:

- NumPy
- Pandas
- OpenCV
- scikit-image
- scikit-learn
- Matplotlib
- Seaborn

The paths used in the notebook are Kaggle-specific and may need to be adjusted when running the project in another environment.

To reproduce the handcrafted experiments:

1. Add the New Plant Diseases Dataset to Kaggle Input.
2. Add the original `split_manifest.csv`.
3. Reconstruct the training, validation, and test DataFrames using the original split.
4. Apply the image preprocessing steps.
5. Extract LBP, HOG, and GLCM features.
6. Construct the combined feature representation.
7. Train the three classifiers using the specified configurations.
8. Evaluate the models on the original test set.
9. Generate the classification reports, metrics, and visualizations.

---

## 17. Final Takeaway

**CNN models substantially outperformed the handcrafted feature methods evaluated in this project. Among the handcrafted approaches, GLCM combined with Random Forest provided the best balance between classification performance and feature dimensionality.**