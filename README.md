# Breast Cancer Classification — Machine Learning Assignment 2

**Name:** Charu Goyal  
**BITS ID:** 2026AJ05682  
**Programme:** M.Tech AIML  
**Course:** Machine Learning  

## a. Problem Statement

The objective is to classify breast cancer samples as benign or malignant using numerical features describing cell nuclei.

Five classification models are trained on the same dataset and evaluated using Accuracy, AUC, Precision, Recall, F1 Score and Matthews Correlation Coefficient (MCC). An interactive Streamlit application allows users to upload labelled test data and compare the saved models without retraining them.

This project is an academic demonstration and is not intended for clinical diagnosis.

## b. Dataset Description

The project uses the Breast Cancer Wisconsin (Diagnostic) dataset from the UCI Machine Learning Repository, loaded through scikit-learn.

**Dataset source:**  
https://archive.ics.uci.edu/dataset/17/breast-cancer-wisconsin-diagnostic

| Property | Description |
| --- | --- |
| Total instances | 569 |
| Input features | 30 numerical features |
| Classification task | Binary classification |
| Benign samples | 357 |
| Malignant samples | 212 |
| Missing feature values | 0 |
| Target encoding | 0 = Benign; 1 = Malignant |
| Training records | 455 |
| Test records | 114 |

The features describe characteristics such as radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry and fractal dimension. The dataset includes mean, standard error and worst-value measurements.

### Data Preparation

- The original target labels are recoded so that malignant is the positive class, represented by 1.
- The dataset is divided into 80% training data and 20% test data.
- Stratified splitting preserves approximately the same class proportions in both sets.
- `random_state=42` ensures a reproducible split.
- All five models use exactly the same training and test records.
- StandardScaler is applied through pipelines for Logistic Regression, KNN and SVM.
- Scaling parameters are learned from training data only.
- Decision Tree and Gaussian Naive Bayes use the original feature values.

## c. GitHub Repository Link

**GitHub repository:**  
(https://github.com/goyalcharu/ml-classification-comparison)

**Live Streamlit application:**  
(https://ml-classification-comparison-bwvtoh5s3upjrimjmjg3a2.streamlit.app/)

## d. Models Used

| ML Model Name | Configuration |
| --- | --- |
| Logistic Regression | StandardScaler; max_iter=2000 |
| Decision Tree | max_depth=5; random_state=42 |
| KNN | StandardScaler; n_neighbors=5 |
| Naive Bayes | GaussianNB with default parameters |
| Support Vector Machine | StandardScaler; RBF kernel; C=1.0; gamma="scale" |

Model settings are fixed before test evaluation. No hyperparameter tuning is performed on the test set.

### Evaluation Metrics

All metrics are calculated on the held-out test set.

- **Accuracy:** Proportion of correctly classified samples.
- **AUC:** Ability to rank malignant samples above benign samples across classification thresholds.
- **Precision:** Proportion of predicted malignant samples that are actually malignant.
- **Recall:** Proportion of actual malignant samples correctly detected.
- **F1 Score:** Harmonic mean of precision and recall.
- **MCC:** Correlation between actual and predicted labels, accounting for all four confusion-matrix counts.

Precision, recall and F1 use malignant (1) as the positive class. AUC is calculated using positive-class probabilities for models supporting them and decision scores for SVM.

### Model Comparison

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| --- | --- | --- | --- | --- | --- | --- |
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9211 | 0.9448 | 0.9459 | 0.8333 | 0.8861 | 0.8299 |
| KNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes | 0.9386 | 0.9934 | 1.0000 | 0.8333 | 0.9091 | 0.8715 |
| Support Vector Machine | 0.9737 | 0.9947 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

### Observations on Model Performance

| ML Model Name | Observation about model performance |
| --- | --- |
| Logistic Regression | Achieved 96.49% accuracy and the highest AUC of 0.9960. It detected 39 of 42 malignant samples, missing three, and incorrectly classified one benign sample as malignant. Its results show that a linear classifier performs well on this split. |
| Decision Tree | Achieved 92.11% accuracy, the lowest among the five models. It missed seven malignant samples and produced two false positives. Its F1 and MCC were also the lowest on this test set. |
| KNN | Achieved 95.61% accuracy and 0.9744 precision. It missed four malignant samples and produced one false positive. Its recall and F1 were slightly lower than Logistic Regression and SVM. |
| Naive Bayes | Achieved 93.86% accuracy with precision of 1.0000, producing no false positives. However, it missed seven malignant samples, giving a recall of 0.8333. Its high AUC of 0.9934 indicates strong ranking performance despite lower recall at the default classification rule. |
| Support Vector Machine | Achieved the highest accuracy of 97.37%, F1 of 0.9630 and MCC of 0.9442. It produced no false positives and missed three malignant samples. It tied Logistic Regression for the highest recall. |
| Overall winner for this dataset | SVM provided the strongest overall performance on this test split, considering accuracy, F1 and MCC while maintaining the joint-highest malignant-class recall. Logistic Regression achieved the highest AUC and is also a strong alternative. |

Recall is particularly relevant because a false negative represents a malignant sample classified as benign. However, the comparison considers all six metrics rather than recall alone.

These conclusions apply to this single test split of 114 records. They do not establish that one model will always outperform the others on new populations.

## Streamlit Application Features

- Upload a labelled test dataset in CSV format.
- Download an example test CSV.
- Evaluate the bundled held-out test dataset.
- Select any of the five classifiers.
- Compare all five models on the same test records.
- Display all six evaluation metrics.
- Display a confusion matrix and classification report.
- Download evaluation results as a CSV.

The application loads saved models and preprocessing pipelines. It does not retrain models on uploaded data.

### CSV Requirements

The uploaded CSV must contain:

- All 30 input feature columns with the same names as the example CSV.
- A `target` column containing 0 for benign and 1 for malignant.
- Numeric, finite and non-missing feature values.

AUC is displayed as N/A if the uploaded data contains only one target class. Undefined precision, recall or F1 denominators are handled using zero.

## Project Files

| File or Folder | Purpose |
| --- | --- |
| classification_experiments.ipynb | Dataset inspection, model training, evaluation and verification |
| train_models.py | Reproducible training and artifact generation |
| evaluation.py | Shared evaluation metrics and test-data validation |
| app.py | Streamlit application |
| model/ | Five saved models, including preprocessing where applicable |
| metadata.json | Feature names, target mapping and model filenames |
| test_data.csv | Held-out test records with target labels |
| model_comparison.csv | Evaluation results for all five models |
| confusion_counts.csv | TN, FP, FN and TP counts for each model |
| requirements.txt | Python package dependencies |
| python_version.txt | Python version used during training |
