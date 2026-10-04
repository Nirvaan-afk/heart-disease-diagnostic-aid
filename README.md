# Explainable and Imbalance-Aware Machine Learning Framework for Heart Disease Risk Prediction

## Problem Statement

> Develop a diagnostic aid for disease prediction. Address dataset imbalances and compare interpretable models like decision trees with more complex neural networks, focusing on both accuracy and explainability.

## Project Overview

This project develops a machine-learning-based research prototype for heart disease risk prediction.

The project focuses on three main objectives:

1. Develop a diagnostic aid for heart disease prediction.
2. Address class imbalance using class weighting and SMOTE.
3. Compare interpretable models such as Decision Trees with more complex models such as Neural Networks, focusing on both predictive performance and explainability.

The project evaluates Logistic Regression, Decision Tree, Random Forest, and Neural Network models. Explainability techniques including feature importance, SHAP, and permutation importance are used to understand model behavior.

> **Disclaimer:** This project is intended for academic, educational, and research purposes only. It is not a clinical diagnostic system and should not be used to make medical decisions.

## Dataset

The project uses the supplied `heart.csv` dataset.

### Dataset Summary

- Original rows: 1,025
- Columns: 14
- Input features: 13
- Target variable: `target`
- Missing values: None
- Exact duplicate rows: 723
- Unique observations after duplicate removal: 302

The dataset contains the following input features:

| Feature | Description |
|---|---|
| `age` | Age of the patient |
| `sex` | Sex |
| `cp` | Chest pain type |
| `trestbps` | Resting blood pressure |
| `chol` | Serum cholesterol |
| `fbs` | Fasting blood sugar |
| `restecg` | Resting electrocardiographic results |
| `thalach` | Maximum heart rate achieved |
| `exang` | Exercise-induced angina |
| `oldpeak` | ST depression induced by exercise |
| `slope` | Slope of the peak exercise ST segment |
| `ca` | Number of major vessels |
| `thal` | Thalassemia-related categorical feature |

The target variable represents the binary heart-disease classification.

The dataset used in this project was obtained from Kaggle:

https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

## Methodology

The overall machine-learning pipeline is:

```text
Heart Disease Dataset
        ↓
Data Auditing
        ↓
Duplicate Removal
        ↓
Feature / Target Separation
        ↓
Stratified Train-Test Split
        ↓
Class Imbalance Handling
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Explainability Analysis
        ↓
Streamlit Application

