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
```

## Machine Learning Models

The following models were evaluated.

### Logistic Regression

Logistic Regression was used as an interpretable baseline classification model.

It estimates the probability of a sample belonging to the positive class.

### Balanced Logistic Regression

A class-weighted version of Logistic Regression was evaluated to investigate the effect of explicitly accounting for class imbalance.

### Logistic Regression with SMOTE

SMOTE was applied to the training data before training Logistic Regression.

The preprocessing and model were combined in a pipeline.

### Decision Tree

A single Decision Tree was used as an interpretable rule-based model.

The tree was configured with:

```text
Maximum depth = 4
```
Decision Trees are useful for explainability because their predictions can be represented through a sequence of decision rules.

### Random Forest

A Random Forest was used as an ensemble learning model.

The Random Forest contained:

```text
300 decision trees

```

It combines predictions from multiple decision trees to produce the final prediction.

### Neural Network

A Multi-Layer Perceptron (MLP) neural network was used as the more complex model in the comparison.

The architecture was:

```text
13 Input Features
       ↓
64 Neurons
       ↓
32 Neurons
       ↓
1 Output
```

The neural network used:

- ReLU activation
- Adam optimizer
- Regularization
- Early stopping
- Standardized input features
- SMOTE on the training data

## Model Evaluation

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Specificity
- ROC-AUC
- PR-AUC

### Results

| Model | Accuracy | Precision | Recall | F1 | Specificity | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| Majority Baseline | 54.10% | 54.10% | 100.00% | 70.21% | 0.00% | 0.5000 | 0.5410 |
| Logistic Regression | 80.33% | 80.00% | 84.85% | 82.35% | 75.00% | 0.8712 | 0.8954 |
| Balanced Logistic Regression | 78.69% | 79.41% | 81.82% | 80.60% | 75.00% | 0.8734 | 0.8972 |
| Logistic Regression + SMOTE | **81.97%** | **84.38%** | 81.82% | **83.08%** | **82.14%** | 0.8810 | 0.9007 |
| Decision Tree | 72.13% | 75.00% | 72.73% | 73.85% | 71.43% | 0.8328 | 0.8198 |
| Random Forest | 77.05% | 78.79% | 78.79% | 78.79% | 75.00% | 0.8745 | 0.8989 |
| Neural Network | **81.97%** | **84.38%** | 81.82% | **83.08%** | **82.14%** | **0.8864** | **0.9124** |

### Main Result

The **Neural Network achieved the highest ROC-AUC of 0.8864** among the evaluated models.

Both Logistic Regression with SMOTE and the Neural Network achieved the highest test accuracy of **81.97%**.

The results are based on a single stratified held-out test split and should not be interpreted as clinical validation.

## Explainability

Explainability is a major component of the project because the original problem requires consideration of both predictive performance and model interpretability.

### Decision Tree Feature Importance

The most influential Decision Tree features included:

```text
cp       0.468613
ca       0.153171
thal     0.111776
oldpeak  0.064609
exang    0.060984
sex      0.051363
restecg  0.038528
chol     0.022373
thalach  0.021700
slope    0.006882
```

### Random Forest Feature Importance

The most influential Random Forest features included:

```text
cp       0.163001
thalach  0.123861
ca       0.113423
thal     0.098275
oldpeak  0.091120
age      0.082666
chol     0.075830
exang    0.073684
trestbps 0.068493
sex      0.039406
```

### SHAP

SHAP stands for **SHapley Additive exPlanations**.

SHAP was used to analyze the contribution of individual features to Random Forest predictions.

It provides a way to investigate how features influence model outputs and helps make complex machine-learning models more interpretable.

### Neural Network Explainability

Permutation importance was used to investigate feature contribution for the Neural Network.

Important features included:

```text
thal
sex
ca
oldpeak
thalach
exang
age
fbs
slope
cp
```

Feature importance represents model-specific predictive contribution. It does not establish medical causation.

## Application

A **Streamlit-based interface** was developed to provide an interactive research and educational demonstration of the trained model.

The application allows users to enter patient-related feature values and obtain:

- A predicted class
- Prediction probability
- An interactive interface for demonstrating the model

The application is intended only as a demonstration of the machine-learning pipeline.

## Project Structure

```text
heart-disease-diagnostic-aid/
│
├── app/
│   └── app.py
│
├── data/
│   └── heart.csv
│
├── models/
│
├── notebooks/
│
├── reports/
│   ├── figures/
│   ├── model_results.csv
│   ├── dataset_audit.csv
│   ├── RESULTS.md
│   └── shap_feature_importance.csv
│
├── src/
│   ├── config.py
│   ├── data_utils.py
│   ├── evaluate.py
│   ├── explain.py
│   └── train.py
│
├── .gitignore
├── LICENSE
├── README.md
├── requirements.txt
└── run_project.py
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Nirvaan-afk/heart-disease-diagnostic-aid.git
```

### 2. Enter the Project Directory

```bash
cd heart-disease-diagnostic-aid
```

### 3. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

## Training the Models

Run:

```bash
python src/train.py
```

This trains the machine-learning models and generates the corresponding model and evaluation outputs.

## Running the Streamlit Application

Run:

```bash
python -m streamlit run app/app.py
```

The Streamlit application will open in your browser.

## Key Findings

- The original dataset contained **1,025 rows and 14 columns**.
- **723 exact duplicate rows** were identified and removed.
- The resulting dataset contained **302 unique observations**.
- The final class distribution was **54.3% positive and 45.7% negative**.
- The Decision Tree provided an interpretable rule-based model.
- The Random Forest provided nonlinear ensemble learning using 300 trees.
- SMOTE was used to address class imbalance on the training data.
- Logistic Regression with SMOTE achieved **81.97% accuracy**.
- The Neural Network achieved **81.97% accuracy**.
- The Neural Network achieved the highest ROC-AUC of **0.8864**.
- SHAP, feature importance, and permutation importance were used to improve model interpretability.

## Limitations

This project has several limitations:

1. The effective dataset after duplicate removal contains only 302 unique observations.
2. Model evaluation is based on one held-out stratified test split.
3. The dataset is not sufficient to establish clinical effectiveness.
4. Feature importance should not be interpreted as medical causation.
5. External clinical validation was not performed.
6. The Streamlit application is a research and educational prototype rather than a medical diagnostic device.

## Future Work

Future improvements could include:

- Evaluation on larger and independent datasets
- Cross-validation and repeated experiments
- Hyperparameter optimization
- Probability calibration
- Additional explainability techniques
- External validation using independent patient populations
- More extensive clinical evaluation
- Testing on clinically validated datasets

## Authors

**Nirvaan Sinha**  
Department of Electronics and Communication Engineering  
Bharati Vidyapeeth, Pune, India

**Sam Mendonca**  
Department of Electronics and Communication Engineering  
Bharati Vidyapeeth, Pune, India

## License

The source code in this repository is licensed under the **MIT License**.

The dataset remains subject to its original source's terms and conditions.

## Disclaimer

This project is developed for **academic, educational, and research purposes**.

It is **not intended to provide medical diagnosis, treatment recommendations, or clinical decision-making**.

The predictions produced by this system should not be used as a substitute for professional medical advice or clinical evaluation.









