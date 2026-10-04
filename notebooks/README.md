# Notebook workflow

The project is intentionally implemented as reproducible Python scripts rather than requiring a large notebook.

Recommended notebook order if you want to create notebooks for a college demonstration:

1. `01_data_exploration.ipynb`
   - Load CSV
   - Inspect shape, columns, dtypes
   - Missing values
   - Duplicate analysis
   - Target distribution
   - Descriptive statistics
   - Correlation heatmap

2. `02_preprocessing_and_imbalance.ipynb`
   - Deduplicate
   - Stratified train/test split
   - Baseline
   - Class-weighted model
   - SMOTE

3. `03_model_comparison.ipynb`
   - Logistic regression
   - Decision tree
   - Random forest
   - Neural network
   - Metrics and confusion matrices

4. `04_explainability.ipynb`
   - Decision tree visualization
   - Feature importance
   - Permutation importance
   - SHAP
