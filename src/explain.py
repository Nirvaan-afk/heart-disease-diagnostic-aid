import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance

from config import DATA_PATH, MODEL_DIR, FIGURE_DIR, FEATURES
from data_utils import load_and_clean_data, split_data


def main():
    df, _ = load_and_clean_data(DATA_PATH)
    X_train, X_test, y_train, y_test = split_data(df)

    # Tree feature importance
    tree = joblib.load(MODEL_DIR / "decision_tree.joblib")
    tree_importance = pd.Series(
        tree.feature_importances_, index=FEATURES
    ).sort_values(ascending=False)

    plt.figure(figsize=(8, 5))
    tree_importance.head(10).sort_values().plot(kind="barh")
    plt.xlabel("Importance")
    plt.title("Decision Tree Feature Importance")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "decision_tree_feature_importance.png", dpi=200)
    plt.close()

    # Random forest feature importance
    rf = joblib.load(MODEL_DIR / "random_forest.joblib")
    rf_importance = pd.Series(
        rf.feature_importances_, index=FEATURES
    ).sort_values(ascending=False)

    plt.figure(figsize=(8, 5))
    rf_importance.head(10).sort_values().plot(kind="barh")
    plt.xlabel("Importance")
    plt.title("Random Forest Feature Importance")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "random_forest_feature_importance.png", dpi=200)
    plt.close()

    # Model-agnostic permutation importance for the neural network.
    nn = joblib.load(MODEL_DIR / "neural_network.joblib")
    perm = permutation_importance(
        nn, X_test, y_test,
        n_repeats=20,
        random_state=42,
        scoring="roc_auc"
    )
    perm_importance = pd.Series(
        perm.importances_mean, index=FEATURES
    ).sort_values(ascending=False)

    plt.figure(figsize=(8, 5))
    perm_importance.head(10).sort_values().plot(kind="barh")
    plt.xlabel("Mean decrease in ROC-AUC")
    plt.title("Neural Network Permutation Importance")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "neural_network_permutation_importance.png", dpi=200)
    plt.close()

    # Optional SHAP: use the Random Forest because TreeExplainer is efficient.
    try:
        import shap

        background = X_train.sample(min(100, len(X_train)), random_state=42)
        sample = X_test.sample(min(80, len(X_test)), random_state=42)

        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(sample)

        # SHAP API differs between versions.
        if isinstance(shap_values, list):
            values = shap_values[1]
        else:
            values = shap_values
            if values.ndim == 3:
                values = values[:, :, 1]

        plt.figure()
        shap.summary_plot(values, sample, show=False)
        plt.tight_layout()
        plt.savefig(FIGURE_DIR / "random_forest_shap_summary.png", dpi=200, bbox_inches="tight")
        plt.close()

        pd.DataFrame({
            "feature": FEATURES,
            "mean_abs_shap": np.abs(values).mean(axis=0)
        }).sort_values("mean_abs_shap", ascending=False).to_csv(
            ROOT / "reports" / "shap_feature_importance.csv", index=False
        )

        print("SHAP summary generated successfully.")
    except Exception as exc:
        print("SHAP output was skipped:", exc)

    print("\nTop Decision Tree features:")
    print(tree_importance.head(10))
    print("\nTop Random Forest features:")
    print(rf_importance.head(10))
    print("\nTop Neural Network permutation features:")
    print(perm_importance.head(10))


if __name__ == "__main__":
    main()
