import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
import pandas as pd

from sklearn.dummy import DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from config import (
    RANDOM_STATE, MODEL_DIR, REPORT_DIR, FIGURE_DIR, FEATURES
)
from data_utils import load_and_clean_data, split_data
from evaluate import get_scores, save_confusion_matrix, save_roc_pr_curves

import matplotlib.pyplot as plt


def main():
    df, info = load_and_clean_data()
    X_train, X_test, y_train, y_test = split_data(df)

    # 1. Majority-class baseline
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)

    models = {
        "Majority Baseline": baseline,
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))
        ]),
        "Logistic Regression Balanced": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=RANDOM_STATE
            ))
        ]),
        "Logistic Regression SMOTE": ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))
        ]),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=4,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),
        "Neural Network": ImbPipeline([
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("scaler", StandardScaler()),
            ("model", MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                learning_rate_init=1e-3,
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=30,
                random_state=RANDOM_STATE
            ))
        ]),
    }

    fitted = {}
    rows = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted[name] = model

        scores = get_scores(model, X_test, y_test)
        scores["model"] = name
        rows.append(scores)

        safe_name = name.lower().replace(" ", "_")
        joblib.dump(model, MODEL_DIR / f"{safe_name}.joblib")
        save_confusion_matrix(model, X_test, y_test, name, FIGURE_DIR)

    results = pd.DataFrame(rows)
    cols = [
        "model", "accuracy", "precision", "recall", "f1",
        "specificity", "roc_auc", "pr_auc", "tn", "fp", "fn", "tp"
    ]
    results = results[cols]
    results.to_csv(REPORT_DIR / "model_results.csv", index=False)

    save_roc_pr_curves(fitted, y_test, X_test, FIGURE_DIR)

    # Save a compact dataset audit.
    audit = {
        **info,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "train_positive": int(y_train.sum()),
        "train_negative": int((y_train == 0).sum()),
        "test_positive": int(y_test.sum()),
        "test_negative": int((y_test == 0).sum()),
    }
    pd.DataFrame([audit]).to_csv(REPORT_DIR / "dataset_audit.csv", index=False)

    # Decision tree plot.
    tree = fitted["Decision Tree"]
    plt.figure(figsize=(20, 10))
    plot_tree(
        tree,
        feature_names=FEATURES,
        class_names=["No Disease", "Disease"],
        filled=True,
        rounded=True,
        fontsize=8,
    )
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "decision_tree.png", dpi=200)
    plt.close()

    print("\nDataset audit:")
    print(audit)
    print("\nModel results:")
    print(results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
