from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)


def get_scores(model, X_test, y_test):
    predictions = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:, 1]
    else:
        probabilities = model.decision_function(X_test)

    tn, fp, fn, tp = confusion_matrix(y_test, predictions).ravel()

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(y_test, predictions, zero_division=0),
        "recall": recall_score(y_test, predictions, zero_division=0),
        "f1": f1_score(y_test, predictions, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) else 0.0,
        "roc_auc": roc_auc_score(y_test, probabilities),
        "pr_auc": average_precision_score(y_test, probabilities),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def save_confusion_matrix(model, X_test, y_test, name, out_dir):
    pred = model.predict(X_test)
    cm = confusion_matrix(y_test, pred)

    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Disease", "Disease"],
        yticklabels=["No Disease", "Disease"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {name}")
    plt.tight_layout()
    plt.savefig(Path(out_dir) / f"{name.lower().replace(' ', '_')}_confusion_matrix.png", dpi=200)
    plt.close()


def save_roc_pr_curves(results, y_test, X_test, out_dir):
    plt.figure(figsize=(7, 5))
    for name, model in results.items():
        if hasattr(model, "predict_proba"):
            score = model.predict_proba(X_test)[:, 1]
        else:
            score = model.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test, score)
        auc = roc_auc_score(y_test, score)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "roc_curves.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 5))
    for name, model in results.items():
        if hasattr(model, "predict_proba"):
            score = model.predict_proba(X_test)[:, 1]
        else:
            score = model.decision_function(X_test)
        precision, recall, _ = precision_recall_curve(y_test, score)
        ap = average_precision_score(y_test, score)
        plt.plot(recall, precision, label=f"{name} (AP={ap:.3f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(Path(out_dir) / "precision_recall_curves.png", dpi=200)
    plt.close()
