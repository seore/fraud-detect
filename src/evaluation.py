import numpy as np

from typing import Dict, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)


def evaluate_classifier(
    y_true,
    y_pred,
    y_proba: Optional[np.ndarray] = None,
    pos_label: int = 1,
) -> Dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
        "recall": recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
        "f1": f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0),
    }

    if y_proba is not None:
        # y_proba can be probability of positive class or a 2D array
        if y_proba.ndim == 2:
            proba_pos = y_proba[:, 1]
        else:
            proba_pos = y_proba

        try:
            metrics["roc_auc"] = roc_auc_score(y_true, proba_pos)
        except ValueError:
            metrics["roc_auc"] = float("nan")

        try:
            metrics["pr_auc"] = average_precision_score(y_true, proba_pos)
        except ValueError:
            metrics["pr_auc"] = float("nan")

    return metrics


def print_evaluation_report(
    y_true,
    y_pred,
    y_proba: Optional[np.ndarray] = None,
    model_name: str = "Model",
):
    print(f"\n===== {model_name} Evaluation =====")
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))

    if y_proba is not None:
        if y_proba.ndim == 2:
            proba_pos = y_proba[:, 1]
        else:
            proba_pos = y_proba

        try:
            roc = roc_auc_score(y_true, proba_pos)
            pr = average_precision_score(y_true, proba_pos)
            print(f"ROC-AUC: {roc:.4f}")
            print(f"PR-AUC:  {pr:.4f}")
        except ValueError:
            pass
