import numpy as np

from typing import Dict
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

from .evaluation import evaluate_classifier, print_evaluation_report


def fit_isolation_forest(
    X_train: np.ndarray,
    contamination: float = 0.0017,
    random_state: int = 42,
) -> IsolationForest:
    iso = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    iso.fit(X_train)
    return iso


def predict_isolation_forest(model: IsolationForest, X: np.ndarray) -> np.ndarray:
    """
    Converts Isolation Forest output to fraud labels.
    -1 -> anomaly -> 1 (fraud)
     1 -> normal  -> 0 (non-fraud)
    """
    raw_pred = model.predict(X)
    return np.where(raw_pred == -1, 1, 0)


def fit_one_class_svm(
    X_train: np.ndarray,
    nu: float = 0.001,
    kernel: str = "rbf",
    gamma: str = "scale",
) -> OneClassSVM:
    ocsvm = OneClassSVM(nu=nu, kernel=kernel, gamma=gamma)
    ocsvm.fit(X_train)
    return ocsvm


def predict_one_class_svm(model: OneClassSVM, X: np.ndarray) -> np.ndarray:
    """
    Converts One-Class SVM output to fraud labels.
    -1 -> anomaly -> 1
     1 -> normal  -> 0
    """
    raw_pred = model.predict(X)
    return np.where(raw_pred == -1, 1, 0)


def run_anomaly_pipeline(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    contamination: float = 0.0017,
    verbose: bool = True,
) -> Dict[str, Dict[str, float]]:
    results = {}

    # Isolation Forest
    print("\n=== Isolation Forest ===")
    iso = fit_isolation_forest(X_train, contamination=contamination)
    y_pred_iso = predict_isolation_forest(iso, X_test)
    metrics_iso = evaluate_classifier(y_test, y_pred_iso)
    results["IsolationForest"] = metrics_iso
    if verbose:
        print_evaluation_report(y_test, y_pred_iso, None, model_name="Isolation Forest")

    # One-Class SVM (optional, can be slow)
    print("\n=== One-Class SVM (may be slow on full dataset) ===")
    ocsvm = fit_one_class_svm(X_train)
    y_pred_svm = predict_one_class_svm(ocsvm, X_test)
    metrics_svm = evaluate_classifier(y_test, y_pred_svm)
    results["OneClassSVM"] = metrics_svm
    if verbose:
        print_evaluation_report(y_test, y_pred_svm, None, model_name="One-Class SVM")

    return results
