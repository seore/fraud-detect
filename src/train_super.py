import numpy as np
from typing import Dict, Tuple

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from .evaluation import evaluate_classifier, print_evaluation_report

# Try optional XGBoost
try:
    from xgboost import XGBClassifier  # type: ignore
    HAS_XGBOOST = True
except Exception:
    XGBClassifier = None  # type: ignore
    HAS_XGBOOST = False
    print(
        "[WARN] XGBoost could not be imported. "
        "XGBoost models will be skipped."
    )


def get_base_models(random_state: int = 42) -> Dict[str, object]:
    """
    Returns a dictionary of supervised classification models.
    XGBoost is included only if available.
    """
    models: Dict[str, object] = {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            n_jobs=-1,
            class_weight="balanced",
            random_state=random_state,
        ),
    }

    if HAS_XGBOOST and XGBClassifier is not None:
        models["XGBoost"] = XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
        )

    return models


def resample_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    method: str = "none",
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies resampling to handle class imbalance.

    method: "none", "smote", "smote_tomek", "undersample"

    For SMOTE-based methods, we adapt k_neighbors to the number
    of available minority samples to avoid errors on very small datasets.
    """
    method = method.lower()
    if method == "none":
        return X_train, y_train

    # Count classes
    unique, counts = np.unique(y_train, return_counts=True)
    class_counts = dict(zip(unique, counts))
    minority_class = min(class_counts, key=class_counts.get)
    n_minority = class_counts[minority_class]

    if method in ("smote", "smote_tomek"):
        # Need at least 2 samples to attempt SMOTE
        if n_minority < 2:
            print(
                f"[WARN] Not enough minority samples ({n_minority}) for {method}. "
                "Skipping resampling."
            )
            return X_train, y_train

        # k_neighbors must be < n_minority
        k = min(5, n_minority - 1)
        if k < 1:
            print(
                f"[WARN] Not enough minority samples ({n_minority}) for {method} "
                f"even with adaptive k_neighbors. Skipping resampling."
            )
            return X_train, y_train

        print(f"[INFO] Using SMOTE with k_neighbors={k} (minority samples: {n_minority})")

        if method == "smote":
            sampler = SMOTE(random_state=random_state, k_neighbors=k)
        else:  # smote_tomek
            sampler = SMOTETomek(
                random_state=random_state,
                smote=SMOTE(random_state=random_state, k_neighbors=k),
            )

    elif method == "undersample":
        sampler = RandomUnderSampler(random_state=random_state)

    else:
        raise ValueError(f"Unknown resampling method: {method}")

    X_res, y_res = sampler.fit_resample(X_train, y_train)
    return X_res, y_res


def train_and_evaluate_supervised(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    resampling: str = "none",
    verbose: bool = True,
) -> Dict[str, Dict[str, float]]:
    """
    Trains all base models on (possibly resampled) data and evaluates them.

    Returns
    -------
    dict: {model_name: {metric_name: metric_value}}
    """
    X_res, y_res = resample_data(X_train, y_train, method=resampling)
    models = get_base_models()
    results: Dict[str, Dict[str, float]] = {}

    for name, model in models.items():
        print(f"\n--- Training {name} with resampling='{resampling}' ---")
        model.fit(X_res, y_res)

        y_pred = model.predict(X_test)
        try:
            y_proba = model.predict_proba(X_test)
        except AttributeError:
            y_proba = None

        metrics = evaluate_classifier(y_test, y_pred, y_proba)
        results[name] = metrics

        if verbose:
            print_evaluation_report(
                y_test, y_pred, y_proba, model_name=f"{name} ({resampling})"
            )

    return results
