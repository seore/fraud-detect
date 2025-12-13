import numpy as np

from typing import Dict, Tuple
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from .evaluation import evaluate_classifier, print_evaluation_report


def get_base_models(random_state: int = 42) -> Dict[str, object]:
    models = {
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
        "XGBoost": XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
        ),
    }
    return models


def resample_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    method: str = "none",
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    method = method.lower()
    if method == "none":
        return X_train, y_train

    if method == "smote":
        sampler = SMOTE(random_state=random_state)
    elif method == "smote_tomek":
        sampler = SMOTETomek(random_state=random_state)
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
    X_res, y_res = resample_data(X_train, y_train, method=resampling)
    models = get_base_models()
    results = {}

    for name, model in models.items():
        print(f"\n--- Training {name} with resampling='{resampling}' ---")
        model.fit(X_res, y_res)

        y_pred = model.predict(X_test)
        try:
            y_proba = model.predict_proba(X_test)
        except AttributeError:
            # Some models may not have predict_proba
            y_proba = None

        metrics = evaluate_classifier(y_test, y_pred, y_proba)
        results[name] = metrics

        if verbose:
            print_evaluation_report(y_test, y_pred, y_proba, model_name=f"{name} ({resampling})")

    return results
