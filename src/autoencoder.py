from typing import Dict, Tuple

import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping

from .evaluation import evaluate_classifier, print_evaluation_report


def build_autoencoder(input_dim: int, encoding_dim: int = 14) -> Tuple[Model, Model]:
    """
    Builds a simple symmetric dense autoencoder.

    Returns
    -------
    autoencoder, encoder
    """
    input_layer = Input(shape=(input_dim,))

    # Encoder
    encoded = Dense(encoding_dim, activation="relu")(input_layer)
    encoded = Dense(encoding_dim // 2, activation="relu")(encoded)

    # Decoder
    decoded = Dense(encoding_dim, activation="relu")(encoded)
    decoded = Dense(input_dim, activation="linear")(decoded)

    autoencoder = Model(inputs=input_layer, outputs=decoded)
    encoder = Model(inputs=input_layer, outputs=encoded)

    autoencoder.compile(optimizer="adam", loss="mse")
    return autoencoder, encoder


def train_autoencoder(
    autoencoder: Model,
    X_train_normal: np.ndarray,
    epochs: int = 20,
    batch_size: int = 256,
    validation_split: float = 0.1,
) -> Model:
    """
    Trains the autoencoder only on non-fraud (normal) transactions.
    """
    es = EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    )

    autoencoder.fit(
        X_train_normal,
        X_train_normal,
        epochs=epochs,
        batch_size=batch_size,
        shuffle=True,
        validation_split=validation_split,
        callbacks=[es],
        verbose=1,
    )
    return autoencoder


def reconstruction_errors(autoencoder: Model, X: np.ndarray) -> np.ndarray:
    recon = autoencoder.predict(X, verbose=0)
    errors = np.mean(np.square(X - recon), axis=1)
    return errors


def choose_threshold(errors_normal: np.ndarray, quantile: float = 0.995) -> float:
    threshold = np.quantile(errors_normal, quantile)
    return float(threshold)


def predict_autoencoder(
    autoencoder: Model,
    X: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """
    Predicts fraud labels using reconstruction error.
    error > threshold -> 1 (fraud)
    else -> 0
    """
    errors = reconstruction_errors(autoencoder, X)
    return np.where(errors > threshold, 1, 0), errors


def run_autoencoder_pipeline(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    encoding_dim: int = 14,
    quantile: float = 0.995,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Full pipeline:
    - Train autoencoder only on normal class from X_train
    - Choose threshold from reconstruction errors on train normals
    - Predict on test set and evaluate
    """
    # Use only non-fraud samples from training for unsupervised learning
    normal_mask = y_train == 0
    X_train_normal = X_train[normal_mask]

    input_dim = X_train.shape[1]
    autoencoder, _ = build_autoencoder(input_dim, encoding_dim=encoding_dim)

    print("\n=== Training Autoencoder on normal transactions only ===")
    train_autoencoder(autoencoder, X_train_normal)

    # Compute threshold
    train_errors_normal = reconstruction_errors(autoencoder, X_train_normal)
    threshold = choose_threshold(train_errors_normal, quantile=quantile)
    print(f"Chosen reconstruction error threshold: {threshold:.6f}")

    # Predict on test set
    y_pred, test_errors = predict_autoencoder(autoencoder, X_test, threshold)
    metrics = evaluate_classifier(y_test, y_pred)
    if verbose:
        print_evaluation_report(y_test, y_pred, None, model_name="Autoencoder Anomaly Detector")

    # Add threshold to metrics for reference
    metrics["threshold"] = threshold
    return metrics
