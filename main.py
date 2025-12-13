import numpy as np

from pprint import pprint
from src.preprocess import load_data, split_data, scale_data
from src.train_super import train_and_evaluate_supervised
from src.anomaly_method import run_anomaly_pipeline
from src.autoencoder import run_autoencoder_pipeline


def main():
    # 1. Load data
    print("Loading data...")
    df = load_data("data/creditcard.csv")
    print(df.head())
    print(df["Class"].value_counts(normalize=True) * 100)

    # 2. Train/Test split
    print("\nSplitting data into train and test...")
    X_train, X_test, y_train, y_test = split_data(df, label_col="Class")

    # 3. Scaling
    print("Scaling features...")
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

    # Convert to numpy arrays (for imblearn / keras compatibility)
    X_train_scaled = np.asarray(X_train_scaled, dtype=float)
    X_test_scaled = np.asarray(X_test_scaled, dtype=float)
    y_train = np.asarray(y_train)
    y_test = np.asarray(y_test)

    # 4. Supervised models WITHOUT resampling
    print("\n===============================================")
    print("Supervised models WITHOUT resampling")
    print("===============================================")
    sup_results_no_resampling = train_and_evaluate_supervised(
        X_train_scaled, X_test_scaled, y_train, y_test, resampling="none"
    )

    # 5. Supervised models WITH SMOTE
    print("\n===============================================")
    print("Supervised models WITH SMOTE")
    print("===============================================")
    sup_results_smote = train_and_evaluate_supervised(
        X_train_scaled, X_test_scaled, y_train, y_test, resampling="smote"
    )

    # 6. Supervised models WITH SMOTE+Tomek
    print("\n===============================================")
    print("Supervised models WITH SMOTE+Tomek")
    print("===============================================")
    sup_results_smote_tomek = train_and_evaluate_supervised(
        X_train_scaled, X_test_scaled, y_train, y_test, resampling="smote_tomek"
    )

    # 7. Anomaly detection methods
    print("\n===============================================")
    print("Anomaly Detection Methods (IsolationForest, OneClassSVM)")
    print("===============================================")
    contamination = (y_train.sum() + y_test.sum()) / (len(y_train) + len(y_test))
    anomaly_results = run_anomaly_pipeline(
        X_train_scaled, X_test_scaled, y_test, contamination=contamination
    )

    # 8. Autoencoder anomaly detection
    print("\n===============================================")
    print("Autoencoder Anomaly Detection")
    print("===============================================")
    autoencoder_results = run_autoencoder_pipeline(
        X_train_scaled, X_test_scaled, y_train, y_test, encoding_dim=14, quantile=0.995
    )

    # 9. Print summary
    print("\n\n=================== SUMMARY ===================")
    print("\nSupervised (No resampling):")
    pprint(sup_results_no_resampling)

    print("\nSupervised (SMOTE):")
    pprint(sup_results_smote)

    print("\nSupervised (SMOTE+Tomek):")
    pprint(sup_results_smote_tomek)

    print("\nAnomaly Methods:")
    pprint(anomaly_results)

    print("\nAutoencoder:")
    pprint(autoencoder_results)


if __name__ == "__main__":
    main()