import os
import sys

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from src.preprocess import load_data, split_data, scale_data
from src.train_super import resample_data
from src.autoencoder import (
    build_autoencoder,
    train_autoencoder,
    reconstruction_errors,
    choose_threshold,
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)


sns.set(style="whitegrid")


@st.cache_resource(show_spinner=True)
def train_supervised_models():
    """Load data, preprocess it, and train baseline supervised models."""

    data_path = os.path.join(PROJECT_ROOT, "data", "creditcard.csv")
    df = load_data(data_path)

    X_train, X_test, y_train, y_test = split_data(df, label_col="Class")
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

    # Resample training data with SMOTE
    X_res, y_res = resample_data(
        X_train_scaled, y_train.values, method="smote"
    )

    models = {}

    log_reg = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        n_jobs=-1,
    )
    log_reg.fit(X_res, y_res)
    models["Logistic Regression"] = log_reg

    rf = RandomForestClassifier(
        n_estimators=300,
        n_jobs=-1,
        class_weight="balanced",
        random_state=42,
    )
    rf.fit(X_res, y_res)
    models["Random Forest"] = rf

    return df, scaler, models, X_train_scaled, X_test_scaled, y_train.values, y_test.values


@st.cache_resource(show_spinner=True)
def train_autoencoder_models(X_train_scaled, X_test_scaled, y_train, y_test):
    """
    Train an autoencoder on normal (non-fraud) samples only.
    Returns metrics + threshold + reconstruction errors.
    """

    normal_mask = y_train == 0
    X_train_normal = X_train_scaled[normal_mask]

    input_dim = X_train_scaled.shape[1]
    autoencoder, _ = build_autoencoder(input_dim, encoding_dim=14)

    autoencoder = train_autoencoder(
        autoencoder,
        X_train_normal,
        epochs=20,
        batch_size=256,
        validation_split=0.1,
    )

    # Threshold based on normal reconstruction error
    train_errors_normal = reconstruction_errors(autoencoder, X_train_normal)
    threshold = choose_threshold(train_errors_normal, quantile=0.995)

    test_errors = reconstruction_errors(autoencoder, X_test_scaled)
    y_pred = (test_errors > threshold).astype(int)

    metrics = {
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, test_errors),
        "pr_auc": average_precision_score(y_test, test_errors),
        "threshold": threshold,
    }

    return autoencoder, metrics, test_errors


def get_model_probas(model, X):
    """Return probability of class 1, or a proxy if no predict_proba."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        raw = model.decision_function(X)
        return (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
    preds = model.predict(X)
    return preds.astype(float)


def compute_basic_metrics(y_true, y_pred, y_proba):
    """Return a dict of standard evaluation metrics."""
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
    }


def threshold_curve(y_true, y_proba, thresholds):
    """Compute precision / recall / F1 for a range of thresholds."""
    precs, recs, f1s = [], [], []
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        precs.append(precision_score(y_true, y_pred, zero_division=0))
        recs.append(recall_score(y_true, y_pred, zero_division=0))
        f1s.append(f1_score(y_true, y_pred, zero_division=0))
    return np.array(precs), np.array(recs), np.array(f1s)


def main():
    st.set_page_config(
        page_title="Credit Card Fraud Detection",
        layout="wide",
        page_icon="💳",
        initial_sidebar_state="expanded",
    )

    # ---------- Custom CSS ----------
    st.markdown(
        """
        <style>
        body {
            background-color: #020517;
        }
        .main {
            background: radial-gradient(circle at top, #1e293b 0, #020617 60%);
            color: #e5e7eb;
        }
        .block-container {
            padding-top: 1.2rem;
            padding-bottom: 2rem;
        }
        h1, h2, h3, h4 {
            color: #e5e7eb !important;
        }
        .metric-card {
            padding: 1.5rem 1.25rem;
            border-radius: 0.9rem;
            border: 1px solid #1f2937;
            background: #1f2937;
            box-shadow: 0 16px 40px rgba(0,0,0,0.45);
        }
        .metric-label {
            font-size: 0.85rem;
            color: #9ca3af;
            text-transform: uppercase;
            letter-spacing: .08em;
        }
        .metric-value {
            font-size: 1.4rem;
            font-weight: 600;
            color: #e5e7eb;
        }
        .metric-caption {
            font-size: 0.8rem;
            color: #6b7280;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ---------- Sidebar ----------
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        st.write("Choose model and fraud probability threshold.")

        model_choice = st.selectbox(
            "Model",
            ["Random Forest", "Logistic Regression"],
            index=0,
        )

        threshold = st.slider(
            "Fraud probability threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.01,
            help="Transactions with probability above this will be flagged as fraud.",
        )

        st.markdown("---")
        st.caption(
            "Lower threshold → catch more fraud (higher recall). "
            "Raise threshold → fewer false positives (higher precision)."
        )

    # ---------- Header ----------
    st.markdown(
        """
        <div style="text-align:center; margin-bottom: 1.6rem;">
            <h1 style="margin-bottom: 0.3rem;">💳 Credit Card Fraud Detection</h1>
            <p style="color:#9ca3af; font-size:0.95rem;">
                End-to-end ML pipeline with class imbalance handling, autoencoder anomaly detection, and interactive analysis.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ---------- Train / load models ----------
    with st.spinner("Loading data and training supervised models..."):
        df_train, scaler, models, X_train_scaled, X_test_scaled, y_train, y_test = (
            train_supervised_models()
        )

    chosen_model = models[model_choice]

    # Precompute proba on test for selected model
    y_proba = get_model_probas(chosen_model, X_test_scaled)
    y_pred = (y_proba >= threshold).astype(int)
    metrics = compute_basic_metrics(y_test, y_pred, y_proba)

    # ---------- High-level stats ----------
    total_tx = len(df_train)
    fraud_tx = int(df_train["Class"].sum())
    fraud_rate = fraud_tx / total_tx * 100 if total_tx > 0 else 0
    n_features = df_train.shape[1] - 1

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Total transactions</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{total_tx:,}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-caption">Rows in training dataset</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Fraud rate</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{fraud_rate:.3f}%</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="metric-caption">{fraud_tx:,} frauds found in training data</div>',
            unsafe_allow_html=True,
        )
        st.markdown('</div>', unsafe_allow_html=True)

    with col_c:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.markdown('<div class="metric-label">Feature count</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="metric-value">{n_features}</div>', unsafe_allow_html=True)
        st.markdown('<div class="metric-caption">Numeric PCA-like inputs + Amount/Time</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("")

    # ---------- Top-level tabs ----------
    tab_overview, tab_metrics, tab_autoencoder, tab_predict = st.tabs(
        ["📊 Data Overview", "📈 Model Metrics & Explainability", "🧬 Autoencoder", "🧮 Predict"]
    )

    # ==================== Overview Tab ====================
    with tab_overview:
        st.markdown("### Sample of Training Data")
        st.dataframe(df_train.head(15))

        st.markdown("#### Class Balance")
        class_counts = df_train["Class"].value_counts()
        class_percent = df_train["Class"].value_counts(normalize=True) * 100

        col1, col2 = st.columns([1, 1.2])
        with col1:
            st.write("Counts:")
            st.write(class_counts)
            st.write("Percentages (%):")
            st.write(class_percent)

        with col2:
            fig, ax = plt.subplots(figsize=(4, 4))
            sns.barplot(x=class_counts.index, y=class_counts.values, ax=ax)
            ax.set_xticklabels(["Non-fraud (0)", "Fraud (1)"])
            ax.set_title("Class Distribution")
            ax.set_xlabel("Class")
            ax.set_ylabel("Count")
            st.pyplot(fig)

        st.markdown("---")

        st.markdown("### Amount Distribution by Class")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sns.histplot(
            data=df_train,
            x="Amount",
            hue="Class",
            bins=50,
            log_scale=(False, True),
            element="step",
            stat="density",
            ax=ax2,
        )
        ax2.set_title("Transaction Amount Distribution (log Y)")
        ax2.set_xlabel("Amount")
        ax2.set_ylabel("Density (log scale)")
        st.pyplot(fig2)

    # ---------- Metrics & Explainability Tab --------
    with tab_metrics:
        st.markdown(f"### {model_choice} – Evaluation on Test Set")

        colm1, colm2 = st.columns([1, 1.4])

        with colm1:
            st.markdown("#### Key Metrics (at selected threshold)")
            st.write(
                pd.DataFrame(
                    {
                        "Metric": ["Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"],
                        "Value": [
                            metrics["precision"],
                            metrics["recall"],
                            metrics["f1"],
                            metrics["roc_auc"],
                            metrics["pr_auc"],
                        ],
                    }
                ).set_index("Metric")
            )

        with colm2:
            st.markdown("#### Threshold vs Precision / Recall / F1")
            thresholds = np.linspace(0.01, 0.99, 40)
            precs, recs, f1s = threshold_curve(y_test, y_proba, thresholds)

            fig_th, ax_th = plt.subplots(figsize=(6, 4))
            ax_th.plot(thresholds, precs, label="Precision")
            ax_th.plot(thresholds, recs, label="Recall")
            ax_th.plot(thresholds, f1s, label="F1-score")
            ax_th.axvline(threshold, color="gray", linestyle="--", label="Current threshold")
            ax_th.set_xlabel("Threshold")
            ax_th.set_ylabel("Score")
            ax_th.set_title("Metric curves vs threshold")
            ax_th.legend()
            st.pyplot(fig_th)

        st.markdown("---")

        colr1, colr2 = st.columns(2)
        with colr1:
            st.markdown("#### ROC Curve")
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            fig_roc, ax_roc = plt.subplots(figsize=(4.5, 4))
            ax_roc.plot(fpr, tpr, label=f"ROC (AUC={metrics['roc_auc']:.3f})")
            ax_roc.plot([0, 1], [0, 1], "k--", label="Random")
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.legend()
            st.pyplot(fig_roc)

        with colr2:
            st.markdown("#### Precision–Recall Curve")
            prec_curve, rec_curve, _ = precision_recall_curve(y_test, y_proba)
            fig_pr, ax_pr = plt.subplots(figsize=(4.5, 4))
            ax_pr.plot(rec_curve, prec_curve, label=f"PR (AUC={metrics['pr_auc']:.3f})")
            ax_pr.set_xlabel("Recall")
            ax_pr.set_ylabel("Precision")
            ax_pr.legend()
            st.pyplot(fig_pr)

        st.markdown("---")

        st.markdown("### Feature Importance / Explainability")
        
        if model_choice == "Random Forest":
            st.markdown("Using SHAP (TreeExplainer) to estimate feature importance.")
            feature_cols = df_train.drop(columns=["Class"]).columns
            n_features = len(feature_cols)
            
            sample_idx = np.random.choice(
                len(X_test_scaled),
                size=min(500, len(X_test_scaled)),
                replace=False,
            )
            
            X_sample = X_test_scaled[sample_idx]

            # Prefer the new SHAP interface which returns an Explanation object
            try:
                explainer = shap.Explainer(chosen_model, X_train_scaled)
                shap_exp = explainer(X_sample)
                shap_arr = np.array(shap_exp.values)   # shape: (n_samples, n_features)
                mean_abs = np.mean(np.abs(shap_arr), axis=0)
            except Exception:
                # Fallback to old TreeExplainer API
                explainer = shap.TreeExplainer(chosen_model)
                shap_values = explainer.shap_values(X_sample)
                
                if isinstance(shap_values, list):
                    shap_arr = np.array(shap_values[1])  # fraud class
                else:
                    shap_arr = np.array(shap_values)
                    
                    if shap_arr.ndim == 2 and shap_arr.shape[1] == n_features:
                        mean_abs = np.mean(np.abs(shap_arr), axis=0)
                    elif shap_arr.ndim == 2 and shap_arr.shape[0] == n_features:
                        mean_abs = np.mean(np.abs(shap_arr), axis=1)
                    else:
                        st.warning(
                            "SHAP output shape still unexpected; "
                            "falling back to RandomForest feature_importances_."
                        )
                        mean_abs = chosen_model.feature_importances_
                
                importance = (
                    pd.Series(mean_abs, index=feature_cols)
                    .sort_values(ascending=False)
                    .head(15)
                )
                
                fig_imp, ax_imp = plt.subplots(figsize=(6, 4))
                importance[::-1].plot(kind="barh", ax=ax_imp)
                ax_imp.set_title("Top 15 features by mean |SHAP| (or RF importance)")
                st.pyplot(fig_imp)
        else:
            st.markdown(
                "Feature importance visualization is only implemented for the "
                "**Random Forest** model. Select it in the sidebar to see SHAP values."
            )


    # --------- Autoencoder Tab ---------
    with tab_autoencoder:
        st.markdown("### Autoencoder Anomaly Detector")

        with st.spinner("Training autoencoder on non-fraud transactions..."):
            autoencoder, ae_metrics, test_errors = train_autoencoder_models(
                X_train_scaled, X_test_scaled, y_train, y_test
            )

        st.markdown("#### Performance on Test Set")
        st.write(
            pd.DataFrame(
                {
                    "Metric": ["Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "Threshold"],
                    "Value": [
                        ae_metrics["precision"],
                        ae_metrics["recall"],
                        ae_metrics["f1"],
                        ae_metrics["roc_auc"],
                        ae_metrics["pr_auc"],
                        ae_metrics["threshold"],
                    ],
                }
            ).set_index("Metric")
        )

        st.markdown("#### Reconstruction Error Distribution")
        fig_err, ax_err = plt.subplots(figsize=(6, 4))
        sns.histplot(
            test_errors[y_test == 0],
            bins=50,
            color="tab:blue",
            stat="density",
            label="Non-fraud",
            ax=ax_err,
        )
        sns.histplot(
            test_errors[y_test == 1],
            bins=50,
            color="tab:red",
            stat="density",
            label="Fraud",
            ax=ax_err,
        )
        ax_err.axvline(ae_metrics["threshold"], color="white", linestyle="--", label="Threshold")
        ax_err.set_xlabel("Reconstruction error")
        ax_err.set_ylabel("Density")
        ax_err.set_title("Autoencoder reconstruction errors on test set")
        ax_err.legend()
        st.pyplot(fig_err)

        st.caption(
            "Fraudulent transactions should have higher reconstruction error, "
            "since the autoencoder only learned to reconstruct normal patterns."
        )

    # ==================== Predict Tab ====================
    with tab_predict:
        st.markdown("### Upload Transactions for Prediction")

        st.write(
            "Upload a CSV file with the same feature columns as the training data. "
            "If a `Class` column is present, it will be ignored."
        )

        uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])

        if uploaded_file is not None:
            try:
                df_new = pd.read_csv(uploaded_file)

                if "Class" in df_new.columns:
                    df_features = df_new.drop(columns=["Class"])
                else:
                    df_features = df_new.copy()

                feature_cols = df_train.drop(columns=["Class"]).columns
                missing_cols = set(feature_cols) - set(df_features.columns)
                if missing_cols:
                    st.error(f"Uploaded file is missing required columns: {missing_cols}")
                else:
                    df_features = df_features[feature_cols]
                    X_new_scaled = scaler.transform(df_features.values)

                    proba_new = get_model_probas(chosen_model, X_new_scaled)
                    df_results = df_new.copy()
                    df_results["fraud_probability"] = proba_new

                    df_sorted = df_results.sort_values(
                        by="fraud_probability", ascending=False
                    )
                    flagged = df_sorted[df_sorted["fraud_probability"] >= threshold]

                    st.success("Prediction completed.")
                    st.write(
                        f"**Flagged {len(flagged)} / {len(df_sorted)} "
                        f"transactions with probability ≥ {threshold:.2f}.**"
                    )

                    st.markdown("#### Top 20 most suspicious transactions")
                    st.dataframe(df_sorted.head(20))

                    st.markdown("#### Flagged transactions (above threshold)")
                    st.dataframe(flagged.head(100))

                    # download button
                    csv_flagged = flagged.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        label="💾 Download flagged transactions as CSV",
                        data=csv_flagged,
                        file_name="flagged_transactions.csv",
                        mime="text/csv",
                    )

            except Exception as e:
                st.error(f"Error reading or processing file: {e}")
        else:
            st.info("Upload a CSV file to run predictions.")


if __name__ == "__main__":
    main()
