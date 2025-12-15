import os
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

sns.set(style="whitegrid")

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.preprocess import load_data, split_data, scale_data
from src.train_super import resample_data
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


@st.cache_resource(show_spinner=True)
def train_models():
    data_path = os.path.join(PROJECT_ROOT, "data", "creditcard.csv")
    df = load_data(data_path)

    X_train, X_test, y_train, y_test = split_data(df, label_col="Class")
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)

    X_res, y_res = resample_data(X_train_scaled, y_train.values, method="smote")
    models = {}

    log_reg = LogisticRegression(
        class_weight="balanced", max_iter=1000, n_jobs=1
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

    return df, scaler, models


def main():
    st.set_page_config(
        page_title="Credit Card Fraud Detection",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("💳 Credit Card Fraud Detection Dashboard")

    with st.sidebar:
        st.header("Settings")
        st.write("This app uses a model trained on the credit card dataset.")
        model_choice = st.selectbox(
            "Choose model",
            ["Random Forest", "Logistic Regression"],
            index=0,
        )
        st.caption("Models are trained with SMOTE to handle class imbalance.")

    with st.spinner("Training / loading models..."):
        df_train, scaler, models = train_models()

    model = models[model_choice]

    # --- Training data summary
    st.subheader("Training Data Overview")

    col1, col2 = st.columns(2)
    with col1:
        st.write("Shape:", df_train.shape)
        st.dataframe(df_train.head())

    with col2:
        class_counts = df_train["Class"].value_counts()
        class_percent = df_train["Class"].value_counts(normalize=True) * 100
        st.write("Class counts:")
        st.write(class_counts)
        st.write("Class percentages (%):")
        st.write(class_percent)

    st.markdown("---")

    # --- Training data vs
    st.subheader("Training Data Visualizations")
    
    tab1, tab2, tab3 = st.tabs(["Class Distribution", "Amounts by Class", "Top Feature Correlations"])
    
    with tab1:
        st.write("Class imbalance in the training dataset.")
        class_counts = df_train["Class"].value_counts()
        fig1, ax1 = plt.subplots()
        sns.barplot(x=class_counts.index, y=class_counts.values, ax=ax1)
        ax1.set_xticklabels(["Non-fraud (0)", "Fraud (1)"])
        ax1.set_title("Class Distribution")
        ax1.set_xlabel("Class")
        ax1.set_ylabel("Count")
        st.pyplot(fig1)

    with tab2:
        st.write("Distribution of transaction amounts split by class.")
        fig2, ax2 = plt.subplots()
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

    with tab3:
        st.write("Most correlated features with the fraud label.")
        # compute only once per session
        numeric_cols = df_train.select_dtypes(include=[np.number]).columns
        corr_with_class = df_train[numeric_cols].corr()["Class"].sort_values(ascending=False)
        top_features = corr_with_class.drop("Class").abs().sort_values(ascending=False).head(8).index.tolist()

        fig3, ax3 = plt.subplots(figsize=(6,4))
        corr_matrix = df_train[top_features + ["Class"]].corr()
        sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", ax=ax3)
        ax3.set_title("Correlation of Top Features with Class")
        st.pyplot(fig3)


    # --- File uploader for new transactions
    st.subheader("Upload Transactions for Fraud Prediction")

    st.write(
        "Upload a CSV file with the same columns as the training data. "
        "You can include `Class` or not; if present, it will be ignored."
    )

    uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            df_new = pd.read_csv(uploaded_file)

            # Drop Class if present
            if "Class" in df_new.columns:
                df_new_features = df_new.drop(columns=["Class"])
            else:
                df_new_features = df_new.copy()

            # Ensure columns align with training features
            train_feature_cols = df_train.drop(columns=["Class"]).columns
            missing_cols = set(train_feature_cols) - set(df_new_features.columns)

            if missing_cols:
                st.error(
                    f"The uploaded file is missing these required columns: {missing_cols}"
                )
            else:
                # Reorder columns to match training data
                df_new_features = df_new_features[train_feature_cols]

                # Scale using training scaler
                X_new_scaled = scaler.transform(df_new_features.values)

                # Predict
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X_new_scaled)[:, 1]
                else:
                    # Fallback: use decision function or raw prediction
                    if hasattr(model, "decision_function"):
                        raw = model.decision_function(X_new_scaled)
                        probs = (raw - raw.min()) / (raw.max() - raw.min() + 1e-8)
                    else:
                        preds = model.predict(X_new_scaled)
                        probs = preds.astype(float)

                df_results = df_new.copy()
                df_results["fraud_probability"] = probs

                # Sort by highest probability
                df_results_sorted = df_results.sort_values(
                    by="fraud_probability", ascending=False
                )

                st.success("Prediction completed.")
                st.write("Top suspicious transactions:")
                st.dataframe(df_results_sorted.head(20))

                # Simple thresholding
                threshold = st.slider(
                    "Flag transactions above probability",
                    min_value=0.0,
                    max_value=1.0,
                    value=0.5,
                    step=0.01,
                )
                flagged = df_results_sorted[df_results_sorted["fraud_probability"] >= threshold]
                st.write(f"Number of flagged transactions (p ≥ {threshold:.2f}): {len(flagged)}")
                st.dataframe(flagged.head(50))

        except Exception as e:
            st.error(f"Error reading or processing file: {e}")
    else:
        st.info("Upload a CSV file to get predictions.")


if __name__ == "__main__":
    main()