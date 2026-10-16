"""Streamlit user interface for the IEEE-CIS fraud-detection capstone."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import sklearn

from inference import InferenceInputError, preprocess_raw_inputs


ROOT = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT / "artefacts"
MODEL_PATH = ARTIFACT_DIR / "rf_fraud_pipeline.joblib"
FREQUENCY_MAPS_PATH = ARTIFACT_DIR / "frequency_maps.joblib"
METADATA_PATH = ARTIFACT_DIR / "model_metadata.json"
MAX_ROWS = 100_000

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="🔎",
    layout="wide",
)

st.title("Credit Card Fraud Detection")
st.caption("TechCrush AI/ML Cohort 8 · Group 17 · IEEE-CIS dataset")
st.write(
    "Upload raw transaction data and its matching identity data. The application "
    "performs the feature engineering and training-fitted encoding before scoring."
)


@st.cache_resource
def load_artefacts():
    missing = [
        str(path.name)
        for path in (MODEL_PATH, FREQUENCY_MAPS_PATH, METADATA_PATH)
        if not path.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing model artefacts in the artefacts/ folder: " +
            ", ".join(missing)
        )

    with METADATA_PATH.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    model = joblib.load(MODEL_PATH)
    frequency_maps = joblib.load(FREQUENCY_MAPS_PATH)
    return model, frequency_maps, metadata


try:
    model, frequency_maps, metadata = load_artefacts()
except Exception as exc:
    st.error(f"Could not load model artefacts: {exc}")
    st.info(
        "Export the model, frequency mappings and metadata from the training notebook, "
        "then place all three files in the application's artefacts/ directory."
    )
    st.stop()

trained_sklearn_version = metadata.get("sklearn_version")
if trained_sklearn_version and trained_sklearn_version != sklearn.__version__:
    st.warning(
        "The model was exported with scikit-learn "
        f"{trained_sklearn_version}, but this app is running "
        f"{sklearn.__version__}. Match the version used in training before relying on results."
    )

threshold = float(metadata.get("threshold", 0.63))

with st.sidebar:
    st.header("Model information")
    st.write(f"Estimator: {metadata.get('model', 'Random Forest')}")
    st.write(f"Decision threshold: **{threshold:.2f}**")
    st.write(f"Features used: **{len(metadata.get('features', [])):,}**")
    st.caption(
        "TransactionID is retained for result identification, not used as a feature.")

st.subheader("1. Upload raw data")
left, right = st.columns(2)
with left:
    transaction_file = st.file_uploader(
        "Transaction CSV (required)", type=["csv"], key="transaction_csv"
    )
with right:
    identity_file = st.file_uploader(
        "Matching identity CSV (recommended)", type=["csv"], key="identity_csv"
    )

st.caption(
    "Use matching transaction and identity files from the IEEE-CIS schema. "
    "For a small demo, upload a representative sample rather than the full dataset."
)

if transaction_file is None:
    st.info("Upload a transaction CSV to begin.")
    st.stop()

try:
    transactions = pd.read_csv(transaction_file, low_memory=False)
    identity = pd.read_csv(
        identity_file, low_memory=False) if identity_file else None
except Exception as exc:
    st.error(f"Could not read the uploaded CSV: {exc}")
    st.stop()

if len(transactions) > MAX_ROWS:
    st.error(
        f"This demo accepts at most {MAX_ROWS:,} transaction rows per upload. "
        "Split larger files into smaller batches."
    )
    st.stop()

st.write(f"Transactions uploaded: **{len(transactions):,}**")
if identity is not None:
    st.write(f"Identity rows uploaded: **{len(identity):,}**")

with st.expander("Preview transaction input"):
    st.dataframe(transactions.head(10), use_container_width=True)

if st.button("Run fraud detection", type="primary", use_container_width=True):
    with st.spinner("Preprocessing records and generating predictions..."):
        try:
            X, transaction_ids, input_warnings = preprocess_raw_inputs(
                transactions=transactions,
                identity=identity,
                metadata=metadata,
                frequency_maps=frequency_maps,
            )
            scores = model.predict_proba(X)[:, 1]
            flags = scores >= threshold
        except InferenceInputError as exc:
            st.error(str(exc))
            st.stop()
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.stop()

    for message in input_warnings:
        st.warning(message)

    if "isFraud" in transactions.columns:
        st.info(
            "The uploaded file contains an isFraud label. It was excluded from inference features."
        )

    results = pd.DataFrame({"TransactionID": transaction_ids})
    for display_col in ("TransactionAmt", "TransactionDT"):
        if display_col in transactions.columns:
            results[display_col] = transactions[display_col].to_numpy()
    results["Fraud_Score"] = scores
    results["Threshold"] = threshold
    results["Prediction"] = np.where(flags, "Potential Fraud", "Not Flagged")

    total = len(results)
    flagged = int(flags.sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Transactions analysed", f"{total:,}")
    c2.metric("Potential fraud flags", f"{flagged:,}")
    c3.metric("Flag rate", f"{flagged / total:.2%}" if total else "0.00%")

    st.subheader("2. Prediction results")
    st.dataframe(
        results.sort_values("Fraud_Score", ascending=False),
        use_container_width=True,
        hide_index=True,
    )
    st.download_button(
        "Download predictions CSV",
        data=results.to_csv(index=False).encode("utf-8"),
        file_name="fraud_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.caption(
        "These are model-generated risk signals for review, not confirmation of fraud. "
        "Fraud scores should not be treated as calibrated probabilities unless calibration has been evaluated."
    )
