# Group 17 — Credit Card Fraud Detection App

This is the deployment application scaffold for the IEEE-CIS fraud-detection capstone. It is separate from the Colab training runtime. The app accepts raw transaction CSV data and an optional matching identity CSV, applies the notebook's inference preprocessing, loads the trained Random Forest artefact, and returns predictions using the saved threshold.

## Files expected from the Colab artefact export

Put the following files inside `artefacts/`:

- `rf_fraud_pipeline.joblib`
- `frequency_maps.joblib`
- `model_metadata.json`

These contain learned state from the training notebook. Do not recreate frequency mappings using uploaded inference data.

## Local setup

1. Export the three artefacts from Colab using the instructions accompanying this scaffold.
2. Copy them into `artefacts/`.
3. Create a Python environment compatible with the training environment (see `python_version` in the metadata).
4. In `requirements.txt`, pin `scikit-learn`, `numpy`, `pandas`, and `joblib` to the exact versions recorded in `model_metadata.json` (`sklearn_version`, `numpy_version`, `pandas_version`, `joblib_version`).
5. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

6. Start the app:

   ```bash
   streamlit run app.py
   ```

7. Upload a transaction CSV and the corresponding identity CSV, then run predictions.

## Input expectations

- Transaction CSV: requires `TransactionID`, `TransactionDT` and `TransactionAmt`.
- Identity CSV: recommended and should use the corresponding IEEE-CIS identity schema with `TransactionID` as the join key. If omitted, identity-derived data will be missing and predictions may be less representative of the original evaluation setup.
- Files may contain `isFraud`; the app removes the target from inference inputs.
- The demo limits each upload to 100,000 transaction rows to avoid exhausting hosted memory. This is configurable in `app.py` after testing the hosting resources.

## Model compatibility and security

Scikit-learn model artefacts are Python pickle-based. Only load artefacts produced by a trusted source, and deploy with the same scikit-learn version used during training. The app warns when versions differ; the warning should be resolved before release.

Do not commit raw IEEE-CIS datasets, Kaggle API credentials, or other secrets. If the model artefact is too large for your chosen Git hosting setup, use an approved external artefact store rather than committing raw training data.

## Deployment

Create a GitHub repository with this scaffold and the exported artefacts, then deploy `app.py` using Streamlit Community Cloud. Select the matching Python runtime and install dependencies from `requirements.txt`. Test the live URL from a separate browser session.
