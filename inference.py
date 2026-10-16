"""Raw-input preprocessing for the IEEE-CIS fraud detection model.

This module mirrors the feature engineering and train-fitted frequency encoding
used in the Group 17 capstone notebook. It deliberately does not retrain a model.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class InferenceInputError(ValueError):
    """Raised when uploaded input files cannot be safely processed."""


def add_features(data: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the feature engineering used by the training notebook."""
    data = data.copy()

    if "TransactionDT" in data.columns:
        data["TransactionHour"] = (
            (data["TransactionDT"] // 3600) % 24
        ).astype("float32")
        data["TransactionDay"] = (
            data["TransactionDT"] // (3600 * 24)
        ).astype("float32")

    if "TransactionAmt" in data.columns:
        data["TransactionAmt_Log"] = np.log1p(
            data["TransactionAmt"].clip(lower=0)
        ).astype("float32")

    if "card1" in data.columns and "addr1" in data.columns:
        data["CardAddr"] = (
            data["card1"].astype("string") + "_" +
            data["addr1"].astype("string")
        )

    return data


def preprocess_raw_inputs(
    transactions: pd.DataFrame,
    identity: pd.DataFrame | None,
    metadata: dict[str, Any],
    frequency_maps: dict[str, pd.Series],
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Convert raw transaction and optional identity tables to model features.

    Returns:
        X: numeric DataFrame in the exact saved model feature order.
        transaction_ids: original transaction IDs aligned row-for-row with X.
        warnings: non-fatal input warnings for the UI.
    """
    if transactions is None or transactions.empty:
        raise InferenceInputError("The transaction CSV is empty.")

    id_col = metadata.get("transaction_id_column", "TransactionID")
    target_col = metadata.get("target_column", "isFraud")
    required_tx = metadata.get(
        "required_transaction_columns",
        [id_col, "TransactionDT", "TransactionAmt"],
    )
    missing_required = [col for col in required_tx if col not in transactions.columns]
    if missing_required:
        raise InferenceInputError(
            "The transaction CSV is missing required columns: "
            + ", ".join(missing_required)
        )

    tx = transactions.copy()
    transaction_ids = tx[id_col].copy().reset_index(drop=True)
    if transaction_ids.isna().any():
        raise InferenceInputError("TransactionID contains missing values.")
    if transaction_ids.duplicated().any():
        raise InferenceInputError(
            "TransactionID contains duplicates. Please deduplicate the transaction CSV first."
        )

    # The label may exist in training CSVs, but it must never enter inference features.
    tx = tx.drop(columns=[target_col], errors="ignore")
    warnings: list[str] = []

    if identity is not None:
        if identity.empty:
            warnings.append("The identity CSV is empty; identity fields will be unavailable.")
        else:
            idt = identity.copy().drop(columns=[target_col], errors="ignore")
            if id_col not in idt.columns:
                raise InferenceInputError(
                    f"The identity CSV must contain {id_col}."
                )
            if idt[id_col].isna().any():
                raise InferenceInputError("Identity CSV TransactionID contains missing values.")
            if idt[id_col].duplicated().any():
                raise InferenceInputError(
                    "The identity CSV contains duplicate TransactionID values."
                )

            overlaps = (set(tx.columns) & set(idt.columns)) - {id_col}
            if overlaps:
                raise InferenceInputError(
                    "The two CSV files have overlapping non-ID columns: "
                    + ", ".join(sorted(overlaps)[:10])
                )

            try:
                tx = tx.merge(idt, on=id_col, how="left", validate="one_to_one")
            except (ValueError, pd.errors.MergeError) as exc:
                raise InferenceInputError(f"Could not merge identity data: {exc}") from exc
    else:
        warnings.append(
            "No identity CSV was uploaded. Identity-derived fields will be missing; "
            "predictions may be less representative of the notebook evaluation setup."
        )

    # Convert core arithmetic inputs to numeric just as the training data was parsed.
    # Malformed values become NaN and are handled by the saved preprocessing path.
    for col in ("TransactionDT", "TransactionAmt", "card1", "addr1"):
        if col in tx.columns:
            tx[col] = pd.to_numeric(tx[col], errors="coerce")

    # Match the training notebook's rule for dropping columns with >90% missingness.
    dropped_columns = metadata.get("dropped_high_missing_columns", [])
    tx = tx.drop(columns=[c for c in dropped_columns if c in tx.columns], errors="ignore")

    # Restore any expected base columns absent from a partial input. The fitted median
    # imputer and saved encodings then handle missing values as in training.
    raw_feature_columns = metadata.get("raw_feature_columns", [])
    for col in raw_feature_columns:
        if col not in tx.columns:
            tx[col] = np.nan

    tx = add_features(tx)

    # TransactionID identifies the output record but was excluded from model features.
    model_data = tx.drop(columns=[id_col, target_col], errors="ignore").copy()

    categorical_cols = metadata.get("categorical_cols", [])
    for col in categorical_cols:
        if col not in model_data.columns:
            model_data[col] = np.nan
        mapping = frequency_maps.get(col)
        if mapping is None:
            raise InferenceInputError(
                f"No saved training frequency mapping exists for categorical feature {col!r}."
            )
        # Use the mapping learned from training. Unseen values fall back to 0,
        # matching the notebook's map(...).fillna(0) logic.
        model_data[col] = model_data[col].astype(object).map(mapping).fillna(0)

    model_data = model_data.replace([np.inf, -np.inf], np.nan)
    for col in model_data.columns:
        model_data[col] = pd.to_numeric(model_data[col], errors="coerce")

    expected_features = metadata.get("features", [])
    if not expected_features:
        raise InferenceInputError("Model metadata does not contain the expected feature list.")

    X = model_data.reindex(columns=expected_features)
    if X.shape[0] != len(transaction_ids):
        raise InferenceInputError("Internal row alignment error during preprocessing.")
    if X.columns.tolist() != expected_features:
        raise InferenceInputError("Could not align the processed features to the model schema.")

    return X, transaction_ids, warnings
