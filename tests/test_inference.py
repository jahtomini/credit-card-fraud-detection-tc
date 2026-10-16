import numpy as np
import pandas as pd

from inference import preprocess_raw_inputs


def test_preprocessing_uses_saved_mappings_and_exact_feature_order():
    transactions = pd.DataFrame({
        "TransactionID": [1001, 1002],
        "TransactionDT": [3600, 7200],
        "TransactionAmt": [10.0, 20.0],
        "card1": [11, 22],
        "addr1": [3, 4],
        "card4": ["visa", "new-brand"],
        "isFraud": [0, 1],  # Must not be used as an inference feature.
    })
    identity = pd.DataFrame({
        "TransactionID": [1001, 1002],
        "DeviceType": ["desktop", "mobile"],
    })
    metadata = {
        "transaction_id_column": "TransactionID",
        "target_column": "isFraud",
        "required_transaction_columns": ["TransactionID", "TransactionDT", "TransactionAmt"],
        "dropped_high_missing_columns": [],
        "raw_feature_columns": ["TransactionDT", "TransactionAmt", "card1", "addr1", "card4", "DeviceType"],
        "categorical_cols": ["card4", "CardAddr", "DeviceType"],
        "features": [
            "TransactionDT", "TransactionAmt", "card1", "addr1", "card4",
            "DeviceType", "TransactionHour", "TransactionDay",
            "TransactionAmt_Log", "CardAddr",
        ],
    }
    maps = {
        "card4": pd.Series({"visa": 0.7, "mastercard": 0.3}),
        "CardAddr": pd.Series({"11_3": 0.5, "22_4": 0.5}),
        "DeviceType": pd.Series({"desktop": 0.6, "mobile": 0.4}),
    }

    X, ids, warnings = preprocess_raw_inputs(transactions, identity, metadata, maps)

    assert X.columns.tolist() == metadata["features"]
    assert ids.tolist() == [1001, 1002]
    assert "TransactionID" not in X.columns
    assert "isFraud" not in X.columns
    assert X.loc[0, "card4"] == 0.7
    assert X.loc[1, "card4"] == 0.0  # unseen category uses the documented fallback
    assert np.isfinite(X["TransactionAmt_Log"]).all()
    assert warnings == []
