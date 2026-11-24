"""
Tests for data pipeline, feature engineering, and model inference.
"""

import os
import joblib
import numpy as np
import pandas as pd
import pytest

from src.data import clean_data, split_data, DROPPED_COLUMNS, TARGET_COLUMN
from src.features import NetworkFeaturePipeline


def test_clean_data_removes_leaky_columns():
    fake_df = pd.DataFrame({
        "uid": ["C123", "C456"],
        "src_ip": ["192.168.0.1", "10.0.0.1"],
        "dst_ip": ["192.168.0.10", "10.0.0.2"],
        "src_port": [54321, 12345],
        "history": ["ShAD", "Ar"],
        "duration": [0.1, 0.2],
        "traffic_type": [0, 1],
    })
    cleaned = clean_data(fake_df)
    for col in DROPPED_COLUMNS:
        assert col not in cleaned.columns, f"Leaky column {col} was not dropped"
    assert "duration" in cleaned.columns
    assert "traffic_type" in cleaned.columns


def test_split_data_integrity():
    # Synthetic group dataset
    data = []
    for cat in ["benign", "dos", "probe"]:
        for i in range(100):
            data.append({
                "attack_category": cat,
                "attack_type": f"{cat}_type",
                "traffic_type": 0 if cat == "benign" else 1,
                "duration": 0.01 * i,
                "src_pkts": i,
                "dst_pkts": i * 2,
                "src_bytes": i * 50,
                "dst_bytes": i * 100,
            })
    df = pd.DataFrame(data)
    train_df, test_df = split_data(df, test_ratio=0.30)
    
    assert len(train_df) + len(test_df) == len(df)
    # Check each group has 70 train, 30 test
    for cat in ["benign", "dos", "probe"]:
        train_cat = train_df[train_df["attack_category"] == cat]
        test_cat = test_df[test_df["attack_category"] == cat]
        assert len(train_cat) == 70
        assert len(test_cat) == 30


def test_feature_pipeline_and_models_inference():
    # Ensure trained models exist
    pipeline_path = "models/feature_pipeline.joblib"
    iso_path = "models/isolation_forest.joblib"
    xgb_path = "models/xgboost_baseline.joblib"
    
    assert os.path.exists(pipeline_path), "Feature pipeline artifact missing"
    assert os.path.exists(iso_path), "Isolation forest artifact missing"
    assert os.path.exists(xgb_path), "XGBoost baseline artifact missing"

    pipeline = joblib.load(pipeline_path)
    iso_model = joblib.load(iso_path)
    xgb_model = joblib.load(xgb_path)

    # Load 10 sample test rows from processed test data
    test_df = pd.read_parquet("data/processed/test.parquet").head(10)
    X_test_trans = pipeline.transform(test_df)

    assert X_test_trans.shape[0] == 10
    assert X_test_trans.shape[1] == 51
    assert not np.isnan(X_test_trans).any(), "Transformed feature matrix contains NaNs"
    assert not np.isinf(X_test_trans).any(), "Transformed feature matrix contains Infs"

    # Isolation Forest inference
    iso_preds = (iso_model.predict(X_test_trans) == -1).astype(int)
    assert len(iso_preds) == 10
    assert set(iso_preds).issubset({0, 1})

    # XGBoost inference
    xgb_preds = xgb_model.predict(X_test_trans)
    xgb_probs = xgb_model.predict_proba(X_test_trans)[:, 1]
    assert len(xgb_preds) == 10
    assert len(xgb_probs) == 10
    assert (xgb_probs >= 0.0).all() and (xgb_probs <= 1.0).all()
