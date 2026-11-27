"""
Unit tests for the streaming scoring function (FlowScorer).
Runs offline without requiring a live Kafka broker.
"""

import numpy as np
import pandas as pd
import pytest

from src.consumer import FlowScorer


def test_flow_scorer_xgboost_and_isolation_forest():
    test_df = pd.read_parquet("data/processed/test.parquet")
    benign_sample = test_df[test_df["traffic_type"] == 0].head(5).to_dict(orient="records")
    attack_sample = test_df[test_df["traffic_type"] == 1].head(5).to_dict(orient="records")
    records = benign_sample + attack_sample

    # Simulate streaming metadata attached by producer.py
    for i, rec in enumerate(records):
        rec["flow_id"] = f"unit-{i}"
        rec["produce_ts"] = 1700000000.0 + i

    # Test XGBoost scorer
    xgb_scorer = FlowScorer(models_dir="models", model_name="xgboost_baseline")
    preds, scores = xgb_scorer.score_batch(records)

    assert len(preds) == 10
    assert len(scores) == 10
    assert set(preds.tolist()).issubset({0, 1})
    assert np.all((scores >= 0.0) & (scores <= 1.0))

    # Single-record wrapper check
    single_pred, single_score = xgb_scorer.score_one(records[0])
    assert single_pred == int(preds[0])
    assert np.isclose(single_score, float(scores[0]))

    # Test Isolation Forest scorer
    iso_scorer = FlowScorer(models_dir="models", model_name="isolation_forest")
    iso_preds, iso_scores = iso_scorer.score_batch(records)

    assert len(iso_preds) == 10
    assert len(iso_scores) == 10
    assert set(iso_preds.tolist()).issubset({0, 1})
    assert not np.isnan(iso_scores).any()


def test_flow_scorer_empty_batch():
    scorer = FlowScorer(models_dir="models", model_name="xgboost_baseline")
    preds, scores = scorer.score_batch([])
    assert len(preds) == 0
    assert len(scores) == 0


def test_flow_scorer_invalid_model():
    with pytest.raises(ValueError):
        FlowScorer(models_dir="models", model_name="unknown_model")
