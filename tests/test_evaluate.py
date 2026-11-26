"""
Tests for evaluation metrics and generated Phase 3 artifacts.
"""

import os
import json
import numpy as np
import pandas as pd

from src.evaluate import compute_binary_metrics, compute_per_group_recall


def test_compute_binary_metrics():
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 1, 0])
    y_score = np.array([0.1, 0.6, 0.9, 0.8, 0.4])

    metrics = compute_binary_metrics(y_true, y_pred, y_score)
    # tn=1, fp=1, fn=1, tp=2
    assert metrics["confusion_matrix"] == {"tn": 1, "fp": 1, "fn": 1, "tp": 2}
    assert metrics["fpr"] == 0.5
    assert round(metrics["precision"], 4) == round(2 / 3, 4)
    assert round(metrics["recall"], 4) == round(2 / 3, 4)


def test_compute_per_group_recall():
    df_meta = pd.DataFrame({"attack_type": ["benign", "benign", "syn", "syn", "ssh"]})
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 1, 1, 1, 0])

    res = compute_per_group_recall(df_meta, y_true, y_pred, "attack_type")
    assert res["benign"]["fpr"] == 0.5
    assert res["benign"]["recall_or_specificity"] == 0.5
    assert res["syn"]["recall"] == 1.0
    assert res["ssh"]["recall"] == 0.0


def test_phase3_artifacts_exist():
    assert os.path.exists("results/metrics.json")
    assert os.path.exists("results/confusion_matrices.png")
    assert os.path.exists("results/pr_curves.png")
    assert os.path.exists("docs/FINDINGS.md")

    with open("results/metrics.json") as f:
        data = json.load(f)

    assert "tracks" in data
    assert "isolation_forest" in data["tracks"]
    assert "xgboost_baseline" in data["tracks"]
    assert "per_attack_type" in data["tracks"]["xgboost_baseline"]
