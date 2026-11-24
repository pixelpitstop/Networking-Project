"""
Training script for Track A (Isolation Forest) and Track B (XGBoost Baseline).
Reproducible, deterministic, and self-contained.
"""

import os
import sys
import time
import json

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

from src.data import prepare_and_save_splits, TARGET_COLUMN
from src.features import NetworkFeaturePipeline

MODELS_DIR = "models"
RANDOM_SEED = 42


def train_models():
    os.makedirs(MODELS_DIR, exist_ok=True)
    np.random.seed(RANDOM_SEED)

    # 1. Prepare / Load splits
    processed_train = "data/processed/train.parquet"
    processed_test = "data/processed/test.parquet"

    if not (os.path.exists(processed_train) and os.path.exists(processed_test)):
        print("Processed splits not found. Preparing from raw data...")
        train_df, test_df = prepare_and_save_splits()
    else:
        print(f"Loading cached splits from {processed_train} and {processed_test}...")
        train_df = pd.read_parquet(processed_train)
        test_df = pd.read_parquet(processed_test)

    print(f"Train set: {len(train_df):,} flows (Benign: {(train_df[TARGET_COLUMN] == 0).sum():,}, Attack: {(train_df[TARGET_COLUMN] == 1).sum():,})")
    print(f"Test set:  {len(test_df):,} flows (Benign: {(test_df[TARGET_COLUMN] == 0).sum():,}, Attack: {(test_df[TARGET_COLUMN] == 1).sum():,})")

    y_train = train_df[TARGET_COLUMN].values
    y_test = test_df[TARGET_COLUMN].values

    # 2. Fit Feature Pipeline on Training Data Only
    print("\n[Step 1/3] Fitting feature pipeline on training data...")
    feature_pipeline = NetworkFeaturePipeline()
    t0 = time.time()
    X_train = feature_pipeline.fit_transform(train_df)
    X_test = feature_pipeline.transform(test_df)
    fit_time = time.time() - t0
    print(f"Feature transformation completed in {fit_time:.2f}s. Feature matrix shape: {X_train.shape}")

    pipeline_path = os.path.join(MODELS_DIR, "feature_pipeline.joblib")
    joblib.dump(feature_pipeline, pipeline_path)
    print(f"Saved feature pipeline to {pipeline_path}")

    # 3. Track A: Isolation Forest (Trained on Benign Traffic ONLY)
    print("\n[Step 2/3] Training Track A: Isolation Forest (unsupervised, benign-only)...")
    benign_mask = (y_train == 0)
    X_train_benign = X_train[benign_mask]
    print(f"Fitting Isolation Forest on {len(X_train_benign):,} strictly benign flows...")

    iso_forest = IsolationForest(
        n_estimators=100,
        max_samples="auto",
        contamination=0.01,
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    t0 = time.time()
    iso_forest.fit(X_train_benign)
    iso_train_time = time.time() - t0
    print(f"Isolation Forest trained in {iso_train_time:.2f}s.")

    iso_path = os.path.join(MODELS_DIR, "isolation_forest.joblib")
    joblib.dump(iso_forest, iso_path)
    print(f"Saved Isolation Forest model to {iso_path}")

    # Quick test check for Isolation Forest
    # IsolationForest: 1 = normal, -1 = anomaly -> map to 0 = normal, 1 = anomaly
    iso_test_pred = (iso_forest.predict(X_test) == -1).astype(int)
    # Anomaly score: higher score = more anomalous
    iso_test_scores = -iso_forest.score_samples(X_test)
    iso_ap = average_precision_score(y_test, iso_test_scores)
    print(f"Isolation Forest Quick Test: PR-AUC = {iso_ap:.4f}")

    # 4. Track B: XGBoost Supervised Baseline (Class-Weighted)
    print("\n[Step 3/3] Training Track B: XGBoost Classifier (supervised, class-weighted)...")
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count
    print(f"Class weighting (neg/pos): {neg_count} / {pos_count} = {scale_pos_weight:.4f}")

    xgb_model = XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        eval_metric="logloss"
    )
    t0 = time.time()
    xgb_model.fit(X_train, y_train)
    xgb_train_time = time.time() - t0
    print(f"XGBoost trained in {xgb_train_time:.2f}s.")

    xgb_path = os.path.join(MODELS_DIR, "xgboost_baseline.joblib")
    joblib.dump(xgb_model, xgb_path)
    print(f"Saved XGBoost model to {xgb_path}")

    xgb_test_probs = xgb_model.predict_proba(X_test)[:, 1]
    xgb_ap = average_precision_score(y_test, xgb_test_probs)
    xgb_roc = roc_auc_score(y_test, xgb_test_probs)
    print(f"XGBoost Quick Test: ROC-AUC = {xgb_roc:.4f}, PR-AUC = {xgb_ap:.4f}")

    # Save training metadata
    meta = {
        "timestamp": time.time(),
        "random_seed": RANDOM_SEED,
        "train_samples": int(len(train_df)),
        "train_benign": int(neg_count),
        "train_attack": int(pos_count),
        "test_samples": int(len(test_df)),
        "test_benign": int((y_test == 0).sum()),
        "test_attack": int((y_test == 1).sum()),
        "features_count": int(X_train.shape[1]),
        "feature_names": feature_pipeline.feature_names_,
        "iso_forest": {
            "training_time_sec": round(iso_train_time, 2),
            "contamination": 0.01,
            "quick_pr_auc": round(float(iso_ap), 4),
            "path": iso_path
        },
        "xgboost": {
            "training_time_sec": round(xgb_train_time, 2),
            "scale_pos_weight": round(float(scale_pos_weight), 4),
            "quick_roc_auc": round(float(xgb_roc), 4),
            "quick_pr_auc": round(float(xgb_ap), 4),
            "path": xgb_path
        }
    }
    meta_path = os.path.join(MODELS_DIR, "training_meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved training metadata to {meta_path}")
    print("\n[SUCCESS] Phase 2 training completed successfully!")


if __name__ == "__main__":
    train_models()
