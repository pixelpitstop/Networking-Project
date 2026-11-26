"""
Evaluation script for Track A (Isolation Forest) and Track B (XGBoost Baseline).
Computes classification metrics, false-positive rate, PR-AUC, per-attack recall,
and generates confusion-matrix and PR-curve plots in results/.
"""

import os
import sys
import json
import time

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    precision_recall_curve,
)

from src.data import TARGET_COLUMN

MODELS_DIR = "models"
RESULTS_DIR = "results"
TEST_DATA_PATH = "data/processed/test.parquet"


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> dict:
    """Compute overall binary classification metrics and confusion matrix counts."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    return {
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "pr_auc": round(float(average_precision_score(y_true, y_score)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, y_score)), 4),
        "fpr": round(fpr, 4),
        "confusion_matrix": {
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        },
        "predicted_anomalies": int(tp + fp),
        "actual_anomalies": int(tp + fn),
        "total_samples": int(len(y_true)),
    }


def compute_per_group_recall(
    df_meta: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_col: str,
) -> dict:
    """
    Compute recall (detection rate) per attack group.
    For benign rows (y_true == 0), reports specificity (true negative rate)
    alongside count and false-positive rate.
    """
    results = {}
    temp = pd.DataFrame({
        "group": df_meta[group_col].values,
        "y_true": y_true,
        "y_pred": y_pred,
    })

    for group_name, grp in temp.groupby("group"):
        n = int(len(grp))
        if group_name == "benign":
            tn = int((grp["y_pred"] == 0).sum())
            fp = int((grp["y_pred"] == 1).sum())
            results[str(group_name)] = {
                "count": n,
                "recall_or_specificity": round(float(tn / n), 4) if n > 0 else 0.0,
                "fpr": round(float(fp / n), 4) if n > 0 else 0.0,
                "detected": tn,
            }
        else:
            tp = int((grp["y_pred"] == 1).sum())
            results[str(group_name)] = {
                "count": n,
                "recall": round(float(tp / n), 4) if n > 0 else 0.0,
                "detected": tp,
                "missed": int(n - tp),
            }
    return results


def plot_confusion_matrices(
    y_true: np.ndarray,
    preds_dict: dict[str, np.ndarray],
    output_path: str,
) -> None:
    """Plot side-by-side confusion matrices for all evaluated tracks."""
    fig, axes = plt.subplots(1, len(preds_dict), figsize=(11, 4.5))
    if len(preds_dict) == 1:
        axes = [axes]

    labels = ["Benign (0)", "Attack (1)"]

    for ax, (name, y_pred) in zip(axes, preds_dict.items()):
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

        im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
        ax.set_title(name, fontsize=11, fontweight="bold", pad=10)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

        for i in range(2):
            for j in range(2):
                pct = cm_norm[i, j] * 100.0
                count = cm[i, j]
                color = "white" if cm_norm[i, j] > 0.55 else "black"
                ax.text(
                    j,
                    i,
                    f"{count:,}\n({pct:.1f}%)",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=10,
                )

    fig.suptitle("Confusion Matrices on Held-Out Test Split", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_pr_curves(
    y_true: np.ndarray,
    scores_dict: dict[str, tuple[np.ndarray, float]],
    output_path: str,
) -> None:
    """Plot Precision-Recall curves for all evaluated tracks."""
    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    prevalence = float((y_true == 1).mean())
    for name, (y_score, pr_auc) in scores_dict.items():
        precision, recall, _ = precision_recall_curve(y_true, y_score)
        # Subsample points if huge for fast, clean plotting
        if len(precision) > 2000:
            idx = np.linspace(0, len(precision) - 1, 2000, dtype=int)
            precision = precision[idx]
            recall = recall[idx]
        ax.plot(recall, precision, linewidth=2, label=f"{name} (PR-AUC = {pr_auc:.4f})")

    ax.axhline(
        prevalence,
        color="gray",
        linestyle="--",
        linewidth=1.2,
        label=f"No-Skill Baseline (Prevalence = {prevalence:.4f})",
    )
    ax.set_title("Precision-Recall Curve (Test Split)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Recall (Attack Detection Rate)")
    ax.set_ylabel("Precision")
    ax.set_xlim([0.0, 1.02])
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def evaluate_models(
    test_path: str = TEST_DATA_PATH,
    models_dir: str = MODELS_DIR,
    results_dir: str = RESULTS_DIR,
) -> dict:
    """Run full offline evaluation and write metrics and plots."""
    os.makedirs(results_dir, exist_ok=True)

    print(f"Loading test split from {test_path}...")
    test_df = pd.read_parquet(test_path)
    y_test = test_df[TARGET_COLUMN].values

    print("Loading feature pipeline and trained models...")
    pipeline = joblib.load(os.path.join(models_dir, "feature_pipeline.joblib"))
    iso_model = joblib.load(os.path.join(models_dir, "isolation_forest.joblib"))
    xgb_model = joblib.load(os.path.join(models_dir, "xgboost_baseline.joblib"))

    print(f"Transforming {len(test_df):,} test flows...")
    t0 = time.time()
    X_test = pipeline.transform(test_df)
    print(f"Transformed in {time.time() - t0:.2f}s.")

    # Track A: Isolation Forest
    print("Scoring Track A: Isolation Forest...")
    t0 = time.time()
    y_pred_iso = (iso_model.predict(X_test) == -1).astype(int)
    y_score_iso = -iso_model.score_samples(X_test)
    iso_score_time = time.time() - t0

    # Track B: XGBoost
    print("Scoring Track B: XGBoost Baseline...")
    t0 = time.time()
    y_pred_xgb = xgb_model.predict(X_test).astype(int)
    y_score_xgb = xgb_model.predict_proba(X_test)[:, 1]
    xgb_score_time = time.time() - t0

    iso_metrics = compute_binary_metrics(y_test, y_pred_iso, y_score_iso)
    iso_metrics["scoring_time_sec"] = round(iso_score_time, 2)
    iso_metrics["per_attack_category"] = compute_per_group_recall(test_df, y_test, y_pred_iso, "attack_category")
    iso_metrics["per_attack_type"] = compute_per_group_recall(test_df, y_test, y_pred_iso, "attack_type")

    xgb_metrics = compute_binary_metrics(y_test, y_pred_xgb, y_score_xgb)
    xgb_metrics["scoring_time_sec"] = round(xgb_score_time, 2)
    xgb_metrics["per_attack_category"] = compute_per_group_recall(test_df, y_test, y_pred_xgb, "attack_category")
    xgb_metrics["per_attack_type"] = compute_per_group_recall(test_df, y_test, y_pred_xgb, "attack_type")

    metrics_payload = {
        "test_split_path": test_path,
        "test_samples": int(len(test_df)),
        "test_benign": int((y_test == 0).sum()),
        "test_attack": int((y_test == 1).sum()),
        "tracks": {
            "isolation_forest": iso_metrics,
            "xgboost_baseline": xgb_metrics,
        },
    }

    metrics_path = os.path.join(results_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"Saved evaluation metrics to {metrics_path}")

    cm_path = os.path.join(results_dir, "confusion_matrices.png")
    plot_confusion_matrices(
        y_test,
        {
            "Track A: Isolation Forest": y_pred_iso,
            "Track B: XGBoost Baseline": y_pred_xgb,
        },
        cm_path,
    )
    print(f"Saved confusion matrices plot to {cm_path}")

    pr_path = os.path.join(results_dir, "pr_curves.png")
    plot_pr_curves(
        y_test,
        {
            "Track A: Isolation Forest": (y_score_iso, iso_metrics["pr_auc"]),
            "Track B: XGBoost Baseline": (y_score_xgb, xgb_metrics["pr_auc"]),
        },
        pr_path,
    )
    print(f"Saved PR curves plot to {pr_path}")

    return metrics_payload


if __name__ == "__main__":
    results = evaluate_models()
    for track_name, m in results["tracks"].items():
        print(f"\n=== {track_name} ===")
        print(
            f"Precision: {m['precision']:.4f} | Recall: {m['recall']:.4f} | "
            f"F1: {m['f1']:.4f} | PR-AUC: {m['pr_auc']:.4f} | FPR: {m['fpr']:.4f}"
        )
        print("Per-attack-type recall:")
        for atype, stats in sorted(m["per_attack_type"].items()):
            if atype == "benign":
                print(f"  benign (n={stats['count']}): specificity={stats['recall_or_specificity']:.4f}, fpr={stats['fpr']:.4f}")
            else:
                print(f"  {atype:20s} (n={stats['count']:6d}): recall={stats['recall']:.4f}")
