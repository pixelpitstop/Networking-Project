"""
Unit tests for the Streamlit alerts dashboard helpers and line-count budget.
"""

import pandas as pd
from src.dashboard import build_alert_rate_series, top_attack_types


def test_dashboard_line_count_under_limit():
    with open("src/dashboard.py") as f:
        lines = f.readlines()
    assert len(lines) <= 150, f"Dashboard should stay under ~150 lines, got {len(lines)}"


def test_build_alert_rate_series():
    df = pd.DataFrame({
        "consume_ts": [1700000000.1, 1700000000.5, 1700000001.2],
        "score": [0.9, 0.95, 0.88],
    })
    rate_df = build_alert_rate_series(df)
    assert len(rate_df) == 2
    assert rate_df["alerts_per_sec"].tolist() == [2, 1]


def test_top_attack_types():
    df = pd.DataFrame({
        "attack_type": ["syn", "syn", "http_flood", "udp", "syn"],
    })
    top = top_attack_types(df, top_n=2)
    assert list(top.index) == ["syn", "http_flood"]
    assert top.iloc[0] == 3
