"""
Feature engineering and transformation pipeline for 5G network traffic.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CATEGORICAL_COLS = ["proto", "service", "conn_state"]

BASE_NUMERICAL_COLS = [
    "duration",
    "src_bytes",
    "dst_bytes",
    "missed_bytes",
    "src_pkts",
    "src_ip_bytes",
    "dst_pkts",
    "dst_ip_bytes",
    "ip_proto",
    "http_trans_depth",
    "files_total_bytes",
    "is_GET_mthd",
    "http_status_error",
    "is_file_transfered",
    "dst_port",
]

ENGINEERED_COLS = [
    "pkt_rate",
    "byte_rate",
    "src_dst_pkt_ratio",
    "src_dst_byte_ratio",
    "avg_pkt_size",
]

ALL_NUMERICAL_COLS = BASE_NUMERICAL_COLS + ENGINEERED_COLS


class FlowFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Computes domain-specific flow rate and ratio features:
    - Packet rate (packets/sec)
    - Byte rate (bytes/sec)
    - Source/destination packet ratio
    - Source/destination byte ratio
    - Average packet size
    """

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        duration_safe = df["duration"].clip(lower=1e-5)
        
        df["pkt_rate"] = (df["src_pkts"] + df["dst_pkts"]) / duration_safe
        df["byte_rate"] = (df["src_bytes"] + df["dst_bytes"]) / duration_safe
        df["src_dst_pkt_ratio"] = df["src_pkts"] / (df["dst_pkts"] + 1.0)
        df["src_dst_byte_ratio"] = df["src_bytes"] / (df["dst_bytes"] + 1.0)
        df["avg_pkt_size"] = (df["src_bytes"] + df["dst_bytes"]) / (df["src_pkts"] + df["dst_pkts"] + 1e-5)
        
        # Replace any infs or nans from edge cases
        for col in ENGINEERED_COLS:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            
        return df


def create_preprocessor() -> ColumnTransformer:
    """Build column transformer for categoricals and numericals."""
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_COLS,
            ),
            (
                "num",
                StandardScaler(),
                ALL_NUMERICAL_COLS,
            ),
        ],
        remainder="drop",
    )
    return preprocessor


class NetworkFeaturePipeline:
    """End-to-end pipeline combining flow feature engineering and transformation."""

    def __init__(self):
        self.engineer = FlowFeatureEngineer()
        self.preprocessor = create_preprocessor()
        self.feature_names_ = []

    def fit(self, df: pd.DataFrame, y=None):
        engineered_df = self.engineer.transform(df)
        self.preprocessor.fit(engineered_df)
        
        # Extract feature names
        cat_encoder = self.preprocessor.named_transformers_["cat"]
        cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_COLS))
        self.feature_names_ = cat_names + ALL_NUMERICAL_COLS
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        engineered_df = self.engineer.transform(df)
        return self.preprocessor.transform(engineered_df)

    def fit_transform(self, df: pd.DataFrame, y=None) -> np.ndarray:
        self.fit(df, y)
        return self.transform(df)
