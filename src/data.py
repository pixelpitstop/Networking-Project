"""
Data loading, cleaning, and scenario-aware train/test splitting.
"""

import os
import pandas as pd
import numpy as np

DROPPED_COLUMNS = [
    "uid",          # Leakage: unique transaction identifier
    "src_ip",       # Leakage: static testbed attacker IP addresses
    "dst_ip",       # Leakage: 88.4% attacks target single victim IP
    "src_port",     # Leakage/Noise: ephemeral high port numbers
    "history",      # High cardinality string; summarized in conn_state
]

CATEGORICAL_FEATURES = ["proto", "service", "conn_state"]

NUMERICAL_FEATURES = [
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

TARGET_COLUMN = "traffic_type"
LABEL_COLUMNS = ["attack_category", "attack_type"]


def load_raw_data(filepath: str = "data/raw/Network_Dataset.csv") -> pd.DataFrame:
    """Load the raw dataset from CSV."""
    if not os.path.exists(filepath):
        # Fallback to rawdata/ if data/raw/ not present
        alt_path = "rawdata/Network_Dataset.csv"
        if os.path.exists(alt_path):
            filepath = alt_path
        else:
            raise FileNotFoundError(f"Dataset not found at {filepath} or {alt_path}")
    df = pd.read_csv(filepath)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop leaky and identifier columns while retaining features and labels."""
    cols_to_drop = [c for c in DROPPED_COLUMNS if c in df.columns]
    return df.drop(columns=cols_to_drop)


def split_data(
    df: pd.DataFrame,
    test_ratio: float = 0.30,
    random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset preserving scenario/attack integrity.
    Within each (attack_category, attack_type) group, the first (1 - test_ratio)
    fraction is assigned to train, and the trailing test_ratio fraction to test.
    This preserves sequential flow ordering without mixing future and past flows,
    and ensures every attack type is present in both train and test.
    """
    train_dfs = []
    test_dfs = []

    for _, group in df.groupby(["attack_category", "attack_type"]):
        n = len(group)
        split_idx = int(n * (1.0 - test_ratio))
        # Ensure at least 1 sample in test if group has > 1 sample
        if split_idx == n and n > 1:
            split_idx = n - 1
        train_dfs.append(group.iloc[:split_idx])
        test_dfs.append(group.iloc[split_idx:])

    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)
    return train_df, test_df


def prepare_and_save_splits(
    raw_path: str = "data/raw/Network_Dataset.csv",
    output_dir: str = "data/processed",
    test_ratio: float = 0.30
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load raw data, clean leaky columns, split, and save processed splits to parquet."""
    os.makedirs(output_dir, exist_ok=True)
    print("Loading raw dataset...")
    df = load_raw_data(raw_path)
    print(f"Loaded {len(df)} rows. Cleaning leaky columns...")
    df_cleaned = clean_data(df)
    
    print("Splitting into train and test sets (group-aware 70/30)...")
    train_df, test_df = split_data(df_cleaned, test_ratio=test_ratio)
    
    train_path = os.path.join(output_dir, "train.parquet")
    test_path = os.path.join(output_dir, "test.parquet")
    
    print(f"Saving train split ({len(train_df)} rows) to {train_path}...")
    train_df.to_parquet(train_path, index=False)
    print(f"Saving test split ({len(test_df)} rows) to {test_path}...")
    test_df.to_parquet(test_path, index=False)
    
    return train_df, test_df


if __name__ == "__main__":
    train, test = prepare_and_save_splits()
    print(f"Train benign: {(train['traffic_type'] == 0).sum()}, attack: {(train['traffic_type'] == 1).sum()}")
    print(f"Test benign: {(test['traffic_type'] == 0).sum()}, attack: {(test['traffic_type'] == 1).sum()}")
