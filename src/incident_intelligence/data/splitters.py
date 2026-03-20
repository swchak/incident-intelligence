from __future__ import annotations

from typing import Tuple
from sklearn.model_selection import train_test_split
import pandas as pd


def split_by_incident(
    feature_df: pd.DataFrame,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    random_seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if abs(train_size + val_size + test_size - 1.0) > 1e-8:
        raise ValueError("train_size + val_size + test_size must sum to 1.0")

    if "root_cause_label" not in feature_df.columns:
        raise ValueError("feature_df must contain root_cause_label")

    train_df, temp_df = train_test_split(
        feature_df,
        test_size=(1.0 - train_size),
        stratify=feature_df["root_cause_label"],
        random_state=random_seed,
    )

    relative_test_size = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test_size,
        stratify=temp_df["root_cause_label"],
        random_state=random_seed,
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)