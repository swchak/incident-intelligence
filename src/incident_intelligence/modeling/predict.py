"""
Module for generating predictions from a fitted model pipeline.
"""

from __future__ import annotations

from typing import Any, Dict

import pandas as pd
from sklearn.pipeline import Pipeline


def predict_outputs(
    model: Pipeline,
    X: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Generate predictions and predicted probabilities (if supported) from a fitted model.

    Parameters
    ----------
    model : Pipeline
        A fitted sklearn Pipeline or estimator that supports predict and optionally predict_proba.
    X : pd.DataFrame
        Input features for generating predictions. Should have the same columns used during training.

    Returns
    -------
    Dict[str, Any]
        A dictionary containing:
            y_pred: predicted class labels
            y_proba: predicted class probabilities if available, otherwise None
    """

    y_pred = model.predict(X)

    y_proba = None
    if hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X)
        except Exception:
            y_proba = None

    return {
        "y_pred": y_pred,
        "y_proba": y_proba,
    }