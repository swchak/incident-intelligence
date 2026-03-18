from __future__ import annotations

from typing import Any, Dict

import pandas as pd
from sklearn.pipeline import Pipeline


def predict_outputs(
    model: Pipeline,
    X: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Run inference for a fitted classification pipeline.

    Args:
        model: Fitted sklearn Pipeline.
        X: Feature dataframe.

    Returns:
        Dictionary containing:
            y_pred: predicted class labels
            y_proba: predicted class probabilities if available
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