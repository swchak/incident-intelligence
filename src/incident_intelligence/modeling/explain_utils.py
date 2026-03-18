"""
Shared utilities for global and local model explainability.

This module centralizes the common helpers used by both global explanation
workflows and row-level local explanation workflows. It handles:

- SHAP import / availability detection
- output directory creation
- model loading
- pipeline estimator/transformer extraction
- feature transformation for explanation-time inputs
- SHAP explainer creation and multiclass output normalization
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.pipeline import Pipeline

try:
    import shap

    _HAS_SHAP = True
except Exception:
    shap = None
    _HAS_SHAP = False


plt.rcParams.update(
    {
        "figure.figsize": (10, 6),
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    }
)


def ensure_dir(path: str | Path) -> Path:
    """Create a directory if needed and return it as a ``Path``."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path



def _safe_name(name: str) -> str:
    """Convert a display name into a filesystem-safe filename stem."""
    return (
        str(name)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("(", "")
        .replace(")", "")
    )



def model_output_dir(cfg: Any, model_name: str) -> Path:
    """Return the output directory for one explained model."""
    base = Path(cfg.out_dir)
    return ensure_dir(base / _safe_name(model_name))



def load_model(path: str | Path) -> Any:
    """Load a serialized model artifact from disk."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)



def transform_X(transformer: Any, X: pd.DataFrame) -> pd.DataFrame:
    """
    Transform an input feature frame using a fitted preprocessing step.

    When no transformer is present, the original dataframe is returned.
    When a non-dataframe output is returned, it is converted back into a
    dataframe using the original column names and index.
    """
    if transformer is None:
        return X

    Xt = transformer.transform(X)

    if isinstance(Xt, pd.DataFrame):
        return Xt

    return pd.DataFrame(Xt, columns=X.columns, index=X.index)



def normalize_multiclass_shap(shap_vals: Any, n_classes: int) -> List[np.ndarray]:
    """
    Normalize SHAP outputs into a list of per-class arrays.

    SHAP may return values as:
    - a list of arrays (already normalized)
    - a 2D array for binary classification
    - a 3D array for multiclass classification
    """
    if isinstance(shap_vals, list):
        return shap_vals

    arr = np.array(shap_vals)

    if arr.ndim == 2:
        if n_classes <= 2:
            return [arr, -arr]
        raise ValueError("Received 2D SHAP values for multiclass model.")

    if arr.ndim == 3:
        return [arr[:, :, i] for i in range(arr.shape[2])]

    raise ValueError(f"Unexpected SHAP shape: {arr.shape}")



def get_estimator_and_transformer(model: Any) -> Tuple[ClassifierMixin, Any]:
    """
    Split a pipeline-like model into final estimator and upstream transformer.

    The transformer is assumed to be the second-to-last step when the model is
    an sklearn ``Pipeline`` with multiple steps.
    """
    if isinstance(model, Pipeline):
        clf = model.steps[-1][1]
        transformer = model.steps[-2][1] if len(model.steps) > 1 else None
        return clf, transformer
    return model, None



def make_explainer(
    clf: ClassifierMixin,
    X_bg: pd.DataFrame,
    cfg: Any,
) -> Tuple[Any, str]:
    """
    Build the most appropriate SHAP explainer for a fitted classifier.

    Returns a tuple of ``(explainer, kind)`` where ``kind`` is one of:
    - ``tree``
    - ``linear``
    - ``kernel``
    - ``none``
    """
    if not _HAS_SHAP:
        return None, "none"

    if hasattr(clf, "predict_proba"):
        type_name = str(type(clf))

        if "RandomForest" in type_name or "XGB" in type_name or "GradientBoosting" in type_name:
            try:
                return shap.TreeExplainer(clf), "tree"
            except Exception:
                pass

        if "LogisticRegression" in type_name:
            try:
                return shap.LinearExplainer(clf, X_bg), "linear"
            except Exception:
                pass

        try:
            X_bg_small = shap.sample(X_bg, min(cfg.kernel_bg, len(X_bg)))
        except Exception:
            X_bg_small = X_bg.sample(min(cfg.kernel_bg, len(X_bg)), random_state=cfg.random_state)

        return shap.KernelExplainer(clf.predict_proba, X_bg_small), "kernel"

    return None, "none"
