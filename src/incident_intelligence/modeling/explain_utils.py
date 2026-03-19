"""
Shared utilities for global and local model explainability.      
It is not intended to be used directly by end users, but rather to support 
    - global explainability workflow in ``explain.py`` 
    - local explainability workflow in ``explain_local.py``. 

It handles:
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


"""Matplotlib global style settings for SHAP plots."""
plt.rcParams.update(
    {
        "figure.figsize": (10, 6),
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    }
)



def ensure_dir(path: str | Path) -> Path:
    """
    Create a directory if needed and return it as a ``Path``.
    This is used to create output directories for explainability artifacts.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path



def _safe_name(name: str) -> str:
    """
    Convert a display name into a filesystem-safe filename stem.
    Removes spaces, slashes, backslashes, and parentheses.
    """
    return (
        str(name)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("(", "")
        .replace(")", "")
    )



def model_output_dir(cfg: Any, model_name: str) -> Path:
    """
    Return the output directory for the model, creating it if needed.
    The directory name is derived from the sanitized model name.
    """
    base = Path(cfg.out_dir)
    return ensure_dir(base / _safe_name(model_name))



def load_model(path: str | Path) -> Any:
    """
    Load a fitted model from a .joblib file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)



def transform_X(transformer: Any, X: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature transformations at explanation time to match the training pipeline.
    Returns the original dataframe if no transformer is provided.
    
    Parameters
    ----------
    transformer: Fitted transformer or None
    X: Input dataframe to transform
    
    Returns
    -------
    Transformed dataframe matching the model's feature space
    """
    if transformer is None:
        return X

    Xt = transformer.transform(X)

    if isinstance(Xt, pd.DataFrame):
        return Xt

    return pd.DataFrame(Xt, columns=X.columns, index=X.index)



def normalize_multiclass_shap(shap_vals: Any, n_classes: int) -> List[np.ndarray]:
    """
    Normalize SHAP output into a consistent list of 2D arrays (one per class).
    
    Handles different SHAP formats:
    - List of arrays: returned as-is
    - 2D array (binary): returns [array, -array]
    - 3D array (multiclass): splits along class dimension
    
    Parameters
    ----------
    shap_vals: Raw SHAP values (list, 2D, or 3D array)
    n_classes: Number of classes
    
    Returns
    -------
    List of 2D arrays, one per class with shape (n_samples, n_features)
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
    Extract the final estimator and preceding transformer from a model pipeline.
    
    For pipelines, returns the final step as the estimator and earlier steps as transformer.
    For non-pipeline models, returns the model as estimator with None for transformer.
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
    Create the most appropriate SHAP explainer for a classifier.

    Parameters
    ----------
    clf: The fitted classifier to explain (e.g., RandomForestClassifier, LogisticRegression, etc.)
    X_bg: Background dataset for SHAP (used for KernelExplainer)
    cfg: Configuration object containing settings like kernel_bg and random_state   
    
    Returns
    -------
    A tuple of (explainer, kind) where 
        explainer is the created SHAP explainer object (or None if creation failed) 
        kind is a string indicating the type of explainer created ("tree", "linear", "kernel", or "none").
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
