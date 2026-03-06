from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import json

from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# Optional SHAP import (fallback gracefully if not installed)
try:
    import shap  # type: ignore
    _HAS_SHAP = True
except Exception:
    shap = None
    _HAS_SHAP = False


@dataclass(frozen=True)
class ExplainConfig:
    label_col: str = "root_cause_label"
    out_dir: str = "artifacts/explain"

    # Sampling controls (for speed)
    background_n: int = 100
    explain_n: int = 200

    # Kernel SHAP controls (slow)
    kernel_bg: int = 40
    kernel_nsamples: int = 80

    # Permutation fallback controls
    perm_repeats: int = 10
    random_state: int = 42

    # Output controls
    top_k: int = 20


# -------------------------
# Loading / utility
# -------------------------

def load_pipeline(model_path: str | Path) -> Pipeline:
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = joblib.load(model_path)
    if not isinstance(model, Pipeline):
        raise TypeError(f"Expected sklearn Pipeline, got {type(model)}")
    return model


def find_models(models_dir: str | Path) -> List[Path]:
    models_dir = Path(models_dir)
    if not models_dir.exists():
        raise FileNotFoundError(f"Models directory not found: {models_dir}")
    return sorted(models_dir.glob("*.joblib"))


def split_xy(df: pd.DataFrame, label_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    if label_col not in df.columns:
        raise ValueError(f"label_col='{label_col}' not found. Columns={list(df.columns)}")
    X = df.drop(columns=[label_col])
    y = df[label_col]
    return X, y


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _safe_feature_names(X: pd.DataFrame) -> List[str]:
    return list(X.columns)


def get_estimator_and_transformer(pipe: Pipeline) -> Tuple[Any, Optional[Any]]:
    """
    If your pipeline is like: ('scaler', StandardScaler) -> ('clf', estimator)
    return (estimator, transformer) where transformer may be scaler/feature transformer.
    Otherwise if only estimator exists, transformer=None.
    """
    if "clf" in pipe.named_steps:
        clf = pipe.named_steps["clf"]
        # "scaler" is common in your notebook-style baselines
        transformer = pipe.named_steps.get("scaler", None)
        return clf, transformer

    # Fallback: assume last step is estimator
    steps = list(pipe.named_steps.items())
    if not steps:
        raise ValueError("Pipeline has no steps")
    clf = steps[-1][1]
    # optional transformer: everything before last
    transformer = None
    if len(steps) >= 2:
        # If there's exactly one preprocessing step, use it
        transformer = steps[-2][1]
    return clf, transformer


def transform_X(transformer: Optional[Any], X_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Apply transformer (e.g., StandardScaler) if present, keeping DataFrame if possible.
    """
    if transformer is None:
        return X_raw.copy()

    Xt = transformer.transform(X_raw)
    # If transformer returns numpy, convert to DataFrame with original columns
    if isinstance(Xt, np.ndarray):
        return pd.DataFrame(Xt, columns=X_raw.columns, index=X_raw.index)
    return Xt


# -------------------------
# SHAP builder (matches your notebook intent)
# -------------------------

def make_explainer(clf: Any, X_bg: pd.DataFrame, cfg: ExplainConfig):
    """
    Returns (explainer, kind) where kind in {"tree","linear","kernel"}.
    If unsupported, returns (None, "unsupported").
    """
    if not _HAS_SHAP:
        return None, "no_shap"

    # sklearn GradientBoostingClassifier: TreeExplainer works well for binary, but multiclass is tricky.
    if isinstance(clf, GradientBoostingClassifier):
        n_classes = len(getattr(clf, "classes_", []))
        if n_classes > 2:
            return None, "skip_gb_multiclass"
        return shap.TreeExplainer(clf), "tree"

    # RandomForest / tree-based w/ feature_importances_
    if isinstance(clf, RandomForestClassifier) or hasattr(clf, "feature_importances_"):
        return shap.TreeExplainer(clf), "tree"

    # Logistic regression (linear)
    if isinstance(clf, LogisticRegression):
        return shap.LinearExplainer(clf, X_bg), "linear"

    # SVM
    if isinstance(clf, SVC):
        if getattr(clf, "kernel", None) == "linear":
            return shap.LinearExplainer(clf, X_bg), "linear"
        return shap.KernelExplainer(clf.predict_proba, X_bg.to_numpy()), "kernel"

    # Fallback: kernel if we can do predict_proba
    if hasattr(clf, "predict_proba"):
        return shap.KernelExplainer(clf.predict_proba, X_bg.to_numpy()), "kernel"

    return None, "unsupported"


def _normalize_multiclass_shap(shap_out, n_classes: int) -> List[np.ndarray]:
    """
    Return list[class] -> (n_samples, n_features)
    """
    if isinstance(shap_out, list):
        return [np.asarray(x) for x in shap_out]

    arr = np.asarray(shap_out)
    if arr.ndim == 3 and arr.shape[2] == n_classes:
        return [arr[:, :, i] for i in range(n_classes)]

    # Binary tree/linear often returns (n_samples, n_features)
    if arr.ndim == 2 and n_classes == 2:
        return [arr, -arr]

    raise ValueError(f"Unsupported SHAP output shape: {arr.shape} for n_classes={n_classes}")


# -------------------------
# Global importance
# -------------------------

def global_importance_for_model(
    model: Pipeline,
    X: pd.DataFrame,
    y: pd.Series,
    cfg: ExplainConfig,
) -> Dict[str, Any]:
    """
    Computes global importance using SHAP if possible; otherwise permutation importance.
    Returns a dict with:
      - method
      - runtime_sec
      - importance: list of (feature, score) sorted desc
      - top_features
    """
    import time

    rng = cfg.random_state
    feature_names = _safe_feature_names(X)

    # sample background + explain set
    X_bg_raw = X.sample(min(cfg.background_n, len(X)), random_state=rng)
    X_ex_raw = X.sample(min(cfg.explain_n, len(X)), random_state=rng)

    clf, transformer = get_estimator_and_transformer(model)
    X_bg = transform_X(transformer, X_bg_raw)
    X_ex = transform_X(transformer, X_ex_raw)

    classes = list(getattr(clf, "classes_", []))
    n_classes = len(classes) if classes else 0

    start = time.time()

    # If SHAP is available and supported, do SHAP; else fallback.
    explainer, kind = make_explainer(clf, X_bg, cfg)

    # Special-case: multiclass GradientBoosting => permutation fallback (your notebook does this)
    if isinstance(clf, GradientBoostingClassifier) and n_classes > 2:
        explainer = None
        kind = "skip_gb_multiclass"

    if explainer is None or kind in {"no_shap", "unsupported", "skip_gb_multiclass"}:
        # Permutation importance on the *pipeline* for correctness
        perm = permutation_importance(
            model,
            X_ex_raw,
            y.loc[X_ex_raw.index],
            n_repeats=cfg.perm_repeats,
            random_state=rng,
            n_jobs=-1,
        )

        scores = pd.Series(perm.importances_mean, index=feature_names).sort_values(ascending=False)
        runtime = round(time.time() - start, 2)

        return {
            "method": "permutation",
            "runtime_sec": runtime,
            "importance": [(f, float(v)) for f, v in scores.items()],
            "top_features": list(scores.head(cfg.top_k).index),
        }

    # SHAP path
    if kind == "kernel":
        # Reduce background for speed
        X_bg_small = shap.sample(X_bg, min(cfg.kernel_bg, len(X_bg)))
        explainer = shap.KernelExplainer(clf.predict_proba, X_bg_small.to_numpy())
        shap_vals = explainer.shap_values(X_ex.to_numpy(), nsamples=cfg.kernel_nsamples)
    else:
        shap_vals = explainer.shap_values(X_ex)

    shap_list = _normalize_multiclass_shap(shap_vals, n_classes=max(n_classes, 2))

    # Global importance: mean(|shap|) aggregated across classes
    stacked = np.vstack(shap_list)  # (n_samples * n_classes, n_features)
    scores = pd.Series(np.mean(np.abs(stacked), axis=0), index=feature_names).sort_values(ascending=False)

    runtime = round(time.time() - start, 2)
    return {
        "method": f"shap_{kind}",
        "runtime_sec": runtime,
        "importance": [(f, float(v)) for f, v in scores.items()],
        "top_features": list(scores.head(cfg.top_k).index),
    }


def write_importance_outputs(
    model_name: str,
    importance_items: List[Tuple[str, float]],
    cfg: ExplainConfig,
) -> Tuple[Path, Path]:
    """
    Writes CSV + PNG bar plot for the top-K features.
    """
    out_dir = _ensure_dir(cfg.out_dir)
    df = pd.DataFrame(importance_items, columns=["feature", "importance"])

    csv_path = out_dir / f"{model_name}_global_importance.csv"
    df.to_csv(csv_path, index=False)

    # Plot (matplotlib only)
    import matplotlib.pyplot as plt

    top = df.head(cfg.top_k).copy()
    plt.figure()
    top.iloc[::-1].plot(x="feature", y="importance", kind="barh", legend=False)
    plt.title(f"{model_name} Global Feature Importance")
    plt.tight_layout()

    png_path = out_dir / f"{model_name}_global_importance.png"
    plt.savefig(png_path, dpi=200)
    plt.close()

    return csv_path, png_path


def explain_models(
    models: List[Path],
    df_eval: pd.DataFrame,
    cfg: ExplainConfig,
) -> Dict[str, Any]:
    """
    Runs global explainability for each model and writes:
      - per-model CSV + PNG
      - summary JSON
    """
    out_dir = _ensure_dir(cfg.out_dir)
    X, y = split_xy(df_eval, cfg.label_col)

    results: List[Dict[str, Any]] = []

    for mp in models:
        model = load_pipeline(mp)
        model_name = mp.stem

        gi = global_importance_for_model(model, X, y, cfg)
        importance_items = gi["importance"]
        csv_path, png_path = write_importance_outputs(model_name, importance_items, cfg)

        results.append(
            {
                "model_name": model_name,
                "model_path": str(mp),
                "method": gi["method"],
                "runtime_sec": gi["runtime_sec"],
                "top_features": gi["top_features"],
                "csv": str(csv_path),
                "png": str(png_path),
            }
        )

        print(f"[OK] {model_name}: method={gi['method']} runtime={gi['runtime_sec']}s -> {csv_path.name}, {png_path.name}")

    summary = {
        "label_col": cfg.label_col,
        "out_dir": str(out_dir),
        "models": results,
    }

    summary_path = out_dir / "explainability_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))  # pandas has a safe JSON writer
    print(f"Wrote summary: {summary_path}")

    return summary