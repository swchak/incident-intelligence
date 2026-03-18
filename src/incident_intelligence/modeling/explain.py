"""
Explain trained models with SHAP or permutation importance.

This module focuses on *global* explainability, such as:
- overall feature importance per model
- SHAP summary bar plots
- permutation-importance fallback when SHAP is unavailable or fails
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from incident_intelligence.modeling.evaluate import load_df
from incident_intelligence.modeling.explain_utils import (
    _HAS_SHAP,
    ensure_dir,
    get_estimator_and_transformer,
    load_model,
    make_explainer,
    model_output_dir,
    normalize_multiclass_shap,
    shap,
    transform_X,
)


@dataclass
class ExplainConfig:
    """Configuration for global explainability generation."""

    label_col: str = "root_cause_label"
    out_dir: str | Path = "artifacts/explain"
    background_n: int = 100
    explain_n: int = 200
    kernel_bg: int = 40
    kernel_nsamples: int = 80
    perm_repeats: int = 10
    random_state: int = 42
    top_k: int = 20



def save_shap_summary_plot(
    shap_list: List[np.ndarray],
    X_ex: pd.DataFrame,
    classes: List[Any],
    model_name: str,
    cfg: ExplainConfig,
) -> Path | None:
    """Save a SHAP summary bar plot for one model."""
    if not _HAS_SHAP:
        return None

    out_dir = ensure_dir(model_output_dir(cfg, model_name) / "global")
    png_path = out_dir / "shap_importance.png"

    plt.figure()
    shap.summary_plot(
        shap_list,
        X_ex,
        class_names=classes if classes else None,
        plot_type="bar",
        show=False,
    )
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()
    return png_path



def save_permutation_plot(
    importance: pd.Series,
    model_name: str,
    cfg: ExplainConfig,
) -> Path:
    """Save a global permutation-importance bar chart for one model."""
    out_dir = ensure_dir(model_output_dir(cfg, model_name) / "global")
    png_path = out_dir / "importance.png"

    plt.figure()
    importance.head(cfg.top_k).plot(
        kind="bar",
        title=f"{model_name} Feature Importance",
    )
    plt.tight_layout()
    plt.savefig(png_path, dpi=200)
    plt.close()
    return png_path



def global_importance_for_model(
    model_name: str,
    model: Any,
    X: pd.DataFrame,
    y: pd.Series,
    cfg: ExplainConfig,
) -> pd.Series:
    """
    Compute global feature importance for a single fitted model.

    SHAP is attempted first. If SHAP is unavailable or fails, permutation
    importance is used instead.
    """
    rng = np.random.RandomState(cfg.random_state)

    bg_n = min(len(X), cfg.background_n)
    X_bg_raw = X.sample(bg_n, random_state=rng)

    remaining = X.drop(X_bg_raw.index)
    if len(remaining) > 0:
        ex_n = min(len(remaining), cfg.explain_n)
        X_ex_raw = remaining.sample(ex_n, random_state=rng)
    else:
        X_ex_raw = X_bg_raw.copy()

    clf, transformer = get_estimator_and_transformer(model)

    X_bg = transform_X(transformer, X_bg_raw)
    X_ex = transform_X(transformer, X_ex_raw)

    feature_names = X_ex.columns
    classes = list(getattr(clf, "classes_", []))
    n_classes = len(classes)

    explainer, kind = make_explainer(clf, X_bg, cfg)

    if explainer is not None:
        try:
            if kind == "kernel":
                shap_vals = explainer.shap_values(X_ex, nsamples=cfg.kernel_nsamples)
            else:
                shap_vals = explainer.shap_values(X_ex)

            shap_list = normalize_multiclass_shap(shap_vals, max(n_classes, 2))

            save_shap_summary_plot(
                shap_list=shap_list,
                X_ex=X_ex,
                classes=classes,
                model_name=model_name,
                cfg=cfg,
            )

            stacked = np.vstack(shap_list)
            scores = pd.Series(
                np.mean(np.abs(stacked), axis=0),
                index=feature_names,
            ).sort_values(ascending=False)

            return scores

        except Exception as e:
            print(f"[WARN] SHAP failed for {model_name}; falling back to permutation importance. Error: {e}")

    r = permutation_importance(
        clf,
        X_ex,
        y.loc[X_ex_raw.index],
        n_repeats=cfg.perm_repeats,
        random_state=cfg.random_state,
        n_jobs=-1,
    )

    importance = pd.Series(
        r.importances_mean,
        index=feature_names,
    ).sort_values(ascending=False)

    save_permutation_plot(importance, model_name, cfg)
    return importance



def explain_models(
    model_paths: List[Path],
    X: pd.DataFrame,
    y: pd.Series,
    cfg: ExplainConfig,
) -> Dict[str, Any]:
    """Generate global explainability artifacts for one or more model files."""
    root_out_dir = ensure_dir(cfg.out_dir)
    summary: Dict[str, Any] = {"models": []}

    for path in model_paths:
        model_name = path.stem
        print(f"[INFO] Explaining {model_name}")

        model = load_model(path)

        scores = global_importance_for_model(
            model_name=model_name,
            model=model,
            X=X,
            y=y,
            cfg=cfg,
        )

        global_dir = ensure_dir(model_output_dir(cfg, model_name) / "global")
        csv_path = global_dir / "global_importance.csv"
        scores.to_csv(csv_path, header=["importance"])

        summary["models"].append(
            {
                "name": model_name,
                "model_dir": str(model_output_dir(cfg, model_name)),
                "global_dir": str(global_dir),
                "top_features": scores.head(cfg.top_k).to_dict(),
                "csv": str(csv_path),
                "shap_plot": str(global_dir / "shap_importance.png"),
                "perm_plot": str(global_dir / "importance.png"),
            }
        )

    summary_path = root_out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[DONE] Explainability completed")

    return {
        "models": summary["models"],
        "out_dir": str(root_out_dir),
        "summary_path": str(summary_path),
    }



def run_explainability(
    data_path: str | Path,
    cfg: ExplainConfig,
    models_dir: str | Path = "artifacts/models",
    model_path: str | Path | None = None,
) -> Dict[str, Any]:
    """Load labeled data and generate global explanation artifacts."""
    data_path = Path(data_path)
    models_dir = Path(models_dir)
    cfg.out_dir = ensure_dir(cfg.out_dir)

    df = load_df(data_path)
    if cfg.label_col not in df.columns:
        raise KeyError(f"Label column '{cfg.label_col}' not found in {data_path}")

    y = df[cfg.label_col]
    X = df.drop(columns=[cfg.label_col])

    if model_path:
        model_paths = [Path(model_path)]
        if not model_paths[0].exists():
            raise FileNotFoundError(f"Model not found: {model_paths[0]}")
    else:
        model_paths = sorted(models_dir.glob("*.joblib"))
        if not model_paths:
            raise FileNotFoundError(f"No .joblib model files found in {models_dir}")

    return explain_models(
        model_paths=model_paths,
        X=X,
        y=y,
        cfg=cfg,
    )