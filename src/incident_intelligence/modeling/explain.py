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
from incident_intelligence.modeling.evaluate import find_model_files
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
    """
    Configuration for global explainability generation.
    This includes parameters for SHAP explainers, permutation importance, and output settings.
    Users can override these settings via CLI arguments or by modifying the default values here.
    """
    
    # Column name in the data containing labels/targets
    label_col: str = "root_cause_label"

    # Output directory where explainability artifacts will be saved
    out_dir: str | Path = "artifacts/explain"

    # Number of samples to use for SHAP background dataset
    background_n: int = 100

    # Number of samples to use for SHAP explanation dataset
    explain_n: int = 200

    # Number of background samples for kernel explainer
    kernel_bg: int = 40

    # Number of samples for kernel explainer SHAP value computation
    kernel_nsamples: int = 80

    # Number of permutations for permutation importance calculation
    perm_repeats: int = 10

    # Random seed for reproducibility
    random_state: int = 42

    # Number of top features to include in output summary
    top_k: int = 20


def save_shap_summary_plot(
    shap_list: List[np.ndarray],
    X_ex: pd.DataFrame,
    classes: List[Any],
    model_name: str,
    cfg: ExplainConfig,
) -> Path | None:
    """
    Save a SHAP summary plot for one model. Returns the path to the saved plot.
    The plot is a bar chart of mean absolute SHAP values for the top features.

    Parameters
    ----------
    shap_list: List of SHAP value arrays (one per class)
    X_ex: DataFrame of samples used for explanation (after transformation)
    classes: List of class labels (if available)
    model_name: Name of the model being explained
    cfg: ExplainConfig with output settings
    
    Returns
    -------
    Path to the saved SHAP summary plot, or None if SHAP is unavailable
    """
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
    """
    Save a permutation importance plot for one model. Returns the path to the saved plot.
    The plot is a bar chart of mean permutation importance scores for the top features.

    Parameters
    ----------
    importance: Series of feature importance scores indexed by feature name
    model_name: Name of the model being explained
    cfg: ExplainConfig with output settings
    
    Returns
    -------
    Path to the saved permutation importance plot
    """
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
    importance is used instead. The resulting importance scores are saved as CSV 
    and plotted, with paths returned in the output summary.
    
    Parameters
    ----------
    model_name: Name of the model (used for labeling outputs)
    model: Fitted model pipeline (e.g. sklearn Pipeline) to explain
    X: DataFrame of input features (after any necessary transformations)
    y: Series of target labels corresponding to X
    cfg: ExplainConfig with settings for SHAP and permutation importance
    
    Returns
    -------
    Series of feature importance scores indexed by feature name, sorted descending
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
    """
    Generate global explainability artifacts for a list of models. Returns a summary dictionary with paths to artifacts.
    For each model, global feature importance is computed using SHAP (if available) or permutation importance (as a fallback). 
    The importance scores are saved as CSV and plotted, with paths included in the summary dictionary.

    Parameters
    ----------
    model_paths: List of file paths to the saved .joblib models to explain
    X: DataFrame of input features for the evaluation dataset
    y: Series of target labels corresponding to X
    cfg: ExplainConfig with settings for explainability generation

    Returns
    -------
    Dictionary containing a list of explained models with paths to their artifacts and a summary JSON path
    """
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
    dataset_kind: str | None = None,
) -> Dict[str, Any]:
    """
    Main entry point for generating explainability artifacts for one or more models.
    Loads the evaluation dataset, identifies models to explain (either a single model or all models in
    a directory), and generates global feature importance explanations for each model using SHAP or permutation importance.
    The resulting artifacts (importance scores, plots) are saved to disk, and a summary dictionary
    with paths to the artifacts is returned.

    Parameters
    ----------
    data_path: File path to the evaluation dataset (CSV) containing features and labels
    cfg: ExplainConfig with settings for explainability generation
    models_dir: Directory containing .joblib model files to explain (ignored if model_path is provided)
    model_path: Optional file path to a single .joblib model to explain (overrides models_dir if provided)

    Returns
    -------
    Dictionary containing a list of explained models with paths to their artifacts and a summary JSON path
    """
    data_path = Path(data_path)
    models_dir = Path(models_dir)
    cfg.out_dir = ensure_dir(cfg.out_dir)

    df = load_df(data_path)
    if cfg.label_col not in df.columns:
        raise KeyError(f"Label column '{cfg.label_col}' not found in {data_path}")

    y = df[cfg.label_col]
    drop_cols = [cfg.label_col]
    if "incident_id" in df.columns:
        drop_cols.append("incident_id")
    X = df.drop(columns=drop_cols)

    if model_path:
        model_paths = [Path(model_path)]
        if not model_paths[0].exists():
            raise FileNotFoundError(f"Model not found: {model_paths[0]}")
    else:
        model_paths = find_model_files(models_dir, dataset_kind=dataset_kind)
        if not model_paths:
            raise FileNotFoundError(f"No .joblib model files found in {models_dir}")

    return explain_models(
        model_paths=model_paths,
        X=X,
        y=y,
        cfg=cfg,
    )


def run_explainability_for_dataset_kind(
    *,
    dataset_kind: str,
    cfg: ExplainConfig,
    model_path: str | Path | None = None,
    models_dir: str | Path = "artifacts/models",
) -> Dict[str, Any]:
    """
    Generate explainability artifacts using the standard processed evaluation
    dataset for the requested dataset family.
    """
    if dataset_kind == "snapshot":
        data_path = Path("data/processed/incident_snapshot_eval.csv")
    elif dataset_kind == "temporal":
        data_path = Path("data/processed/incident_temporal_eval.csv")
    else:
        raise ValueError(
            f"Unsupported dataset_kind='{dataset_kind}'. "
            "Expected one of: ['snapshot', 'temporal']"
        )

    return run_explainability(
        data_path=data_path,
        cfg=cfg,
        models_dir=models_dir,
        model_path=model_path,
        dataset_kind=dataset_kind,
    )
