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


def load_df(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path.suffix} (use .csv or .parquet)")

def run_explainability(
    *,
    data_path: str | Path,
    cfg: ExplainConfig,
    model_path: str | Path | None = None,
    models_dir: str | Path = "artifacts/models",
) -> Dict[str, Any]:
    df_eval = load_df(data_path)

    if model_path:
        models = [Path(model_path)]
    else:
        models = find_models(models_dir)

    return explain_models(models, df_eval, cfg)


def build_local_explainer_for_model(
    model: Pipeline,
    X_bg_raw: pd.DataFrame,
    cfg: ExplainConfig,
) -> Tuple[Any, str, Any]:
    """
    Returns (explainer, kind, clf) for local explanations.
    Uses a reduced background for kernel explainers.
    """
    clf, transformer = get_estimator_and_transformer(model)
    X_bg = transform_X(transformer, X_bg_raw)

    classes = list(getattr(clf, "classes_", []))

    # Force kernel for multiclass GradientBoosting
    if isinstance(clf, GradientBoostingClassifier) and len(classes) > 2:
        if not _HAS_SHAP:
            raise RuntimeError("SHAP is required for local explainability")
        X_bg_small = shap.sample(X_bg, min(cfg.kernel_bg, len(X_bg)))
        explainer = shap.KernelExplainer(clf.predict_proba, X_bg_small.to_numpy())
        return explainer, "kernel", clf

    explainer, kind = make_explainer(clf, X_bg)

    if explainer is None:
        raise ValueError(f"Local explanation unsupported for model type: {type(clf)}")

    if kind == "kernel":
        X_bg_small = shap.sample(X_bg, min(cfg.kernel_bg, len(X_bg)))
        explainer = shap.KernelExplainer(clf.predict_proba, X_bg_small.to_numpy())

    return explainer, kind, clf


def get_shap_list_for_one(
    explainer: Any,
    kind: str,
    x_one_df: pd.DataFrame,
    n_classes: int,
    cfg: ExplainConfig,
) -> List[np.ndarray]:
    """
    Returns list[class] -> (1, n_features) for one example.
    """
    if kind == "kernel":
        shap_one = explainer.shap_values(
            x_one_df.to_numpy(),
            nsamples=cfg.kernel_nsamples,
        )
    else:
        shap_one = explainer.shap_values(x_one_df)

    return _normalize_multiclass_shap(shap_one, n_classes=n_classes)


def _get_base_value(explainer: Any, class_index: int) -> float | None:
    """
    Robust base value extraction for multiclass explainers.
    """
    base = getattr(explainer, "expected_value", None)
    if base is None:
        return None

    if isinstance(base, (list, np.ndarray)):
        base = np.asarray(base).reshape(-1)
        if len(base) > class_index:
            return float(base[class_index])
        return float(base[0])

    return float(base)


def local_waterfall_for_pred_class(
    explainer: Any,
    kind: str,
    clf: Any,
    x_one_df: pd.DataFrame,
    cfg: ExplainConfig,
    out_path: str | Path | None = None,
) -> Dict[str, Any]:
    """
    Builds a SHAP waterfall for the predicted class of one row.
    Optionally saves the figure.
    """
    if not _HAS_SHAP:
        raise RuntimeError("SHAP is required for local waterfall plots")

    classes = list(clf.classes_)
    pred = clf.predict(x_one_df)[0]
    class_index = classes.index(pred)

    shap_list = get_shap_list_for_one(
        explainer=explainer,
        kind=kind,
        x_one_df=x_one_df,
        n_classes=len(classes),
        cfg=cfg,
    )
    sv = shap_list[class_index][0]
    base = _get_base_value(explainer, class_index)

    exp = shap.Explanation(
        values=sv,
        base_values=base,
        data=x_one_df.iloc[0].values,
        feature_names=list(x_one_df.columns),
    )

    import matplotlib.pyplot as plt

    shap.plots.waterfall(exp, max_display=cfg.top_k, show=False)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.tight_layout()
        plt.savefig(out_path, dpi=200, bbox_inches="tight")

    plt.close()

    return {
        "predicted_class": pred,
        "class_index": class_index,
        "feature_contributions": pd.Series(
            sv,
            index=x_one_df.columns,
        ).sort_values(key=np.abs, ascending=False).to_dict(),
    }


def topk_rca(
    explainer: Any,
    kind: str,
    clf: Any,
    x_one_df: pd.DataFrame,
    cfg: ExplainConfig,
    k: int = 3,
    top_feats: int = 8,
) -> List[Dict[str, Any]]:
    """
    Returns the top-k predicted classes and their most influential features.
    """
    classes = list(clf.classes_)
    proba = clf.predict_proba(x_one_df)[0]
    order = np.argsort(proba)[::-1][:k]

    shap_list = get_shap_list_for_one(
        explainer=explainer,
        kind=kind,
        x_one_df=x_one_df,
        n_classes=len(classes),
        cfg=cfg,
    )

    results: List[Dict[str, Any]] = []

    for class_index in order:
        class_name = classes[class_index]
        prob = float(proba[class_index])

        sv = shap_list[class_index][0]
        s = (
            pd.Series(sv, index=x_one_df.columns)
            .sort_values(key=np.abs, ascending=False)
            .head(top_feats)
        )

        results.append(
            {
                "class_name": class_name,
                "probability": prob,
                "top_features": [
                    {"feature": feature, "shap_value": float(value)}
                    for feature, value in s.items()
                ],
            }
        )

    return results


def run_local_explainability(
    *,
    data_path: str | Path,
    model_path: str | Path,
    cfg: ExplainConfig,
    row_indices: List[int] | None = None,
    n_examples: int = 3,
    top_k_classes: int = 3,
    top_features_per_class: int = 8,
) -> Dict[str, Any]:
    """
    Generates local explainability artifacts for one model on selected eval rows.
    Saves waterfall plots and a JSON summary.
    """
    df_eval = load_df(data_path)
    X, y = split_xy(df_eval, cfg.label_col)

    model = load_pipeline(model_path)
    model_name = Path(model_path).stem

    X_bg_raw = X.sample(min(cfg.background_n, len(X)), random_state=cfg.random_state)
    explainer, kind, clf = build_local_explainer_for_model(model, X_bg_raw, cfg)

    if row_indices is None:
        row_indices = list(
            X.sample(min(n_examples, len(X)), random_state=cfg.random_state).index
        )

    out_dir = _ensure_dir(Path(cfg.out_dir) / model_name / "local")
    local_results: List[Dict[str, Any]] = []

    clf, transformer = get_estimator_and_transformer(model)

    for idx in row_indices:
        x_one_raw = X.loc[[idx]]
        x_one = transform_X(transformer, x_one_raw)
        true_label = y.loc[idx]
        pred_label = clf.predict(x_one)[0]

        waterfall_path = out_dir / f"row_{idx}_waterfall.png"
        waterfall = local_waterfall_for_pred_class(
            explainer=explainer,
            kind=kind,
            clf=clf,
            x_one_df=x_one,
            cfg=cfg,
            out_path=waterfall_path,
        )

        rca = topk_rca(
            explainer=explainer,
            kind=kind,
            clf=clf,
            x_one_df=x_one,
            cfg=cfg,
            k=top_k_classes,
            top_feats=top_features_per_class,
        )

        local_results.append(
            {
                "row_index": int(idx),
                "true_label": true_label,
                "predicted_label": pred_label,
                "waterfall_png": str(waterfall_path),
                "waterfall": waterfall,
                "topk_rca": rca,
            }
        )

    summary = {
        "model_name": model_name,
        "model_path": str(model_path),
        "method": kind,
        "rows": local_results,
    }

    written_reports = write_local_html_reports(summary)
    summary["html_reports"] = [str(p) for p in written_reports]
    
    written_reports = write_local_markdown_reports(summary)
    summary["markdown_reports"] = [str(p) for p in written_reports]

    summary_path = out_dir / "local_explainability_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    return summary


def write_local_markdown_reports(
    summary: Dict[str, Any],
) -> List[Path]:
    """
    Writes one markdown RCA report per explained row plus an index report.
    Expects the output from run_local_explainability(...).
    """
    model_name = summary["model_name"]
    rows = summary["rows"]

    if not rows:
        return []

    first_waterfall = Path(rows[0]["waterfall_png"])
    out_dir = first_waterfall.parent

    written: List[Path] = []

    # One report per row
    for row in rows:
        row_index = row["row_index"]
        report_path = out_dir / f"row_{row_index}_report.md"

        waterfall_name = Path(row["waterfall_png"]).name

        lines: List[str] = [
            f"# Local RCA Report: row {row_index}",
            "",
            f"- Model: `{model_name}`",
            f"- True label: `{row['true_label']}`",
            f"- Predicted label: `{row['predicted_label']}`",
            "",
            "## Predicted Class Waterfall",
            "",
            f"![Waterfall]({waterfall_name})",
            "",
            "## Top Candidate Root Causes",
            "",
        ]

        for i, candidate in enumerate(row["topk_rca"], start=1):
            lines.extend(
                [
                    f"### {i}. {candidate['class_name']} ({candidate['probability']:.4f})",
                    "",
                    "| Feature | SHAP value |",
                    "|---|---:|",
                ]
            )

            for feat in candidate["top_features"]:
                lines.append(f"| {feat['feature']} | {feat['shap_value']:.6f} |")

            lines.append("")

        report_path.write_text("\n".join(lines))
        written.append(report_path)

    # Index report
    index_path = out_dir / "local_explainability_index.md"
    index_lines: List[str] = [
        f"# Local Explainability Index: {model_name}",
        "",
        f"- Method: `{summary['method']}`",
        f"- Number of explained rows: `{len(rows)}`",
        "",
        "## Reports",
        "",
    ]

    for row in rows:
        row_index = row["row_index"]
        index_lines.append(
            f"- [row {row_index}](row_{row_index}_report.md) "
            f"(true=`{row['true_label']}`, pred=`{row['predicted_label']}`)"
        )

    index_lines.append("")
    index_path.write_text("\n".join(index_lines))
    written.append(index_path)

    return written

from html import escape


def write_local_html_reports(
    summary: Dict[str, Any],
) -> List[Path]:
    """
    Writes one HTML RCA report per explained row plus an index HTML page.
    Expects the output from run_local_explainability(...).
    """
    model_name = summary["model_name"]
    rows = summary["rows"]

    if not rows:
        return []

    first_waterfall = Path(rows[0]["waterfall_png"])
    out_dir = first_waterfall.parent

    written: List[Path] = []

    css = """
    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
        margin: 32px;
        line-height: 1.5;
        color: #222;
        max-width: 1100px;
    }
    h1, h2, h3 {
        color: #111;
    }
    .meta {
        background: #f6f8fa;
        border: 1px solid #d0d7de;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 24px;
    }
    .card {
        border: 1px solid #d0d7de;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
        background: #fff;
    }
    .pill {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 999px;
        background: #eef2ff;
        border: 1px solid #c7d2fe;
        margin-right: 8px;
        font-size: 0.95em;
    }
    img {
        max-width: 100%;
        height: auto;
        border: 1px solid #d0d7de;
        border-radius: 8px;
        background: #fff;
        padding: 6px;
    }
    table {
        border-collapse: collapse;
        width: 100%;
        margin: 12px 0 24px 0;
    }
    th, td {
        border: 1px solid #d0d7de;
        padding: 8px 10px;
        text-align: left;
    }
    th {
        background: #f6f8fa;
    }
    .small {
        color: #57606a;
        font-size: 0.95em;
    }
    a {
        color: #0969da;
        text-decoration: none;
    }
    a:hover {
        text-decoration: underline;
    }
    """

    # Per-row reports
    for row in rows:
        row_index = row["row_index"]
        report_path = out_dir / f"row_{row_index}_report.html"
        waterfall_name = Path(row["waterfall_png"]).name

        candidates_html: List[str] = []

        for i, candidate in enumerate(row["topk_rca"], start=1):
            rows_html = "\n".join(
                f"<tr><td>{escape(str(feat['feature']))}</td><td>{feat['shap_value']:.6f}</td></tr>"
                for feat in candidate["top_features"]
            )

            candidates_html.append(
                f"""
                <div class="card">
                    <h3>{i}. {escape(str(candidate['class_name']))}</h3>
                    <p><span class="pill">Probability: {candidate['probability']:.4f}</span></p>
                    <table>
                        <thead>
                            <tr>
                                <th>Feature</th>
                                <th>SHAP value</th>
                            </tr>
                        </thead>
                        <tbody>
                            {rows_html}
                        </tbody>
                    </table>
                </div>
                """
            )

        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>Local RCA Report: row {row_index}</title>
            <style>{css}</style>
        </head>
        <body>
            <h1>Local RCA Report: row {row_index}</h1>

            <div class="meta">
                <p><strong>Model:</strong> {escape(str(model_name))}</p>
                <p><strong>True label:</strong> <span class="pill">{escape(str(row['true_label']))}</span></p>
                <p><strong>Predicted label:</strong> <span class="pill">{escape(str(row['predicted_label']))}</span></p>
            </div>

            <h2>Predicted Class Waterfall</h2>
            <p class="small">This plot shows the feature contributions driving the predicted class.</p>
            <img src="{escape(waterfall_name)}" alt="Waterfall plot for row {row_index}">

            <h2>Top Candidate Root Causes</h2>
            <p class="small">These are the top predicted classes with their strongest contributing features.</p>

            {''.join(candidates_html)}

            <p><a href="local_explainability_index.html">Back to index</a></p>
        </body>
        </html>
        """

        report_path.write_text(html, encoding="utf-8")
        written.append(report_path)

    # Index page
    links_html = "\n".join(
        f"""
        <tr>
            <td>{row['row_index']}</td>
            <td>{escape(str(row['true_label']))}</td>
            <td>{escape(str(row['predicted_label']))}</td>
            <td><a href="row_{row['row_index']}_report.html">Open report</a></td>
        </tr>
        """
        for row in rows
    )

    index_path = out_dir / "local_explainability_index.html"
    index_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Local Explainability Index: {escape(str(model_name))}</title>
        <style>{css}</style>
    </head>
    <body>
        <h1>Local Explainability Index</h1>

        <div class="meta">
            <p><strong>Model:</strong> {escape(str(model_name))}</p>
            <p><strong>Method:</strong> <span class="pill">{escape(str(summary['method']))}</span></p>
            <p><strong>Explained rows:</strong> <span class="pill">{len(rows)}</span></p>
        </div>

        <table>
            <thead>
                <tr>
                    <th>Row index</th>
                    <th>True label</th>
                    <th>Predicted label</th>
                    <th>Report</th>
                </tr>
            </thead>
            <tbody>
                {links_html}
            </tbody>
        </table>
    </body>
    </html>
    """

    index_path.write_text(index_html, encoding="utf-8")
    written.append(index_path)

    return written