"""
Explain trained models with SHAP + permutation importance.

Generates:
- Global feature importance (SHAP or permutation)
- SHAP summary bar plots
- Local SHAP waterfall explanations

Designed to reproduce notebook visual outputs with the same look.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline

try:
    import shap

    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False


plt.rcParams.update(
    {
        "figure.figsize": (10, 6),
        "axes.titlesize": 14,
        "axes.labelsize": 12,
    }
)


@dataclass
class ExplainConfig:
    label_col: str = "root_cause_label"
    out_dir: str | Path = "artifacts/explain"
    background_n: int = 100
    explain_n: int = 200
    kernel_bg: int = 40
    kernel_nsamples: int = 80
    perm_repeats: int = 10
    random_state: int = 42
    top_k: int = 20

def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path

def _safe_name(name: str) -> str:
    return (
        str(name)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("(", "")
        .replace(")", "")
    )

def model_output_dir(cfg: ExplainConfig, model_name: str) -> Path:
    base = Path(cfg.out_dir)
    return ensure_dir(base / _safe_name(model_name))


def load_model(path: Path) -> Any:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    return joblib.load(path)


def transform_X(transformer: Any, X: pd.DataFrame) -> pd.DataFrame:
    if transformer is None:
        return X

    Xt = transformer.transform(X)

    if isinstance(Xt, pd.DataFrame):
        return Xt

    return pd.DataFrame(Xt, columns=X.columns, index=X.index)


def normalize_multiclass_shap(shap_vals: Any, n_classes: int) -> List[np.ndarray]:
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
    if isinstance(model, Pipeline):
        clf = model.steps[-1][1]
        transformer = model.steps[-2][1] if len(model.steps) > 1 else None
        return clf, transformer
    return model, None


def make_explainer(
    clf: ClassifierMixin,
    X_bg: pd.DataFrame,
    cfg: ExplainConfig,
) -> Tuple[Any, str]:
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


def save_shap_summary_plot(
    shap_list: List[np.ndarray],
    X_ex: pd.DataFrame,
    classes: List[Any],
    model_name: str,
    cfg: ExplainConfig,
) -> Path | None:
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
                shap_vals = explainer.shap_values(
                    X_ex,
                    nsamples=cfg.kernel_nsamples,
                )
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
    with open(summary_path, "w") as f:
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
    models_dir: str | Path,
) -> Dict[str, Any]:
    data_path = Path(data_path)
    models_dir = Path(models_dir)
    cfg.out_dir = ensure_dir(cfg.out_dir)

    df = pd.read_csv(data_path)
    if cfg.label_col not in df.columns:
        raise KeyError(f"Label column '{cfg.label_col}' not found in {data_path}")

    y = df[cfg.label_col]
    X = df.drop(columns=[cfg.label_col])

    model_paths = sorted(models_dir.glob("*.joblib"))
    if not model_paths:
        raise FileNotFoundError(f"No .joblib model files found in {models_dir}")

    return explain_models(
        model_paths=model_paths,
        X=X,
        y=y,
        cfg=cfg,
    )

def save_local_waterfall_plot(
    explanation: Any,
    model_name: str,
    row_index: int,
    class_name: Any,
    cfg: ExplainConfig,
) -> Path | None:
    if not _HAS_SHAP:
        return None

    out_dir = ensure_dir(model_output_dir(cfg, model_name) / "local")
    png_path = out_dir / (
        f"row_{row_index}_class_{_safe_name(class_name)}_waterfall.png"
    )

    plt.figure()
    shap.plots.waterfall(explanation, max_display=cfg.top_k, show=False)
    plt.tight_layout()
    plt.savefig(png_path, dpi=200, bbox_inches="tight")
    plt.close()
    return png_path

def save_local_json(
    payload: Dict[str, Any],
    model_name: str,
    row_index: int,
    cfg: ExplainConfig,
) -> Path:
    out_dir = ensure_dir(model_output_dir(cfg, model_name) / "local")
    json_path = out_dir / f"row_{row_index}.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return json_path

def save_local_markdown(
    payload: Dict[str, Any],
    model_name: str,
    row_index: int,
    cfg: ExplainConfig,
) -> Path:
    out_dir = ensure_dir(model_output_dir(cfg, model_name) / "local")
    md_path = out_dir / f"row_{row_index}.md"

    lines = [
        f"# Local Explanation - {model_name}",
        "",
        f"- Row index: {payload['row_index']}",
        f"- True label: {payload.get('true_label')}",
        f"- Predicted label: {payload.get('predicted_label')}",
        "",
    ]

    for cls in payload.get("classes", []):
        lines.append(f"## Class: {cls['class']}")
        lines.append("")
        lines.append("| Feature | SHAP value | |SHAP| |")
        lines.append("|---|---:|---:|")
        for item in cls.get("top_features", []):
            lines.append(
                f"| {item['feature']} | {item['shap_value']:.6f} | {item['abs_shap_value']:.6f} |"
            )
        if cls.get("waterfall_plot"):
            lines.append("")
            lines.append(f"- Waterfall plot: `{cls['waterfall_plot']}`")
        lines.append("")

    md_path.write_text("\n".join(lines), encoding="utf-8")
    return md_path


def run_local_explainability(
    *,
    data_path: str | Path,
    cfg: ExplainConfig,
    model_path: str | Path,
    n_examples: int = 5,
    top_k_classes: int = 3,
    top_features_per_class: int = 8,
) -> Dict[str, Any]:
    data_path = Path(data_path)
    cfg.out_dir = ensure_dir(cfg.out_dir)

    df = pd.read_csv(data_path)
    if cfg.label_col not in df.columns:
        raise KeyError(f"Label column '{cfg.label_col}' not found in {data_path}")

    y = df[cfg.label_col]
    X = df.drop(columns=[cfg.label_col])

    model_path = Path(model_path)
    model = load_model(model_path)
    model_name = model_path.stem

    clf, transformer = get_estimator_and_transformer(model)
    X_trans = transform_X(transformer, X)

    rng = np.random.RandomState(cfg.random_state)
    example_idx = rng.choice(len(X_trans), size=min(n_examples, len(X_trans)), replace=False)

    rows = []

    X_bg = X_trans.sample(min(cfg.background_n, len(X_trans)), random_state=cfg.random_state)
    explainer, kind = make_explainer(clf, X_bg, cfg)

    if explainer is None:
        raise RuntimeError("No SHAP explainer available for local explainability.")

    classes = list(getattr(clf, "classes_", []))

    for idx in example_idx:
        row_X = X_trans.iloc[[idx]]

        predicted_label = None
        try:
            pred = clf.predict(row_X)
            predicted_label = pred[0] if len(pred) else None
        except Exception:
            pass

        try:
            if kind == "kernel":
                shap_vals = explainer.shap_values(row_X, nsamples=cfg.kernel_nsamples)
            else:
                shap_vals = explainer.shap_values(row_X)

            shap_list = normalize_multiclass_shap(
                shap_vals,
                max(len(classes), 2),
            )

            class_rows = []
            class_strength = []

            for c_idx, vals in enumerate(shap_list):
                class_name = classes[c_idx] if c_idx < len(classes) else c_idx
                contrib = pd.Series(vals[0], index=row_X.columns)
                ranked = contrib.reindex(contrib.abs().sort_values(ascending=False).index)

                top_items = [
                    {
                        "feature": feat,
                        "shap_value": float(ranked.loc[feat]),
                        "abs_shap_value": float(abs(ranked.loc[feat])),
                    }
                    for feat in ranked.head(top_features_per_class).index
                ]

                class_rows.append(
                    {
                        "class_idx": c_idx,
                        "class": class_name,
                        "top_features": top_items,
                    }
                )

                class_strength.append(
                    (c_idx, sum(item["abs_shap_value"] for item in top_items))
                )

            keep_class_idx = {
                c_idx
                for c_idx, _ in sorted(class_strength, key=lambda x: x[1], reverse=True)[:top_k_classes]
            }

            filtered_classes = []

            for cls_row in class_rows:
                if cls_row["class_idx"] not in keep_class_idx:
                    continue

                c_idx = cls_row["class_idx"]
                class_name = cls_row["class"]

                base_values = getattr(explainer, "expected_value", 0.0)
                if isinstance(base_values, (list, np.ndarray)):
                    base_value = base_values[c_idx] if len(base_values) > c_idx else base_values[0]
                else:
                    base_value = base_values

                explanation = shap.Explanation(
                    values=np.array(shap_list[c_idx][0]),
                    base_values=base_value,
                    data=row_X.iloc[0].values,
                    feature_names=list(row_X.columns),
                )

                waterfall_path = save_local_waterfall_plot(
                    explanation=explanation,
                    model_name=model_name,
                    row_index=int(idx),
                    class_name=class_name,
                    cfg=cfg,
                )

                filtered_classes.append(
                    {
                        "class": class_name,
                        "top_features": cls_row["top_features"],
                        "waterfall_plot": str(waterfall_path) if waterfall_path else None,
                    }
                )

            row_payload = {
                "row_index": int(idx),
                "true_label": y.iloc[idx],
                "predicted_label": predicted_label,
                "classes": filtered_classes,
            }

            json_path = save_local_json(
                payload=row_payload,
                model_name=model_name,
                row_index=int(idx),
                cfg=cfg,
            )
            md_path = save_local_markdown(
                payload=row_payload,
                model_name=model_name,
                row_index=int(idx),
                cfg=cfg,
            )

            row_payload["json_path"] = str(json_path)
            row_payload["markdown_path"] = str(md_path)

            rows.append(row_payload)

        except Exception as e:
            print(f"[WARN] Local explanation failed for row {idx}: {e}")

    return {
        "rows": rows,
        "model": model_name,
        "out_dir": str(cfg.out_dir),
    }

