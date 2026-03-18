"""
Generate row-level SHAP explanations for trained models.

This module focuses on *local* explainability, such as:
- per-row SHAP contribution rankings
- SHAP waterfall plots
- JSON and markdown explanation reports
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from incident_intelligence.modeling.evaluate import load_df
from incident_intelligence.modeling.explain_utils import (
    _HAS_SHAP,
    _safe_name,
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
class ExplainLocalConfig:
    """Configuration for row-level local explainability generation."""
    label_col: str = "root_cause_label"
    out_dir: str | Path = "artifacts/explain"
    background_n: int = 100
    kernel_bg: int = 40
    kernel_nsamples: int = 80
    random_state: int = 42
    top_k: int = 20



def save_local_waterfall_plot(
    explanation: Any,
    model_name: str,
    row_index: int,
    class_name: Any,
    cfg: ExplainLocalConfig,
) -> Path | None:
    """Save a SHAP waterfall plot for one row and one class."""
    if not _HAS_SHAP:
        return None

    out_dir = ensure_dir(model_output_dir(cfg, model_name) / "local")
    png_path = out_dir / f"row_{row_index}_class_{_safe_name(class_name)}_waterfall.png"

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
    cfg: ExplainLocalConfig,
) -> Path:
    """Persist one local explanation payload as JSON."""
    out_dir = ensure_dir(model_output_dir(cfg, model_name) / "local")
    json_path = out_dir / f"row_{row_index}.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return json_path



def save_local_markdown(
    payload: Dict[str, Any],
    model_name: str,
    row_index: int,
    cfg: ExplainLocalConfig,
) -> Path:
    """Persist one local explanation payload as a markdown report."""
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
    cfg: ExplainLocalConfig,
    model_path: str | Path,
    row_indices: list[int] | None = None,
    n_examples: int = 5,
    top_k_classes: int = 3,
    top_features_per_class: int = 8,
) -> Dict[str, Any]:
    """Generate local SHAP explanations for selected rows from one fitted model."""
    df = load_df(data_path)
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

    if row_indices:
        example_idx = [idx for idx in row_indices if 0 <= idx < len(X_trans)]
    else:
        example_idx = rng.choice(
            len(X_trans),
            size=min(n_examples, len(X_trans)),
            replace=False,
        ).tolist()

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

            shap_list = normalize_multiclass_shap(shap_vals, max(len(classes), 2))

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

