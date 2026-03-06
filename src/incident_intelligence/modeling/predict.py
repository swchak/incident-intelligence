from __future__ import annotations

from pathlib import Path
import argparse

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline


def load_model(path: str | Path) -> Pipeline:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    model = joblib.load(path)
    if not isinstance(model, Pipeline):
        raise TypeError(f"Expected sklearn Pipeline, got {type(model)}")
    return model


def load_inputs(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Inputs not found: {path}")
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path.suffix} (use .csv or .parquet)")


def predict_df(model: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    out = X.copy()
    out["prediction"] = model.predict(X)

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        classes = getattr(model.named_steps.get("clf", model), "classes_", None) or [f"class_{i}" for i in range(proba.shape[1])]
        for i, c in enumerate(classes):
            out[f"proba_{c}"] = proba[:, i]

    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run inference with a saved pipeline.")
    parser.add_argument("--model", type=str, required=True, help="Path to saved pipeline .joblib")
    parser.add_argument("--inputs", type=str, required=True, help="CSV/Parquet of features (no label column).")
    parser.add_argument("--out", type=str, default="artifacts/predictions.csv")
    args = parser.parse_args()

    model = load_model(args.model)
    X = load_inputs(args.inputs)
    preds = predict_df(model, X)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(out_path, index=False)
    print(f"Saved predictions to: {out_path}")


if __name__ == "__main__":
    main()