from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import joblib
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC


@dataclass(frozen=True)
class BaselineTrainConfig:
    label_col: str = "root_cause_label"
    test_size: float = 0.2
    random_state: int = 42
    cv: int = 5
    n_jobs: int = -1
    verbose: int = 1


def needs_scaling(estimator: BaseEstimator) -> bool:
    """Matches your notebook: scale for LogisticRegression and SVC."""
    return isinstance(estimator, (LogisticRegression, SVC))


def get_models_to_run(random_state: int = 42) -> List[Dict[str, Any]]:
    """Matches your notebook model list + grids."""
    return [
        {
            "name": "Logistic Regression",
            "estimator": LogisticRegression(max_iter=1000, solver="lbfgs"),
            "param_grid": {"clf__C": [0.01, 0.1, 1, 10]},
        },
        {
            "name": "Random Forest",
            "estimator": RandomForestClassifier(random_state=random_state),
            "param_grid": {"clf__n_estimators": [100, 200], "clf__max_depth": [None, 10, 20]},
        },
        {
            "name": "Gradient Boosting",
            "estimator": GradientBoostingClassifier(random_state=random_state),
            "param_grid": {"clf__n_estimators": [100, 200], "clf__learning_rate": [0.05, 0.1]},
        },
        {
            "name": "SVM (RBF)",
            "estimator": SVC(probability=True),
            "param_grid": {"clf__C": [0.1, 1, 10], "clf__gamma": ["scale", "auto"]},
        },
    ]


def make_pipeline(estimator: BaseEstimator) -> Pipeline:
    """Add StandardScaler only when needed (same logic as notebook)."""
    if needs_scaling(estimator):
        return Pipeline(
            [
                ("scaler", StandardScaler().set_output(transform="pandas")),
                ("clf", estimator),
            ]
        )
    return Pipeline([("clf", estimator)])


def train_and_evaluate(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    pipeline: Pipeline,
    param_grid: Dict[str, Any],
    *,
    model_name: str,
    cfg: BaselineTrainConfig,
) -> Tuple[GridSearchCV, Dict[str, Any]]:
    """
    Trains with GridSearchCV and returns:
      - fitted GridSearchCV
      - evaluation dict (report + confusion matrix)
    """
    grid = GridSearchCV(
        pipeline,
        param_grid,
        cv=cfg.cv,
        n_jobs=cfg.n_jobs,
        verbose=cfg.verbose,
    )
    grid.fit(X_train, y_train)

    y_pred = grid.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    eval_out = {
        "model_name": model_name,
        "best_params": grid.best_params_,
        "classification_report": report,
        "confusion_matrix": cm.tolist(),  # convert to list for JSON serialization
    }
    return grid, eval_out


def split_xy(df: pd.DataFrame, label_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    if label_col not in df.columns:
        raise ValueError(f"label_col='{label_col}' not found. Columns={list(df.columns)}")
    X = df.drop(columns=[label_col])
    y = df[label_col]
    return X, y


def train_all_models(
    df: pd.DataFrame,
    cfg: Optional[BaselineTrainConfig] = None,
) -> Tuple[Dict[str, GridSearchCV], List[Dict[str, Any]]]:
    """
    Runs the full notebook loop: trains all models and returns:
      - grids dict (model_name -> GridSearchCV)
      - evaluations list (per-model metrics)
    """
    cfg = cfg or BaselineTrainConfig()
    X, y = split_xy(df, cfg.label_col)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg.test_size,
        stratify=y,
        random_state=cfg.random_state,
    )

    grids: Dict[str, GridSearchCV] = {}
    evaluations: List[Dict[str, Any]] = []

    for model_info in get_models_to_run(cfg.random_state):
        name = model_info["name"]
        est = model_info["estimator"]
        pipe = make_pipeline(est)

        grid, eval_out = train_and_evaluate(
            X_train,
            y_train,
            X_test,
            y_test,
            pipe,
            model_info["param_grid"],
            model_name=name,
            cfg=cfg,
        )

        grids[name] = grid
        evaluations.append(eval_out)

    return grids, evaluations


def save_best_pipeline(grid: GridSearchCV, out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(grid.best_estimator_, out_path)
    return out_path


def save_all_best_pipelines(grids: Dict[str, GridSearchCV], out_dir: str | Path) -> List[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []

    for model_name, grid in grids.items():
        fname = f"{model_name.replace(' ', '_')}_pipeline.joblib"
        paths.append(save_best_pipeline(grid, out_dir / fname))

    return paths