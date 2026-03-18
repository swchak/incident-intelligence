from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import pandas as pd

from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


@dataclass(frozen=True)
class BaselineTrainConfig:
    """
    Configuration for baseline model training and evaluation.

    Attributes:
        label_col: Name of the target column in the input dataframe.
        test_size: Fraction of data reserved for holdout evaluation.
        random_state: Random seed used for train/test split and model reproducibility
            where supported.
        cv: Number of cross-validation folds used by GridSearchCV.
        n_jobs: Number of parallel jobs for GridSearchCV.
        verbose: Verbosity level passed to GridSearchCV.
        scoring: Metric used by GridSearchCV to select the best hyperparameters.
            If None, GridSearchCV uses the estimator's default score method,
            which is typically accuracy for classifiers.
    """

    label_col: str = "root_cause_label"
    test_size: float = 0.2
    random_state: int = 42
    cv: int = 5
    n_jobs: int = -1
    verbose: int = 1
    scoring: str = "f1_macro"


def _safe_model_name(name: str) -> str:
    """
    Convert a model display name into a filesystem-friendly stem.

    Example:
        "SVM (RBF)" -> "SVM_RBF"
    """
    return (
        str(name)
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
        .replace("(", "")
        .replace(")", "")
    )


def needs_scaling(estimator: BaseEstimator) -> bool:
    """
    Return True when the estimator benefits from feature scaling.

    Currently scaling is applied to:
    - LogisticRegression
    - SVC

    Tree-based models are left unscaled.
    """
    return isinstance(estimator, (LogisticRegression, SVC))


def get_models_to_run(random_state: int = 42) -> List[Dict[str, Any]]:
    """
    Return the baseline model definitions and hyperparameter grids.

    Each item contains:
    - name: human-readable model name
    - estimator: unfitted estimator instance
    - param_grid: GridSearchCV parameter grid targeting the pipeline's 'clf' step
    """
    return [
        {
            "name": "Logistic Regression",
            "estimator": LogisticRegression(max_iter=1000, solver="lbfgs"),
            "param_grid": {"clf__C": [0.01, 0.1, 1, 10]},
        },
        {
            "name": "Random Forest",
            "estimator": RandomForestClassifier(random_state=random_state),
            "param_grid": {
                "clf__n_estimators": [100, 200],
                "clf__max_depth": [None, 10, 20],
            },
        },
        {
            "name": "Gradient Boosting",
            "estimator": GradientBoostingClassifier(random_state=random_state),
            "param_grid": {
                "clf__n_estimators": [100, 200],
                "clf__learning_rate": [0.05, 0.1],
            },
        },
        {
            "name": "SVM (RBF)",
            "estimator": SVC(probability=True),
            "param_grid": {
                "clf__C": [0.1, 1, 10],
                "clf__gamma": ["scale", "auto"],
            },
        },
    ]


def make_pipeline(estimator: BaseEstimator) -> Pipeline:
    """
    Build a training pipeline for the provided estimator.

    A StandardScaler is added only for models that are sensitive to feature
    magnitude. The scaler is configured to emit pandas output so downstream
    steps preserve column names when supported by the installed scikit-learn
    version.
    """
    if needs_scaling(estimator):
        return Pipeline(
            [
                ("scaler", StandardScaler().set_output(transform="pandas")),
                ("clf", estimator),
            ]
        )
    return Pipeline([("clf", estimator)])


def split_xy(df: pd.DataFrame, label_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split a dataframe into features and target.

    Args:
        df: Input dataframe containing feature columns and the target column.
        label_col: Name of the target column.

    Returns:
        A tuple of (X, y), where X is the feature dataframe and y is the target series.

    Raises:
        ValueError: If label_col is not present in df.
    """
    if label_col not in df.columns:
        raise ValueError(f"label_col='{label_col}' not found. Columns={list(df.columns)}")
    X = df.drop(columns=[label_col])
    y = df[label_col]
    return X, y


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
    Fit a hyperparameter-tuned pipeline and evaluate it on the holdout split.

    GridSearchCV is used to select the best parameter set according to
    cfg.scoring. If cfg.scoring is None, the estimator's default score method
    is used, which is typically accuracy for classifiers.

    Args:
        X_train: Training features.
        y_train: Training labels.
        X_test: Holdout features.
        y_test: Holdout labels.
        pipeline: Pipeline to tune and train.
        param_grid: Hyperparameter grid using pipeline step names.
        model_name: Human-readable model name for reporting.
        cfg: Training configuration.

    Returns:
        A tuple of:
        - fitted GridSearchCV object
        - evaluation dictionary containing best params, classification report,
          confusion matrix, and best CV score
    """
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cfg.cv,
        n_jobs=cfg.n_jobs,
        verbose=cfg.verbose,
        scoring=cfg.scoring,
        refit=True,
    )
    grid.fit(X_train, y_train)

    y_pred = grid.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    eval_out = {
        "model_name": model_name,
        "best_params": grid.best_params_,
        "best_cv_score": float(grid.best_score_),
        "scoring": cfg.scoring or "estimator_default_score",
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
    }
    return grid, eval_out


def train_all_models(
    df: pd.DataFrame,
    cfg: Optional[BaselineTrainConfig] = None,
) -> Tuple[Dict[str, GridSearchCV], List[Dict[str, Any]]]:
    """
    Train all configured baseline models and evaluate them on a holdout split.

    Workflow:
    1. Split the dataframe into features and target
    2. Create a stratified train/test split
    3. Run GridSearchCV for each baseline model
    4. Evaluate the best fitted model on the holdout set

    Args:
        df: Input dataframe containing features and the target column.
        cfg: Optional training configuration. Defaults to BaselineTrainConfig().

    Returns:
        A tuple of:
        - grids: mapping from model name to fitted GridSearchCV
        - evaluations: list of per-model evaluation summaries

    Raises:
        ValueError: If the label column is missing.
        ValueError: If stratified splitting is not possible because one or more
            classes do not have enough samples for the requested split.
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
    """
    Persist the best estimator from a fitted GridSearchCV object.

    Args:
        grid: Fitted GridSearchCV instance.
        out_path: Destination path for the serialized best estimator.

    Returns:
        The resolved output path used for saving.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(grid.best_estimator_, out_path)
    return out_path


def save_all_best_pipelines(
    grids: Dict[str, GridSearchCV],
    out_dir: str | Path,
) -> List[Path]:
    """
    Save the best estimator from each fitted GridSearchCV result.

    Filenames are derived from model names and sanitized for filesystem use.

    Args:
        grids: Mapping from model name to fitted GridSearchCV.
        out_dir: Directory where pipeline artifacts should be written.

    Returns:
        List of output paths for the saved pipeline files.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []

    for model_name, grid in grids.items():
        fname = f"{_safe_model_name(model_name)}_pipeline.joblib"
        paths.append(save_best_pipeline(grid, out_dir / fname))

    return paths