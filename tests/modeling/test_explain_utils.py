from __future__ import annotations

import unittest
from dataclasses import dataclass
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from incident_intelligence.modeling.explain_utils import (
    ensure_dir,
    get_estimator_and_transformer,
    model_output_dir,
    normalize_multiclass_shap,
    transform_X,
)


@dataclass
class _Cfg:
    out_dir: str


class _DummyTransformer:
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return X.to_numpy() * 2


class ExplainabilityHelperTests(unittest.TestCase):
    def test_ensure_dir_and_model_output_dir_create_expected_directory(self) -> None:
        with TemporaryDirectory() as temp_dir:
            out = model_output_dir(_Cfg(out_dir=temp_dir), "Random Forest (v1)")

            self.assertTrue(out.exists())
            self.assertEqual(out.name, "Random_Forest_v1")
            self.assertEqual(ensure_dir(out), out)

    def test_transform_x_preserves_dataframe_shape_and_columns(self) -> None:
        X = pd.DataFrame({"cpu": [1.0, 2.0], "latency": [3.0, 4.0]})

        transformed = transform_X(_DummyTransformer(), X)

        self.assertIsInstance(transformed, pd.DataFrame)
        self.assertEqual(list(transformed.columns), ["cpu", "latency"])
        self.assertEqual(transformed.iloc[1, 1], 8.0)

    def test_normalize_multiclass_shap_handles_2d_and_3d_shapes(self) -> None:
        binary = np.array([[1.0, 2.0], [3.0, 4.0]])
        binary_out = normalize_multiclass_shap(binary, n_classes=2)
        self.assertEqual(len(binary_out), 2)
        np.testing.assert_array_equal(binary_out[0], binary)
        np.testing.assert_array_equal(binary_out[1], -binary)

        multiclass = np.arange(24, dtype=float).reshape(2, 4, 3)
        multiclass_out = normalize_multiclass_shap(multiclass, n_classes=3)
        self.assertEqual(len(multiclass_out), 3)
        np.testing.assert_array_equal(multiclass_out[2], multiclass[:, :, 2])

    def test_get_estimator_and_transformer_from_pipeline(self) -> None:
        model = Pipeline(
            steps=[
                ("scale", StandardScaler()),
                ("clf", SVC()),
            ]
        )

        clf, transformer = get_estimator_and_transformer(model)

        self.assertIsInstance(clf, SVC)
        self.assertIsInstance(transformer, StandardScaler)
