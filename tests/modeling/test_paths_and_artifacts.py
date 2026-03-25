from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from incident_intelligence.modeling.evaluate import find_model_files, load_eval_data as load_eval_data_eval
from incident_intelligence.modeling.train import (
    load_eval_data as load_eval_data_train,
    with_dataset_suffix,
    with_parent_dir_suffix,
)


class PathAndArtifactTests(unittest.TestCase):
    def test_with_dataset_suffix_appends_to_filename(self) -> None:
        out = with_dataset_suffix("artifacts/models/best_model.joblib", "temporal")
        self.assertEqual(out, "artifacts/models/best_model_temporal.joblib")

    def test_with_parent_dir_suffix_appends_to_parent_directory(self) -> None:
        out = with_parent_dir_suffix("artifacts/metrics/evaluation.json", "temporal")
        self.assertEqual(out, "artifacts/metrics_temporal/evaluation.json")

    def test_find_model_files_filters_snapshot_legacy_temporal_artifacts(self) -> None:
        with TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            (temp_path / "best_model.joblib").write_text("x", encoding="utf-8")
            (temp_path / "best_model_temporal.joblib").write_text("x", encoding="utf-8")

            snapshot_files = find_model_files(temp_path, dataset_kind="snapshot")
            temporal_files = find_model_files(temp_path, dataset_kind="temporal")

        self.assertEqual([path.name for path in snapshot_files], ["best_model.joblib"])
        self.assertEqual(
            [path.name for path in temporal_files],
            ["best_model.joblib", "best_model_temporal.joblib"],
        )

    @patch("incident_intelligence.modeling.train.load_df")
    def test_train_load_eval_data_uses_dataset_kind_defaults(self, load_df_mock) -> None:
        load_eval_data_train("temporal")

        load_df_mock.assert_called_once_with("data/processed/incident_temporal_eval.csv")

    @patch("incident_intelligence.modeling.evaluate.load_df")
    def test_evaluate_load_eval_data_uses_dataset_kind_defaults(self, load_df_mock) -> None:
        load_eval_data_eval("snapshot")

        load_df_mock.assert_called_once_with("data/processed/incident_snapshot_eval.csv")
