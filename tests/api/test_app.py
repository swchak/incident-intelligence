from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi import HTTPException

from incident_intelligence.api.app import (
    PipelineJob,
    PipelineRunRequest,
    _load_jobs,
    _save_jobs,
    artifacts,
    dashboard_summary,
    delete_pipeline_job,
    get_pipeline_job_log,
    health,
    get_project_file,
    run_pipeline,
)


class DashboardApiTests(unittest.TestCase):
    def test_health_endpoint(self) -> None:
        self.assertEqual(health(), {"status": "ok"})

    @patch("incident_intelligence.api.app._dataset_paths")
    @patch("incident_intelligence.api.app._dashboard_paths")
    @patch("incident_intelligence.api.app._safe_load_json")
    def test_dashboard_summary_endpoint(
        self,
        safe_load_json_mock,
        dashboard_paths_mock,
        dataset_paths_mock,
    ) -> None:
        dataset_paths_mock.return_value = {
            "train": "data/processed/incident_temporal_train.csv",
            "val": "data/processed/incident_temporal_val.csv",
            "eval": "data/processed/incident_temporal_eval.csv",
        }
        dashboard_paths_mock.return_value = {
            "models_dir": "/tmp/models_temporal",
            "best_model": "/tmp/models_temporal/best_model.joblib",
            "train_metrics": "/tmp/metrics_temporal/train_val_results.json",
            "leaderboard": "/tmp/metrics_temporal/leaderboard_val.csv",
            "evaluation_metrics": "/tmp/metrics_temporal/evaluation.json",
            "evaluation_summary": "/tmp/metrics_temporal/evaluation_summary.csv",
            "plots_dir": "/tmp/plots_temporal",
            "reports_dir": "/tmp/reports_temporal",
            "explain_dir": "/tmp/explain_temporal",
        }
        safe_load_json_mock.side_effect = [
            {"models": [{"model_name": "Random_Forest_pipeline"}]},
            {"models": [{"model_name": "Random_Forest_pipeline", "metrics": {"accuracy": 0.88}}]},
        ]

        body = dashboard_summary("temporal")

        self.assertEqual(body["dataset_kind"], "temporal")
        self.assertEqual(body["datasets"]["eval"], "data/processed/incident_temporal_eval.csv")
        self.assertEqual(body["evaluation_metrics"]["models"][0]["model_name"], "Random_Forest_pipeline")

    @patch("incident_intelligence.api.app._dashboard_paths")
    @patch("incident_intelligence.api.app._list_artifacts")
    def test_artifacts_endpoint(self, list_artifacts_mock, dashboard_paths_mock) -> None:
        dashboard_paths_mock.return_value = {
            "models_dir": "/tmp/models",
            "best_model": "/tmp/models/best_model.joblib",
            "train_metrics": "/tmp/metrics/train_val_results.json",
            "leaderboard": "/tmp/metrics/leaderboard_val.csv",
            "evaluation_metrics": "/tmp/metrics/evaluation.json",
            "evaluation_summary": "/tmp/metrics/evaluation_summary.csv",
            "plots_dir": "/tmp/plots",
            "reports_dir": "/tmp/reports",
            "explain_dir": "/tmp/explain",
        }

        def fake_list_artifacts(path_str: str) -> list[dict[str, str | int]]:
            return [{"path": path_str, "kind": "file", "size_bytes": 123}]

        list_artifacts_mock.side_effect = fake_list_artifacts

        body = artifacts("snapshot")

        self.assertEqual(body["dataset_kind"], "snapshot")
        self.assertIn("plots_dir", body["artifacts"])
        self.assertEqual(body["artifacts"]["best_model"][0]["path"], "/tmp/models/best_model.joblib")

    @patch("incident_intelligence.api.app.threading.Thread")
    @patch("incident_intelligence.api.app._build_pipeline_command")
    @patch("incident_intelligence.api.app._save_jobs")
    def test_run_pipeline_queues_job(self, save_jobs_mock, build_command_mock, thread_mock) -> None:
        build_command_mock.return_value = [
            "python",
            "-m",
            "incident_intelligence.cli.pipeline",
            "--dataset-kind",
            "snapshot",
        ]

        response = run_pipeline(
            PipelineRunRequest(
                dataset_kind="snapshot",
                fast_mode=True,
                models=["logistic", "rf"],
                cv=3,
            )
        )

        self.assertEqual(response.dataset_kind, "snapshot")
        self.assertEqual(response.status, "queued")
        self.assertEqual(response.command, build_command_mock.return_value)
        save_jobs_mock.assert_called_once()
        thread_mock.return_value.start.assert_called_once()

    def test_get_project_file_rejects_outside_allowed_roots(self) -> None:
        with self.assertRaises(HTTPException) as exc_info:
            get_project_file("../secrets.txt")

        self.assertEqual(exc_info.exception.status_code, 403)

    def test_missing_job_log_returns_404(self) -> None:
        with self.assertRaises(HTTPException) as exc_info:
            get_pipeline_job_log("does-not-exist")

        self.assertEqual(exc_info.exception.status_code, 404)

    def test_existing_job_log_is_returned(self) -> None:
        with TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "job.log"
            log_path.write_text("pipeline finished", encoding="utf-8")

            with patch(
                "incident_intelligence.api.app._JOBS",
                {"job-123": type("Job", (), {"log_path": str(log_path)})()},
            ):
                response = get_pipeline_job_log("job-123")

        self.assertEqual(response["log"], "pipeline finished")

    def test_delete_pipeline_job_removes_job_and_log(self) -> None:
        with TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "job.log"
            log_path.write_text("pipeline finished", encoding="utf-8")
            jobs = {
                "job-123": PipelineJob(
                    job_id="job-123",
                    dataset_kind="snapshot",
                    command=["python", "-m", "incident_intelligence.cli.pipeline"],
                    log_path=str(log_path),
                    status="completed",
                )
            }

            with patch("incident_intelligence.api.app._JOBS", jobs), patch(
                "incident_intelligence.api.app._save_jobs"
            ) as save_jobs_mock:
                response = delete_pipeline_job("job-123")

        self.assertEqual(response.job_id, "job-123")
        self.assertFalse(log_path.exists())
        self.assertEqual(jobs, {})
        save_jobs_mock.assert_called_once()

    def test_jobs_are_persisted_to_sqlite(self) -> None:
        with TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "jobs.sqlite3"
            jobs = {
                "job-123": PipelineJob(
                    job_id="job-123",
                    dataset_kind="snapshot",
                    command=["python", "-m", "incident_intelligence.cli.pipeline"],
                    log_path=str(Path(temp_dir) / "job-123.log"),
                    status="completed",
                )
            }

            with patch("incident_intelligence.api.app.JOBS_DB_PATH", db_path), patch(
                "incident_intelligence.api.app.API_RUNS_DIR", Path(temp_dir)
            ), patch("incident_intelligence.api.app._JOBS", jobs):
                _save_jobs()
                loaded_jobs = _load_jobs()
                self.assertTrue(db_path.exists())
                self.assertIn("job-123", loaded_jobs)
                self.assertEqual(loaded_jobs["job-123"].status, "completed")


if __name__ == "__main__":
    unittest.main()
