from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi import HTTPException

from incident_intelligence.api.app import (
    PipelineRun,
    PipelineRunRequest,
    PipelineStage,
    _build_stages,
    _load_jobs,
    _save_jobs,
    artifacts,
    cancel_pipeline_job,
    dashboard_summary,
    delete_pipeline_job,
    get_pipeline_job_log,
    get_project_file,
    health,
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
            {
                "models": [
                    {
                        "model_name": "Random_Forest_pipeline",
                        "metrics": {"accuracy": 0.88},
                    }
                ]
            },
        ]

        body = dashboard_summary("temporal")

        self.assertEqual(body["dataset_kind"], "temporal")
        self.assertEqual(
            body["datasets"]["eval"],
            "data/processed/incident_temporal_eval.csv",
        )
        self.assertEqual(
            body["evaluation_metrics"]["models"][0]["model_name"],
            "Random_Forest_pipeline",
        )

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
        self.assertEqual(
            body["artifacts"]["best_model"][0]["path"],
            "/tmp/models/best_model.joblib",
        )

    @patch("incident_intelligence.api.app.threading.Thread")
    @patch("incident_intelligence.api.app._save_jobs")
    def test_run_pipeline_queues_staged_run(self, save_jobs_mock, thread_mock) -> None:
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
        self.assertEqual(response.mode, "full")
        self.assertEqual(
            [stage.stage_name for stage in response.stages],
            [
                "generate_snapshot",
                "train_snapshot",
                "evaluate_snapshot",
                "explain_snapshot",
            ],
        )
        save_jobs_mock.assert_called_once()
        thread_mock.return_value.start.assert_called_once()

    @patch("incident_intelligence.api.app.threading.Thread")
    @patch("incident_intelligence.api.app._save_jobs")
    def test_run_pipeline_custom_stages_keep_canonical_order(
        self, save_jobs_mock, thread_mock
    ) -> None:
        response = run_pipeline(
            PipelineRunRequest(
                dataset_kind="temporal",
                mode="custom",
                stages=["evaluate_temporal", "generate_sequence"],
            )
        )

        self.assertEqual(
            [stage.stage_name for stage in response.stages],
            ["generate_sequence", "evaluate_temporal"],
        )
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

    def test_existing_job_log_is_combined_and_returned(self) -> None:
        with TemporaryDirectory() as temp_dir:
            log_path = Path(temp_dir) / "01_train_snapshot.log"
            log_path.write_text("train finished", encoding="utf-8")
            run = PipelineRun(
                job_id="job-123",
                dataset_kind="snapshot",
                mode="full",
                requested_stages=["train_snapshot"],
                stages=[
                    PipelineStage(
                        stage_id="job-123:train_snapshot",
                        stage_name="train_snapshot",
                        stage_order=0,
                        command=["python", "-m", "incident_intelligence.cli.train"],
                        log_path=str(log_path),
                        status="completed",
                    )
                ],
            )

            with patch("incident_intelligence.api.app._JOBS", {"job-123": run}):
                response = get_pipeline_job_log("job-123")

        self.assertIn("train finished", response["log"])
        self.assertEqual(response["stages"][0]["stage_name"], "train_snapshot")

    def test_delete_pipeline_job_removes_run_directory_and_logs(self) -> None:
        with TemporaryDirectory() as temp_dir:
            log_dir = Path(temp_dir) / "job-123"
            log_dir.mkdir()
            log_path = log_dir / "01_train_snapshot.log"
            log_path.write_text("pipeline finished", encoding="utf-8")
            data_root = Path(temp_dir) / "data" / "runs" / "job-123" / "processed"
            data_root.mkdir(parents=True)
            train_csv = data_root / "incident_snapshot_train.csv"
            train_csv.write_text("label\nnormal\n", encoding="utf-8")
            jobs = {
                "job-123": PipelineRun(
                    job_id="job-123",
                    dataset_kind="snapshot",
                    mode="full",
                    requested_stages=["train_snapshot"],
                    stages=[
                        PipelineStage(
                            stage_id="job-123:train_snapshot",
                            stage_name="train_snapshot",
                            stage_order=0,
                            command=["python", "-m", "incident_intelligence.cli.train"],
                            log_path=str(log_path),
                            status="completed",
                        )
                    ],
                    status="completed",
                )
            }

            with patch("incident_intelligence.api.app._JOBS", jobs), patch(
                "incident_intelligence.api.app._save_jobs"
            ) as save_jobs_mock, patch(
                "incident_intelligence.api.app.API_RUNS_DIR", Path(temp_dir)
            ), patch(
                "incident_intelligence.api.app.PROJECT_ROOT", Path(temp_dir)
            ):
                response = delete_pipeline_job("job-123")

        self.assertEqual(response.job_id, "job-123")
        self.assertFalse(log_dir.exists())
        self.assertFalse(train_csv.exists())
        self.assertEqual(jobs, {})
        save_jobs_mock.assert_called_once()

    def test_build_stages_uses_run_scoped_dataset_paths(self) -> None:
        request = PipelineRunRequest(
            dataset_kind="snapshot",
            mode="full",
            models=["logistic", "rf"],
        )

        stages = _build_stages("job-123", request)
        generate_command = stages[0].command
        train_command = stages[1].command
        evaluate_command = stages[2].command
        explain_command = stages[3].command

        self.assertIn("--raw-out", generate_command)
        self.assertTrue(
            any("data/runs/job-123/raw/incidents_raw.csv" in arg for arg in generate_command)
        )
        self.assertIn("--processed-dir", generate_command)
        self.assertTrue(any("data/runs/job-123/processed" in arg for arg in generate_command))

        self.assertIn("--train", train_command)
        self.assertTrue(
            any("data/runs/job-123/processed/incident_snapshot_train.csv" in arg for arg in train_command)
        )
        self.assertIn("--val", train_command)
        self.assertTrue(
            any("data/runs/job-123/processed/incident_snapshot_val.csv" in arg for arg in train_command)
        )

        self.assertIn("--data", evaluate_command)
        self.assertTrue(
            any("data/runs/job-123/processed/incident_snapshot_eval.csv" in arg for arg in evaluate_command)
        )
        self.assertIn("--data", explain_command)
        self.assertTrue(
            any("data/runs/job-123/processed/incident_snapshot_eval.csv" in arg for arg in explain_command)
        )

    def test_cancel_pipeline_job_marks_queued_run_cancelled(self) -> None:
        jobs = {
            "job-123": PipelineRun(
                job_id="job-123",
                dataset_kind="snapshot",
                mode="full",
                requested_stages=["generate_snapshot", "train_snapshot"],
                stages=[
                    PipelineStage(
                        stage_id="job-123:generate_snapshot",
                        stage_name="generate_snapshot",
                        stage_order=0,
                        command=["python", "-m", "incident_intelligence.cli.generator"],
                        log_path="/tmp/01_generate_snapshot.log",
                        status="queued",
                    ),
                    PipelineStage(
                        stage_id="job-123:train_snapshot",
                        stage_name="train_snapshot",
                        stage_order=1,
                        command=["python", "-m", "incident_intelligence.cli.train"],
                        log_path="/tmp/02_train_snapshot.log",
                        status="queued",
                    ),
                ],
                status="queued",
            )
        }

        with patch("incident_intelligence.api.app._JOBS", jobs), patch(
            "incident_intelligence.api.app._save_jobs"
        ) as save_jobs_mock:
            response = cancel_pipeline_job("job-123")

        self.assertEqual(response.status, "cancelled")
        self.assertTrue(all(stage.status == "skipped" for stage in jobs["job-123"].stages))
        save_jobs_mock.assert_called_once()

    def test_cancel_pipeline_job_marks_running_run_cancelling(self) -> None:
        jobs = {
            "job-123": PipelineRun(
                job_id="job-123",
                dataset_kind="snapshot",
                mode="full",
                requested_stages=["train_snapshot"],
                stages=[
                    PipelineStage(
                        stage_id="job-123:train_snapshot",
                        stage_name="train_snapshot",
                        stage_order=0,
                        command=["python", "-m", "incident_intelligence.cli.train"],
                        log_path="/tmp/01_train_snapshot.log",
                        status="running",
                    )
                ],
                status="running",
                current_stage_name="train_snapshot",
            )
        }
        process_mock = unittest.mock.Mock()

        with patch("incident_intelligence.api.app._JOBS", jobs), patch(
            "incident_intelligence.api.app._save_jobs"
        ) as save_jobs_mock, patch(
            "incident_intelligence.api.app._RUN_PROCESSES", {"job-123": process_mock}
        ):
            response = cancel_pipeline_job("job-123")

        self.assertEqual(response.status, "cancelling")
        process_mock.terminate.assert_called_once()
        save_jobs_mock.assert_called_once()

    def test_runs_are_persisted_to_sqlite(self) -> None:
        with TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "jobs.sqlite3"
            jobs = {
                "job-123": PipelineRun(
                    job_id="job-123",
                    dataset_kind="snapshot",
                    mode="full",
                    requested_stages=["train_snapshot"],
                    stages=[
                        PipelineStage(
                            stage_id="job-123:train_snapshot",
                            stage_name="train_snapshot",
                            stage_order=0,
                            command=["python", "-m", "incident_intelligence.cli.train"],
                            log_path=str(Path(temp_dir) / "job-123" / "01_train_snapshot.log"),
                            status="completed",
                        )
                    ],
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
                self.assertEqual(
                    loaded_jobs["job-123"].stages[0].stage_name,
                    "train_snapshot",
                )


if __name__ == "__main__":
    unittest.main()
