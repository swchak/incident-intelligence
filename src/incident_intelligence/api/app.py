from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from incident_intelligence.config import (
    EvaluateCLIConfig,
    ExplainCLIConfig,
    TrainCLIConfig,
    load_config,
)
from incident_intelligence.modeling.train import with_dataset_suffix, with_parent_dir_suffix


DatasetKind = Literal["snapshot", "temporal"]
PROJECT_ROOT = Path(__file__).resolve().parents[3]


class PipelineRunRequest(BaseModel):
    dataset_kind: DatasetKind = "snapshot"
    fast_mode: bool = False
    models: list[str] | None = None
    cv: int | None = Field(default=None, ge=2)
    n_jobs: int | None = None
    verbose: int | None = None
    scoring: str | None = None


class PipelineJobResponse(BaseModel):
    job_id: str
    status: str
    command: list[str]
    dataset_kind: DatasetKind
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    return_code: int | None = None
    log_path: str


@dataclass
class PipelineJob:
    job_id: str
    dataset_kind: DatasetKind
    command: list[str]
    log_path: str
    status: str = "queued"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    started_at: str | None = None
    finished_at: str | None = None
    return_code: int | None = None


_JOBS: dict[str, PipelineJob] = {}
_JOBS_LOCK = threading.Lock()


def _artifact_dir(base_dir: str, dataset_kind: DatasetKind) -> str:
    if dataset_kind == "snapshot":
        return base_dir
    return with_dataset_suffix(base_dir, dataset_kind)


def _artifact_file(base_path: str, dataset_kind: DatasetKind) -> str:
    if dataset_kind == "snapshot":
        return base_path
    return with_parent_dir_suffix(base_path, dataset_kind)


def _dataset_paths(dataset_kind: DatasetKind) -> dict[str, str]:
    if dataset_kind == "snapshot":
        return {
            "train": "data/processed/incident_snapshot_train.csv",
            "val": "data/processed/incident_snapshot_val.csv",
            "eval": "data/processed/incident_snapshot_eval.csv",
        }
    return {
        "train": "data/processed/incident_temporal_train.csv",
        "val": "data/processed/incident_temporal_val.csv",
        "eval": "data/processed/incident_temporal_eval.csv",
    }


def _dashboard_paths(dataset_kind: DatasetKind) -> dict[str, str]:
    train_settings = load_config(TrainCLIConfig, "train")
    eval_settings = load_config(EvaluateCLIConfig, "evaluate")
    explain_settings = load_config(ExplainCLIConfig, "explain")

    return {
        "models_dir": str(PROJECT_ROOT / _artifact_dir(train_settings.models_out_dir, dataset_kind)),
        "best_model": str(PROJECT_ROOT / _artifact_file(train_settings.best_model_out, dataset_kind)),
        "train_metrics": str(PROJECT_ROOT / _artifact_file(train_settings.metrics_out_json, dataset_kind)),
        "leaderboard": str(PROJECT_ROOT / _artifact_file(train_settings.leaderboard_out_csv, dataset_kind)),
        "evaluation_metrics": str(PROJECT_ROOT / _artifact_file(eval_settings.metrics_out, dataset_kind)),
        "evaluation_summary": str(PROJECT_ROOT / _artifact_file(eval_settings.summary_csv_out, dataset_kind)),
        "plots_dir": str(PROJECT_ROOT / _artifact_dir("artifacts/plots", dataset_kind)),
        "reports_dir": str(PROJECT_ROOT / _artifact_dir("artifacts/reports", dataset_kind)),
        "explain_dir": str(PROJECT_ROOT / _artifact_dir(explain_settings.out_dir, dataset_kind)),
    }


def _safe_load_json(path_str: str) -> dict | list | None:
    path = Path(path_str)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _list_artifacts(path_str: str) -> list[dict[str, str | int]]:
    path = Path(path_str)
    if not path.exists():
        return []
    if path.is_file():
        stat = path.stat()
        return [{
            "path": str(path.relative_to(PROJECT_ROOT)),
            "kind": "file",
            "size_bytes": stat.st_size,
        }]

    items: list[dict[str, str | int]] = []
    for item in sorted(path.rglob("*")):
        if item.is_file():
            stat = item.stat()
            items.append(
                {
                    "path": str(item.relative_to(PROJECT_ROOT)),
                    "kind": "file",
                    "size_bytes": stat.st_size,
                }
            )
    return items


def _build_pipeline_command(request: PipelineRunRequest) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "incident_intelligence.cli.pipeline",
        "--dataset-kind",
        request.dataset_kind,
    ]
    if request.fast_mode:
        command.append("--fast-mode")
    if request.models:
        command.extend(["--models", ",".join(request.models)])
    if request.cv is not None:
        command.extend(["--cv", str(request.cv)])
    if request.n_jobs is not None:
        command.extend(["--n-jobs", str(request.n_jobs)])
    if request.verbose is not None:
        command.extend(["--verbose", str(request.verbose)])
    if request.scoring is not None:
        command.extend(["--scoring", request.scoring])
    return command


def _run_pipeline_job(job_id: str) -> None:
    with _JOBS_LOCK:
        job = _JOBS[job_id]
        job.status = "running"
        job.started_at = datetime.now(timezone.utc).isoformat()
        command = list(job.command)
        log_path = Path(job.log_path)

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(PROJECT_ROOT / "src"))
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", "/tmp")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )

    with _JOBS_LOCK:
        job = _JOBS[job_id]
        job.return_code = process.returncode
        job.finished_at = datetime.now(timezone.utc).isoformat()
        job.status = "completed" if process.returncode == 0 else "failed"


def _job_response(job: PipelineJob) -> PipelineJobResponse:
    return PipelineJobResponse(
        job_id=job.job_id,
        status=job.status,
        command=job.command,
        dataset_kind=job.dataset_kind,
        created_at=job.created_at,
        started_at=job.started_at,
        finished_at=job.finished_at,
        return_code=job.return_code,
        log_path=job.log_path,
    )


app = FastAPI(
    title="Incident Intelligence API",
    version="0.1.0",
    description="Backend API for the Incident Intelligence demo dashboard.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/config")
def config_summary() -> dict[str, object]:
    return {
        "project_root": str(PROJECT_ROOT),
        "dataset_kinds": ["snapshot", "temporal"],
        "datasets": {
            "snapshot": _dataset_paths("snapshot"),
            "temporal": _dataset_paths("temporal"),
        },
    }


@app.get("/api/dashboard/summary/{dataset_kind}")
def dashboard_summary(dataset_kind: DatasetKind) -> dict[str, object]:
    paths = _dashboard_paths(dataset_kind)
    return {
        "dataset_kind": dataset_kind,
        "datasets": _dataset_paths(dataset_kind),
        "artifacts": paths,
        "train_metrics": _safe_load_json(paths["train_metrics"]),
        "evaluation_metrics": _safe_load_json(paths["evaluation_metrics"]),
    }


@app.get("/api/artifacts/{dataset_kind}")
def artifacts(dataset_kind: DatasetKind) -> dict[str, object]:
    paths = _dashboard_paths(dataset_kind)
    return {
        "dataset_kind": dataset_kind,
        "artifacts": {
            name: _list_artifacts(path_str)
            for name, path_str in paths.items()
            if name.endswith("_dir") or name in {"best_model", "train_metrics", "leaderboard", "evaluation_metrics", "evaluation_summary"}
        },
    }


@app.post("/api/pipeline/run", response_model=PipelineJobResponse)
def run_pipeline(request: PipelineRunRequest) -> PipelineJobResponse:
    job_id = uuid.uuid4().hex
    log_dir = PROJECT_ROOT / "artifacts" / "api_runs"
    log_path = log_dir / f"{job_id}.log"
    job = PipelineJob(
        job_id=job_id,
        dataset_kind=request.dataset_kind,
        command=_build_pipeline_command(request),
        log_path=str(log_path),
    )
    with _JOBS_LOCK:
        _JOBS[job_id] = job

    worker = threading.Thread(target=_run_pipeline_job, args=(job_id,), daemon=True)
    worker.start()
    return _job_response(job)


@app.get("/api/pipeline/jobs", response_model=list[PipelineJobResponse])
def list_pipeline_jobs() -> list[PipelineJobResponse]:
    with _JOBS_LOCK:
        jobs = list(_JOBS.values())
    return [_job_response(job) for job in sorted(jobs, key=lambda item: item.created_at, reverse=True)]


@app.get("/api/pipeline/jobs/{job_id}", response_model=PipelineJobResponse)
def get_pipeline_job(job_id: str) -> PipelineJobResponse:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job_id '{job_id}'")
    return _job_response(job)


@app.get("/api/pipeline/jobs/{job_id}/log")
def get_pipeline_job_log(job_id: str) -> dict[str, str]:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job_id '{job_id}'")

    log_path = Path(job.log_path)
    if not log_path.exists():
        return {"job_id": job_id, "log": ""}
    return {"job_id": job_id, "log": log_path.read_text(encoding="utf-8")}


def main() -> None:
    uvicorn.run(
        "incident_intelligence.api.app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
