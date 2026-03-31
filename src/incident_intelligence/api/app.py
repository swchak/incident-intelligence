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
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, model_validator
from sqlalchemy import ForeignKey, Integer, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship
import uvicorn

from incident_intelligence.config import (
    EvaluateCLIConfig,
    ExplainCLIConfig,
    TrainCLIConfig,
    load_config,
)
from incident_intelligence.modeling.train import with_dataset_suffix, with_parent_dir_suffix


DatasetKind = Literal["snapshot", "temporal"]
RunMode = Literal["full", "custom"]
RunStatus = Literal["queued", "running", "completed", "failed"]
StageStatus = Literal["queued", "running", "completed", "failed", "skipped"]
StageName = Literal[
    "generate_snapshot",
    "generate_sequence",
    "build_temporal_features",
    "train_snapshot",
    "train_temporal",
    "evaluate_snapshot",
    "evaluate_temporal",
    "explain_snapshot",
    "explain_temporal",
    "pipeline",
]

PROJECT_ROOT = Path(__file__).resolve().parents[3]
API_RUNS_DIR = PROJECT_ROOT / "artifacts" / "api_runs"
JOBS_STATE_PATH = API_RUNS_DIR / "jobs.json"
JOBS_DB_PATH = API_RUNS_DIR / "jobs.sqlite3"

SNAPSHOT_STAGE_ORDER: list[StageName] = [
    "generate_snapshot",
    "train_snapshot",
    "evaluate_snapshot",
    "explain_snapshot",
]
TEMPORAL_STAGE_ORDER: list[StageName] = [
    "generate_sequence",
    "build_temporal_features",
    "train_temporal",
    "evaluate_temporal",
    "explain_temporal",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cors_origins() -> list[str]:
    value = os.getenv("API_CORS_ORIGINS", "*")
    if value.strip() == "*":
        return ["*"]
    return [item.strip() for item in value.split(",") if item.strip()]


class PipelineRunRequest(BaseModel):
    dataset_kind: DatasetKind = "snapshot"
    mode: RunMode = "full"
    stages: list[str] | None = None
    fast_mode: bool = False
    models: list[str] | None = None
    cv: int | None = Field(default=None, ge=2)
    n_jobs: int | None = None
    verbose: int | None = None
    scoring: str | None = None

    @model_validator(mode="after")
    def validate_custom_stages(self) -> "PipelineRunRequest":
        if self.mode == "custom" and not self.stages:
            raise ValueError("Custom pipeline runs require at least one stage")
        return self


class PipelineStageResponse(BaseModel):
    stage_id: str
    stage_name: str
    stage_order: int
    status: str
    command: list[str]
    log_path: str
    started_at: str | None = None
    finished_at: str | None = None
    return_code: int | None = None


class PipelineJobResponse(BaseModel):
    job_id: str
    dataset_kind: DatasetKind
    mode: RunMode
    requested_stages: list[str]
    status: str
    current_stage_name: str | None = None
    created_at: str
    started_at: str | None = None
    finished_at: str | None = None
    return_code: int | None = None
    log_path: str
    stages: list[PipelineStageResponse]


@dataclass
class PipelineStage:
    stage_id: str
    stage_name: str
    stage_order: int
    command: list[str]
    log_path: str
    status: StageStatus = "queued"
    started_at: str | None = None
    finished_at: str | None = None
    return_code: int | None = None
    error_message: str | None = None


@dataclass
class PipelineRun:
    job_id: str
    dataset_kind: DatasetKind
    mode: RunMode
    requested_stages: list[str]
    stages: list[PipelineStage]
    status: RunStatus = "queued"
    created_at: str = field(default_factory=_utc_now)
    started_at: str | None = None
    finished_at: str | None = None
    current_stage_name: str | None = None
    error_message: str | None = None

    @property
    def return_code(self) -> int | None:
        if not self.stages:
            return None
        failed_stage = next((stage for stage in self.stages if stage.return_code), None)
        if failed_stage is not None:
            return failed_stage.return_code
        completed_codes = [stage.return_code for stage in self.stages if stage.return_code is not None]
        return completed_codes[-1] if completed_codes else None

    @property
    def log_path(self) -> str:
        return str(API_RUNS_DIR / self.job_id)


class Base(DeclarativeBase):
    pass


class PipelineRunRecord(Base):
    __tablename__ = "pipeline_runs"

    job_id: Mapped[str] = mapped_column(String, primary_key=True)
    dataset_kind: Mapped[str] = mapped_column(String, nullable=False)
    mode: Mapped[str] = mapped_column(String, nullable=False)
    requested_stages_json: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    started_at: Mapped[str | None] = mapped_column(String, nullable=True)
    finished_at: Mapped[str | None] = mapped_column(String, nullable=True)
    current_stage_name: Mapped[str | None] = mapped_column(String, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    stages: Mapped[list["PipelineStageRecord"]] = relationship(
        back_populates="run",
        cascade="all, delete-orphan",
        order_by="PipelineStageRecord.stage_order",
    )


class PipelineStageRecord(Base):
    __tablename__ = "pipeline_run_stages"

    stage_id: Mapped[str] = mapped_column(String, primary_key=True)
    job_id: Mapped[str] = mapped_column(
        String,
        ForeignKey("pipeline_runs.job_id"),
        nullable=False,
        index=True,
    )
    stage_name: Mapped[str] = mapped_column(String, nullable=False)
    stage_order: Mapped[int] = mapped_column(Integer, nullable=False)
    command_json: Mapped[str] = mapped_column(Text, nullable=False)
    log_path: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    started_at: Mapped[str | None] = mapped_column(String, nullable=True)
    finished_at: Mapped[str | None] = mapped_column(String, nullable=True)
    return_code: Mapped[int | None] = mapped_column(Integer, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    run: Mapped[PipelineRunRecord] = relationship(back_populates="stages")


class LegacyPipelineJobRecord(Base):
    __tablename__ = "pipeline_jobs"

    job_id: Mapped[str] = mapped_column(String, primary_key=True)
    dataset_kind: Mapped[str] = mapped_column(String, nullable=False)
    command_json: Mapped[str] = mapped_column(Text, nullable=False)
    log_path: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[str] = mapped_column(String, nullable=False)
    started_at: Mapped[str | None] = mapped_column(String, nullable=True)
    finished_at: Mapped[str | None] = mapped_column(String, nullable=True)
    return_code: Mapped[int | None] = mapped_column(Integer, nullable=True)


def _db_engine():
    API_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    return create_engine(f"sqlite:///{JOBS_DB_PATH}", future=True)


def _init_job_db() -> None:
    Base.metadata.create_all(_db_engine())


def _record_to_stage(record: PipelineStageRecord) -> PipelineStage:
    return PipelineStage(
        stage_id=record.stage_id,
        stage_name=record.stage_name,
        stage_order=record.stage_order,
        command=json.loads(record.command_json),
        log_path=record.log_path,
        status=record.status,
        started_at=record.started_at,
        finished_at=record.finished_at,
        return_code=record.return_code,
        error_message=record.error_message,
    )


def _record_to_run(record: PipelineRunRecord) -> PipelineRun:
    return PipelineRun(
        job_id=record.job_id,
        dataset_kind=record.dataset_kind,
        mode=record.mode,
        requested_stages=json.loads(record.requested_stages_json),
        stages=[_record_to_stage(stage) for stage in sorted(record.stages, key=lambda item: item.stage_order)],
        status=record.status,
        created_at=record.created_at,
        started_at=record.started_at,
        finished_at=record.finished_at,
        current_stage_name=record.current_stage_name,
        error_message=record.error_message,
    )


def _legacy_rows_to_runs(rows: list[LegacyPipelineJobRecord]) -> dict[str, PipelineRun]:
    runs: dict[str, PipelineRun] = {}
    for row in rows:
        stage = PipelineStage(
            stage_id=f"{row.job_id}:pipeline",
            stage_name="pipeline",
            stage_order=0,
            command=json.loads(row.command_json),
            log_path=row.log_path,
            status=row.status,
            started_at=row.started_at,
            finished_at=row.finished_at,
            return_code=row.return_code,
        )
        runs[row.job_id] = PipelineRun(
            job_id=row.job_id,
            dataset_kind=row.dataset_kind,
            mode="full",
            requested_stages=["pipeline"],
            stages=[stage],
            status=row.status,
            created_at=row.created_at,
            started_at=row.started_at,
            finished_at=row.finished_at,
            current_stage_name=None,
        )
    return runs


def _load_jobs() -> dict[str, PipelineRun]:
    _init_job_db()
    with Session(_db_engine()) as session:
        run_rows = session.scalars(
            select(PipelineRunRecord).order_by(PipelineRunRecord.created_at.desc())
        ).all()
        if run_rows:
            return {row.job_id: _record_to_run(row) for row in run_rows}

        legacy_rows = session.scalars(
            select(LegacyPipelineJobRecord).order_by(LegacyPipelineJobRecord.created_at.desc())
        ).all()
    if legacy_rows:
        runs = _legacy_rows_to_runs(legacy_rows)
        _persist_jobs(runs.values())
        return runs

    if not JOBS_STATE_PATH.exists():
        return {}

    raw_jobs = json.loads(JOBS_STATE_PATH.read_text(encoding="utf-8"))
    runs = {
        item["job_id"]: PipelineRun(
            job_id=item["job_id"],
            dataset_kind=item["dataset_kind"],
            mode="full",
            requested_stages=["pipeline"],
            stages=[
                PipelineStage(
                    stage_id=f"{item['job_id']}:pipeline",
                    stage_name="pipeline",
                    stage_order=0,
                    command=item["command"],
                    log_path=item["log_path"],
                    status=item["status"],
                    started_at=item.get("started_at"),
                    finished_at=item.get("finished_at"),
                    return_code=item.get("return_code"),
                )
            ],
            status=item["status"],
            created_at=item["created_at"],
            started_at=item.get("started_at"),
            finished_at=item.get("finished_at"),
        )
        for item in raw_jobs
    }
    if runs:
        _persist_jobs(runs.values())
    return runs


def _persist_jobs(runs: list[PipelineRun] | tuple[PipelineRun, ...] | dict.values) -> None:
    _init_job_db()
    ordered_runs = list(runs)
    with Session(_db_engine()) as session:
        session.query(PipelineStageRecord).delete()
        session.query(PipelineRunRecord).delete()
        for run in ordered_runs:
            session.add(
                PipelineRunRecord(
                    job_id=run.job_id,
                    dataset_kind=run.dataset_kind,
                    mode=run.mode,
                    requested_stages_json=json.dumps(run.requested_stages),
                    status=run.status,
                    created_at=run.created_at,
                    started_at=run.started_at,
                    finished_at=run.finished_at,
                    current_stage_name=run.current_stage_name,
                    error_message=run.error_message,
                    stages=[
                        PipelineStageRecord(
                            stage_id=stage.stage_id,
                            job_id=run.job_id,
                            stage_name=stage.stage_name,
                            stage_order=stage.stage_order,
                            command_json=json.dumps(stage.command),
                            log_path=stage.log_path,
                            status=stage.status,
                            started_at=stage.started_at,
                            finished_at=stage.finished_at,
                            return_code=stage.return_code,
                            error_message=stage.error_message,
                        )
                        for stage in run.stages
                    ],
                )
            )
        session.commit()


def _save_jobs() -> None:
    _persist_jobs(sorted(_JOBS.values(), key=lambda item: item.created_at, reverse=True))


def _stage_order_for_dataset(dataset_kind: DatasetKind) -> list[StageName]:
    return SNAPSHOT_STAGE_ORDER if dataset_kind == "snapshot" else TEMPORAL_STAGE_ORDER


def _resolve_requested_stages(request: PipelineRunRequest) -> list[StageName]:
    valid_stages = _stage_order_for_dataset(request.dataset_kind)
    if request.mode == "full":
        return list(valid_stages)

    assert request.stages is not None
    requested = set(request.stages)
    invalid = requested.difference(valid_stages)
    if invalid:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unsupported stages for dataset_kind='{request.dataset_kind}': "
                f"{sorted(invalid)}"
            ),
        )
    return [stage for stage in valid_stages if stage in requested]


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


def _stage_command(stage_name: StageName, request: PipelineRunRequest) -> list[str]:
    command = [sys.executable, "-m"]

    if stage_name == "generate_snapshot":
        return command + ["incident_intelligence.cli.generator"]
    if stage_name == "generate_sequence":
        return command + ["incident_intelligence.cli.generate_sequence"]
    if stage_name == "build_temporal_features":
        return command + ["incident_intelligence.cli.build_temporal_features"]

    if stage_name in {"train_snapshot", "train_temporal"}:
        command.extend(
            [
                "incident_intelligence.cli.train",
                "--dataset-kind",
                request.dataset_kind,
            ]
        )
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

    if stage_name in {"evaluate_snapshot", "evaluate_temporal"}:
        return command + [
            "incident_intelligence.cli.evaluate",
            "--dataset-kind",
            request.dataset_kind,
        ]

    if stage_name in {"explain_snapshot", "explain_temporal"}:
        return command + [
            "incident_intelligence.cli.explain",
            "--dataset-kind",
            request.dataset_kind,
        ]

    raise HTTPException(status_code=500, detail=f"Unhandled stage '{stage_name}'")


def _build_stages(job_id: str, request: PipelineRunRequest) -> list[PipelineStage]:
    stages: list[PipelineStage] = []
    for order, stage_name in enumerate(_resolve_requested_stages(request)):
        log_dir = API_RUNS_DIR / job_id
        log_path = log_dir / f"{order + 1:02d}_{stage_name}.log"
        stages.append(
            PipelineStage(
                stage_id=f"{job_id}:{stage_name}",
                stage_name=stage_name,
                stage_order=order,
                command=_stage_command(stage_name, request),
                log_path=str(log_path),
            )
        )
    return stages


def _combined_log_text(run: PipelineRun) -> str:
    parts: list[str] = []
    for stage in run.stages:
        log_path = Path(stage.log_path)
        header = f"=== {stage.stage_name} ({stage.status}) ==="
        if log_path.exists():
            body = log_path.read_text(encoding="utf-8")
        else:
            body = ""
        parts.append(f"{header}\n{body}".rstrip())
    return "\n\n".join(part for part in parts if part.strip())


def _run_pipeline_job(job_id: str) -> None:
    with _JOBS_LOCK:
        run = _JOBS.get(job_id)
        if run is None:
            return
        run.status = "running"
        run.started_at = _utc_now()
        _save_jobs()

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(PROJECT_ROOT / "src"))
    env.setdefault("MPLBACKEND", "Agg")
    env.setdefault("MPLCONFIGDIR", "/tmp")

    for stage in run.stages:
        with _JOBS_LOCK:
            current = _JOBS.get(job_id)
            if current is None:
                return
            current.current_stage_name = stage.stage_name
            active_stage = next(
                item for item in current.stages if item.stage_id == stage.stage_id
            )
            active_stage.status = "running"
            active_stage.started_at = _utc_now()
            _save_jobs()

        log_path = Path(stage.log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w", encoding="utf-8") as log_file:
            process = subprocess.run(
                stage.command,
                cwd=PROJECT_ROOT,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )

        with _JOBS_LOCK:
            current = _JOBS.get(job_id)
            if current is None:
                return
            active_stage = next(
                item for item in current.stages if item.stage_id == stage.stage_id
            )
            active_stage.return_code = process.returncode
            active_stage.finished_at = _utc_now()
            active_stage.status = "completed" if process.returncode == 0 else "failed"
            if process.returncode != 0:
                current.status = "failed"
                current.finished_at = _utc_now()
                current.current_stage_name = active_stage.stage_name
                for pending in current.stages:
                    if pending.stage_order > active_stage.stage_order and pending.status == "queued":
                        pending.status = "skipped"
                _save_jobs()
                return
            _save_jobs()

    with _JOBS_LOCK:
        current = _JOBS.get(job_id)
        if current is None:
            return
        current.status = "completed"
        current.finished_at = _utc_now()
        current.current_stage_name = None
        _save_jobs()


def _stage_response(stage: PipelineStage) -> PipelineStageResponse:
    return PipelineStageResponse(
        stage_id=stage.stage_id,
        stage_name=stage.stage_name,
        stage_order=stage.stage_order,
        status=stage.status,
        command=stage.command,
        log_path=stage.log_path,
        started_at=stage.started_at,
        finished_at=stage.finished_at,
        return_code=stage.return_code,
    )


def _job_response(run: PipelineRun) -> PipelineJobResponse:
    return PipelineJobResponse(
        job_id=run.job_id,
        dataset_kind=run.dataset_kind,
        mode=run.mode,
        requested_stages=run.requested_stages,
        status=run.status,
        current_stage_name=run.current_stage_name,
        created_at=run.created_at,
        started_at=run.started_at,
        finished_at=run.finished_at,
        return_code=run.return_code,
        log_path=run.log_path,
        stages=[_stage_response(stage) for stage in run.stages],
    )


def _delete_job(job_id: str) -> PipelineRun:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail=f"Unknown job_id '{job_id}'")
        if job.status in {"queued", "running"}:
            raise HTTPException(
                status_code=409,
                detail=f"Cannot delete job '{job_id}' while it is {job.status}",
            )
        _JOBS.pop(job_id, None)
        _save_jobs()

    log_dir = Path(job.log_path)
    if log_dir.exists():
        for item in sorted(log_dir.rglob("*"), reverse=True):
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                item.rmdir()
        log_dir.rmdir()
    return job


app = FastAPI(
    title="Incident Intelligence API",
    version="0.1.0",
    description="Backend API for the Incident Intelligence demo dashboard.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


_JOBS: dict[str, PipelineRun] = _load_jobs()
_JOBS_LOCK = threading.Lock()


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
        "stages": {
            "snapshot": SNAPSHOT_STAGE_ORDER,
            "temporal": TEMPORAL_STAGE_ORDER,
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
            if name.endswith("_dir")
            or name
            in {
                "best_model",
                "train_metrics",
                "leaderboard",
                "evaluation_metrics",
                "evaluation_summary",
            }
        },
    }


@app.get("/api/files/{file_path:path}")
def get_project_file(file_path: str) -> FileResponse:
    requested = (PROJECT_ROOT / file_path).resolve()
    allowed_roots = [
        (PROJECT_ROOT / "artifacts").resolve(),
        (PROJECT_ROOT / "docs" / "images").resolve(),
    ]
    if not any(root == requested or root in requested.parents for root in allowed_roots):
        raise HTTPException(status_code=403, detail="File path is not allowed")
    if not requested.exists() or not requested.is_file():
        raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
    return FileResponse(requested)


@app.post("/api/pipeline/run", response_model=PipelineJobResponse)
def run_pipeline(request: PipelineRunRequest) -> PipelineJobResponse:
    job_id = uuid.uuid4().hex
    run = PipelineRun(
        job_id=job_id,
        dataset_kind=request.dataset_kind,
        mode=request.mode,
        requested_stages=list(_resolve_requested_stages(request)),
        stages=_build_stages(job_id, request),
    )
    with _JOBS_LOCK:
        _JOBS[job_id] = run
        _save_jobs()

    worker = threading.Thread(target=_run_pipeline_job, args=(job_id,), daemon=True)
    worker.start()
    return _job_response(run)


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
def get_pipeline_job_log(job_id: str) -> dict[str, object]:
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job_id '{job_id}'")

    return {
        "job_id": job_id,
        "current_stage_name": job.current_stage_name,
        "log": _combined_log_text(job),
        "stages": [
            {
                "stage_name": stage.stage_name,
                "status": stage.status,
                "log_path": stage.log_path,
            }
            for stage in job.stages
        ],
    }


@app.delete("/api/pipeline/jobs/{job_id}", response_model=PipelineJobResponse)
def delete_pipeline_job(job_id: str) -> PipelineJobResponse:
    return _job_response(_delete_job(job_id))


def main() -> None:
    uvicorn.run(
        "incident_intelligence.api.app:app",
        host=os.getenv("API_HOST", "127.0.0.1"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=False,
    )


if __name__ == "__main__":
    main()
