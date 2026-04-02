import { useEffect, useMemo, useRef, useState } from "react";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "";
const DATASET_KINDS = ["snapshot", "temporal"];
const STAGE_OPTIONS = {
  snapshot: [
    "generate_snapshot",
    "train_snapshot",
    "evaluate_snapshot",
    "explain_snapshot",
  ],
  temporal: [
    "generate_sequence",
    "build_temporal_features",
    "train_temporal",
    "evaluate_temporal",
    "explain_temporal",
  ],
};

async function fetchJson(path, options) {
  const response = await fetch(`${API_BASE_URL}${path}`, options);
  if (!response.ok) {
    const body = await response.text();
    throw new Error(body || `Request failed: ${response.status}`);
  }
  return response.json();
}

function withOptionalJob(path, jobId) {
  if (!jobId) {
    return path;
  }
  const separator = path.includes("?") ? "&" : "?";
  return `${path}${separator}job_id=${encodeURIComponent(jobId)}`;
}

function formatScore(value) {
  return typeof value === "number" ? value.toFixed(4) : "n/a";
}

function fileUrl(path, version = "") {
  const suffix = version ? `?v=${encodeURIComponent(version)}` : "";
  return `${API_BASE_URL}/api/files/${path}${suffix}`;
}

function humanizeStageName(stageName) {
  return stageName.replace(/_/g, " ");
}

function shortRunId(jobId) {
  return jobId.slice(0, 8);
}

function isVisualArtifact(path) {
  return /\.(png|jpg|jpeg|svg|webp)$/i.test(path);
}

function completedStageNamesForJobs(jobs, datasetKind, selectedJob) {
  if (selectedJob && selectedJob.dataset_kind === datasetKind) {
    return new Set(
      (Array.isArray(selectedJob.stages) ? selectedJob.stages : [])
        .filter((stage) => stage.status === "completed")
        .map((stage) => stage.stage_name),
    );
  }
  return new Set(
    jobs
      .filter((job) => job.dataset_kind === datasetKind)
      .flatMap((job) =>
        (Array.isArray(job.stages) ? job.stages : [])
          .filter((stage) => stage.status === "completed")
          .map((stage) => stage.stage_name),
      ),
  );
}

function stageReadiness(stageName, datasetKind, completedStages, summary, artifactEntries) {
  const hasModels = (artifactEntries.models_dir || []).length > 0 || Boolean(summary?.artifacts?.best_model);
  const hasEvalMetrics = Boolean(summary?.evaluation_metrics?.models?.length);
  const hasExplainArtifacts = (artifactEntries.explain_dir || []).length > 0;

  const prerequisites = {
    snapshot: {
      generate_snapshot: [],
      train_snapshot: ["generate_snapshot"],
      evaluate_snapshot: ["train_snapshot"],
      explain_snapshot: ["evaluate_snapshot"],
    },
    temporal: {
      generate_sequence: [],
      build_temporal_features: ["generate_sequence"],
      train_temporal: ["build_temporal_features"],
      evaluate_temporal: ["train_temporal"],
      explain_temporal: ["evaluate_temporal"],
    },
  };

  const missingPrereqs = (prerequisites[datasetKind][stageName] || []).filter(
    (prereq) => !completedStages.has(prereq),
  );

  if (!missingPrereqs.length) {
    return { enabled: true, reason: "" };
  }

  if (
    stageName.startsWith("evaluate_") &&
    hasModels
  ) {
    return { enabled: true, reason: "" };
  }

  if (
    stageName.startsWith("explain_") &&
    (hasEvalMetrics || hasExplainArtifacts || hasModels)
  ) {
    return { enabled: true, reason: "" };
  }

  return {
    enabled: false,
    reason: `Requires ${missingPrereqs.join(" -> ")} first`,
  };
}

function customStageReadiness(stageName, datasetKind, completedStages) {
  const orderedStages = STAGE_OPTIONS[datasetKind];
  const nextStage = orderedStages.find((item) => !completedStages.has(item)) || null;

  if (completedStages.has(stageName)) {
    return {
      enabled: false,
      reason: "Already completed in this staged run",
    };
  }

  if (stageName === nextStage) {
    return { enabled: true, reason: "" };
  }

  return {
    enabled: false,
    reason: nextStage ? `Run ${nextStage} first` : "No remaining stages to run",
  };
}

function lockedCustomStageReadiness(stageName, completedStages, nextStage, status) {
  if (completedStages.has(stageName)) {
    return {
      enabled: false,
      reason: "Already completed in this staged run",
    };
  }

  if (stageName === nextStage) {
    return {
      enabled: false,
      reason:
        status === "queued"
          ? "Current staged run is queued"
          : "Wait for the current stage to finish",
    };
  }

  return {
    enabled: false,
    reason: nextStage ? `Run ${nextStage} first` : "No remaining stages to run",
  };
}

function nextStageAfterCurrent(datasetKind, currentStageName) {
  if (!currentStageName) {
    return null;
  }
  const orderedStages = STAGE_OPTIONS[datasetKind];
  const currentIndex = orderedStages.indexOf(currentStageName);
  if (currentIndex === -1) {
    return null;
  }
  return orderedStages[currentIndex + 1] || null;
}

function StatCard({ title, value, subtitle }) {
  return (
    <div className="card stat-card">
      <div className="stat-title">{title}</div>
      <div className="stat-value">{value}</div>
      {subtitle ? <div className="stat-subtitle">{subtitle}</div> : null}
    </div>
  );
}

function SummaryList({ items }) {
  return (
    <div className="summary-list">
      {items.map((item) => (
        <div className="summary-list-row" key={item.title}>
          <div className="summary-list-label">{item.title}</div>
          <div className="summary-list-value">
            <span>{item.value}</span>
            {item.subtitle ? (
              <span className="summary-list-subtitle">{item.subtitle}</span>
            ) : null}
          </div>
        </div>
      ))}
    </div>
  );
}

function ArtifactSection({ title, items, sectionId }) {
  const [expanded, setExpanded] = useState(false);
  const previewItems = expanded ? items : items.slice(0, 2);
  const remainingCount = Math.max(items.length - 2, 0);

  return (
    <div className="artifact-section" id={sectionId}>
      <div className="artifact-section-title">{title}</div>
      {items.length === 0 ? (
        <div className="muted">No artifacts found yet.</div>
      ) : (
        <ul className="artifact-list">
          {previewItems.map((item) => (
            <li key={item.path}>
              <a
                className="artifact-link"
                href={fileUrl(item.path)}
                target="_blank"
                rel="noreferrer"
              >
                <code>{item.path}</code>
              </a>
            </li>
          ))}
          {remainingCount ? (
            <li className="artifact-more">
              <button
                type="button"
                className="artifact-more-button"
                onClick={() => setExpanded((current) => !current)}
              >
                {expanded ? "Show less" : `+${remainingCount} more`}
              </button>
            </li>
          ) : null}
        </ul>
      )}
    </div>
  );
}

function MetricTable({ rows }) {
  if (!rows?.length) {
    return <div className="muted">No model rows available yet.</div>;
  }

  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th>Accuracy</th>
            <th>F1 Macro</th>
            <th>Precision</th>
            <th>Recall</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.model_name}>
              <td>{row.model_name}</td>
              <td>{formatScore(row.accuracy)}</td>
              <td>{formatScore(row.f1_macro)}</td>
              <td>{formatScore(row.precision_macro)}</td>
              <td>{formatScore(row.recall_macro)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function JobList({
  jobs,
  onSelect,
  onDelete,
  onCancel,
  selectedJobId,
  deletingJobId,
  cancellingJobId,
}) {
  if (!jobs.length) {
    return <div className="muted">No pipeline jobs started yet.</div>;
  }

  const groupedJobs = [
    {
      key: "custom",
      title: "Staged Runs",
      items: jobs.filter((job) => job.mode === "custom"),
    },
    {
      key: "full",
      title: "Full Pipeline Runs",
      items: jobs.filter((job) => job.mode !== "custom"),
    },
  ].filter((group) => group.items.length);

  return (
    <div className="job-list">
      {groupedJobs.map((group) => (
        <div className="job-group" key={group.key}>
          <div className="job-group-title">{group.title}</div>
          <div className="job-list-header">
            <div className="job-list-header-main">
              <span>Status</span>
              <span>Kind</span>
              <span>Time</span>
            </div>
            <span>Action</span>
          </div>
          {group.items.map((job) => (
            (() => {
              const stages = Array.isArray(job.stages) ? job.stages : [];
              const completedStageCount = stages.filter(
                (stage) => stage.status === "completed",
              ).length;
              return (
                <div
                  key={job.job_id}
                  className={`job-item ${
                    selectedJobId === job.job_id ? "selected" : ""
                  }`}
                >
                  <button
                    className="job-select"
                    onClick={() => onSelect(job)}
                    type="button"
                  >
                    <div className="job-primary-row">
                      <span className={`status-chip ${job.status}`}>{job.status}</span>
                      <span className="job-kind">{job.dataset_kind}</span>
                      <span className="job-meta">
                        {job.finished_at
                          ? `finished ${job.finished_at}`
                          : job.current_stage_name
                            ? `running ${job.current_stage_name}`
                            : job.created_at}
                      </span>
                    </div>
                    <div className="job-secondary-row">
                      <span className="job-mode">{job.mode}</span>
                      <span className="job-run-id">run {shortRunId(job.job_id)}</span>
                      <span className="job-stage-summary">
                        {completedStageCount}/{stages.length} stages
                      </span>
                      <span className="job-stage-list">
                        {stages.map((stage) => (
                          <span
                            key={stage.stage_id}
                            className={`stage-chip ${stage.status}`}
                          >
                            {stage.stage_name}
                          </span>
                        ))}
                      </span>
                    </div>
                  </button>
                  {job.status === "queued" || job.status === "running" || job.status === "cancelling" ? (
                    <button
                      className="job-cancel"
                      type="button"
                      onClick={() => onCancel(job.job_id)}
                      disabled={cancellingJobId === job.job_id || job.status === "cancelling"}
                      aria-label={`Cancel job ${job.job_id}`}
                      title="Cancel job"
                    >
                      {cancellingJobId === job.job_id || job.status === "cancelling"
                        ? "Cancelling..."
                        : "Cancel Run"}
                    </button>
                  ) : (
                    <button
                      className="job-delete"
                      type="button"
                      onClick={() => onDelete(job.job_id)}
                      disabled={deletingJobId === job.job_id}
                      aria-label={`Delete job ${job.job_id}`}
                      title="Delete job"
                    >
                      {deletingJobId === job.job_id ? "Deleting..." : "Delete Run"}
                    </button>
                  )}
                </div>
              );
            })()
          ))}
        </div>
      ))}
    </div>
  );
}

function VisualGallery({ visuals, assetVersion }) {
  if (!visuals.length) {
    return (
      <div className="muted">
        No confusion matrices or feature importance plots found yet.
      </div>
    );
  }

  return (
    <div className="visual-grid">
      {visuals.map((visual) => (
        <figure className="visual-card" key={visual.path}>
          <a
            className="visual-link"
            href={fileUrl(visual.path, assetVersion)}
            target="_blank"
            rel="noreferrer"
          >
            <img alt={visual.title} src={fileUrl(visual.path, assetVersion)} />
            <span className="visual-hover-preview" aria-hidden="true">
              <img alt="" src={fileUrl(visual.path, assetVersion)} />
            </span>
          </a>
          <figcaption>
            <div className="visual-title">{visual.title}</div>
          </figcaption>
        </figure>
      ))}
    </div>
  );
}

export default function App() {
  const [datasetKind, setDatasetKind] = useState("snapshot");
  const [summary, setSummary] = useState(null);
  const [artifacts, setArtifacts] = useState(null);
  const [jobs, setJobs] = useState([]);
  const [selectedJobId, setSelectedJobId] = useState(null);
  const [selectedJobLog, setSelectedJobLog] = useState("");
  const [forceNewCustomRun, setForceNewCustomRun] = useState(false);
  const [deletingJobId, setDeletingJobId] = useState(null);
  const [cancellingJobId, setCancellingJobId] = useState(null);
  const [loadingSummary, setLoadingSummary] = useState(false);
  const [loadingArtifacts, setLoadingArtifacts] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [assetVersion, setAssetVersion] = useState("");
  const lastCompletedJobRef = useRef("");
  const previousModeRef = useRef("full");
  const [runForm, setRunForm] = useState({
    mode: "full",
    stages: STAGE_OPTIONS.snapshot,
    fast_mode: false,
    models: "logistic,rf",
    cv: datasetKind === "temporal" ? "3" : "",
    n_jobs: "1",
    verbose: "0",
    scoring: "f1_macro",
  });

  useEffect(() => {
    refreshDashboard(datasetKind, selectedJob?.dataset_kind === datasetKind ? selectedJob.job_id : null);
  }, [datasetKind, selectedJobId, jobs]);

  useEffect(() => {
    setRunForm((current) => ({
      ...current,
      stages: STAGE_OPTIONS[datasetKind],
      cv: datasetKind === "temporal" && !current.cv ? "3" : current.cv,
    }));
    setForceNewCustomRun(false);
  }, [datasetKind]);

  useEffect(() => {
    refreshJobs();
    const timer = window.setInterval(refreshJobs, 3000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    if (!selectedJobId) {
      setSelectedJobLog("");
      return;
    }
    loadJobLog(selectedJobId);
  }, [selectedJobId, jobs]);

  useEffect(() => {
    const latestCompletedJob = jobs.find(
      (job) =>
        job.dataset_kind === datasetKind &&
        (job.status === "completed" || job.status === "failed") &&
        job.finished_at,
    );

    const completedKey = latestCompletedJob
      ? `${latestCompletedJob.job_id}:${latestCompletedJob.status}:${latestCompletedJob.finished_at}`
      : "";

    if (completedKey && completedKey !== lastCompletedJobRef.current) {
      lastCompletedJobRef.current = completedKey;
      refreshDashboard(datasetKind);
    }
  }, [jobs, datasetKind]);

  async function refreshDashboard(kind, jobId = null) {
    setError("");
    setLoadingSummary(true);
    setLoadingArtifacts(true);
    try {
      const [summaryData, artifactData] = await Promise.all([
        fetchJson(withOptionalJob(`/api/dashboard/summary/${kind}`, jobId)),
        fetchJson(withOptionalJob(`/api/artifacts/${kind}`, jobId)),
      ]);
      setSummary(summaryData);
      setArtifacts(artifactData);
      setAssetVersion(`${kind}-${Date.now()}`);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoadingSummary(false);
      setLoadingArtifacts(false);
    }
  }

  async function refreshJobs() {
    try {
      const data = await fetchJson("/api/pipeline/jobs");
      setJobs(data);
    } catch (err) {
      setError(err.message);
    }
  }

  async function loadJobLog(jobId) {
    try {
      const data = await fetchJson(`/api/pipeline/jobs/${jobId}/log`);
      setSelectedJobLog(data.log);
    } catch (err) {
      setError(err.message);
    }
  }

  async function deleteJob(jobId) {
    setError("");
    setDeletingJobId(jobId);
    try {
      await fetchJson(`/api/pipeline/jobs/${jobId}`, { method: "DELETE" });
      const nextJobs = jobs.filter((job) => job.job_id !== jobId);
      if (selectedJobId === jobId) {
        setSelectedJobId(null);
        setSelectedJobLog("");
      }
      await refreshJobs();
    } catch (err) {
      setError(err.message);
    } finally {
      setDeletingJobId(null);
    }
  }

  async function cancelJob(jobId) {
    setError("");
    setCancellingJobId(jobId);
    try {
      await fetchJson(`/api/pipeline/jobs/${jobId}/cancel`, { method: "POST" });
      await refreshJobs();
    } catch (err) {
      setError(err.message);
    } finally {
      setCancellingJobId(null);
    }
  }

  function selectDatasetKind(kind) {
    setForceNewCustomRun(false);
    setSelectedJobId(null);
    setSelectedJobLog("");
    setDatasetKind(kind);
  }

  function selectJob(job) {
    setForceNewCustomRun(false);
    setRunForm((current) => ({
      ...current,
      mode: job.mode === "custom" ? "custom" : "full",
      stages:
        job.mode === "full"
          ? STAGE_OPTIONS[job.dataset_kind]
          : current.stages,
      cv:
        job.dataset_kind === "temporal" && !current.cv ? "3" : current.cv,
    }));
    setSelectedJobId(job.job_id);
    setSelectedJobLog("");
    loadJobLog(job.job_id);
    setDatasetKind(job.dataset_kind);
    refreshDashboard(job.dataset_kind, job.job_id);
  }

  async function submitRun(event) {
    event.preventDefault();
    await startRun();
  }

  async function startRun(stageOverride = null, modeOverride = null) {
    setSubmitting(true);
    setError("");
    try {
      const nextMode = modeOverride || runForm.mode;
      const nextStages = stageOverride || (nextMode === "custom" ? runForm.stages : null);
      const payload = {
        dataset_kind: datasetKind,
        mode: nextMode,
        stages: nextMode === "custom" ? nextStages : null,
        source_job_id:
          nextMode === "custom" &&
          sourceJob?.dataset_kind === datasetKind &&
          sourceJob?.status === "completed"
            ? sourceJob.job_id
            : null,
        force_new_run: nextMode === "custom" && forceNewCustomRun,
        fast_mode: runForm.fast_mode,
        models: runForm.models
          ? runForm.models
              .split(",")
              .map((item) => item.trim())
              .filter(Boolean)
          : null,
        cv: runForm.cv ? Number(runForm.cv) : null,
        n_jobs: runForm.n_jobs ? Number(runForm.n_jobs) : null,
        verbose: runForm.verbose ? Number(runForm.verbose) : null,
        scoring: runForm.scoring || null,
      };
      const job = await fetchJson("/api/pipeline/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      setSelectedJobId(job.job_id);
      await refreshJobs();
    } catch (err) {
      setError(err.message);
    } finally {
      setSubmitting(false);
    }
  }

  const modelRows = useMemo(() => {
    return (
      summary?.evaluation_metrics?.models?.map((model) => ({
        model_name: model.model_name,
        ...model.metrics,
      })) || []
    );
  }, [summary]);

  const bestModel = useMemo(() => {
    if (!modelRows.length) {
      return null;
    }
    return [...modelRows].sort(
      (left, right) => (right.f1_macro ?? -1) - (left.f1_macro ?? -1),
    )[0];
  }, [modelRows]);

  const artifactEntries = artifacts?.artifacts || {};
  const visuals = useMemo(() => {
    const plotFiles = artifactEntries.plots_dir || [];
    const explainFiles = artifactEntries.explain_dir || [];

    const evaluationVisuals = plotFiles.filter((item) =>
      isVisualArtifact(item.path),
    );

    const explainabilityVisuals = explainFiles.filter((item) =>
      isVisualArtifact(item.path),
    );

    return [...evaluationVisuals, ...explainabilityVisuals].map((item) => ({
      path: item.path,
      title: item.path
        .split("/")
        .slice(-1)[0]
        .replace(/_/g, " ")
        .replace(".png", ""),
    }));
  }, [artifactEntries]);
  const featuredVisual = visuals[0] || null;

  const headlineStats = [
    {
      title: "Best F1 Macro",
      value: bestModel ? formatScore(bestModel.f1_macro) : "Run a pipeline",
      subtitle: bestModel ? bestModel.model_name : "No evaluation results yet",
    },
    {
      title: "Tracked Jobs",
      value: jobs.length,
      subtitle: jobs.length ? "Persisted run history" : "No runs recorded yet",
    },
    {
      title: "Plots & Reports",
      value:
        (artifactEntries.plots_dir?.length || 0) +
        (artifactEntries.reports_dir?.length || 0),
      subtitle: `${datasetKind} evaluation visuals and reports`,
    },
  ];

  const visibleJobs = useMemo(() => jobs, [jobs]);
  const selectedJob = useMemo(
    () => jobs.find((job) => job.job_id === selectedJobId) || null,
    [jobs, selectedJobId],
  );
  const selectedCustomRun = useMemo(() => {
    if (runForm.mode !== "custom") {
      return null;
    }
    if (forceNewCustomRun) {
      return null;
    }
    if (
      selectedJob &&
      selectedJob.dataset_kind === datasetKind &&
      selectedJob.mode === "custom"
    ) {
      return selectedJob;
    }
    return null;
  }, [datasetKind, forceNewCustomRun, runForm.mode, selectedJob]);
  const customRunLocked = Boolean(
    selectedCustomRun &&
      ["queued", "running", "cancelling"].includes(selectedCustomRun.status),
  );
  const sourceJob =
    selectedCustomRun &&
    ["completed", "failed", "cancelled"].includes(selectedCustomRun.status)
      ? selectedCustomRun
      : null;
  const completedStages = useMemo(() => {
    if (runForm.mode === "custom" && !selectedCustomRun) {
      return new Set();
    }
    return completedStageNamesForJobs(jobs, datasetKind, selectedCustomRun);
  }, [jobs, datasetKind, runForm.mode, selectedCustomRun]);
  const nextIncompleteStage = useMemo(
    () =>
      STAGE_OPTIONS[datasetKind].find((stageName) => !completedStages.has(stageName)) ||
      null,
    [completedStages, datasetKind],
  );
  const lockedPreviewStage = useMemo(() => {
    if (!customRunLocked || !selectedCustomRun) {
      return nextIncompleteStage;
    }
    if (selectedCustomRun.status === "queued") {
      return selectedCustomRun.current_stage_name || nextIncompleteStage;
    }
    return (
      nextStageAfterCurrent(datasetKind, selectedCustomRun.current_stage_name) ||
      nextIncompleteStage
    );
  }, [customRunLocked, datasetKind, nextIncompleteStage, selectedCustomRun]);
  const stageAvailability = useMemo(
    () =>
      Object.fromEntries(
        STAGE_OPTIONS[datasetKind].map((stageName) => [
          stageName,
          runForm.mode === "custom"
            ? customRunLocked
              ? lockedCustomStageReadiness(
                  stageName,
                  completedStages,
                  lockedPreviewStage,
                  selectedCustomRun?.status ?? "running",
                )
              : customStageReadiness(stageName, datasetKind, completedStages)
            : stageReadiness(
                stageName,
                datasetKind,
                completedStages,
                summary,
                artifactEntries,
              ),
        ]),
      ),
    [
      artifactEntries,
      completedStages,
      customRunLocked,
      datasetKind,
      lockedPreviewStage,
      runForm.mode,
      selectedCustomRun?.status,
      summary,
    ],
  );
  const lastSuccessfulStage = useMemo(() => {
    const ordered = STAGE_OPTIONS[datasetKind];
    const completedOrdered = ordered.filter((stageName) =>
      completedStages.has(stageName),
    );
    return completedOrdered.at(-1) || null;
  }, [completedStages, datasetKind]);
  const nextRunnableStage = useMemo(
    () => (customRunLocked ? lockedPreviewStage : STAGE_OPTIONS[datasetKind].find(
      (stageName) =>
        !completedStages.has(stageName) && stageAvailability[stageName]?.enabled,
    ) || null),
    [completedStages, customRunLocked, datasetKind, lockedPreviewStage, stageAvailability],
  );
  const customRunComplete =
    runForm.mode === "custom" &&
    Boolean(selectedCustomRun) &&
    completedStages.size === STAGE_OPTIONS[datasetKind].length;
  const canStartNewCustomRun =
    runForm.mode === "custom" &&
    !forceNewCustomRun &&
    Boolean(selectedCustomRun);
  const selectedStageSummary = useMemo(() => {
    if (runForm.mode === "full") {
      return {
        primary: `All ${datasetKind} stages will run in order.`,
        secondary: null,
      };
    }
    if (customRunComplete && selectedCustomRun) {
      return {
        primary: `Staged run ${shortRunId(selectedCustomRun.job_id)} is complete.`,
        secondary: "There are no remaining stages to run.",
      };
    }
    if (customRunLocked && selectedCustomRun) {
      return {
        primary: `Staged run ${shortRunId(selectedCustomRun.job_id)} is ${selectedCustomRun.status}.`,
        secondary:
          selectedCustomRun.current_stage_name
            ? `Wait for ${selectedCustomRun.current_stage_name} to finish before continuing, or start a new staged run.`
            : "Wait for the active stage to finish before continuing, or start a new staged run.",
      };
    }
    if (!runForm.stages.length) {
      return {
        primary: "Select at least one stage to run.",
        secondary: null,
      };
    }
    if (sourceJob) {
      return {
        primary: `Continuing staged run ${shortRunId(sourceJob.job_id)} with: ${runForm.stages.join(", ")}`,
        secondary: null,
      };
    }
    return {
      primary: `Starting a staged run with: ${runForm.stages.join(", ")}`,
      secondary: "Select a staged run below to continue it later.",
    };
  }, [
    customRunComplete,
    customRunLocked,
    datasetKind,
    runForm.mode,
    runForm.stages,
    selectedCustomRun,
    sourceJob,
  ]);

  const trainStageName = datasetKind === "snapshot" ? "train_snapshot" : "train_temporal";
  const tuningEnabled =
    runForm.mode === "full" || runForm.stages.includes(trainStageName);
  const runButtonLabel = useMemo(() => {
    if (submitting) {
      return "Starting...";
    }
    if (runForm.mode === "full") {
      return `Run full ${datasetKind} pipeline`;
    }
    if (runForm.stages.length === 1) {
      return `Run ${humanizeStageName(runForm.stages[0])}`;
    }
    return `Run custom ${datasetKind} stages`;
  }, [datasetKind, runForm.mode, runForm.stages, submitting]);

  useEffect(() => {
    const previousMode = previousModeRef.current;
    previousModeRef.current = runForm.mode;
    if (runForm.mode !== "custom") {
      setForceNewCustomRun(false);
    }
    if (runForm.mode !== "custom" || previousMode === "custom") {
      return;
    }
    setRunForm((current) => ({
      ...current,
      stages: nextRunnableStage ? [nextRunnableStage] : [],
    }));
  }, [nextRunnableStage, runForm.mode]);

  useEffect(() => {
    if (runForm.mode !== "custom") {
      return;
    }
    setRunForm((current) => {
      const filteredStages = customRunLocked
        ? current.stages.filter((stageName) => stageName === nextIncompleteStage)
        : current.stages.filter((stageName) => stageAvailability[stageName]?.enabled);
      if (filteredStages.length === current.stages.length) {
        return current;
      }
      return {
        ...current,
        stages:
          filteredStages.length > 0
            ? filteredStages
            : nextRunnableStage
              ? [nextRunnableStage]
              : [],
      };
    });
  }, [customRunLocked, nextIncompleteStage, nextRunnableStage, runForm.mode, stageAvailability]);

  useEffect(() => {
    if (runForm.mode !== "custom" || !forceNewCustomRun) {
      return;
    }
    setRunForm((current) => {
      const nextStages = nextRunnableStage ? [nextRunnableStage] : [];
      const sameStages =
        current.stages.length === nextStages.length &&
        current.stages.every((stage, index) => stage === nextStages[index]);
      if (sameStages) {
        return current;
      }
      return {
        ...current,
        stages: nextStages,
      };
    });
  }, [forceNewCustomRun, nextRunnableStage, runForm.mode]);

  return (
    <div className="app-shell">
      <section className="hero-shell">
        <div className="hero-eyebrow-row">
          <div className="eyebrow">Incident Intelligence</div>
        </div>
        <header className="hero">
          <div className="hero-copy">
            <div className="hero-title-row">
              <h1>
                Incident root cause modeling - from synthetic telemetry and
                temporal features to explainable root-cause results.
              </h1>
            </div>
            <div className="hero-description">
              <p>
                root-cause analysis: generate data, train baseline models,
                compare snapshot versus temporal features, and inspect the
                artifacts those runs produce.
              </p>
            </div>
            <div className="narrative-grid">
              <div className="narrative-item">
                <div className="narrative-title">Problem</div>
                <div className="narrative-body">
                  Distinguish deployment, dependency, traffic, CPU, and memory
                  failure modes from telemetry patterns.
                </div>
              </div>
              <div className="narrative-item">
                <div className="narrative-title">Why Temporal</div>
                <div className="narrative-body">
                  Many incidents are about shape over time, not just one static
                  row of metrics.
                </div>
              </div>
              <div className="narrative-item">
                <div className="narrative-title">What You Can Inspect</div>
                <div className="narrative-body">
                  Evaluation tables, confusion matrices, feature importance
                  plots, explainability outputs, and job logs.
                </div>
              </div>
            </div>
          </div>
        </header>
      </section>

      {error ? <div className="error-banner">{error}</div> : null}

      <section className="workflow-tabs-row">
        <div className="hero-side-inner workflow-tabs">
          <div className="hero-control-group workflow-tabs-group">
            <div className="hero-control-label">Workflow</div>
            <div className="dataset-toggle">
              {DATASET_KINDS.map((kind) => (
                <button
                  key={kind}
                  type="button"
                  className={datasetKind === kind ? "active" : ""}
                  onClick={() => selectDatasetKind(kind)}
                >
                  {kind}
                </button>
              ))}
            </div>
          </div>
        </div>
      </section>

      <section className="grid">
        <div className="card form-card">
          <div className="run-ppl-col">
            <div className="section-title">Run Pipeline</div>
            <form onSubmit={submitRun} className="run-form">
              <div className="tabbed-form-shell">
                <div className="stage-mode-row">
                  <span className="form-label">Mode</span>
                  <div className="stage-mode-toggle">
                    {["full", "custom"].map((mode) => (
                      <button
                        key={mode}
                        type="button"
                        className={runForm.mode === mode ? "active" : ""}
                        onClick={() =>
                          setRunForm((current) => ({
                            ...current,
                            mode,
                            stages:
                              mode === "full"
                                ? STAGE_OPTIONS[datasetKind]
                                : current.stages,
                          }))
                        }
                      >
                        {mode}
                      </button>
                    ))}
                  </div>
                </div>
                <div className="run-form-main">
                  <div className="stage-picker">
                    <span className="form-label">Stages</span>
                    <div className="stage-options">
                      {STAGE_OPTIONS[datasetKind].map((stageName) => {
                        const checked =
                          !customRunComplete &&
                          (customRunLocked
                            ? stageName === nextIncompleteStage
                            : runForm.stages.includes(stageName));
                        const availability = stageAvailability[stageName];
                        return (
                          <label
                            key={stageName}
                            className={`stage-option ${availability?.enabled ? "" : "disabled"}`}
                          >
                            <input
                              type="checkbox"
                              checked={checked}
                              disabled={
                                runForm.mode === "full" ||
                                customRunComplete ||
                                !availability?.enabled
                              }
                              onChange={(event) =>
                                setRunForm((current) => {
                                  const nextStages = event.target.checked
                                    ? [...current.stages, stageName]
                                    : current.stages.filter(
                                        (item) => item !== stageName,
                                      );
                                  return {
                                    ...current,
                                    stages: STAGE_OPTIONS[datasetKind].filter((item) =>
                                      nextStages.includes(item),
                                    ),
                                  };
                                })
                              }
                            />
                            <div className="stage-option-copy">
                              <span className="stage-option-name">{stageName}</span>
                              {!availability?.enabled ? (
                                <span className="stage-option-hint">{availability.reason}</span>
                              ) : null}
                            </div>
                          </label>
                        );
                      })}
                    </div>
                    <div className="stage-helper-copy">
                      <span className="stage-helper-line stage-helper-line-primary">
                        {selectedStageSummary.primary}
                      </span>
                      {selectedStageSummary.secondary ? (
                        <span className="stage-helper-line stage-helper-line-secondary">
                          * {selectedStageSummary.secondary}
                        </span>
                      ) : null}
                    </div>
                    {canStartNewCustomRun ? (
                      <button
                        className="secondary-button"
                        type="button"
                        onClick={() => {
                          setSelectedJobId(null);
                          setSelectedJobLog("");
                          setForceNewCustomRun(true);
                        }}
                      >
                        Start new staged run
                      </button>
                    ) : null}
                  </div>
                  <div className="run-form-fields">
                    <div className="tuning-inline-grid">
                      <label className="compact-field compact-field-wide">
                        <span className="form-label">Models</span>
                        <input
                          value={runForm.models}
                          disabled={!tuningEnabled}
                          onChange={(event) =>
                            setRunForm((current) => ({
                              ...current,
                              models: event.target.value,
                            }))
                          }
                          placeholder="logistic,rf"
                        />
                      </label>
                      <label className="compact-field">
                        <span className="form-label">CV</span>
                        <input
                          value={runForm.cv}
                          disabled={!tuningEnabled}
                          onChange={(event) =>
                            setRunForm((current) => ({
                              ...current,
                              cv: event.target.value,
                            }))
                          }
                          placeholder="3"
                        />
                      </label>
                      <label className="compact-field">
                        <span className="form-label">n_jobs</span>
                        <input
                          value={runForm.n_jobs}
                          disabled={!tuningEnabled}
                          onChange={(event) =>
                            setRunForm((current) => ({
                              ...current,
                              n_jobs: event.target.value,
                            }))
                          }
                          placeholder="1"
                        />
                      </label>
                      <label className="compact-field">
                        <span className="form-label">Verbose</span>
                        <input
                          value={runForm.verbose}
                          disabled={!tuningEnabled}
                          onChange={(event) =>
                            setRunForm((current) => ({
                              ...current,
                              verbose: event.target.value,
                            }))
                          }
                          placeholder="0"
                        />
                      </label>
                      <label className="compact-field compact-field-wide">
                        <span className="form-label">Scoring</span>
                        <input
                          value={runForm.scoring}
                          disabled={!tuningEnabled}
                          onChange={(event) =>
                            setRunForm((current) => ({
                              ...current,
                              scoring: event.target.value,
                            }))
                          }
                          placeholder="f1_macro"
                        />
                      </label>
                      <label className="compact-field compact-checkbox-field">
                        <span className="form-label">Fast mode</span>
                        <input
                          type="checkbox"
                          checked={runForm.fast_mode}
                          disabled={!tuningEnabled}
                          onChange={(event) =>
                            setRunForm((current) => ({
                              ...current,
                              fast_mode: event.target.checked,
                            }))
                          }
                        />
                      </label>
                    </div>
                    {!tuningEnabled ? (
                      <div className="stage-option-hint">
                        Tuning parameters apply only when a train stage is selected.
                      </div>
                    ) : null}
                    {!customRunComplete ? (
                      <div className="tuning-actions">
                        <button
                          className="primary-button"
                          type="submit"
                          disabled={
                            submitting ||
                            (runForm.mode === "custom" && runForm.stages.length === 0)
                          }
                        >
                          {runButtonLabel}
                        </button>
                      </div>
                    ) : null}
                  </div>
                </div>
              </div>
            </form>
          </div>
          <div className="form-subsection jobs-subsection">
            <div className="section-title section-title-compact">
              Recent Jobs
            </div>
            <JobList
              jobs={visibleJobs}
              onSelect={selectJob}
              onDelete={deleteJob}
              onCancel={cancelJob}
              selectedJobId={selectedJobId}
              deletingJobId={deletingJobId}
              cancellingJobId={cancellingJobId}
            />
          </div>
          <div className="form-subsection log-subsection">
            <div className="section-title section-title-compact">
              Selected Job Log
            </div>
            {selectedJob ? (
              <div className="selected-job-meta">
                <span className={`status-chip ${selectedJob.status}`}>
                  {selectedJob.status}
                </span>
                <span>
                  {(Array.isArray(selectedJob.stages) ? selectedJob.stages : []).filter(
                    (stage) => stage.status === "completed",
                  ).length}
                  /{(Array.isArray(selectedJob.stages) ? selectedJob.stages : []).length}{" "}
                  stages complete
                </span>
                {selectedJob.current_stage_name ? (
                  <span>Current: {selectedJob.current_stage_name}</span>
                ) : null}
              </div>
            ) : null}
            <pre className="log-view">
              {selectedJobLog ||
                "No log selected yet. Launch a pipeline run to watch logs stream here."}
            </pre>
          </div>
        </div>
      </section>

      <section className="grid">
        <div className="card summary-card">
          {loadingSummary ? (
            <div className="muted">Loading summary...</div>
          ) : (
            <div>
              <div className="card-band-top">
                <div className="section-title">Latest Summary</div>
                <SummaryList
                  items={[
                    ...headlineStats,
                    {
                      title: "Models Dir",
                      value: summary?.artifacts?.models_dir ? (
                        <a className="stat-link" href="#artifact-models">
                          {summary.artifacts.models_dir.split("/").slice(-1)[0]}
                        </a>
                      ) : (
                        "n/a"
                      ),
                    },
                    {
                      title: "Best Model",
                      value: summary?.artifacts?.best_model ? (
                        <a
                          className="stat-link"
                          href={fileUrl(summary.artifacts.best_model)}
                          target="_blank"
                          rel="noreferrer"
                        >
                          {summary.artifacts.best_model.split("/").slice(-1)[0]}
                        </a>
                      ) : (
                        "n/a"
                      ),
                    },
                    {
                      title: "Eval Models",
                      value: summary?.evaluation_metrics?.models?.length ?? 0,
                      subtitle: datasetKind,
                    },
                    {
                      title: "Metrics File",
                      value: summary?.artifacts?.evaluation_metrics ? (
                        <a
                          className="stat-link"
                          href={fileUrl(summary.artifacts.evaluation_metrics)}
                          target="_blank"
                          rel="noreferrer"
                        >
                          {
                            summary.artifacts.evaluation_metrics
                              .split("/")
                              .slice(-1)[0]
                          }
                        </a>
                      ) : (
                        "n/a"
                      ),
                    },
                  ]}
                />
              </div>
              <div className="summary-feature evaluation-feature">
                <div className="section-title section-title-compact">
                  Model Evaluation
                </div>
                <MetricTable rows={modelRows} />
              </div>
              <div className="summary-feature featured-result-feature">
                <div className="section-title section-title-compact">
                  Featured Result
                </div>
                {featuredVisual ? (
                  <figure className="featured-visual">
                    <a
                      className="featured-visual-link"
                      href={fileUrl(featuredVisual.path, assetVersion)}
                      target="_blank"
                      rel="noreferrer"
                    >
                      <img
                        alt={featuredVisual.title}
                        src={fileUrl(featuredVisual.path, assetVersion)}
                      />
                      <span
                        className="featured-hover-preview"
                        aria-hidden="true"
                      >
                        <img
                          alt=""
                          src={fileUrl(featuredVisual.path, assetVersion)}
                        />
                      </span>
                    </a>
                    <figcaption>
                      <div className="visual-title">{featuredVisual.title}</div>
                    </figcaption>
                  </figure>
                ) : (
                  <div className="empty-state">
                    Run a pipeline to surface confusion matrices, model
                    comparison plots, feature-importance visuals, and
                    explainability outputs here.
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </section>

      <section className="grid">
        <div className="card">
          <div className="section-title">
            Evaluation & Explainability Visuals
          </div>
          {visuals.length ? (
            <VisualGallery
              visuals={visuals.slice(1)}
              assetVersion={assetVersion}
            />
          ) : (
            <div className="empty-state">
              No visuals yet. Start a run to populate this gallery with
              evaluation plots and explainability outputs.
            </div>
          )}
        </div>
      </section>

      <section className="grid">
        <div className="card">
          <div className="section-title">Artifact Inventory</div>
          {loadingArtifacts ? (
            <div className="muted">Loading artifacts...</div>
          ) : (
            <div className="artifact-inventory-block">
              <ArtifactSection
                title="Models"
                items={artifactEntries.models_dir || []}
                sectionId="artifact-models"
              />
              <ArtifactSection
                title="Metrics"
                items={artifactEntries.train_metrics || []}
                sectionId={undefined}
              />
              <ArtifactSection
                title="Plots"
                items={artifactEntries.plots_dir || []}
                sectionId={undefined}
              />
              <ArtifactSection
                title="Reports"
                items={artifactEntries.reports_dir || []}
                sectionId={undefined}
              />
              <ArtifactSection
                title="Explainability"
                items={artifactEntries.explain_dir || []}
                sectionId={undefined}
              />
            </div>
          )}
        </div>
      </section>
    </div>
  );
}
