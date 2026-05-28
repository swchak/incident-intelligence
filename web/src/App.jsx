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
const MODEL_OPTIONS = [
  { value: "logistic", label: "Logistic Regression" },
  { value: "rf", label: "Random Forest" },
  { value: "gb", label: "Gradient Boosting" },
  { value: "svm", label: "SVM (RBF)" },
];
const SCORING_OPTIONS = [
  { value: "f1_macro", label: "F1 Macro" },
  { value: "accuracy", label: "Accuracy" },
  { value: "precision_macro", label: "Precision Macro" },
  { value: "recall_macro", label: "Recall Macro" },
  { value: "f1_weighted", label: "F1 Weighted" },
  { value: "precision_weighted", label: "Precision Weighted" },
  { value: "recall_weighted", label: "Recall Weighted" },
];
const SECTION_LINKS = [
  { id: "run-pipeline", label: "Run Pipeline" },
  { id: "recent-jobs", label: "Recent Jobs" },
  { id: "selected-job-log", label: "Selected Job Log" },
  { id: "latest-run-results", label: "Latest Run Results" },
  { id: "knowledge-search", label: "Knowledge Base" },
  { id: "visuals", label: "Visuals" },
  { id: "artifact-inventory", label: "Artifact Inventory" },
];
const KNOWLEDGE_LOG_EMPTY_MESSAGE =
  "No knowledge-base activity yet. Generate KB docs, build the RAG index, or search to see activity here.";

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

function stageSummaryLabel(stageName, datasetKind) {
  if (!stageName) {
    return "No stages yet";
  }
  const suffix = `_${datasetKind}`;
  const trimmed =
    stageName.endsWith(suffix) ? stageName.slice(0, -suffix.length) : stageName;
  return humanizeStageName(trimmed);
}

function shortRunId(jobId) {
  return jobId.slice(0, 8);
}

function titleCaseLabel(value) {
  return value
    .replace(/[_-]/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());
}

function formatKnowledgeLogEntry(title, lines = []) {
  const timestamp = new Intl.DateTimeFormat(undefined, {
    hour: "numeric",
    minute: "2-digit",
    second: "2-digit",
  }).format(new Date());
  return [`[${timestamp}] ${title}`, ...lines].join("\n");
}

function summarizeKnowledgeActionPayload(payload) {
  return Object.entries(payload || {})
    .filter(([key, value]) => !["status", "message"].includes(key) && value != null)
    .map(([key, value]) => {
      const formattedValue =
        typeof value === "object" ? JSON.stringify(value) : String(value);
      return `${titleCaseLabel(key)}: ${formattedValue}`;
    });
}

function knowledgeActionLabel(action) {
  return action === "generate" ? "KB document generation" : "RAG index build";
}

function latestJobForDataset(jobs, datasetKind) {
  const timestampForJob = (job) => {
    const raw = job.finished_at || job.created_at || "";
    const parsed = raw ? Date.parse(raw) : Number.NaN;
    return Number.isNaN(parsed) ? 0 : parsed;
  };

  return (
    [...jobs]
      .filter((job) => job.dataset_kind === datasetKind)
      .sort((left, right) => timestampForJob(right) - timestampForJob(left))[0] ||
    null
  );
}

function formatJobTime(timestamp) {
  if (!timestamp) {
    return "";
  }
  const parsed = new Date(timestamp);
  if (Number.isNaN(parsed.getTime())) {
    return timestamp;
  }
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(parsed);
}

function lastStageLabel(job) {
  const stages = Array.isArray(job.stages) ? job.stages : [];
  if (job.current_stage_name) {
    return stageSummaryLabel(job.current_stage_name, job.dataset_kind);
  }
  if (stages.length) {
    const ordered = [...stages].sort((left, right) => left.stage_order - right.stage_order);
    return stageSummaryLabel(ordered[ordered.length - 1].stage_name, job.dataset_kind);
  }
  return "No stages yet";
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
  currentMode,
  deletingJobId,
  cancellingJobId,
}) {
  if (!jobs.length) {
    return <div className="muted">No pipeline jobs started yet.</div>;
  }

  const groupedJobs = [
    {
      key: "custom",
      title: "Custom Pipeline Runs",
      items: jobs.filter((job) => job.mode === "custom"),
    },
    {
      key: "full",
      title: "Full Pipeline Runs",
      items: jobs.filter((job) => job.mode !== "custom"),
    },
  ]
    .map((group) => ({
      ...group,
      items: [...group.items].sort((left, right) => {
        if (left.job_id === selectedJobId && right.job_id !== selectedJobId) {
          return -1;
        }
        if (right.job_id === selectedJobId && left.job_id !== selectedJobId) {
          return 1;
        }
        return 0;
      }),
    }))
    .filter((group) => group.items.length)
    .sort((left, right) => {
      const activeGroupKey = currentMode === "custom" ? "custom" : "full";
      if (left.key === activeGroupKey && right.key !== activeGroupKey) {
        return -1;
      }
      if (right.key === activeGroupKey && left.key !== activeGroupKey) {
        return 1;
      }
      return 0;
    });

  return (
    <div className="job-list">
      {groupedJobs.map((group) => (
        <div
          className={`job-group ${
            group.key === (currentMode === "custom" ? "custom" : "full")
              ? "active"
              : ""
          }`}
          key={group.key}
        >
          <div className="job-group-title">{group.title}</div>
          <div className="job-list-header">
            <div className="job-list-header-main">
              <span>Status</span>
              <span>Kind</span>
              <span>Run ID</span>
              <span>Last Stage</span>
            </div>
            <span>Action</span>
          </div>
          {group.items.map((job) => (
            (() => {
              const stages = Array.isArray(job.stages) ? job.stages : [];
              const completedStageCount = stages.filter(
                (stage) => stage.status === "completed",
              ).length;
              const isExpanded = selectedJobId === job.job_id;
              return (
                <div
                  key={job.job_id}
                  className={`job-item ${
                    isExpanded ? "selected" : ""
                  }`}
                >
                  <button
                    className="job-select"
                    onClick={() => onSelect(job)}
                    type="button"
                    aria-expanded={isExpanded}
                  >
                    <div className="job-primary-row">
                      <span className={`status-chip ${job.status}`}>{job.status}</span>
                      <span className="job-kind">{job.dataset_kind}</span>
                      <span className="job-run-id">{shortRunId(job.job_id)}</span>
                      <span className="job-last-stage">
                        <span>{lastStageLabel(job)}</span>
                        <span className="job-toggle-indicator" aria-hidden="true">
                          {isExpanded ? "▴" : "▾"}
                        </span>
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
                      data-tooltip="Cancel"
                    >
                      {cancellingJobId === job.job_id || job.status === "cancelling"
                        ? "…"
                        : (
                          <svg
                            aria-hidden="true"
                            className="job-cancel-icon"
                            viewBox="0 0 24 24"
                            fill="none"
                            stroke="currentColor"
                            strokeWidth="1.9"
                            strokeLinecap="round"
                            strokeLinejoin="round"
                          >
                            <circle cx="12" cy="12" r="8.5" />
                            <path d="M9 9l6 6" />
                            <path d="M15 9l-6 6" />
                          </svg>
                        )}
                    </button>
                  ) : (
                    <button
                      className="job-delete"
                      type="button"
                      onClick={() => onDelete(job.job_id)}
                      disabled={deletingJobId === job.job_id}
                      aria-label={`Delete job ${job.job_id}`}
                      data-tooltip="Delete"
                    >
                      {deletingJobId === job.job_id ? (
                        "…"
                      ) : (
                        <svg
                          aria-hidden="true"
                          className="job-delete-icon"
                          viewBox="0 0 24 24"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="1.9"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        >
                          <path d="M3 6h18" />
                          <path d="M8 6V4.8C8 3.81 8.81 3 9.8 3h4.4C15.19 3 16 3.81 16 4.8V6" />
                          <path d="M6.5 6l.9 12.2A2 2 0 0 0 9.39 20h5.22a2 2 0 0 0 1.99-1.8L17.5 6" />
                          <path d="M10 10.25v6.5" />
                          <path d="M14 10.25v6.5" />
                        </svg>
                      )}
                    </button>
                  )}
                  {isExpanded ? (
                    <div className="job-details">
                      <div className="job-secondary-row">
                        <span className="job-stage-summary">
                          {completedStageCount}/{stages.length} stages
                        </span>
                      </div>
                      <div className="job-stage-list" role="list" aria-label={`Stages for ${job.job_id}`}>
                        {stages.length ? (
                          stages.map((stage) => (
                            <div
                              key={stage.stage_id}
                              className="job-stage-row"
                              role="listitem"
                            >
                              <span className="job-stage-name">
                                {humanizeStageName(stage.stage_name)}
                              </span>
                              <span
                                className={`job-stage-indicator ${
                                  stage.status === "completed" ? "done" : "not-done"
                                }`}
                                title={stage.status}
                                aria-hidden="true"
                              >
                                {stage.status === "completed" ? "✓" : "✕"}
                              </span>
                            </div>
                          ))
                        ) : (
                          <div className="muted">No stages recorded yet.</div>
                        )}
                      </div>
                    </div>
                  ) : null}
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
  const [activeCustomRunByDataset, setActiveCustomRunByDataset] = useState({});
  const [deletingJobId, setDeletingJobId] = useState(null);
  const [cancellingJobId, setCancellingJobId] = useState(null);
  const [loadingSummary, setLoadingSummary] = useState(false);
  const [loadingArtifacts, setLoadingArtifacts] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [assetVersion, setAssetVersion] = useState("");
  const [scoringMenuOpen, setScoringMenuOpen] = useState(false);
  const [activeSectionId, setActiveSectionId] = useState("run-pipeline");
  const [knowledgeQuery, setKnowledgeQuery] = useState("");
  const [knowledgeResults, setKnowledgeResults] = useState([]);
  const [knowledgeContext, setKnowledgeContext] = useState("");
  const [knowledgeLoading, setKnowledgeLoading] = useState(false);
  const [knowledgeError, setKnowledgeError] = useState("");
  const [knowledgeActionLoading, setKnowledgeActionLoading] = useState("");
  const [knowledgeActionMessage, setKnowledgeActionMessage] = useState("");
  const [knowledgeLog, setKnowledgeLog] = useState(KNOWLEDGE_LOG_EMPTY_MESSAGE);
  const [knowledgeDiagnosis, setKnowledgeDiagnosis] = useState(null);
  const [knowledgeTaskStatus, setKnowledgeTaskStatus] = useState({
    generate: { status: "idle" },
    index: { status: "idle" },
  });
  const lastCompletedJobRef = useRef("");
  const previousModeRef = useRef("full");
  const scoringMenuRef = useRef(null);
  const knowledgeTaskStatusRef = useRef({
    generate: { status: "idle" },
    index: { status: "idle" },
  });
  const [runForm, setRunForm] = useState({
    mode: "full",
    stages: STAGE_OPTIONS.snapshot,
    fast_mode: false,
    models: ["logistic", "rf"],
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

  useEffect(() => {
    refreshKnowledgeStatus();
  }, []);

  useEffect(() => {
    const hasRunningKnowledgeTask = Object.values(knowledgeTaskStatus).some(
      (task) => task?.status === "running",
    );
    if (!hasRunningKnowledgeTask) {
      return undefined;
    }
    const timer = window.setInterval(() => {
      refreshKnowledgeStatus();
    }, 2000);
    return () => window.clearInterval(timer);
  }, [knowledgeTaskStatus]);

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

  function appendKnowledgeLog(title, lines = []) {
    setKnowledgeLog((current) => {
      const entry = formatKnowledgeLogEntry(title, lines);
      return current === KNOWLEDGE_LOG_EMPTY_MESSAGE
        ? entry
        : `${current}\n\n${entry}`;
    });
  }

  function syncKnowledgeTaskStatus(data) {
    const nextTasks = data?.tasks || {};
    setKnowledgeTaskStatus((current) => ({
      ...current,
      ...nextTasks,
    }));

    for (const action of ["generate", "index"]) {
      const previousStatus = knowledgeTaskStatusRef.current[action]?.status;
      const nextTask = nextTasks[action];
      if (!nextTask) {
        continue;
      }
      const nextStatus = nextTask.status;

      if (previousStatus !== nextStatus) {
        if (nextStatus === "completed") {
          appendKnowledgeLog(`Finished ${knowledgeActionLabel(action)}`, [
            nextTask.message || "Knowledge-base action completed.",
            ...summarizeKnowledgeActionPayload(nextTask.detail || {}),
          ]);
          setKnowledgeActionMessage(nextTask.message || "Knowledge-base action completed.");
          if (knowledgeActionLoading === action) {
            setKnowledgeActionLoading("");
          }
        } else if (nextStatus === "failed") {
          appendKnowledgeLog(`Failed ${knowledgeActionLabel(action)}`, [
            nextTask.error || nextTask.message || "Knowledge-base action failed.",
          ]);
          setKnowledgeError(
            nextTask.error || nextTask.message || "Knowledge-base action failed.",
          );
          if (knowledgeActionLoading === action) {
            setKnowledgeActionLoading("");
          }
        }
      }
    }

    knowledgeTaskStatusRef.current = {
      generate: nextTasks.generate || knowledgeTaskStatusRef.current.generate,
      index: nextTasks.index || knowledgeTaskStatusRef.current.index,
    };
  }

  async function refreshKnowledgeStatus(jobId = null) {
    try {
      const [statusData, diagnoseData] = await Promise.all([
        fetchJson("/api/knowledge-base/status"),
        fetchJson(withOptionalJob("/api/rag/diagnose", jobId)),
      ]);
      syncKnowledgeTaskStatus(statusData);
      setKnowledgeDiagnosis(diagnoseData);
    } catch {
      // Keep the knowledge-base actions usable even if status polling fails transiently.
    }
  }

  async function searchKnowledgeBase(event) {
    event.preventDefault();
    const trimmedQuery = knowledgeQuery.trim();
    if (!trimmedQuery) {
      setKnowledgeResults([]);
      setKnowledgeContext("");
      setKnowledgeError("");
      appendKnowledgeLog("Search skipped", [
        "Enter a query before searching the knowledge base.",
      ]);
      return;
    }

    if (!knowledgeTargetJobId) {
      setKnowledgeError("Select a run first so search can use its run-scoped knowledge-base index.");
      appendKnowledgeLog("Search skipped", [
        "Select a run before searching the knowledge base.",
      ]);
      return;
    }

    setKnowledgeLoading(true);
    setKnowledgeError("");
    appendKnowledgeLog(`Searching knowledge base for "${trimmedQuery}"`, [
      "Retrieving the top 5 semantic matches.",
    ]);
    try {
      const data = await fetchJson(withOptionalJob("/api/rag/search", knowledgeTargetJobId), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          query: trimmedQuery,
          top_k: 5,
        }),
      });
      setKnowledgeResults(data.matches || []);
      setKnowledgeContext(data.grounded_context || "");
      const topMatches = (data.matches || [])
        .slice(0, 3)
        .map(
          (match, index) =>
            `${index + 1}. ${match.metadata?.title || "Untitled document"} (${match.metadata?.source_path || "unknown source"})`,
        );
      appendKnowledgeLog(`Search completed for "${trimmedQuery}"`, [
        `Matches: ${(data.matches || []).length}`,
        ...topMatches,
      ]);
    } catch (err) {
      setKnowledgeError(err.message);
      setKnowledgeResults([]);
      setKnowledgeContext("");
      appendKnowledgeLog(`Search failed for "${trimmedQuery}"`, [err.message]);
    } finally {
      setKnowledgeLoading(false);
    }
  }

  async function runKnowledgeAction(action) {
    if (!knowledgeTargetJobId) {
      setKnowledgeError("Select a run first so knowledge-base actions can use that run's data.");
      appendKnowledgeLog(`Skipped ${knowledgeActionLabel(action)}`, [
        "Select a run before starting knowledge-base actions.",
      ]);
      return;
    }
    setKnowledgeActionLoading(action);
    setKnowledgeError("");
    setKnowledgeActionMessage("");
    const actionLabel = knowledgeActionLabel(action);
    appendKnowledgeLog(`Started ${actionLabel}`, [
      action === "generate"
        ? "Creating synthetic incidents, runbooks, and postmortems."
        : "Embedding knowledge-base documents into the local vector index.",
    ]);
    try {
      const endpoint =
        action === "generate"
          ? "/api/knowledge-base/generate"
          : "/api/rag/index";
      const data = await fetchJson(withOptionalJob(endpoint, knowledgeTargetJobId), {
        method: "POST",
      });
      if (data.task) {
        syncKnowledgeTaskStatus({ tasks: { [action]: data.task } });
      }
      setKnowledgeActionMessage(data.message || `${actionLabel} started.`);
      await refreshKnowledgeStatus(knowledgeTargetJobId);
    } catch (err) {
      setKnowledgeError(err.message);
      appendKnowledgeLog(`Failed ${actionLabel}`, [err.message]);
      setKnowledgeActionLoading("");
    } finally {
      const anyRunning = ["generate", "index"].some(
        (key) => knowledgeTaskStatusRef.current[key]?.status === "running",
      );
      if (!anyRunning) {
        setKnowledgeActionLoading("");
      }
    }
  }

  async function deleteJob(jobId) {
    setError("");
    setDeletingJobId(jobId);
    try {
      const deletedJob = jobs.find((job) => job.job_id === jobId) || null;
      await fetchJson(`/api/pipeline/jobs/${jobId}`, { method: "DELETE" });
      if (selectedJobId === jobId) {
        setSelectedJobId(null);
        setSelectedJobLog("");
      }
      if (deletedJob?.mode === "custom") {
        setActiveCustomRunByDataset((current) => {
          if (current[deletedJob.dataset_kind] !== jobId) {
            return current;
          }
          return {
            ...current,
            [deletedJob.dataset_kind]: null,
          };
        });
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
    const latestJob = latestJobForDataset(jobs, kind);
    if (latestJob?.mode === "custom") {
      setActiveCustomRunByDataset((current) => ({
        ...current,
        [kind]: latestJob.job_id,
      }));
    }
    setRunForm((current) => ({
      ...current,
      mode: latestJob?.mode === "custom" ? "custom" : "full",
      stages:
        latestJob?.mode === "full"
          ? STAGE_OPTIONS[kind]
          : current.stages,
      cv: kind === "temporal" && !current.cv ? "3" : current.cv,
    }));
    setSelectedJobId(latestJob?.job_id || null);
    setSelectedJobLog("");
    setDatasetKind(kind);
  }

  function selectJob(job) {
    if (selectedJobId === job.job_id) {
      if (job.mode === "custom") {
        setActiveCustomRunByDataset((current) => ({
          ...current,
          [job.dataset_kind]: null,
        }));
        setForceNewCustomRun(true);
      } else {
        setForceNewCustomRun(false);
      }
      setSelectedJobId(null);
      setSelectedJobLog("");
      refreshDashboard(datasetKind);
      return;
    }
    setForceNewCustomRun(false);
    if (job.mode === "custom") {
      setActiveCustomRunByDataset((current) => ({
        ...current,
        [job.dataset_kind]: job.job_id,
      }));
    }
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
        models: runForm.models.length ? runForm.models : null,
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
      if (nextMode === "custom") {
        setActiveCustomRunByDataset((current) => ({
          ...current,
          [datasetKind]: job.job_id,
        }));
        setForceNewCustomRun(false);
      }
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
  const evaluationVisuals = useMemo(() => {
    const plotFiles = artifactEntries.plots_dir || [];
    return plotFiles.filter((item) => isVisualArtifact(item.path)).map((item) => ({
      path: item.path,
      title: item.path
        .split("/")
        .slice(-1)[0]
        .replace(/_/g, " ")
        .replace(".png", ""),
    }));
  }, [artifactEntries]);
  const explainabilityVisuals = useMemo(() => {
    const explainFiles = artifactEntries.explain_dir || [];
    return explainFiles.filter((item) => isVisualArtifact(item.path)).map((item) => ({
      path: item.path,
      title: item.path
        .split("/")
        .slice(-1)[0]
        .replace(/_/g, " ")
        .replace(".png", ""),
    }));
  }, [artifactEntries]);
  const featuredVisual = evaluationVisuals[0] || explainabilityVisuals[0] || null;

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
    const rememberedJobId = activeCustomRunByDataset[datasetKind];
    if (!rememberedJobId) {
      return null;
    }
    const rememberedJob =
      jobs.find(
        (job) =>
          job.job_id === rememberedJobId &&
          job.dataset_kind === datasetKind &&
          job.mode === "custom",
      ) || null;
    if (!rememberedJob) {
      return null;
    }
    const rememberedCompletedStages = completedStageNamesForJobs(
      jobs,
      datasetKind,
      rememberedJob,
    );
    const rememberedRunComplete =
      rememberedCompletedStages.size === STAGE_OPTIONS[datasetKind].length;
    if (
      rememberedRunComplete &&
      !["queued", "running", "cancelling"].includes(rememberedJob.status)
    ) {
      return null;
    }
    return rememberedJob;
  }, [
    activeCustomRunByDataset,
    datasetKind,
    forceNewCustomRun,
    jobs,
    runForm.mode,
    selectedJob,
  ]);
  const customRunLocked = Boolean(
    selectedCustomRun &&
      ["queued", "running", "cancelling"].includes(selectedCustomRun.status),
  );
  const sourceJob =
    selectedCustomRun &&
    ["completed", "failed", "cancelled"].includes(selectedCustomRun.status)
      ? selectedCustomRun
      : null;
  const selectedResultsJob =
    selectedJob && selectedJob.dataset_kind === datasetKind
      ? selectedJob
      : selectedCustomRun;
  const completedStages = useMemo(() => {
    if (runForm.mode === "custom" && !selectedCustomRun) {
      return new Set();
    }
    return completedStageNamesForJobs(jobs, datasetKind, selectedCustomRun);
  }, [jobs, datasetKind, runForm.mode, selectedCustomRun]);
  const resultCompletedStages = useMemo(() => {
    if (!selectedResultsJob) {
      return new Set();
    }
    return completedStageNamesForJobs(jobs, datasetKind, selectedResultsJob);
  }, [datasetKind, jobs, selectedResultsJob]);
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
  const knowledgeIndexReady = Boolean(knowledgeDiagnosis?.index_exists);
  const knowledgeGenerateRunning =
    knowledgeActionLoading === "generate" ||
    knowledgeTaskStatus.generate?.status === "running";
  const knowledgeIndexRunning =
    knowledgeActionLoading === "index" ||
    knowledgeTaskStatus.index?.status === "running";
  const knowledgeTargetJob =
    selectedResultsJob && selectedResultsJob.dataset_kind === datasetKind
      ? selectedResultsJob
      : null;
  const knowledgeTargetJobId = knowledgeTargetJob?.job_id || null;
  const knowledgeControlsEnabled = Boolean(knowledgeTargetJobId);
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
    !customRunLocked &&
    (runForm.mode === "full" || runForm.stages.includes(trainStageName));
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
  const summaryItems = useMemo(() => {
    const items = [
      {
        title: "Tracked Jobs",
        value: jobs.length,
        subtitle: jobs.length ? "Persisted run history" : "No runs recorded yet",
      },
    ];

    if (bestModel) {
      items.unshift({
        title: "Best F1 Macro",
        value: formatScore(bestModel.f1_macro),
        subtitle: bestModel.model_name,
      });
    }

    if ((artifactEntries.plots_dir?.length || 0) + (artifactEntries.reports_dir?.length || 0) > 0) {
      items.push({
        title: "Plots & Reports",
        value:
          (artifactEntries.plots_dir?.length || 0) +
          (artifactEntries.reports_dir?.length || 0),
        subtitle: `${datasetKind} evaluation visuals and reports`,
      });
    }

    if (summary?.artifacts?.models_dir) {
      items.push({
        title: "Models Dir",
        value: (
          <a className="stat-link" href="#artifact-models">
            {summary.artifacts.models_dir.split("/").slice(-1)[0]}
          </a>
        ),
      });
    }

    if (summary?.artifacts?.best_model) {
      items.push({
        title: "Best Model",
        value: (
          <a
            className="stat-link"
            href={fileUrl(summary.artifacts.best_model)}
            target="_blank"
            rel="noreferrer"
          >
            {summary.artifacts.best_model.split("/").slice(-1)[0]}
          </a>
        ),
      });
    }

    if (summary?.evaluation_metrics?.models?.length) {
      items.push({
        title: "Eval Models",
        value: summary.evaluation_metrics.models.length,
        subtitle: datasetKind,
      });
    }

    if (summary?.artifacts?.evaluation_metrics) {
      items.push({
        title: "Metrics File",
        value: (
          <a
            className="stat-link"
            href={fileUrl(summary.artifacts.evaluation_metrics)}
            target="_blank"
            rel="noreferrer"
          >
            {summary.artifacts.evaluation_metrics.split("/").slice(-1)[0]}
          </a>
        ),
      });
    }

    return items;
  }, [artifactEntries, bestModel, datasetKind, jobs.length, summary]);
  const resultStageNote = useMemo(() => {
    if (!selectedResultsJob || selectedResultsJob.mode !== "custom") {
      return null;
    }

    if (datasetKind === "snapshot") {
      if (resultCompletedStages.has("generate_snapshot") && !resultCompletedStages.has("train_snapshot")) {
        return "Generated datasets are ready. Run train snapshot to create model artifacts.";
      }
      if (resultCompletedStages.has("train_snapshot") && !resultCompletedStages.has("evaluate_snapshot")) {
        return "Training artifacts are ready. Run evaluate snapshot to see model metrics and plots.";
      }
      if (resultCompletedStages.has("evaluate_snapshot") && !resultCompletedStages.has("explain_snapshot")) {
        return "Evaluation results are ready. Run explain snapshot to generate explainability outputs.";
      }
    }

    if (datasetKind === "temporal") {
      if (resultCompletedStages.has("generate_sequence") && !resultCompletedStages.has("build_temporal_features")) {
        return "Sequence data is ready. Run build temporal features to prepare training data.";
      }
      if (resultCompletedStages.has("build_temporal_features") && !resultCompletedStages.has("train_temporal")) {
        return "Temporal features are ready. Run train temporal to create model artifacts.";
      }
      if (resultCompletedStages.has("train_temporal") && !resultCompletedStages.has("evaluate_temporal")) {
        return "Training artifacts are ready. Run evaluate temporal to see model metrics and plots.";
      }
      if (resultCompletedStages.has("evaluate_temporal") && !resultCompletedStages.has("explain_temporal")) {
        return "Evaluation results are ready. Run explain temporal to generate explainability outputs.";
      }
    }

    return null;
  }, [datasetKind, resultCompletedStages, selectedResultsJob]);

  useEffect(() => {
    refreshKnowledgeStatus(knowledgeTargetJobId);
  }, [knowledgeTargetJobId]);
  const showModelEvaluation = modelRows.length > 0;
  const showFeaturedResult = Boolean(featuredVisual);

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
        ? current.stages.filter((stageName) => stageName === lockedPreviewStage)
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
  }, [customRunLocked, lockedPreviewStage, nextRunnableStage, runForm.mode, stageAvailability]);

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

  useEffect(() => {
    function handlePointerDown(event) {
      if (!scoringMenuRef.current?.contains(event.target)) {
        setScoringMenuOpen(false);
      }
    }

    document.addEventListener("mousedown", handlePointerDown);
    return () => document.removeEventListener("mousedown", handlePointerDown);
  }, []);

  useEffect(() => {
    if (!tuningEnabled) {
      setScoringMenuOpen(false);
    }
  }, [tuningEnabled]);

  useEffect(() => {
    const sections = SECTION_LINKS.map(({ id }) => document.getElementById(id)).filter(Boolean);
    if (!sections.length) {
      return undefined;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        const visibleEntries = entries
          .filter((entry) => entry.isIntersecting)
          .sort((left, right) => right.intersectionRatio - left.intersectionRatio);

        if (visibleEntries.length) {
          setActiveSectionId(visibleEntries[0].target.id);
        }
      },
      {
        rootMargin: "-20% 0px -55% 0px",
        threshold: [0.15, 0.3, 0.5, 0.75],
      },
    );

    sections.forEach((section) => observer.observe(section));
    return () => observer.disconnect();
  }, []);

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

      <div className="page-layout page-layout-with-nav">
      <aside className="quick-nav" aria-label="Quick navigation">
        <div className="quick-nav-title">Quick Links</div>
        {SECTION_LINKS.map((section) => (
          <a
            key={section.id}
            href={`#${section.id}`}
            className={activeSectionId === section.id ? "active" : ""}
          >
            {section.label}
          </a>
        ))}
      </aside>

      <div className="page-content">
      <section className="grid" id="run-pipeline">
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
                          setActiveCustomRunByDataset((current) => ({
                            ...current,
                            [datasetKind]: null,
                          }));
                          setForceNewCustomRun(true);
                        }}
                      >
                        Start new staged run
                      </button>
                    ) : null}
                  </div>
                  <div className="run-form-fields">
                    <div className="tuning-inline-grid">
                      <div className="compact-field compact-field-wide compact-model-field">
                        <span className="form-label">Models</span>
                        <div className="model-options">
                          {MODEL_OPTIONS.map((modelOption) => (
                            <label
                              key={modelOption.value}
                              className="model-option"
                            >
                              <input
                                type="checkbox"
                                checked={runForm.models.includes(modelOption.value)}
                                disabled={!tuningEnabled}
                                onChange={(event) =>
                                  setRunForm((current) => ({
                                    ...current,
                                    models: event.target.checked
                                      ? [...current.models, modelOption.value]
                                      : current.models.filter(
                                          (model) => model !== modelOption.value,
                                        ),
                                  }))
                                }
                              />
                              <span>{modelOption.label}</span>
                            </label>
                          ))}
                        </div>
                      </div>
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
                        <div className="compact-field-stack">
                          <div
                            className={`custom-select ${scoringMenuOpen ? "open" : ""} ${
                              tuningEnabled ? "" : "disabled"
                            }`}
                            ref={scoringMenuRef}
                          >
                            <button
                              type="button"
                              className="custom-select-trigger"
                              disabled={!tuningEnabled}
                              aria-haspopup="listbox"
                              aria-expanded={scoringMenuOpen}
                              onClick={() => setScoringMenuOpen((current) => !current)}
                            >
                              <span>
                                {SCORING_OPTIONS.find(
                                  (option) => option.value === runForm.scoring,
                                )?.label || runForm.scoring}
                              </span>
                              <span className="custom-select-caret" aria-hidden="true">
                                ▾
                              </span>
                            </button>
                            {scoringMenuOpen ? (
                              <div className="custom-select-menu" role="listbox" aria-label="Scoring">
                                {SCORING_OPTIONS.map((option) => (
                                  <button
                                    key={option.value}
                                    type="button"
                                    role="option"
                                    className={`custom-select-option ${
                                      option.value === runForm.scoring ? "selected" : ""
                                    }`}
                                    aria-selected={option.value === runForm.scoring}
                                    onClick={() => {
                                      setRunForm((current) => ({
                                        ...current,
                                        scoring: option.value,
                                      }));
                                      setScoringMenuOpen(false);
                                    }}
                                  >
                                    {option.label}
                                  </button>
                                ))}
                              </div>
                            ) : null}
                          </div>
                          <span className="field-helper-text">
                            Used for cross-validation tuning only.
                          </span>
                        </div>
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
                            customRunLocked ||
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
          <div className="form-subsection run-activity-subsection">
            <div className="run-activity-grid">
              <div className="jobs-subsection" id="recent-jobs">
                <div className="section-title section-title-compact">
                  Recent Jobs
                </div>
                <JobList
                  jobs={visibleJobs}
                  onSelect={selectJob}
                  onDelete={deleteJob}
                  onCancel={cancelJob}
                  selectedJobId={selectedJobId}
                  currentMode={runForm.mode}
                  deletingJobId={deletingJobId}
                  cancellingJobId={cancellingJobId}
                />
              </div>
              <div className="log-subsection" id="selected-job-log">
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
          </div>
        </div>
      </section>

      <section className="grid" id="latest-run-results">
        <div className="card summary-card">
          {loadingSummary ? (
            <div className="muted">Loading summary...</div>
          ) : (
            <div>
              <div className="card-band-top">
                <div className="section-title">Latest Run Results</div>
                <SummaryList items={summaryItems} />
                {resultStageNote ? (
                  <div className="stage-option-hint">{resultStageNote}</div>
                ) : null}
              </div>
              {showModelEvaluation ? (
                <div className="summary-feature evaluation-feature">
                  <div className="section-title section-title-compact">
                    Model Evaluation
                  </div>
                  <MetricTable rows={modelRows} />
                </div>
              ) : null}
              {showFeaturedResult ? (
                <div className="summary-feature featured-result-feature">
                  <div className="section-title section-title-compact">
                    Featured Result
                  </div>
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
                </div>
              ) : null}
            </div>
          )}
        </div>
      </section>

      <section className="grid" id="knowledge-search">
        <div className="card">
          <div className="section-title">Knowledge Base</div>
          <div className="knowledge-actions-panel">
            <div className="knowledge-actions-copy">
              <div className="artifact-section-title">Prepare Knowledge Artifacts</div>
              <div className="muted">
                Generate synthetic incident markdown docs, then build the local vector index used for retrieval.
              </div>
              <div className="muted">
                {knowledgeTargetJobId
                  ? `Using selected run ${shortRunId(knowledgeTargetJobId)}.`
                  : "Select a run to use its run-scoped data and artifacts."}
              </div>
            </div>
            {knowledgeDiagnosis ? (
              <div className="knowledge-diagnose-panel">
                <div className="knowledge-diagnose-header">
                  <div className="artifact-section-title">Index Diagnostics</div>
                  <span className={`status-chip ${knowledgeIndexReady ? "completed" : "failed"}`}>
                    {knowledgeIndexReady ? "ready" : "not built"}
                  </span>
                </div>
                <div className="knowledge-diagnose-grid">
                  <div className="knowledge-diagnose-item">
                    <span className="knowledge-diagnose-label">Documents</span>
                    <span className="knowledge-diagnose-value">
                      {knowledgeDiagnosis.rag?.n_documents ?? 0}
                    </span>
                  </div>
                  <div className="knowledge-diagnose-item">
                    <span className="knowledge-diagnose-label">Run</span>
                    <span className="knowledge-diagnose-value">
                      {knowledgeDiagnosis.rag?.job_id
                        ? shortRunId(knowledgeDiagnosis.rag.job_id)
                        : "shared"}
                    </span>
                  </div>
                  <div className="knowledge-diagnose-item">
                    <span className="knowledge-diagnose-label">Collection</span>
                    <span className="knowledge-diagnose-value">
                      {knowledgeDiagnosis.rag?.collection_name || "unknown"}
                    </span>
                  </div>
                  <div className="knowledge-diagnose-item">
                    <span className="knowledge-diagnose-label">Embedding Model</span>
                    <span className="knowledge-diagnose-value">
                      {knowledgeDiagnosis.rag?.model_name || "unknown"}
                    </span>
                  </div>
                  <div className="knowledge-diagnose-item">
                    <span className="knowledge-diagnose-label">Manifest</span>
                    <span className="knowledge-diagnose-value">
                      {knowledgeDiagnosis.rag?.manifest_exists ? "present" : "missing"}
                    </span>
                  </div>
                </div>
                <div className="knowledge-diagnose-paths">
                  {knowledgeDiagnosis.rag?.knowledge_base_dir ? (
                    <span>
                      <strong>KB:</strong> {knowledgeDiagnosis.rag.knowledge_base_dir}
                    </span>
                  ) : null}
                  <span>
                    <strong>Chroma:</strong> {knowledgeDiagnosis.rag?.chroma_dir || "n/a"}
                  </span>
                  <span>
                    <strong>Manifest:</strong> {knowledgeDiagnosis.rag?.manifest_path || "n/a"}
                  </span>
                </div>
              </div>
            ) : null}
            <div className="knowledge-actions">
              <button
                className="secondary-button"
                type="button"
                disabled={!knowledgeControlsEnabled || knowledgeGenerateRunning}
                onClick={() => runKnowledgeAction("generate")}
              >
                {knowledgeGenerateRunning ? "Generating..." : "Generate KB Docs"}
              </button>
              <button
                className="secondary-button"
                type="button"
                disabled={!knowledgeControlsEnabled || knowledgeIndexRunning}
                onClick={() => runKnowledgeAction("index")}
              >
                {knowledgeIndexRunning ? "Building..." : "Build RAG Index"}
              </button>
            </div>
          </div>
          {knowledgeActionMessage ? (
            <div className="knowledge-action-message">{knowledgeActionMessage}</div>
          ) : null}
          <div className="artifact-section">
            <div className="artifact-section-title">Search the Knowledge Base</div>
          <form className="knowledge-search-form" onSubmit={searchKnowledgeBase}>
            <label className="knowledge-search-field">
              <span className="form-label">Search Query</span>
              <input
                value={knowledgeQuery}
                onChange={(event) => setKnowledgeQuery(event.target.value)}
                placeholder="Try: memory leak symptoms and OOM logs"
              />
            </label>
            <div className="knowledge-search-actions">
              <button
                className="primary-button knowledge-search-button"
                type="submit"
                disabled={!knowledgeControlsEnabled || knowledgeLoading}
              >
                {knowledgeLoading ? "Searching..." : "Search Knowledge Base"}
              </button>
              <span className="field-helper-text">
                Uses the local RAG index built from incidents, runbooks, and postmortems.
              </span>
            </div>
          </form>
          </div>
          {knowledgeError ? (
            <div className="knowledge-search-empty">{knowledgeError}</div>
          ) : null}
          {!knowledgeError && !knowledgeResults.length && !knowledgeContext && !knowledgeLoading ? (
            <div className="knowledge-search-empty">
              Search the synthetic incident knowledge base to find similar incidents, relevant runbooks,
              and postmortem context.
            </div>
          ) : null}
          {knowledgeResults.length ? (
            <div className="knowledge-results">
              {knowledgeResults.map((match, index) => (
                <article className="knowledge-result-card" key={`${match.metadata?.source_path || "match"}-${index}`}>
                  <div className="knowledge-result-meta">
                    <span className="status-chip completed">
                      {titleCaseLabel(match.metadata?.doc_type || "document")}
                    </span>
                    <span className="knowledge-result-source">
                      {match.metadata?.source_path || "unknown source"}
                    </span>
                    {typeof match.distance === "number" ? (
                      <span className="knowledge-result-distance">
                        distance {match.distance.toFixed(3)}
                      </span>
                    ) : null}
                  </div>
                  <div className="knowledge-result-title">
                    {match.metadata?.title || "Untitled document"}
                  </div>
                  <pre className="knowledge-result-snippet">{match.document}</pre>
                </article>
              ))}
            </div>
          ) : null}
          {knowledgeContext ? (
            <div className="knowledge-context-block">
              <div className="artifact-section-title">Grounded Context</div>
              <pre className="knowledge-context">{knowledgeContext}</pre>
            </div>
          ) : null}
          <div className="artifact-section knowledge-log-subsection">
            <div className="artifact-section-title">Knowledge Base Log</div>
            <pre className="log-view knowledge-log-view">{knowledgeLog}</pre>
          </div>
        </div>
      </section>

      <section className="grid" id="visuals">
        <div className="card">
          <div className="section-title">Visuals</div>
          {evaluationVisuals.length ? (
            <div className="artifact-section">
              <div className="artifact-section-title">Evaluation</div>
              <VisualGallery
                visuals={evaluationVisuals.filter(
                  (visual) => visual.path !== featuredVisual?.path,
                )}
                assetVersion={assetVersion}
              />
            </div>
          ) : null}
          {explainabilityVisuals.length ? (
            <div className="artifact-section">
              <div className="artifact-section-title">Explainability</div>
              <VisualGallery
                visuals={explainabilityVisuals.filter(
                  (visual) => visual.path !== featuredVisual?.path,
                )}
                assetVersion={assetVersion}
              />
            </div>
          ) : null}
          {!evaluationVisuals.length && !explainabilityVisuals.length ? (
            <div className="empty-state">
              No visuals yet. Start a run to populate this gallery with
              evaluation plots and explainability outputs.
            </div>
          ) : null}
        </div>
      </section>

      <section className="grid" id="artifact-inventory">
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

      <footer className="app-footer">
        <div className="app-footer-copy">
          <div className="app-footer-title">Incident Intelligence</div>
          <p>
            Explainable incident root-cause modeling across snapshot and
            temporal workflows.
          </p>
          <div className="app-footer-meta">
            <span>Built by Swetha Chakravarthy</span>
            <span>© 2026 Incident Intelligence</span>
          </div>
        </div>
        <div className="app-footer-links">
          <a
            href="https://swethachakravarthy.wixsite.com/tech-adventures/blog"
            target="_blank"
            rel="noreferrer"
          >
            Blog
          </a>
          <span className="app-footer-separator" aria-hidden="true">
            /
          </span>
          <a
            href="https://github.com/swchak/incident-intelligence"
            target="_blank"
            rel="noreferrer"
          >
            Demo
          </a>
          <span className="app-footer-separator" aria-hidden="true">
            /
          </span>
          <a
            href="https://swethachakravarthy.wixsite.com/tech-adventures/contact"
            target="_blank"
            rel="noreferrer"
          >
            Contact
          </a>
          <span className="app-footer-separator" aria-hidden="true">
            /
          </span>
          <a
            href="https://github.com/swchak/incident-intelligence"
            target="_blank"
            rel="noreferrer"
          >
            GitHub
          </a>
        </div>
      </footer>
      </div>
      </div>
    </div>
  );
}
