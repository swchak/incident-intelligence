import { useEffect, useMemo, useRef, useState } from "react";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "";
const DATASET_KINDS = ["snapshot", "temporal"];

async function fetchJson(path, options) {
  const response = await fetch(`${API_BASE_URL}${path}`, options);
  if (!response.ok) {
    const body = await response.text();
    throw new Error(body || `Request failed: ${response.status}`);
  }
  return response.json();
}

function formatScore(value) {
  return typeof value === "number" ? value.toFixed(4) : "n/a";
}

function fileUrl(path, version = "") {
  const suffix = version ? `?v=${encodeURIComponent(version)}` : "";
  return `${API_BASE_URL}/api/files/${path}${suffix}`;
}

function isVisualArtifact(path) {
  return /\.(png|jpg|jpeg|svg|webp)$/i.test(path);
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

function JobList({ jobs, onSelect, onDelete, selectedJobId, deletingJobId }) {
  if (!jobs.length) {
    return <div className="muted">No pipeline jobs started yet.</div>;
  }

  return (
    <div className="job-list">
      <div className="job-list-header">
        <div className="job-list-header-main">
          <span>Status</span>
          <span>Kind</span>
          <span>Time</span>
        </div>
        <span>Action</span>
      </div>
      {jobs.map((job) => (
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
            <span className={`status-chip ${job.status}`}>{job.status}</span>
            <span className="job-kind">{job.dataset_kind}</span>
            <span className="job-meta">
              {job.finished_at ? `finished ${job.finished_at}` : job.created_at}
            </span>
          </button>
          <button
            className="job-delete"
            type="button"
            onClick={() => onDelete(job.job_id)}
            disabled={
              deletingJobId === job.job_id ||
              job.status === "queued" ||
              job.status === "running"
            }
            aria-label={`Delete job ${job.job_id}`}
            title={
              job.status === "queued" || job.status === "running"
                ? "Cannot delete a job while it is queued or running"
                : "Delete job"
            }
          >
            {deletingJobId === job.job_id ? "Deleting..." : "Delete Run"}
          </button>
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
  const [deletingJobId, setDeletingJobId] = useState(null);
  const [loadingSummary, setLoadingSummary] = useState(false);
  const [loadingArtifacts, setLoadingArtifacts] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [assetVersion, setAssetVersion] = useState("");
  const lastCompletedJobRef = useRef("");
  const [runForm, setRunForm] = useState({
    fast_mode: false,
    models: "logistic,rf",
    cv: datasetKind === "temporal" ? "3" : "",
    n_jobs: "1",
    verbose: "0",
    scoring: "f1_macro",
  });

  useEffect(() => {
    refreshDashboard(datasetKind);
  }, [datasetKind]);

  useEffect(() => {
    refreshJobs();
    const timer = window.setInterval(refreshJobs, 3000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    if (!selectedJobId && jobs.length) {
      setSelectedJobId(jobs[0].job_id);
    }
  }, [jobs, selectedJobId]);

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

  async function refreshDashboard(kind) {
    setError("");
    setLoadingSummary(true);
    setLoadingArtifacts(true);
    try {
      const [summaryData, artifactData] = await Promise.all([
        fetchJson(`/api/dashboard/summary/${kind}`),
        fetchJson(`/api/artifacts/${kind}`),
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
        setSelectedJobId(nextJobs[0]?.job_id ?? null);
        if (!nextJobs.length) {
          setSelectedJobLog("");
        }
      }
      await refreshJobs();
    } catch (err) {
      setError(err.message);
    } finally {
      setDeletingJobId(null);
    }
  }

  function selectJob(job) {
    setSelectedJobId(job.job_id);
    setSelectedJobLog("");
    loadJobLog(job.job_id);
    if (job.dataset_kind !== datasetKind) {
      setDatasetKind(job.dataset_kind);
      return;
    }
    if (job.status === "completed" || job.status === "failed") {
      refreshDashboard(job.dataset_kind);
    }
  }

  async function submitRun(event) {
    event.preventDefault();
    setSubmitting(true);
    setError("");
    try {
      const payload = {
        dataset_kind: datasetKind,
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
                  onClick={() => setDatasetKind(kind)}
                >
                  {kind}
                </button>
              ))}
            </div>
          </div>
        </div>
      </section>

      <section className="grid two-up">
        <div className="card form-card">
          <div className="run-ppl-col">
            <div className="section-kicker">Start Here</div>
            <div className="section-title">Run Pipeline</div>
            <form onSubmit={submitRun} className="run-form">
              <label className="form-row">
                <span className="form-label">Models</span>
                <input
                  value={runForm.models}
                  onChange={(event) =>
                    setRunForm((current) => ({
                      ...current,
                      models: event.target.value,
                    }))
                  }
                  placeholder="logistic,rf"
                />
              </label>
              <label className="form-row">
                <span className="form-label">CV</span>
                <input
                  value={runForm.cv}
                  onChange={(event) =>
                    setRunForm((current) => ({
                      ...current,
                      cv: event.target.value,
                    }))
                  }
                  placeholder="3"
                />
              </label>
              <label className="form-row">
                <span className="form-label">n_jobs</span>
                <input
                  value={runForm.n_jobs}
                  onChange={(event) =>
                    setRunForm((current) => ({
                      ...current,
                      n_jobs: event.target.value,
                    }))
                  }
                  placeholder="1"
                />
              </label>
              <label className="form-row">
                <span className="form-label">Verbose</span>
                <input
                  value={runForm.verbose}
                  onChange={(event) =>
                    setRunForm((current) => ({
                      ...current,
                      verbose: event.target.value,
                    }))
                  }
                  placeholder="0"
                />
              </label>
              <label className="form-row">
                <span className="form-label">Scoring</span>
                <input
                  value={runForm.scoring}
                  onChange={(event) =>
                    setRunForm((current) => ({
                      ...current,
                      scoring: event.target.value,
                    }))
                  }
                  placeholder="f1_macro"
                />
              </label>
              <label className="checkbox-row">
                <span className="form-label">Fast mode</span>
                <input
                  type="checkbox"
                  checked={runForm.fast_mode}
                  onChange={(event) =>
                    setRunForm((current) => ({
                      ...current,
                      fast_mode: event.target.checked,
                    }))
                  }
                />
              </label>
              <button
                className="primary-button"
                type="submit"
                disabled={submitting}
              >
                {submitting ? "Starting..." : `Run ${datasetKind} pipeline`}
              </button>
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
              selectedJobId={selectedJobId}
              deletingJobId={deletingJobId}
            />
          </div>
          <div className="form-subsection log-subsection">
            <div className="section-title section-title-compact">
              Selected Job Log
            </div>
            <pre className="log-view">
              {selectedJobLog ||
                "No log selected yet. Launch a pipeline run to watch logs stream here."}
            </pre>
          </div>
        </div>

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
