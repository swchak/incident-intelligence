import { useEffect, useMemo, useState } from "react";

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

function fileUrl(path) {
  return `${API_BASE_URL}/api/files/${path}`;
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

function NarrativeCard({ title, body }) {
  return (
    <div className="narrative-card">
      <div className="narrative-title">{title}</div>
      <div className="narrative-body">{body}</div>
    </div>
  );
}

function ArtifactSection({ title, items }) {
  return (
    <div className="card">
      <div className="section-title">{title}</div>
      {items.length === 0 ? (
        <div className="muted">No artifacts found yet.</div>
      ) : (
        <ul className="artifact-list">
          {items.slice(0, 12).map((item) => (
            <li key={item.path}>
              <code>{item.path}</code>
            </li>
          ))}
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

function JobList({ jobs, onSelect, selectedJobId }) {
  if (!jobs.length) {
    return <div className="muted">No pipeline jobs started yet.</div>;
  }

  return (
    <div className="job-list">
      {jobs.map((job) => (
        <button
          key={job.job_id}
          className={`job-item ${selectedJobId === job.job_id ? "selected" : ""}`}
          onClick={() => onSelect(job.job_id)}
          type="button"
        >
          <div className="job-header">
            <span className={`status-chip ${job.status}`}>{job.status}</span>
            <span>{job.dataset_kind}</span>
          </div>
          <div className="job-id">{job.job_id}</div>
          <div className="job-meta">{job.created_at}</div>
          {job.finished_at ? <div className="job-finished">finished {job.finished_at}</div> : null}
        </button>
      ))}
    </div>
  );
}

function VisualGallery({ visuals }) {
  if (!visuals.length) {
    return <div className="muted">No confusion matrices or feature importance plots found yet.</div>;
  }

  return (
    <div className="visual-grid">
      {visuals.map((visual) => (
        <figure className="visual-card" key={visual.path}>
          <img alt={visual.title} src={fileUrl(visual.path)} />
          <figcaption>
            <div className="visual-title">{visual.title}</div>
            <code>{visual.path}</code>
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
  const [loadingSummary, setLoadingSummary] = useState(false);
  const [loadingArtifacts, setLoadingArtifacts] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [runForm, setRunForm] = useState({
    fast_mode: false,
    models: "logistic,rf",
    cv: datasetKind === "temporal" ? "3" : "",
    n_jobs: "1",
    verbose: "0",
    scoring: "f1_macro"
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
    fetchJson(`/api/pipeline/jobs/${selectedJobId}/log`)
      .then((data) => setSelectedJobLog(data.log))
      .catch((err) => setError(err.message));
  }, [selectedJobId, jobs]);

  async function refreshDashboard(kind) {
    setError("");
    setLoadingSummary(true);
    setLoadingArtifacts(true);
    try {
      const [summaryData, artifactData] = await Promise.all([
        fetchJson(`/api/dashboard/summary/${kind}`),
        fetchJson(`/api/artifacts/${kind}`)
      ]);
      setSummary(summaryData);
      setArtifacts(artifactData);
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

  async function submitRun(event) {
    event.preventDefault();
    setSubmitting(true);
    setError("");
    try {
      const payload = {
        dataset_kind: datasetKind,
        fast_mode: runForm.fast_mode,
        models: runForm.models
          ? runForm.models.split(",").map((item) => item.trim()).filter(Boolean)
          : null,
        cv: runForm.cv ? Number(runForm.cv) : null,
        n_jobs: runForm.n_jobs ? Number(runForm.n_jobs) : null,
        verbose: runForm.verbose ? Number(runForm.verbose) : null,
        scoring: runForm.scoring || null
      };
      const job = await fetchJson("/api/pipeline/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
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
    return summary?.evaluation_metrics?.models?.map((model) => ({
      model_name: model.model_name,
      ...model.metrics
    })) || [];
  }, [summary]);

  const bestModel = useMemo(() => {
    if (!modelRows.length) {
      return null;
    }
    return [...modelRows].sort((left, right) => (right.f1_macro ?? -1) - (left.f1_macro ?? -1))[0];
  }, [modelRows]);

  const artifactEntries = artifacts?.artifacts || {};
  const visuals = useMemo(() => {
    const plotFiles = artifactEntries.plots_dir || [];
    return plotFiles
      .filter((item) => /confusion_matrix|feature_importance|model_comparison/.test(item.path))
      .slice(0, 6)
      .map((item) => ({
        path: item.path,
        title: item.path.split("/").slice(-1)[0].replace(/_/g, " ").replace(".png", "")
      }));
  }, [artifactEntries]);

  const headlineStats = [
    {
      title: "Best F1 Macro",
      value: bestModel ? formatScore(bestModel.f1_macro) : "n/a",
      subtitle: bestModel ? bestModel.model_name : "no evaluation yet"
    },
    {
      title: "Tracked Jobs",
      value: jobs.length,
      subtitle: jobs.length ? "persisted run history" : "no runs yet"
    },
    {
      title: "Artifacts",
      value: (artifactEntries.plots_dir?.length || 0) + (artifactEntries.reports_dir?.length || 0),
      subtitle: `${datasetKind} plots and reports`
    }
  ];

  return (
    <div className="app-shell">
      <header className="hero">
        <div className="hero-copy">
          <div className="eyebrow">Incident Intelligence</div>
          <h1>Root-cause modeling, from synthetic telemetry to explainable results.</h1>
          <p>
            This dashboard demonstrates a full ML workflow for incident root-cause analysis: generate data,
            train baseline models, compare snapshot versus temporal features, and inspect the artifacts those runs produce.
          </p>
          <div className="narrative-grid">
            <NarrativeCard
              title="Problem"
              body="Distinguish deployment, dependency, traffic, CPU, and memory failure modes from telemetry patterns."
            />
            <NarrativeCard
              title="Why Temporal"
              body="Many incidents are about shape over time, not just one static row of metrics."
            />
            <NarrativeCard
              title="What You Can Inspect"
              body="Evaluation tables, confusion matrices, feature importance plots, explainability outputs, and job logs."
            />
          </div>
        </div>
        <div className="hero-side">
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
          <div className="hero-stats">
            {headlineStats.map((item) => (
              <StatCard key={item.title} title={item.title} value={item.value} subtitle={item.subtitle} />
            ))}
          </div>
        </div>
      </header>

      {error ? <div className="error-banner">{error}</div> : null}

      <section className="grid two-up">
        <div className="card form-card">
          <div className="section-title">Run Pipeline</div>
          <form onSubmit={submitRun} className="run-form">
            <label>
              Models
              <input
                value={runForm.models}
                onChange={(event) => setRunForm((current) => ({ ...current, models: event.target.value }))}
                placeholder="logistic,rf"
              />
            </label>
            <label>
              CV
              <input
                value={runForm.cv}
                onChange={(event) => setRunForm((current) => ({ ...current, cv: event.target.value }))}
                placeholder="3"
              />
            </label>
            <label>
              n_jobs
              <input
                value={runForm.n_jobs}
                onChange={(event) => setRunForm((current) => ({ ...current, n_jobs: event.target.value }))}
                placeholder="1"
              />
            </label>
            <label>
              Verbose
              <input
                value={runForm.verbose}
                onChange={(event) => setRunForm((current) => ({ ...current, verbose: event.target.value }))}
                placeholder="0"
              />
            </label>
            <label>
              Scoring
              <input
                value={runForm.scoring}
                onChange={(event) => setRunForm((current) => ({ ...current, scoring: event.target.value }))}
                placeholder="f1_macro"
              />
            </label>
            <label className="checkbox-row">
              <input
                type="checkbox"
                checked={runForm.fast_mode}
                onChange={(event) => setRunForm((current) => ({ ...current, fast_mode: event.target.checked }))}
              />
              Fast mode
            </label>
            <button className="primary-button" type="submit" disabled={submitting}>
              {submitting ? "Starting..." : `Run ${datasetKind} pipeline`}
            </button>
          </form>
        </div>

        <div className="card">
          <div className="section-title">Latest Summary</div>
          {loadingSummary ? (
            <div className="muted">Loading summary...</div>
          ) : (
            <div className="stats-grid">
              <StatCard
                title="Models Dir"
                value={summary?.artifacts?.models_dir?.split("/").slice(-1)[0] || "n/a"}
                subtitle={summary?.artifacts?.models_dir}
              />
              <StatCard
                title="Best Model"
                value={summary?.artifacts?.best_model?.split("/").slice(-1)[0] || "n/a"}
                subtitle={summary?.artifacts?.best_model}
              />
              <StatCard
                title="Eval Models"
                value={summary?.evaluation_metrics?.models?.length ?? 0}
                subtitle={datasetKind}
              />
              <StatCard
                title="Metrics File"
                value={summary?.artifacts?.evaluation_metrics?.split("/").slice(-1)[0] || "n/a"}
                subtitle={summary?.artifacts?.evaluation_metrics}
              />
            </div>
          )}
        </div>
      </section>

      <section className="grid two-up">
        <div className="card">
          <div className="section-title">Model Evaluation</div>
          <MetricTable rows={modelRows} />
        </div>
        <div className="card">
          <div className="section-title">Pipeline Jobs</div>
          <JobList jobs={jobs} onSelect={setSelectedJobId} selectedJobId={selectedJobId} />
        </div>
      </section>

      <section className="grid">
        <div className="card">
          <div className="section-title">Evaluation Visuals</div>
          <VisualGallery visuals={visuals} />
        </div>
      </section>

      <section className="grid two-up">
        <div className="card">
          <div className="section-title">Selected Job Log</div>
          <pre className="log-view">{selectedJobLog || "No log selected yet."}</pre>
        </div>
        <div className="card">
          <div className="section-title">Artifact Inventory</div>
          {loadingArtifacts ? (
            <div className="muted">Loading artifacts...</div>
          ) : (
            <div className="artifact-grid">
              <ArtifactSection title="Metrics" items={artifactEntries.train_metrics || []} />
              <ArtifactSection title="Plots" items={artifactEntries.plots_dir || []} />
              <ArtifactSection title="Reports" items={artifactEntries.reports_dir || []} />
              <ArtifactSection title="Explainability" items={artifactEntries.explain_dir || []} />
            </div>
          )}
        </div>
      </section>
    </div>
  );
}
