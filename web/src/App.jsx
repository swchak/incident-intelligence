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

function StatCard({ title, value, subtitle }) {
  return (
    <div className="card stat-card">
      <div className="stat-title">{title}</div>
      <div className="stat-value">{value}</div>
      {subtitle ? <div className="stat-subtitle">{subtitle}</div> : null}
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

function formatScore(value) {
  return typeof value === "number" ? value.toFixed(4) : "n/a";
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
        </button>
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

  const artifactEntries = artifacts?.artifacts || {};

  return (
    <div className="app-shell">
      <header className="hero">
        <div>
          <div className="eyebrow">Incident Intelligence</div>
          <h1>Pipeline Demo Dashboard</h1>
          <p>
            Run the snapshot or temporal ML workflow, inspect artifacts, and track pipeline jobs from one place.
          </p>
        </div>
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
