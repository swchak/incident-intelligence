import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import App from "./App";

describe("App", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  function checkboxFor(labelPattern) {
    return screen
      .getAllByLabelText(labelPattern)
      .find(
        (element) =>
          element.tagName === "INPUT" &&
          element.getAttribute("type") === "checkbox",
      );
  }

  it("smoke test: mounts the dashboard with empty and legacy job data", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
      const url = String(input);

      if (url.includes("/api/dashboard/summary/snapshot")) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              dataset_kind: "snapshot",
              evaluation_metrics: { models: [] },
              artifacts: {
                models_dir: "artifacts/models",
                best_model: "artifacts/models/best_model.joblib",
                evaluation_metrics: "artifacts/metrics/evaluation.json",
              },
            }),
          ),
        );
      }

      if (url.includes("/api/artifacts/snapshot")) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              dataset_kind: "snapshot",
              artifacts: {
                models_dir: [],
                train_metrics: [],
                plots_dir: [],
                reports_dir: [],
                explain_dir: [],
              },
            }),
          ),
        );
      }

      if (url.endsWith("/api/pipeline/jobs")) {
        return Promise.resolve(
          new Response(
            JSON.stringify([
              {
                job_id: "legacy-job-1",
                dataset_kind: "snapshot",
                mode: "full",
                requested_stages: ["pipeline"],
                status: "completed",
                created_at: "2026-03-31T00:00:00+00:00",
                finished_at: "2026-03-31T00:01:00+00:00",
                current_stage_name: null,
                log_path: "artifacts/api_runs/legacy-job-1.log",
              },
            ]),
          ),
        );
      }

      if (url.endsWith("/api/pipeline/jobs/legacy-job-1/log")) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              log: "legacy pipeline log",
              stages: [],
            }),
          ),
        );
      }

      return Promise.reject(new Error(`Unhandled fetch: ${url}`));
    });

    render(<App />);

    expect(
      await screen.findByRole("heading", {
        name: /Incident root cause modeling - from synthetic telemetry and temporal features to explainable root-cause results\./i,
      }),
    ).toBeInTheDocument();
    expect(await screen.findByText("Run Pipeline")).toBeInTheDocument();
    expect(await screen.findByText("Latest Summary")).toBeInTheDocument();
    expect(await screen.findByText("Recent Jobs")).toBeInTheDocument();
    expect(await screen.findByText("Selected Job Log")).toBeInTheDocument();
    expect(await screen.findByText("Evaluation & Explainability Visuals")).toBeInTheDocument();
    expect(await screen.findByText("Artifact Inventory")).toBeInTheDocument();
    fireEvent.click(screen.getByText("legacy-j").closest("button"));
    expect(await screen.findByText("0/0 stages")).toBeInTheDocument();
    expect(await screen.findByRole("button", { name: /delete job legacy-job-1/i })).toBeInTheDocument();
  });

  it("disables tuning fields for custom generate-only runs", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockImplementation((input, options = {}) => {
        const url = String(input);

      if (url.includes("/api/dashboard/summary/snapshot")) {
          return Promise.resolve(
            new Response(
              JSON.stringify({
                dataset_kind: "snapshot",
                evaluation_metrics: {
                  models: [
                    {
                      model_name: "Logistic",
                      metrics: {
                        accuracy: 0.9,
                        f1_macro: 0.85,
                        precision_macro: 0.84,
                        recall_macro: 0.83,
                      },
                    },
                  ],
                },
                artifacts: {
                  models_dir: "artifacts/models",
                  best_model: "artifacts/models/best_model.joblib",
                  evaluation_metrics: "artifacts/metrics/evaluation.json",
                },
              }),
            ),
          );
        }

      if (url.includes("/api/artifacts/snapshot")) {
          return Promise.resolve(
            new Response(
              JSON.stringify({
                dataset_kind: "snapshot",
                artifacts: {
                  models_dir: [
                    { path: "artifacts/models/best_model.joblib", type: "file" },
                  ],
                  train_metrics: [
                    {
                      path: "artifacts/metrics/train_val_results.json",
                      type: "file",
                    },
                  ],
                  plots_dir: [
                    {
                      path: "artifacts/plots/confusion_matrix.png",
                      type: "file",
                    },
                  ],
                  reports_dir: [
                    { path: "artifacts/reports/report.md", type: "file" },
                  ],
                  explain_dir: [
                    {
                      path: "artifacts/explain/best_model/global/shap_importance.png",
                      type: "file",
                    },
                  ],
                },
              }),
            ),
          );
        }

        if (url.endsWith("/api/pipeline/jobs")) {
          return Promise.resolve(
            new Response(
              JSON.stringify([
                {
                  job_id: "job-1",
                  dataset_kind: "snapshot",
                  mode: "full",
                  requested_stages: [
                    "generate_snapshot",
                    "train_snapshot",
                    "evaluate_snapshot",
                    "explain_snapshot",
                  ],
                  status: "completed",
                  created_at: "2026-03-31T00:00:00+00:00",
                  finished_at: "2026-03-31T00:05:00+00:00",
                  current_stage_name: null,
                  log_path: "artifacts/api_runs/job-1",
                  stages: [
                    {
                      stage_id: "job-1:generate_snapshot",
                      stage_name: "generate_snapshot",
                      stage_order: 0,
                      status: "completed",
                      command: ["python", "-m", "incident_intelligence.cli.generator"],
                      log_path: "artifacts/api_runs/job-1/01_generate_snapshot.log",
                    },
                    {
                      stage_id: "job-1:train_snapshot",
                      stage_name: "train_snapshot",
                      stage_order: 1,
                      status: "completed",
                      command: ["python", "-m", "incident_intelligence.cli.train"],
                      log_path: "artifacts/api_runs/job-1/02_train_snapshot.log",
                    },
                  ],
                },
              ]),
            ),
          );
        }

        if (url.endsWith("/api/pipeline/jobs/job-1/log")) {
          return Promise.resolve(
            new Response(
              JSON.stringify({
                log: "=== train_snapshot (completed) ===\ntrain finished",
                stages: [],
              }),
            ),
          );
        }

        if (url.endsWith("/api/pipeline/jobs/job-2/log")) {
          return Promise.resolve(
            new Response(
              JSON.stringify({
                log: "",
                stages: [],
              }),
            ),
          );
        }

        if (url.endsWith("/api/pipeline/run") && options.method === "POST") {
          return Promise.resolve(
            new Response(
              JSON.stringify({
                job_id: "job-2",
                dataset_kind: "snapshot",
                mode: "custom",
                requested_stages: ["generate_snapshot"],
                status: "queued",
                created_at: "2026-03-31T00:10:00+00:00",
                current_stage_name: null,
                log_path: "artifacts/api_runs/job-2",
                stages: [
                  {
                    stage_id: "job-2:generate_snapshot",
                    stage_name: "generate_snapshot",
                    stage_order: 0,
                    status: "queued",
                    command: ["python", "-m", "incident_intelligence.cli.generator"],
                    log_path: "artifacts/api_runs/job-2/01_generate_snapshot.log",
                  },
                ],
              }),
            ),
          );
        }

        return Promise.reject(new Error(`Unhandled fetch: ${url}`));
      });

    render(<App />);

    expect(
      await screen.findByText(
        /Incident root cause modeling - from synthetic telemetry and temporal features to explainable root-cause results\./i,
      ),
    ).toBeInTheDocument();
    expect(await screen.findAllByText("0.8500")).toHaveLength(2);
    expect(await screen.findByText("shap importance")).toBeInTheDocument();
    fireEvent.click(screen.getByText("job-1").closest("button"));
    expect(await screen.findByText("2/2 stages")).toBeInTheDocument();
    expect(await screen.findAllByText("train_snapshot")).not.toHaveLength(0);

    fireEvent.click(screen.getByRole("button", { name: /^custom$/i }));
    expect(screen.getByLabelText(/models/i)).toBeDisabled();
    expect(screen.getByLabelText(/scoring/i)).toBeDisabled();
    fireEvent.click(
      screen.getByRole("button", { name: /run generate snapshot/i }),
    );

    await waitFor(() => {
      const runCall = fetchMock.mock.calls.find(([url]) =>
        String(url).includes("/api/pipeline/run"),
      );
      expect(runCall).toBeTruthy();
      const payload = JSON.parse(runCall[1].body);
      expect(payload.mode).toBe("custom");
      expect(payload.stages).toEqual(["generate_snapshot"]);
      expect(payload.source_job_id).toBeNull();
      expect(payload.force_new_run).toBe(false);
    });
  });

  it("continues a selected staged run instead of creating separate custom runs", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockImplementation((input, options = {}) => {
        const url = String(input);

        if (url.includes("/api/dashboard/summary/snapshot")) {
          return Promise.resolve(
            new Response(
              JSON.stringify({
                dataset_kind: "snapshot",
                evaluation_metrics: { models: [] },
                artifacts: {
                  models_dir: "artifacts/models",
                  best_model: "artifacts/models/best_model.joblib",
                  evaluation_metrics: "artifacts/metrics/evaluation.json",
                },
              }),
            ),
          );
        }

        if (url.includes("/api/artifacts/snapshot")) {
          return Promise.resolve(
            new Response(
              JSON.stringify({
                dataset_kind: "snapshot",
                artifacts: {
                  models_dir: [],
                  train_metrics: [],
                  plots_dir: [],
                  reports_dir: [],
                  explain_dir: [],
                },
              }),
            ),
          );
        }

        if (url.endsWith("/api/pipeline/jobs")) {
          return Promise.resolve(
            new Response(
              JSON.stringify([
                {
                  job_id: "job-custom",
                  dataset_kind: "snapshot",
                  mode: "custom",
                  requested_stages: ["generate_snapshot"],
                  status: "completed",
                  created_at: "2026-03-31T00:00:00+00:00",
                  finished_at: "2026-03-31T00:02:00+00:00",
                  current_stage_name: null,
                  log_path: "artifacts/api_runs/job-custom",
                  stages: [
                    {
                      stage_id: "job-custom:generate_snapshot",
                      stage_name: "generate_snapshot",
                      stage_order: 0,
                      status: "completed",
                      command: ["python", "-m", "incident_intelligence.cli.generator"],
                      log_path: "artifacts/api_runs/job-custom/01_generate_snapshot.log",
                    },
                  ],
                },
                {
                  job_id: "job-full",
                  dataset_kind: "snapshot",
                  mode: "full",
                  requested_stages: ["generate_snapshot", "train_snapshot"],
                  status: "completed",
                  created_at: "2026-03-31T00:00:00+00:00",
                  finished_at: "2026-03-31T00:05:00+00:00",
                  current_stage_name: null,
                  log_path: "artifacts/api_runs/job-full",
                  stages: [],
                },
              ]),
            ),
          );
        }

        if (url.endsWith("/api/pipeline/jobs/job-custom/log")) {
          return Promise.resolve(new Response(JSON.stringify({ log: "custom log", stages: [] })));
        }

        if (url.endsWith("/api/pipeline/jobs/job-full/log")) {
          return Promise.resolve(new Response(JSON.stringify({ log: "full log", stages: [] })));
        }

        if (url.endsWith("/api/pipeline/run") && options.method === "POST") {
          return Promise.resolve(
            new Response(
              JSON.stringify({
                job_id: "job-custom",
                dataset_kind: "snapshot",
                mode: "custom",
                requested_stages: ["generate_snapshot", "train_snapshot"],
                status: "queued",
                created_at: "2026-03-31T00:10:00+00:00",
                current_stage_name: null,
                log_path: "artifacts/api_runs/job-custom",
                stages: [
                  {
                    stage_id: "job-custom:generate_snapshot",
                    stage_name: "generate_snapshot",
                    stage_order: 0,
                    status: "completed",
                    command: ["python", "-m", "incident_intelligence.cli.generator"],
                    log_path: "artifacts/api_runs/job-custom/01_generate_snapshot.log",
                  },
                  {
                    stage_id: "job-custom:train_snapshot",
                    stage_name: "train_snapshot",
                    stage_order: 1,
                    status: "queued",
                    command: ["python", "-m", "incident_intelligence.cli.train"],
                    log_path: "artifacts/api_runs/job-custom/02_train_snapshot.log",
                  },
                ],
              }),
            ),
          );
        }

        return Promise.reject(new Error(`Unhandled fetch: ${url}`));
      });

    render(<App />);

    expect(await screen.findByText("Custom Pipeline Runs")).toBeInTheDocument();
    expect(await screen.findByText("Full Pipeline Runs")).toBeInTheDocument();

    fireEvent.click(screen.getByText("job-cust").closest("button"));
    expect(screen.getByDisplayValue("logistic,rf")).not.toBeDisabled();
    expect(screen.getByRole("button", { name: /^custom$/i })).toHaveClass("active");
    expect(screen.getByLabelText(/generate_snapshot/i)).toBeDisabled();
    expect(
      screen
        .getAllByLabelText(/train_snapshot/i)
        .find((element) => element.tagName === "INPUT" && element.getAttribute("type") === "checkbox"),
    ).not.toBeDisabled();
    expect(screen.getByLabelText(/evaluate_snapshot/i)).toBeDisabled();
    expect(
      screen.getByText(/Continuing staged run job-cust with: train_snapshot/i),
    ).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /run train snapshot/i }));

    await waitFor(() => {
      const runCall = fetchMock.mock.calls.find(([url]) =>
        String(url).includes("/api/pipeline/run"),
      );
      expect(runCall).toBeTruthy();
      const payload = JSON.parse(runCall[1].body);
      expect(payload.mode).toBe("custom");
      expect(payload.stages).toEqual(["train_snapshot"]);
      expect(payload.source_job_id).toBe("job-custom");
    });
  });

  it("selecting a full temporal run syncs workflow tab and pipeline mode", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
      const url = String(input);

      if (url.includes("/api/dashboard/summary/snapshot")) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              dataset_kind: "snapshot",
              evaluation_metrics: { models: [] },
              artifacts: {
                models_dir: "artifacts/models",
                best_model: "artifacts/models/best_model.joblib",
                evaluation_metrics: "artifacts/metrics/evaluation.json",
              },
            }),
          ),
        );
      }

      if (url.includes("/api/artifacts/snapshot")) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              dataset_kind: "snapshot",
              artifacts: {
                models_dir: [],
                train_metrics: [],
                plots_dir: [],
                reports_dir: [],
                explain_dir: [],
              },
            }),
          ),
        );
      }

      if (url.includes("/api/dashboard/summary/temporal")) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              dataset_kind: "temporal",
              evaluation_metrics: { models: [] },
              artifacts: {
                models_dir: "artifacts/models_temporal",
                best_model: "artifacts/models_temporal/best_model.joblib",
                evaluation_metrics: "artifacts/metrics_temporal/evaluation.json",
              },
            }),
          ),
        );
      }

      if (url.includes("/api/artifacts/temporal")) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              dataset_kind: "temporal",
              artifacts: {
                models_dir: [],
                train_metrics: [],
                plots_dir: [],
                reports_dir: [],
                explain_dir: [],
              },
            }),
          ),
        );
      }

      if (url.endsWith("/api/pipeline/jobs")) {
        return Promise.resolve(
          new Response(
            JSON.stringify([
              {
                job_id: "job-temp-full",
                dataset_kind: "temporal",
                mode: "full",
                requested_stages: [
                  "generate_sequence",
                  "build_temporal_features",
                  "train_temporal",
                  "evaluate_temporal",
                  "explain_temporal",
                ],
                status: "completed",
                created_at: "2026-03-31T00:00:00+00:00",
                finished_at: "2026-03-31T00:05:00+00:00",
                current_stage_name: null,
                log_path: "artifacts/api_runs/job-temp-full",
                stages: [
                  {
                    stage_id: "job-temp-full:generate_sequence",
                    stage_name: "generate_sequence",
                    stage_order: 0,
                    status: "completed",
                    command: ["python"],
                    log_path: "artifacts/api_runs/job-temp-full/01_generate_sequence.log",
                  },
                ],
              },
            ]),
          ),
        );
      }

      if (url.endsWith("/api/pipeline/jobs/job-temp-full/log")) {
        return Promise.resolve(new Response(JSON.stringify({ log: "temporal full log", stages: [] })));
      }

      return Promise.reject(new Error(`Unhandled fetch: ${url}`));
    });

    render(<App />);

    expect(screen.getByRole("button", { name: /^full$/i })).toHaveClass("active");
    expect(screen.getByRole("button", { name: /^snapshot$/i })).toHaveClass("active");

    fireEvent.click((await screen.findByText("job-temp")).closest("button"));

    await waitFor(() => {
      expect(screen.getByRole("button", { name: /^full$/i })).toHaveClass("active");
      expect(screen.getByRole("button", { name: /^temporal$/i })).toHaveClass("active");
      expect(screen.getByLabelText("generate_sequence")).toBeChecked();
      expect(screen.getByLabelText("build_temporal_features")).toBeChecked();
    });
  });

  it("clears the selected run when switching workflow tabs manually", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
      const url = String(input);

      if (url.includes("/api/dashboard/summary/snapshot")) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              dataset_kind: "snapshot",
              evaluation_metrics: { models: [] },
              artifacts: {
                models_dir: "artifacts/models",
                best_model: "artifacts/models/best_model.joblib",
                evaluation_metrics: "artifacts/metrics/evaluation.json",
              },
            }),
          ),
        );
      }

      if (url.includes("/api/artifacts/snapshot")) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              dataset_kind: "snapshot",
              artifacts: {
                models_dir: [],
                train_metrics: [],
                plots_dir: [],
                reports_dir: [],
                explain_dir: [],
              },
            }),
          ),
        );
      }

      if (url.includes("/api/dashboard/summary/temporal")) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              dataset_kind: "temporal",
              evaluation_metrics: { models: [] },
              artifacts: {
                models_dir: "artifacts/models_temporal",
                best_model: "artifacts/models_temporal/best_model.joblib",
                evaluation_metrics: "artifacts/metrics_temporal/evaluation.json",
              },
            }),
          ),
        );
      }

      if (url.includes("/api/artifacts/temporal")) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              dataset_kind: "temporal",
              artifacts: {
                models_dir: [],
                train_metrics: [],
                plots_dir: [],
                reports_dir: [],
                explain_dir: [],
              },
            }),
          ),
        );
      }

      if (url.endsWith("/api/pipeline/jobs")) {
        return Promise.resolve(
          new Response(
            JSON.stringify([
              {
                job_id: "job-snapshot",
                dataset_kind: "snapshot",
                mode: "full",
                requested_stages: [
                  "generate_snapshot",
                  "train_snapshot",
                  "evaluate_snapshot",
                  "explain_snapshot",
                ],
                status: "completed",
                created_at: "2026-03-31T00:00:00+00:00",
                finished_at: "2026-03-31T00:05:00+00:00",
                current_stage_name: null,
                log_path: "artifacts/api_runs/job-snapshot",
                stages: [
                  {
                    stage_id: "job-snapshot:generate_snapshot",
                    stage_name: "generate_snapshot",
                    stage_order: 0,
                    status: "completed",
                    command: ["python"],
                    log_path: "artifacts/api_runs/job-snapshot/01_generate_snapshot.log",
                  },
                ],
              },
            ]),
          ),
        );
      }

      if (url.endsWith("/api/pipeline/jobs/job-snapshot/log")) {
        return Promise.resolve(new Response(JSON.stringify({ log: "snapshot log", stages: [] })));
      }

      return Promise.reject(new Error(`Unhandled fetch: ${url}`));
    });

    render(<App />);

    fireEvent.click((await screen.findByText("job-snap")).closest("button"));

    await waitFor(() => {
      expect(screen.getByText("snapshot log")).toBeInTheDocument();
    });

    fireEvent.click(screen.getByRole("button", { name: /^temporal$/i }));

    await waitFor(() => {
      expect(screen.queryByText("snapshot log")).not.toBeInTheDocument();
      expect(screen.getByText(/No log selected yet/i)).toBeInTheDocument();
    });
  });

  it("restores the in-progress custom run for a workflow after switching away and back", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
      const url = String(input);

      if (url.includes("/api/dashboard/summary/snapshot")) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              dataset_kind: "snapshot",
              evaluation_metrics: { models: [] },
              artifacts: {
                models_dir: "artifacts/models",
                best_model: "artifacts/models/best_model.joblib",
                evaluation_metrics: "artifacts/metrics/evaluation.json",
              },
            }),
          ),
        );
      }

      if (url.includes("/api/artifacts/snapshot")) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              dataset_kind: "snapshot",
              artifacts: {
                models_dir: [],
                train_metrics: [],
                plots_dir: [],
                reports_dir: [],
                explain_dir: [],
              },
            }),
          ),
        );
      }

      if (url.includes("/api/dashboard/summary/temporal")) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              dataset_kind: "temporal",
              evaluation_metrics: { models: [] },
              artifacts: {
                models_dir: "artifacts/models_temporal",
                best_model: "artifacts/models_temporal/best_model.joblib",
                evaluation_metrics: "artifacts/metrics_temporal/evaluation.json",
              },
            }),
          ),
        );
      }

      if (url.includes("/api/artifacts/temporal")) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              dataset_kind: "temporal",
              artifacts: {
                models_dir: [],
                train_metrics: [],
                plots_dir: [],
                reports_dir: [],
                explain_dir: [],
              },
            }),
          ),
        );
      }

      if (url.endsWith("/api/pipeline/jobs")) {
        return Promise.resolve(
          new Response(
            JSON.stringify([
              {
                job_id: "job-snap-custom",
                dataset_kind: "snapshot",
                mode: "custom",
                requested_stages: ["generate_snapshot"],
                status: "completed",
                created_at: "2026-04-06T09:00:00+00:00",
                finished_at: "2026-04-06T09:01:00+00:00",
                current_stage_name: null,
                log_path: "artifacts/api_runs/job-snap-custom",
                stages: [
                  {
                    stage_id: "job-snap-custom:generate_snapshot",
                    stage_name: "generate_snapshot",
                    stage_order: 0,
                    status: "completed",
                    command: ["python"],
                    log_path:
                      "artifacts/api_runs/job-snap-custom/01_generate_snapshot.log",
                  },
                ],
              },
              {
                job_id: "job-temp-custom",
                dataset_kind: "temporal",
                mode: "custom",
                requested_stages: ["generate_sequence"],
                status: "completed",
                created_at: "2026-04-06T09:05:00+00:00",
                finished_at: "2026-04-06T09:06:00+00:00",
                current_stage_name: null,
                log_path: "artifacts/api_runs/job-temp-custom",
                stages: [
                  {
                    stage_id: "job-temp-custom:generate_sequence",
                    stage_name: "generate_sequence",
                    stage_order: 0,
                    status: "completed",
                    command: ["python"],
                    log_path:
                      "artifacts/api_runs/job-temp-custom/01_generate_sequence.log",
                  },
                ],
              },
            ]),
          ),
        );
      }

      if (url.endsWith("/api/pipeline/jobs/job-snap-custom/log")) {
        return Promise.resolve(
          new Response(JSON.stringify({ log: "snapshot staged log", stages: [] })),
        );
      }

      if (url.endsWith("/api/pipeline/jobs/job-temp-custom/log")) {
        return Promise.resolve(
          new Response(JSON.stringify({ log: "temporal staged log", stages: [] })),
        );
      }

      return Promise.reject(new Error(`Unhandled fetch: ${url}`));
    });

    render(<App />);

    fireEvent.click(screen.getByRole("button", { name: /^custom$/i }));
    fireEvent.click((await screen.findByText("job-snap")).closest("button"));

    await waitFor(() => {
      expect(
        screen.getByText(/Continuing staged run job-snap with: train_snapshot/i),
      ).toBeInTheDocument();
    });
    expect(checkboxFor(/generate_snapshot/i)).toBeDisabled();
    expect(checkboxFor(/train_snapshot/i)).toBeEnabled();

    fireEvent.click(screen.getByRole("button", { name: /^temporal$/i }));
    fireEvent.click(screen.getByRole("button", { name: /^custom$/i }));
    fireEvent.click((await screen.findByText("job-temp")).closest("button"));

    await waitFor(() => {
      expect(
        screen.getByText(/Continuing staged run job-temp with: build_temporal_features/i),
      ).toBeInTheDocument();
    });

    fireEvent.click(screen.getByRole("button", { name: /^snapshot$/i }));
    fireEvent.click(screen.getByRole("button", { name: /^custom$/i }));

    await waitFor(() => {
      expect(
        screen.getByText(/Continuing staged run job-snap with: train_snapshot/i),
      ).toBeInTheDocument();
    });
    expect(checkboxFor(/generate_snapshot/i)).toBeDisabled();
    expect(checkboxFor(/train_snapshot/i)).toBeEnabled();
  });

  it("restores the selected custom run log and run-scoped artifacts when switching away and back to its workflow", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
      const url = String(input);

      if (url.includes("/api/dashboard/summary/snapshot")) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              dataset_kind: "snapshot",
              evaluation_metrics: { models: [] },
              artifacts: {
                models_dir: "artifacts/models",
                best_model: "artifacts/models/best_model.joblib",
                evaluation_metrics: "artifacts/metrics/evaluation.json",
              },
            }),
          ),
        );
      }

      if (url.includes("/api/artifacts/snapshot")) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              dataset_kind: "snapshot",
              artifacts: {
                models_dir: [],
                train_metrics: [],
                plots_dir: [],
                reports_dir: [],
                explain_dir: [],
              },
            }),
          ),
        );
      }

      if (url.includes("/api/dashboard/summary/temporal")) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              dataset_kind: "temporal",
              evaluation_metrics: { models: [] },
              artifacts: {
                models_dir: "artifacts/models_temporal",
                best_model: "artifacts/models_temporal/best_model.joblib",
                evaluation_metrics: "artifacts/metrics_temporal/evaluation.json",
              },
            }),
          ),
        );
      }

      if (url.includes("/api/artifacts/temporal")) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              dataset_kind: "temporal",
              artifacts: {
                models_dir: [],
                train_metrics: [],
                plots_dir: [],
                reports_dir: [],
                explain_dir: [],
              },
            }),
          ),
        );
      }

      if (url.endsWith("/api/pipeline/jobs")) {
        return Promise.resolve(
          new Response(
            JSON.stringify([
              {
                job_id: "job-temp-custom",
                dataset_kind: "temporal",
                mode: "custom",
                requested_stages: ["generate_sequence"],
                status: "completed",
                created_at: "2026-04-06T10:00:00+00:00",
                finished_at: "2026-04-06T10:01:00+00:00",
                current_stage_name: null,
                log_path: "artifacts/api_runs/job-temp-custom",
                stages: [
                  {
                    stage_id: "job-temp-custom:generate_sequence",
                    stage_name: "generate_sequence",
                    stage_order: 0,
                    status: "completed",
                    command: ["python"],
                    log_path:
                      "artifacts/api_runs/job-temp-custom/01_generate_sequence.log",
                  },
                ],
              },
            ]),
          ),
        );
      }

      if (url.endsWith("/api/pipeline/jobs/job-temp-custom/log")) {
        return Promise.resolve(
          new Response(
            JSON.stringify({ log: "temporal staged log", stages: [] }),
          ),
        );
      }

      return Promise.reject(new Error(`Unhandled fetch: ${url}`));
    });

    render(<App />);

    fireEvent.click(screen.getByRole("button", { name: /^temporal$/i }));
    fireEvent.click(screen.getByRole("button", { name: /^custom$/i }));
    fireEvent.click((await screen.findByText("job-temp")).closest("button"));

    await waitFor(() => {
      expect(screen.getByText("temporal staged log")).toBeInTheDocument();
    });

    fireEvent.click(screen.getByRole("button", { name: /^snapshot$/i }));

    await waitFor(() => {
      expect(screen.queryByText("temporal staged log")).not.toBeInTheDocument();
      expect(screen.getByText(/No log selected yet/i)).toBeInTheDocument();
    });

    fireEvent.click(screen.getByRole("button", { name: /^temporal$/i }));

    await waitFor(() => {
      expect(screen.getByText("temporal staged log")).toBeInTheDocument();
      expect(
        screen.getByText(/Continuing staged run job-temp with: build_temporal_features/i),
      ).toBeInTheDocument();
    });

    expect(
      fetchMock.mock.calls.some(([url]) =>
        String(url).includes("/api/dashboard/summary/temporal?job_id=job-temp-custom"),
      ),
    ).toBe(true);
    expect(
      fetchMock.mock.calls.some(([url]) =>
        String(url).includes("/api/artifacts/temporal?job_id=job-temp-custom"),
      ),
    ).toBe(true);
  });

  it("advances a fresh custom run even when another same-workflow staged run is active", async () => {
    let jobsResponse = [
      {
        job_id: "job-temp-old",
        dataset_kind: "temporal",
        mode: "custom",
        requested_stages: [
          "generate_sequence",
          "build_temporal_features",
          "train_temporal",
        ],
        status: "running",
        created_at: "2026-04-06T10:00:00+00:00",
        current_stage_name: "train_temporal",
        log_path: "artifacts/api_runs/job-temp-old",
        stages: [
          {
            stage_id: "job-temp-old:generate_sequence",
            stage_name: "generate_sequence",
            stage_order: 0,
            status: "completed",
            command: ["python"],
            log_path: "artifacts/api_runs/job-temp-old/01_generate_sequence.log",
          },
          {
            stage_id: "job-temp-old:build_temporal_features",
            stage_name: "build_temporal_features",
            stage_order: 1,
            status: "completed",
            command: ["python"],
            log_path:
              "artifacts/api_runs/job-temp-old/02_build_temporal_features.log",
          },
          {
            stage_id: "job-temp-old:train_temporal",
            stage_name: "train_temporal",
            stage_order: 2,
            status: "running",
            command: ["python"],
            log_path: "artifacts/api_runs/job-temp-old/03_train_temporal.log",
          },
        ],
      },
    ];
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockImplementation((input, options = {}) => {
        const url = String(input);

        if (url.includes("/api/dashboard/summary/temporal")) {
          return Promise.resolve(
            new Response(
              JSON.stringify({
                dataset_kind: "temporal",
                evaluation_metrics: { models: [] },
                artifacts: {
                  models_dir: "artifacts/models_temporal",
                  best_model: "artifacts/models_temporal/best_model.joblib",
                  evaluation_metrics: "artifacts/metrics_temporal/evaluation.json",
                },
              }),
            ),
          );
        }

        if (url.includes("/api/artifacts/temporal")) {
          return Promise.resolve(
            new Response(
              JSON.stringify({
                dataset_kind: "temporal",
                artifacts: {
                  models_dir: [],
                  train_metrics: [],
                  plots_dir: [],
                  reports_dir: [],
                  explain_dir: [],
                },
              }),
            ),
          );
        }

        if (url.endsWith("/api/pipeline/jobs")) {
          return Promise.resolve(new Response(JSON.stringify(jobsResponse)));
        }

        if (url.endsWith("/api/pipeline/jobs/job-temp-old/log")) {
          return Promise.resolve(
            new Response(JSON.stringify({ log: "old temporal log", stages: [] })),
          );
        }

        if (url.endsWith("/api/pipeline/jobs/job-temp-new/log")) {
          return Promise.resolve(
            new Response(JSON.stringify({ log: "new temporal log", stages: [] })),
          );
        }

        if (url.endsWith("/api/pipeline/run") && options.method === "POST") {
          jobsResponse = [
            {
              job_id: "job-temp-old",
              dataset_kind: "temporal",
              mode: "custom",
              requested_stages: [
                "generate_sequence",
                "build_temporal_features",
                "train_temporal",
              ],
              status: "running",
              created_at: "2026-04-06T10:00:00+00:00",
              current_stage_name: "train_temporal",
              log_path: "artifacts/api_runs/job-temp-old",
              stages: [
                {
                  stage_id: "job-temp-old:generate_sequence",
                  stage_name: "generate_sequence",
                  stage_order: 0,
                  status: "completed",
                  command: ["python"],
                  log_path:
                    "artifacts/api_runs/job-temp-old/01_generate_sequence.log",
                },
                {
                  stage_id: "job-temp-old:build_temporal_features",
                  stage_name: "build_temporal_features",
                  stage_order: 1,
                  status: "completed",
                  command: ["python"],
                  log_path:
                    "artifacts/api_runs/job-temp-old/02_build_temporal_features.log",
                },
                {
                  stage_id: "job-temp-old:train_temporal",
                  stage_name: "train_temporal",
                  stage_order: 2,
                  status: "running",
                  command: ["python"],
                  log_path: "artifacts/api_runs/job-temp-old/03_train_temporal.log",
                },
              ],
            },
            {
              job_id: "job-temp-new",
              dataset_kind: "temporal",
              mode: "custom",
              requested_stages: ["generate_sequence"],
              status: "completed",
              created_at: "2026-04-06T10:10:00+00:00",
              finished_at: "2026-04-06T10:11:00+00:00",
              current_stage_name: null,
              log_path: "artifacts/api_runs/job-temp-new",
              stages: [
                {
                  stage_id: "job-temp-new:generate_sequence",
                  stage_name: "generate_sequence",
                  stage_order: 0,
                  status: "completed",
                  command: ["python"],
                  log_path: "artifacts/api_runs/job-temp-new/01_generate_sequence.log",
                },
              ],
            },
          ];
          return Promise.resolve(
            new Response(
              JSON.stringify({
                job_id: "job-temp-new",
                dataset_kind: "temporal",
                mode: "custom",
                requested_stages: ["generate_sequence"],
                status: "queued",
                created_at: "2026-04-06T10:10:00+00:00",
                current_stage_name: "generate_sequence",
                log_path: "artifacts/api_runs/job-temp-new",
                stages: [],
              }),
            ),
          );
        }

        return Promise.reject(new Error(`Unhandled fetch: ${url}`));
      });

    render(<App />);

    fireEvent.click(screen.getByRole("button", { name: /^temporal$/i }));
    fireEvent.click(screen.getByRole("button", { name: /^custom$/i }));
    fireEvent.click((await screen.findByText("job-temp")).closest("button"));

    await screen.findByText(/Wait for train_temporal to finish before continuing/i);
    fireEvent.click(screen.getByRole("button", { name: /start new staged run/i }));
    fireEvent.click(screen.getByRole("button", { name: /run generate sequence/i }));

    await waitFor(() => {
      expect(
        screen.getByText(/Continuing staged run job-temp with: build_temporal_features/i),
      ).toBeInTheDocument();
    });
    expect(checkboxFor(/generate_sequence/i)).toBeDisabled();
    expect(checkboxFor(/build_temporal_features/i)).toBeEnabled();
    expect(
      fetchMock.mock.calls.some(([url]) =>
        String(url).includes("/api/pipeline/jobs/job-temp-new/log"),
      ),
    ).toBe(true);
  });

  it("can start a brand-new staged run instead of continuing an existing one", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockImplementation((input, options = {}) => {
        const url = String(input);

        if (url.includes("/api/dashboard/summary/snapshot")) {
          return Promise.resolve(
            new Response(
              JSON.stringify({
                dataset_kind: "snapshot",
                evaluation_metrics: { models: [] },
                artifacts: {
                  models_dir: "artifacts/models",
                  best_model: "artifacts/models/best_model.joblib",
                  evaluation_metrics: "artifacts/metrics/evaluation.json",
                },
              }),
            ),
          );
        }

        if (url.includes("/api/artifacts/snapshot")) {
          return Promise.resolve(
            new Response(
              JSON.stringify({
                dataset_kind: "snapshot",
                artifacts: {
                  models_dir: [],
                  train_metrics: [],
                  plots_dir: [],
                  reports_dir: [],
                  explain_dir: [],
                },
              }),
            ),
          );
        }

        if (url.endsWith("/api/pipeline/jobs")) {
          return Promise.resolve(
            new Response(
              JSON.stringify([
                {
                  job_id: "job-custom",
                  dataset_kind: "snapshot",
                  mode: "custom",
                  requested_stages: ["generate_snapshot"],
                  status: "completed",
                  created_at: "2026-03-31T00:00:00+00:00",
                  finished_at: "2026-03-31T00:02:00+00:00",
                  current_stage_name: null,
                  log_path: "artifacts/api_runs/job-custom",
                  stages: [
                    {
                      stage_id: "job-custom:generate_snapshot",
                      stage_name: "generate_snapshot",
                      stage_order: 0,
                      status: "completed",
                      command: ["python", "-m", "incident_intelligence.cli.generator"],
                      log_path: "artifacts/api_runs/job-custom/01_generate_snapshot.log",
                    },
                  ],
                },
              ]),
            ),
          );
        }

        if (url.endsWith("/api/pipeline/jobs/job-custom/log")) {
          return Promise.resolve(new Response(JSON.stringify({ log: "custom log", stages: [] })));
        }

        if (url.endsWith("/api/pipeline/run") && options.method === "POST") {
          return Promise.resolve(
            new Response(
              JSON.stringify({
                job_id: "job-new",
                dataset_kind: "snapshot",
                mode: "custom",
                requested_stages: ["generate_snapshot"],
                status: "queued",
                created_at: "2026-03-31T00:10:00+00:00",
                current_stage_name: null,
                log_path: "artifacts/api_runs/job-new",
                stages: [],
              }),
            ),
          );
        }

        return Promise.reject(new Error(`Unhandled fetch: ${url}`));
      });

    render(<App />);

    fireEvent.click(screen.getByRole("button", { name: /^custom$/i }));
    fireEvent.click((await screen.findByText("job-cust")).closest("button"));
    expect(await screen.findByText("custom log")).toBeInTheDocument();
    expect(
      await screen.findByRole("button", { name: /start new staged run/i }),
    ).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /start new staged run/i }));

    expect(screen.queryByText("custom log")).not.toBeInTheDocument();
    expect(
      await screen.findByText(/Starting a staged run with: generate_snapshot/i),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/Select a staged run below to continue it later\./i),
    ).toBeInTheDocument();

    fireEvent.click(
      screen.getByRole("button", { name: /run generate snapshot/i }),
    );

    await waitFor(() => {
      const runCall = fetchMock.mock.calls.find(([url]) =>
        String(url).includes("/api/pipeline/run"),
      );
      expect(runCall).toBeTruthy();
      const payload = JSON.parse(runCall[1].body);
      expect(payload.mode).toBe("custom");
      expect(payload.stages).toEqual(["generate_snapshot"]);
      expect(payload.source_job_id).toBeNull();
      expect(payload.force_new_run).toBe(true);
    });
  });

  it("keeps a running staged run selected and locks stage continuation until it finishes", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
      const url = String(input);

      if (url.includes("/api/dashboard/summary/temporal")) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              dataset_kind: "temporal",
              evaluation_metrics: { models: [] },
              artifacts: {
                models_dir: "artifacts/models_temporal",
                best_model: "artifacts/models_temporal/best_model.joblib",
                evaluation_metrics: "artifacts/metrics_temporal/evaluation.json",
              },
            }),
          ),
        );
      }

      if (url.includes("/api/artifacts/temporal")) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              dataset_kind: "temporal",
              artifacts: {
                models_dir: [],
                train_metrics: [],
                plots_dir: [],
                reports_dir: [],
                explain_dir: [],
              },
            }),
          ),
        );
      }

      if (url.includes("/api/dashboard/summary/snapshot")) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              dataset_kind: "snapshot",
              evaluation_metrics: { models: [] },
              artifacts: {
                models_dir: "artifacts/models",
                best_model: "artifacts/models/best_model.joblib",
                evaluation_metrics: "artifacts/metrics/evaluation.json",
              },
            }),
          ),
        );
      }

      if (url.includes("/api/artifacts/snapshot")) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              dataset_kind: "snapshot",
              artifacts: {
                models_dir: [],
                train_metrics: [],
                plots_dir: [],
                reports_dir: [],
                explain_dir: [],
              },
            }),
          ),
        );
      }

      if (url.endsWith("/api/pipeline/jobs")) {
        return Promise.resolve(
          new Response(
            JSON.stringify([
              {
                job_id: "job-temp-run",
                dataset_kind: "temporal",
                mode: "custom",
                requested_stages: [
                  "generate_sequence",
                  "build_temporal_features",
                  "train_temporal",
                ],
                status: "running",
                created_at: "2026-04-02T20:00:00+00:00",
                current_stage_name: "build_temporal_features",
                log_path: "artifacts/api_runs/job-temp-run",
                stages: [
                  {
                    stage_id: "job-temp-run:generate_sequence",
                    stage_name: "generate_sequence",
                    stage_order: 0,
                    status: "completed",
                    command: ["python"],
                    log_path: "artifacts/api_runs/job-temp-run/01_generate_sequence.log",
                  },
                  {
                    stage_id: "job-temp-run:build_temporal_features",
                    stage_name: "build_temporal_features",
                    stage_order: 1,
                    status: "running",
                    command: ["python"],
                    log_path: "artifacts/api_runs/job-temp-run/02_build_temporal_features.log",
                  },
                  {
                    stage_id: "job-temp-run:train_temporal",
                    stage_name: "train_temporal",
                    stage_order: 2,
                    status: "queued",
                    command: ["python"],
                    log_path: "artifacts/api_runs/job-temp-run/03_train_temporal.log",
                  },
                ],
              },
            ]),
          ),
        );
      }

      if (url.endsWith("/api/pipeline/jobs/job-temp-run/log")) {
        return Promise.resolve(
          new Response(JSON.stringify({ log: "temporal staged log", stages: [] })),
        );
      }

      return Promise.reject(new Error(`Unhandled fetch: ${url}`));
    });

    render(<App />);

    fireEvent.click(screen.getByRole("button", { name: /^temporal$/i }));
    fireEvent.click(screen.getByRole("button", { name: /^custom$/i }));
    fireEvent.click((await screen.findByText("job-temp")).closest("button"));

    await waitFor(() => {
      expect(
        screen.getByText(/Staged run job-temp is running\./i),
      ).toBeInTheDocument();
      expect(
        screen.getByText(/Wait for build_temporal_features to finish before continuing/i),
      ).toBeInTheDocument();
      expect(
        screen.getByRole("button", { name: /start new staged run/i }),
      ).toBeInTheDocument();
      expect(checkboxFor(/generate_sequence/i)).toBeDisabled();
      expect(checkboxFor(/build_temporal_features/i)).toBeDisabled();
      const nextStageCheckbox = checkboxFor(/train_temporal/i);
      expect(nextStageCheckbox).toBeChecked();
      expect(nextStageCheckbox).toBeDisabled();
      expect(screen.getByLabelText(/models/i)).toBeDisabled();
      expect(screen.getByLabelText(/scoring/i)).toBeDisabled();
      expect(
        screen.getByRole("button", { name: /run train temporal/i }),
      ).toBeDisabled();
    });
  });

  it("does not offer stage reruns when a staged run is already complete", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation((input) => {
      const url = String(input);

      if (url.includes("/api/dashboard/summary/snapshot")) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              dataset_kind: "snapshot",
              evaluation_metrics: { models: [] },
              artifacts: {
                models_dir: "artifacts/models",
                best_model: "artifacts/models/best_model.joblib",
                evaluation_metrics: "artifacts/metrics/evaluation.json",
              },
            }),
          ),
        );
      }

      if (url.includes("/api/artifacts/snapshot")) {
        return Promise.resolve(
          new Response(
            JSON.stringify({
              dataset_kind: "snapshot",
              artifacts: {
                models_dir: [],
                train_metrics: [],
                plots_dir: [],
                reports_dir: [],
                explain_dir: [],
              },
            }),
          ),
        );
      }

      if (url.endsWith("/api/pipeline/jobs")) {
        return Promise.resolve(
          new Response(
            JSON.stringify([
              {
                job_id: "job-complete",
                dataset_kind: "snapshot",
                mode: "custom",
                requested_stages: [
                  "generate_snapshot",
                  "train_snapshot",
                  "evaluate_snapshot",
                  "explain_snapshot",
                ],
                status: "completed",
                created_at: "2026-03-31T00:00:00+00:00",
                finished_at: "2026-03-31T00:10:00+00:00",
                current_stage_name: null,
                log_path: "artifacts/api_runs/job-complete",
                stages: [
                  {
                    stage_id: "job-complete:generate_snapshot",
                    stage_name: "generate_snapshot",
                    stage_order: 0,
                    status: "completed",
                    command: ["python", "-m", "incident_intelligence.cli.generator"],
                    log_path: "artifacts/api_runs/job-complete/01_generate_snapshot.log",
                  },
                  {
                    stage_id: "job-complete:train_snapshot",
                    stage_name: "train_snapshot",
                    stage_order: 1,
                    status: "completed",
                    command: ["python", "-m", "incident_intelligence.cli.train"],
                    log_path: "artifacts/api_runs/job-complete/02_train_snapshot.log",
                  },
                  {
                    stage_id: "job-complete:evaluate_snapshot",
                    stage_name: "evaluate_snapshot",
                    stage_order: 2,
                    status: "completed",
                    command: ["python", "-m", "incident_intelligence.cli.evaluate"],
                    log_path: "artifacts/api_runs/job-complete/03_evaluate_snapshot.log",
                  },
                  {
                    stage_id: "job-complete:explain_snapshot",
                    stage_name: "explain_snapshot",
                    stage_order: 3,
                    status: "completed",
                    command: ["python", "-m", "incident_intelligence.cli.explain"],
                    log_path: "artifacts/api_runs/job-complete/04_explain_snapshot.log",
                  },
                ],
              },
            ]),
          ),
        );
      }

      if (url.endsWith("/api/pipeline/jobs/job-complete/log")) {
        return Promise.resolve(new Response(JSON.stringify({ log: "done", stages: [] })));
      }

      return Promise.reject(new Error(`Unhandled fetch: ${url}`));
    });

    render(<App />);

    fireEvent.click((await screen.findByText("job-comp")).closest("button"));

    expect(
      await screen.findByText(/Staged run .* is complete\./i),
    ).toBeInTheDocument();
    expect(
      screen.getByText(/There are no remaining stages to run\./i),
    ).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: /run (full|custom|generate|train|evaluate|explain)/i })).not.toBeInTheDocument();
    expect(screen.getByLabelText(/generate_snapshot/i)).toBeDisabled();
    expect(screen.getByLabelText(/train_snapshot/i)).toBeDisabled();
    expect(screen.getByLabelText(/evaluate_snapshot/i)).toBeDisabled();
    expect(screen.getByLabelText(/explain_snapshot/i)).toBeDisabled();
  });
});
