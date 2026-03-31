import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import App from "./App";

describe("App", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

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
    expect(await screen.findByText("0/0 stages")).toBeInTheDocument();
    expect(await screen.findByRole("button", { name: /delete job legacy-job-1/i })).toBeInTheDocument();
  });

  it("renders staged dashboard data and can start a custom run", async () => {
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
                requested_stages: ["train_snapshot", "evaluate_snapshot"],
                status: "queued",
                created_at: "2026-03-31T00:10:00+00:00",
                current_stage_name: null,
                log_path: "artifacts/api_runs/job-2",
                stages: [
                  {
                    stage_id: "job-2:train_snapshot",
                    stage_name: "train_snapshot",
                    stage_order: 0,
                    status: "queued",
                    command: ["python", "-m", "incident_intelligence.cli.train"],
                    log_path: "artifacts/api_runs/job-2/01_train_snapshot.log",
                  },
                  {
                    stage_id: "job-2:evaluate_snapshot",
                    stage_name: "evaluate_snapshot",
                    stage_order: 1,
                    status: "queued",
                    command: ["python", "-m", "incident_intelligence.cli.evaluate"],
                    log_path: "artifacts/api_runs/job-2/02_evaluate_snapshot.log",
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
    expect(await screen.findByText("2/2 stages")).toBeInTheDocument();
    expect(await screen.findByText("Last successful stage")).toBeInTheDocument();
    expect(await screen.findAllByText("train_snapshot")).not.toHaveLength(0);
    expect(
      await screen.findByRole("button", {
        name: /run next stage: evaluate_snapshot/i,
      }),
    ).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: /custom/i }));
    fireEvent.click(screen.getByLabelText("generate_snapshot"));
    fireEvent.click(screen.getByLabelText("explain_snapshot"));
    fireEvent.click(
      screen.getByRole("button", { name: /run custom snapshot stages/i }),
    );

    await waitFor(() => {
      const runCall = fetchMock.mock.calls.find(([url]) =>
        String(url).includes("/api/pipeline/run"),
      );
      expect(runCall).toBeTruthy();
      const payload = JSON.parse(runCall[1].body);
      expect(payload.mode).toBe("custom");
      expect(payload.stages).toEqual(["train_snapshot", "evaluate_snapshot"]);
      expect(payload.source_job_id).toBe("job-1");
    });

    fireEvent.click(
      screen.getByRole("button", { name: /run next stage: evaluate_snapshot/i }),
    );

    await waitFor(() => {
      const runCalls = fetchMock.mock.calls.filter(([url]) =>
        String(url).includes("/api/pipeline/run"),
      );
      expect(runCalls).toHaveLength(2);
      const payload = JSON.parse(runCalls[1][1].body);
      expect(payload.mode).toBe("custom");
      expect(payload.stages).toEqual(["evaluate_snapshot"]);
      expect(payload.source_job_id).toBe("job-1");
    });
  });
});
