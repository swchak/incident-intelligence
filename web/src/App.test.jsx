import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";

import App from "./App";

function okJson(body) {
  return Promise.resolve({
    ok: true,
    json: async () => body
  });
}

describe("App", () => {
  beforeEach(() => {
    global.fetch = vi.fn((url, options) => {
      const target = String(url);

      if (target.endsWith("/api/dashboard/summary/snapshot")) {
        return okJson({
          dataset_kind: "snapshot",
          artifacts: {
            models_dir: "/tmp/models",
            best_model: "/tmp/models/best_model.joblib",
            evaluation_metrics: "/tmp/metrics/evaluation.json"
          },
          evaluation_metrics: {
            models: [
              {
                model_name: "Logistic_Regression_pipeline",
                metrics: {
                  accuracy: 0.81,
                  f1_macro: 0.79,
                  precision_macro: 0.8,
                  recall_macro: 0.78
                }
              }
            ]
          }
        });
      }

      if (target.endsWith("/api/dashboard/summary/temporal")) {
        return okJson({
          dataset_kind: "temporal",
          artifacts: {
            models_dir: "/tmp/models_temporal",
            best_model: "/tmp/models_temporal/best_model.joblib",
            evaluation_metrics: "/tmp/metrics_temporal/evaluation.json"
          },
          evaluation_metrics: {
            models: []
          }
        });
      }

      if (target.endsWith("/api/artifacts/snapshot") || target.endsWith("/api/artifacts/temporal")) {
        return okJson({
          artifacts: {
            train_metrics: [{ path: "artifacts/metrics/train_val_results.json" }],
            plots_dir: [],
            reports_dir: [],
            explain_dir: []
          }
        });
      }

      if (target.endsWith("/api/pipeline/jobs")) {
        if (options?.method === "POST") {
          return okJson({
            job_id: "job-1",
            status: "queued",
            command: ["python", "-m", "incident_intelligence.cli.pipeline"],
            dataset_kind: "snapshot",
            created_at: "2026-03-25T10:00:00Z",
            log_path: "/tmp/job-1.log"
          });
        }

        return okJson([
          {
            job_id: "job-1",
            status: "queued",
            command: ["python", "-m", "incident_intelligence.cli.pipeline"],
            dataset_kind: "snapshot",
            created_at: "2026-03-25T10:00:00Z",
            log_path: "/tmp/job-1.log"
          }
        ]);
      }

      if (target.endsWith("/api/pipeline/jobs/job-1/log")) {
        return okJson({ job_id: "job-1", log: "pipeline log output" });
      }

      throw new Error(`Unhandled fetch for ${target}`);
    });
  });

  afterEach(() => {
    cleanup();
    vi.restoreAllMocks();
  });

  it("renders snapshot summary and model metrics", async () => {
    render(<App />);

    expect(screen.getByText("Pipeline Demo Dashboard")).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText("Logistic_Regression_pipeline")).toBeInTheDocument();
    });

    expect(screen.getByText("0.8100")).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText("pipeline log output")).toBeInTheDocument();
    });
  });

  it("submits a pipeline run request", async () => {
    render(<App />);

    const runButton = await screen.findByRole("button", { name: "Run snapshot pipeline" });
    fireEvent.click(runButton);

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith(
        "/api/pipeline/run",
        expect.objectContaining({
          method: "POST",
          headers: { "Content-Type": "application/json" }
        })
      );
    });
  });

  it("switches to temporal dataset view", async () => {
    render(<App />);

    const temporalButton = screen.getByRole("button", { name: "temporal" });
    fireEvent.click(temporalButton);

    await waitFor(() => {
      expect(global.fetch).toHaveBeenCalledWith(
        "/api/dashboard/summary/temporal",
        undefined
      );
    });
  });
});
