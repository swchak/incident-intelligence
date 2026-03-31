import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import App from "./App";

describe("App", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("renders dashboard data and can start a job", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockImplementation((input, options = {}) => {
        const url = String(input);
        if (url.endsWith("/api/dashboard/summary/snapshot")) {
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
                        recall_macro: 0.83
                      }
                    }
                  ]
                },
                artifacts: {
                  models_dir: "/app/artifacts/models",
                  best_model: "/app/artifacts/models/best_model.joblib",
                  evaluation_metrics: "/app/artifacts/metrics/evaluation.json"
                }
              })
            )
          );
        }
        if (url.endsWith("/api/artifacts/snapshot")) {
          return Promise.resolve(
            new Response(
              JSON.stringify({
                dataset_kind: "snapshot",
                artifacts: {
                  train_metrics: [{ path: "artifacts/metrics/train_val_results.json", type: "file" }],
                  plots_dir: [{ path: "artifacts/plots/confusion_matrix.png", type: "file" }],
                  reports_dir: [{ path: "artifacts/reports/report.md", type: "file" }],
                  explain_dir: []
                }
              })
            )
          );
        }
        if (url.endsWith("/api/pipeline/jobs")) {
          return Promise.resolve(new Response(JSON.stringify([])));
        }
        if (url.endsWith("/api/pipeline/run") && options.method === "POST") {
          return Promise.resolve(
            new Response(
              JSON.stringify({
                job_id: "job-123",
                dataset_kind: "snapshot",
                status: "queued"
              })
            )
          );
        }
        if (url.endsWith("/api/pipeline/jobs/job-123/log")) {
          return Promise.resolve(new Response(JSON.stringify({ log: "hello log" })));
        }

        return Promise.reject(new Error(`Unhandled fetch: ${url}`));
      });

    render(<App />);

    expect(
      await screen.findByText(
        /Incident root cause modeling - from synthetic telemetry and temporal features to explainable root-cause results\./i
      )
    ).toBeInTheDocument();
    expect(await screen.findAllByText("0.8500")).toHaveLength(2);
    expect(screen.getAllByText("artifacts/plots/confusion_matrix.png").length).toBeGreaterThan(0);
    fireEvent.click(screen.getByRole("button", { name: /run snapshot pipeline/i }));

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        expect.stringContaining("/api/pipeline/run"),
        expect.objectContaining({ method: "POST" })
      );
    });
  });
});
