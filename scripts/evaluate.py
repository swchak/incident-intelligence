from incident_intelligence.modeling.evaluate import EvalConfig, run_evaluation


def main() -> None:
    cfg = EvalConfig(
        label_col="root_cause_label",
        metrics_out="artifacts/metrics/evaluation.json",
        summary_csv_out="artifacts/metrics/evaluation_summary.csv",
    )

    results = run_evaluation(
        data_path="data/processed/incident_root_cause_eval.csv",
        cfg=cfg,
        models_dir="artifacts/models",
    )

    best = None
    for model_result in results["models"]:
        acc = model_result["metrics"].get("accuracy", -1)
        if best is None or acc > best["metrics"].get("accuracy", -1):
            best = model_result

    if best:
        print(f"Evaluated {len(results['models'])} model(s).")
        print(f"Best by accuracy: {best['model_name']} ({best['metrics']['accuracy']:.4f})")
        print(f"Metrics saved to: {cfg.metrics_out}")
        if cfg.summary_csv_out:
            print(f"Summary saved to: {cfg.summary_csv_out}")


if __name__ == "__main__":
    main()