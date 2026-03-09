from incident_intelligence.modeling.train import TrainValidateConfig, run_training


def main() -> None:
    cfg = TrainValidateConfig(
        label_col="root_cause_label",
        models_out_dir="artifacts/models",
        metrics_out_json="artifacts/metrics/train_val_results.json",
        leaderboard_out_csv="artifacts/metrics/leaderboard_val.csv",
        best_model_out="artifacts/models/best_model.joblib",
    )

    result = run_training(
        train_path="data/processed/incident_root_cause_train.csv",
        val_path="data/processed/incident_root_cause_val.csv",
        cfg=cfg,
    )

    best = result["best_model"]
    print(
        f"Training complete. Best model: {best['model_name']} "
        f"(val_f1_macro={best['val_f1_macro']:.4f}, "
        f"val_accuracy={best['val_accuracy']:.4f})"
    )
    print(f"Models saved to: {cfg.models_out_dir}")
    print(f"Metrics saved to: {cfg.metrics_out_json}")
    print(f"Leaderboard saved to: {cfg.leaderboard_out_csv}")


if __name__ == "__main__":
    main()