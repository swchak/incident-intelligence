from incident_intelligence.modeling.explain import ExplainConfig, run_explainability


def main() -> None:
    cfg = ExplainConfig(
        label_col="root_cause_label",
        out_dir="artifacts/explain",
        background_n=100,
        explain_n=200,
        kernel_bg=40,
        kernel_nsamples=80,
        perm_repeats=10,
        random_state=42,
        top_k=20,
    )

    results = run_explainability(
        data_path="data/processed/incident_root_cause_eval.csv",
        cfg=cfg,
        models_dir="artifacts/models",
    )

    print(f"Generated explainability for {len(results['models'])} model(s).")
    print(f"Artifacts saved to: {cfg.out_dir}")


if __name__ == "__main__":
    main()