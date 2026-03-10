from incident_intelligence.modeling.explain import ExplainConfig, run_local_explainability


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

    result = run_local_explainability(
        data_path="data/processed/incident_root_cause_eval.csv",
        model_path="artifacts/models/best_model.joblib",
        cfg=cfg,
        n_examples=3,
        top_k_classes=3,
        top_features_per_class=8,
    )

    print(f"Generated local explainability for {len(result['rows'])} example(s).")


if __name__ == "__main__":
    main()