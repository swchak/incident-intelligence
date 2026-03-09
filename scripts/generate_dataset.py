from incident_intelligence.data.generator import GeneratorConfig, generate_and_save_datasets


def main() -> None:
    cfg = GeneratorConfig(
        n_samples=4000,
        seed=42,
        label_col="root_cause_label",
        train_size=0.70,
        val_size=0.15,
        raw_out="raw/incidents_raw.csv",
        processed_dir="processed",
    )

    result = generate_and_save_datasets(cfg)

    print(f"Wrote raw:   {result['raw_path']}")
    print(f"Wrote train: {result['train_path']}")
    print(f"Wrote val:   {result['val_path']}")
    print(f"Wrote eval:  {result['eval_path']}")
    print(
        f"Rows -> raw={result['n_raw']}, "
        f"train={result['n_train']}, "
        f"val={result['n_val']}, "
        f"eval={result['n_eval']}"
    )


if __name__ == "__main__":
    main()