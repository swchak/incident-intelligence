import argparse
from pathlib import Path

from sklearn.model_selection import train_test_split

from incident_intelligence.settings import SETTINGS, load_class_config
from incident_intelligence.data.generator import validate_configs, generate_dataset

ROOT_CAUSE_PROBS = {
    "bad_deployment": 0.2,
    "external_dependency_failure": 0.2,
    "traffic_spike": 0.15,
    "memory_leak": 0.15,
    "cpu_saturation": 0.15,
    "normal": 0.15,
}


def stratified_splits(df, label_col: str, seed: int, train_size: float, val_size: float):
    """
    Splits df into train/val/eval with stratification on label_col.

    train_size: fraction of total in train
    val_size: fraction of total in val
    eval_size is computed as (1 - train_size - val_size)
    """
    if train_size <= 0 or train_size >= 1:
        raise ValueError("train_size must be between 0 and 1")

    if val_size <= 0 or val_size >= 1:
        raise ValueError("val_size must be between 0 and 1")

    eval_size = 1.0 - train_size - val_size
    if eval_size <= 0:
        raise ValueError("train_size + val_size must be < 1.0")

    train_df, temp_df = train_test_split(
        df,
        test_size=(1.0 - train_size),
        stratify=df[label_col],
        random_state=seed,
    )

    # Split the remaining temp into val and eval:
    # val fraction within temp = val_size / (val_size + eval_size)
    val_share_of_temp = val_size / (val_size + eval_size)

    val_df, eval_df = train_test_split(
        temp_df,
        test_size=(1.0 - val_share_of_temp),
        stratify=temp_df[label_col],
        random_state=seed,
    )

    return train_df, val_df, eval_df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-samples", type=int, default=4000)
    p.add_argument("--seed", type=int, default=42)

    # Where to write files (relative to SETTINGS.data_dir)
    p.add_argument("--raw-out", type=str, default="raw/incidents_raw.csv")
    p.add_argument("--processed-dir", type=str, default="processed")

    # Splits
    p.add_argument("--train-size", type=float, default=0.70)
    p.add_argument("--val-size", type=float, default=0.15)

    p.add_argument("--label-col", type=str, default="root_cause_label")
    args = p.parse_args()

    class_config = load_class_config()
    validate_configs(class_config)

    # 1) Generate full unsplit dataset
    df = generate_dataset(args.n_samples, ROOT_CAUSE_PROBS, class_config, seed=args.seed)

    # 2) Save raw
    raw_path = SETTINGS.data_dir / args.raw_out
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(raw_path, index=False)

    # 3) Create splits
    train_df, val_df, eval_df = stratified_splits(
        df,
        label_col=args.label_col,
        seed=args.seed,
        train_size=args.train_size,
        val_size=args.val_size,
    )

    # 4) Save processed splits
    processed_dir = SETTINGS.data_dir / args.processed_dir
    processed_dir.mkdir(parents=True, exist_ok=True)

    train_path = processed_dir / "incident_root_cause_train.csv"
    val_path = processed_dir / "incident_root_cause_val.csv"
    eval_path = processed_dir / "incident_root_cause_eval.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    eval_df.to_csv(eval_path, index=False)

    # 5) Print summary
    print(f"Wrote raw:      {raw_path}  (n={len(df)})")
    print(f"Wrote train:    {train_path}  (n={len(train_df)})")
    print(f"Wrote val:      {val_path}  (n={len(val_df)})")
    print(f"Wrote eval:     {eval_path}  (n={len(eval_df)})")

    print("\nRaw class distribution:\n", df[args.label_col].value_counts(normalize=True))
    print("\nTrain class distribution:\n", train_df[args.label_col].value_counts(normalize=True))
    print("\nVal class distribution:\n", val_df[args.label_col].value_counts(normalize=True))
    print("\nEval class distribution:\n", eval_df[args.label_col].value_counts(normalize=True))


if __name__ == "__main__":
    main()