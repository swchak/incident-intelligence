"""
pipeline.py

End-to-end pipeline orchestrator for the incident intelligence project.

Pipeline stages:
1. Generate synthetic dataset
2. Train models
3. Evaluate trained models
4. Produce explainability artifacts

This script is useful for running the full ML workflow in sequence.
"""

from __future__ import annotations

from incident_intelligence.config import (
    EvaluateCLIConfig,
    ExplainCLIConfig,
    GeneratorCLIConfig,
    TrainCLIConfig,
    load_config,
)
from incident_intelligence.data.generator import (
    GeneratorConfig,
    generate_and_save_datasets,
)
from incident_intelligence.modeling.evaluate import (
    EvalConfig,
    run_evaluation,
)
from incident_intelligence.modeling.explain import (
    ExplainConfig,
    run_explainability,
)

from incident_intelligence.modeling.train import (
    TrainValidateConfig,
    run_training,
)


def main() -> None:
    print("Running pipeline...\n")

    # --------------------------------------------------
    # Generate synthetic dataset
    # --------------------------------------------------
    gen_settings = load_config(GeneratorCLIConfig, "generator")
    gen_cfg = GeneratorConfig(
        n_samples=gen_settings.n_samples,
        seed=gen_settings.seed,
        raw_out=gen_settings.raw_out,
        processed_dir=gen_settings.processed_dir,
        train_size=gen_settings.train_size,
        val_size=gen_settings.val_size,
        label_col=gen_settings.label_col,
    )

    gen_result = generate_and_save_datasets(gen_cfg)

    print(f"[generate] raw:   {gen_result['raw_path']}")
    print(f"[generate] train: {gen_result['train_path']}")
    print(f"[generate] val:   {gen_result['val_path']}")
    print(f"[generate] eval:  {gen_result['eval_path']}")

    # --------------------------------------------------
    # Train models using generated dataset
    # --------------------------------------------------
    train_settings = load_config(TrainCLIConfig, "train")
    train_cfg = TrainValidateConfig(
        label_col=train_settings.label_col,
        models_out_dir=train_settings.models_out_dir,
        metrics_out_json=train_settings.metrics_out_json,
        leaderboard_out_csv=train_settings.leaderboard_out_csv,
        best_model_out=train_settings.best_model_out,
    )

    train_result = run_training(
        train_path=train_settings.train,
        val_path=train_settings.val,
        cfg=train_cfg,
    )

    print(f"[train] best model: {train_result['best_model']['model_name']}")
    print(f"[train] best model path: {train_result['best_model']['model_path']}")

    # --------------------------------------------------
    # Evaluate trained models
    # --------------------------------------------------
    eval_settings = load_config(EvaluateCLIConfig, "evaluate")
    eval_cfg = EvalConfig(
        label_col=eval_settings.label_col,
        metrics_out=eval_settings.metrics_out,
        summary_csv_out=eval_settings.summary_csv_out,
    )

    eval_result = run_evaluation(
        data_path=eval_settings.data,
        cfg=eval_cfg,
        model_path=eval_settings.model,
        models_dir=eval_settings.models_dir,
    )

    print(f"[evaluate] evaluated {len(eval_result['models'])} model(s)")

    # --------------------------------------------------
    # Generate explainability artifacts
    # --------------------------------------------------
    explain_settings = load_config(ExplainCLIConfig, "explain")
    explain_cfg = ExplainConfig(
        label_col=explain_settings.label_col,
        out_dir=explain_settings.out_dir,
        background_n=explain_settings.background_n,
        explain_n=explain_settings.explain_n,
        kernel_bg=explain_settings.kernel_bg,
        kernel_nsamples=explain_settings.kernel_nsamples,
        perm_repeats=explain_settings.perm_repeats,
        random_state=explain_settings.random_state,
        top_k=explain_settings.top_k,
    )

    explain_result = run_explainability(
        data_path=explain_settings.data,
        cfg=explain_cfg,
        model_path=explain_settings.model,
        models_dir=explain_settings.models_dir,
    )

    print(f"[explain] generated artifacts for {len(explain_result['models'])} model(s)")
    print(f"[explain] out dir: {explain_result['out_dir']}")

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()