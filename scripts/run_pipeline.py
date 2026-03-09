from scripts.generate_dataset import main as generate_main
from scripts.train import main as train_main
from scripts.evaluate import main as evaluate_main
from scripts.explain import main as explain_main


def main() -> None:
    print("=== Generating dataset ===")
    generate_main()

    print("\n=== Training models ===")
    train_main()

    print("\n=== Evaluating models ===")
    evaluate_main()

    print("\n=== Generating explainability ===")
    explain_main()

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()