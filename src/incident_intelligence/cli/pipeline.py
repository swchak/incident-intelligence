from incident_intelligence.cli.generator import main as generate
from incident_intelligence.cli.train import main as train
from incident_intelligence.cli.evaluate import main as evaluate
from incident_intelligence.cli.explain import main as explain


def main():
    print("Running pipeline...\n")

    generate()
    train()
    evaluate()
    explain()

    print("\nPipeline complete.")