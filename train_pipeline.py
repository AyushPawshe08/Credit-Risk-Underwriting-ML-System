from pathlib import Path

from credit_underwriting.pipeline import save_model_bundle, train_model_bundle


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "credit_risk_dataset.csv"
ARTIFACT_PATH = ROOT / "artifacts" / "model_bundle.joblib"


def main() -> None:
    bundle = train_model_bundle(DATA_PATH)
    save_model_bundle(bundle, ARTIFACT_PATH)

    print(f"Saved model bundle to: {ARTIFACT_PATH}")
    print(f"Best model: {bundle['best_model_name']}")
    print("\nBenchmark results:")
    print(bundle["benchmark_results"].to_string(index=False))


if __name__ == "__main__":
    main()
