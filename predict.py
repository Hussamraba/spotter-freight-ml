from pathlib import Path

import numpy as np
import pandas as pd

from model import build_boosting_model, build_signal_model


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = PROJECT_DIR / "outputs"


def check_predictions(values, label):
    if not np.isfinite(values).all():
        raise ValueError(f"{label}: non-finite predictions")

    if (values <= 0).any():
        raise ValueError(f"{label}: non-positive predictions")


def main():
    development = pd.read_csv(DATA_DIR / "train-test.csv")
    validation = pd.read_csv(DATA_DIR / "validation.csv")
    template = pd.read_csv(
        DATA_DIR / "validation-predictions-template.csv"
    )
    december = pd.read_csv(DATA_DIR / "december-chart-inputs.csv")

    for name, frame in [
        ("validation", validation),
        ("template", template),
    ]:
        if frame["load_id"].isna().any():
            raise ValueError(f"{name}: missing load IDs")
        if frame["load_id"].duplicated().any():
            raise ValueError(f"{name}: duplicate load IDs")

    if len(validation) != 12000 or len(template) != 12000:
        raise ValueError("Expected 12,000 validation and template rows")

    if set(validation["load_id"]) != set(template["load_id"]):
        raise ValueError("Validation and template IDs do not match")

    X = development.drop(columns="posted_rate")
    y = development["posted_rate"]

    # Train the primary model on all labeled development data
    print("Training primary model with signal...", flush=True)
    primary_model = build_signal_model()
    primary_model.fit(X, y)

    validation_predictions = primary_model.predict(validation)
    check_predictions(validation_predictions, "Validation")

    # Match predictions by ID, preserving the template order
    predictions_by_id = pd.Series(
        validation_predictions,
        index=validation["load_id"],
    )

    submission = template[["load_id"]].copy()
    submission["predicted_rate"] = submission["load_id"].map(
        predictions_by_id
    )
    check_predictions(
        submission["predicted_rate"].to_numpy(),
        "Mapped submission",
    )

    # December has no quote_signal, so use the evaluated fallback
    print("Training December fallback without signal...", flush=True)
    fallback_model = build_boosting_model()
    fallback_model.fit(X, y)

    december_output = december.copy()
    december_output["predicted_rate"] = fallback_model.predict(
        december
    )
    check_predictions(
        december_output["predicted_rate"].to_numpy(),
        "December",
    )

    OUTPUT_DIR.mkdir(exist_ok=True)

    submission.to_csv(
        OUTPUT_DIR / "validation_predictions.csv",
        index=False,
    )
    december_output.to_csv(
        OUTPUT_DIR / "december_predictions.csv",
        index=False,
    )

    print(f"\nSaved {len(submission):,} validation predictions.")
    print(f"Saved {len(december_output)} December predictions.")

    print("\nValidation prediction summary:")
    print(submission["predicted_rate"].describe().round(2))

    print("\nDecember preview:")
    print(
        december_output[["date", "predicted_rate"]]
        .head()
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()