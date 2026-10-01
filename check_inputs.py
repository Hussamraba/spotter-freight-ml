from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"


def main():
    development = pd.read_csv(DATA_DIR / "train-test.csv")
    validation = pd.read_csv(DATA_DIR / "validation.csv")
    template = pd.read_csv(
        DATA_DIR / "validation-predictions-template.csv"
    )
    december = pd.read_csv(DATA_DIR / "december-chart-inputs.csv")

    print("Row counts:")
    print(f"Development: {len(development):,}")
    print(f"Validation: {len(validation):,}")
    print(f"Template: {len(template):,}")
    print(f"December: {len(december):,}")

    for name, frame in [
        ("validation", validation),
        ("template", template),
    ]:
        if frame["load_id"].isna().any():
            raise ValueError(f"{name}: missing load IDs")
        if frame["load_id"].duplicated().any():
            raise ValueError(f"{name}: duplicate load IDs")

    if set(validation["load_id"]) != set(template["load_id"]):
        raise ValueError("Validation and template IDs do not match")

    print("\nValidation/template IDs match.")

    required = ["pickup", "delivery", "distance", "equipment", "weight"]

    for name, frame in [
        ("validation", validation),
        ("december", december),
    ]:
        missing = set(required) - set(frame.columns)
        if missing:
            raise ValueError(f"{name}: missing columns {sorted(missing)}")

        print(f"\n{name} missing values:")
        print(frame[required].isna().sum())

        for column in ["pickup", "delivery", "equipment"]:
            unseen = set(frame[column].dropna()) - set(
                development[column].dropna()
            )
            print(f"Unseen {column}: {sorted(unseen)}")

        distance = pd.to_numeric(frame["distance"], errors="coerce")
        if not (np.isfinite(distance) & (distance > 0)).all():
            raise ValueError(f"{name}: invalid distances")

        weight = pd.to_numeric(frame["weight"], errors="coerce")
        print(f"Non-positive weights: {(weight <= 0).sum()}")

    if "quote_signal" not in validation.columns:
        raise ValueError("Validation is missing quote_signal")

    signal = pd.to_numeric(validation["quote_signal"], errors="coerce")
    print("\nValidation signal missing/non-numeric:", signal.isna().sum())
    print("Validation signal infinite:", np.isinf(signal).sum())


if __name__ == "__main__":
    main()