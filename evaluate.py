from pathlib import Path

import numpy as np
import pandas as pd

from model import (
    build_model,
    build_location_model,
    build_boosting_model,
    build_signal_model,
    build_seasonal_model,
)

PROJECT_DIR = Path(__file__).resolve().parent

WINDOWS = [
    ("2025-05-01", "2025-07-01"),
    ("2025-07-01", "2025-09-01"),
    ("2025-09-01", "2025-11-01"),
]


def main():
    data = pd.read_csv(PROJECT_DIR / "data" / "train-test.csv")
    data["date"] = pd.to_datetime(data["date"], errors="raise")

    if data["date"].isna().any():
        raise ValueError("Missing dates found")

    model_builders = {
        "Base": build_model,
        "Weekday": lambda: build_model(use_calendar=True),
        "Locations": build_location_model,
        "Boosting": build_boosting_model,
        "Boosting + signal": build_signal_model,
        "Locations + season": build_seasonal_model,
    }

    results = []

    for start, end in WINDOWS:
        train = data.loc[data["date"] < start]
        holdout = data.loc[
            (data["date"] >= start) & (data["date"] < end)
        ]

        X_train = train.drop(columns="posted_rate")
        y_train = train["posted_rate"]

        X_holdout = holdout.drop(columns="posted_rate")
        y_holdout = holdout["posted_rate"].to_numpy()

        for model_name, builder in model_builders.items():
            print(
                f"Training {model_name} | evaluation starts {start}",
                flush=True,
            )

            model = builder()
            model.fit(X_train, y_train)

            predictions = model.predict(X_holdout)
            errors = predictions - y_holdout

            results.append({
                "eval_start": start,
                "model": model_name,
                "rows": len(holdout),
                "mae": np.abs(errors).mean(),
                "rmse": np.sqrt((errors ** 2).mean()),
            })

    results = pd.DataFrame(results)

    print("\nTemporal validation results:")
    print(results.round(2).to_string(index=False))

    output_dir = PROJECT_DIR / "results"
    output_dir.mkdir(exist_ok=True)

    results.to_csv(
        output_dir / "temporal_validation.csv",
        index=False,
    )


if __name__ == "__main__":
    main()