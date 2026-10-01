from pathlib import Path

import numpy as np
import pandas as pd

from model import build_model


PROJECT_DIR = Path(__file__).resolve().parent


def main():
    data = pd.read_csv(PROJECT_DIR / "data" / "train-test.csv")
    data["date"] = pd.to_datetime(data["date"], errors="raise")

    if data["date"].isna().any():
        raise ValueError("Missing dates found")

    cutoff = pd.Timestamp("2025-09-01")
    train = data.loc[data["date"] < cutoff].copy()
    holdout = data.loc[data["date"] >= cutoff].copy()

    # Separate inputs from the target
    X_train = train.drop(columns="posted_rate")
    y_train = train["posted_rate"]

    X_holdout = holdout.drop(columns="posted_rate")
    y_holdout = holdout["posted_rate"].to_numpy()

    # Model without weekday features
    model = build_model()
    model.fit(X_train, y_train)

    predictions = model.predict(X_holdout)
    errors = predictions - y_holdout

    print(f"Training rows: {len(train):,}")
    print(f"Holdout rows: {len(holdout):,}")

    print("\nWithout weekday features:")
    print(f"MAE: ${np.abs(errors).mean():,.2f}")
    print(f"RMSE: ${np.sqrt((errors ** 2).mean()):,.2f}")
    print(f"Non-positive predictions: {(predictions <= 0).sum()}")

    # Model with weekday features
    calendar_model = build_model(use_calendar=True)
    calendar_model.fit(X_train, y_train)

    calendar_predictions = calendar_model.predict(X_holdout)
    calendar_errors = calendar_predictions - y_holdout

    print("\nWith weekday features:")
    print(f"MAE: ${np.abs(calendar_errors).mean():,.2f}")
    print(f"RMSE: ${np.sqrt((calendar_errors ** 2).mean()):,.2f}")
    print(
        "Non-positive predictions:",
        (calendar_predictions <= 0).sum(),
    )


if __name__ == "__main__":
    main()