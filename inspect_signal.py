from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent


def main():
    data = pd.read_csv(PROJECT_DIR / "data" / "train-test.csv")
    data["date"] = pd.to_datetime(data["date"], errors="raise")

    train = data.loc[
        data["date"] < pd.Timestamp("2025-09-01")
    ].copy()

    train["rate_per_mile"] = (
        train["posted_rate"] / train["distance"]
    )

    columns = [
        "quote_signal",
        "rate_per_mile",
        "posted_rate",
        "distance",
    ]

    print("Training correlations:")
    print(train[columns].corr().round(3).to_string())

    print("\nSignal and rate-per-mile examples:")
    print(
        train[
            ["quote_signal", "rate_per_mile", "distance", "posted_rate"]
        ].head(10).round(4).to_string(index=False)
    )

    print("\nSignal relationship by training month:")
    monthly_results = []

    for month, group in train.groupby(
        train["date"].dt.to_period("M")
    ):
        monthly_results.append({
            "month": str(month),
            "rows": len(group),
            "signal_mean": group["quote_signal"].mean(),
            "rate_per_mile_median": group["rate_per_mile"].median(),
            "correlation": group["quote_signal"].corr(
                group["rate_per_mile"]
            ),
        })

    print(
        pd.DataFrame(monthly_results)
        .round(3)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()