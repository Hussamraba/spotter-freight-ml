from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent


def main():
    results = pd.read_csv(
        PROJECT_DIR / "results" / "temporal_validation.csv"
    )

    summary = []

    for model_name, group in results.groupby("model"):
        counts = group["rows"].to_numpy()
        maes = group["mae"].to_numpy()
        rmses = group["rmse"].to_numpy()

        summary.append({
            "model": model_name,
            "evaluated_rows": counts.sum(),
            "pooled_mae": np.average(maes, weights=counts),
            "pooled_rmse": np.sqrt(
                np.average(rmses ** 2, weights=counts)
            ),
            "worst_window_mae": maes.max(),
        })

    summary = pd.DataFrame(summary).sort_values("pooled_mae")

    print(summary.round(2).to_string(index=False))

    summary.to_csv(
        PROJECT_DIR / "results" / "model_summary.csv",
        index=False,
    )


if __name__ == "__main__":
    main()