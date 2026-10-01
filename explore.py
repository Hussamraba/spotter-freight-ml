from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone
from features import make_interaction_features

# Locate the data relative to this script
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"

# Load the labeled development data
df = pd.read_csv(DATA_DIR / "train-test.csv")

print("Dataset shape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head().to_string())

print("\nColumn information:")
df.info()

print("\nMissing values per column:")
print(df.isna().sum())

# Convert date strings into actual dates
df["date"] = pd.to_datetime(df["date"], errors="coerce")

print("\nInvalid or missing dates:")
print(df["date"].isna().sum())

print("\nDevelopment date range:")
print(df["date"].min(), "to", df["date"].max())

print("\nRows per month:")
print(df.groupby(df["date"].dt.to_period("M")).size())

print("\nDuplicate load IDs:")
print(df["load_id"].duplicated().sum())

print("\nFully duplicated rows:")
print(df.duplicated().sum())

print("\nNumeric summary:")
numeric_columns = [
    "distance", "weight", "market_index", "quote_signal", "posted_rate"
]
print(df[numeric_columns].describe().round(2).to_string())

# Inspect the dates of the final prediction set
validation = pd.read_csv(DATA_DIR / "validation.csv")
validation_dates = pd.to_datetime(validation["date"], errors="coerce")

print("\nFinal prediction date range:")
print(validation_dates.min(), "to", validation_dates.max())

print("\nInvalid or missing final prediction dates:")
print(validation_dates.isna().sum())

# Split chronologically before learning any preprocessing statistics
cutoff = pd.Timestamp("2025-09-01")

train = df.loc[df["date"] < cutoff].copy()
holdout = df.loc[df["date"] >= cutoff].copy()

print("\nTemporal split:")
for name, subset in [("train", train), ("holdout", holdout)]:
    print(
        f"{name}: {len(subset):,} rows | "
        f"{subset['date'].min().date()} to "
        f"{subset['date'].max().date()}"
    )

# Check that training ends before evaluation begins
assert train["date"].max() < holdout["date"].min()

print("\nWeight quality by split:")
for name, subset in [("train", train), ("holdout", holdout)]:
    print(
        f"{name}: "
        f"missing={subset['weight'].isna().sum()}, "
        f"non_positive={(subset['weight'] <= 0).sum()}"
    )

print("\nExamples of invalid training weights:")
print(
    train.loc[
        train["weight"] <= 0,
        ["load_id", "equipment", "distance", "weight", "posted_rate"]
    ].head(10).to_string(index=False)
)

# Replace invalid weights with missing values in both splits
for subset in (train, holdout):
    subset["weight"] = subset["weight"].mask(
        subset["weight"] <= 0
    )

    # Preserve information about which values need imputation
    subset["weight_missing"] = subset["weight"].isna().astype(int)
    subset["market_index_missing"] = (
        subset["market_index"].isna().astype(int)
    )

# Learn replacement values from the training split ONLY
columns_to_fill = ["weight", "market_index"]
training_medians = train[columns_to_fill].median()

print("\nMedians learned from training:")
print(training_medians)

# Apply the same training medians to both splits
train[columns_to_fill] = train[columns_to_fill].fillna(training_medians)
holdout[columns_to_fill] = holdout[columns_to_fill].fillna(training_medians)

print("\nMissing values after imputation:")
for name, subset in [("train", train), ("holdout", holdout)]:
    print(name)
    print(subset[columns_to_fill].isna().sum())

print("\nTraining weight examples after imputation:")
print(
    train.loc[
        train["weight_missing"] == 1,
        ["load_id", "weight", "weight_missing"]
    ].head().to_string(index=False)
)

# Actual prices for the evaluation period
y_true = holdout["posted_rate"].to_numpy()

# Baseline 1: predict the training median price for every load
median_rate = train["posted_rate"].median()
constant_predictions = np.full(len(holdout), median_rate)

# Baseline 2: distance multiplied by the training median rate per mile
training_rate_per_mile = train["posted_rate"] / train["distance"]
median_rate_per_mile = training_rate_per_mile.median()

distance_predictions = (
    holdout["distance"] * median_rate_per_mile
).to_numpy()


def evaluate_predictions(name, actual, predicted):
    errors = actual - predicted

    mae = np.abs(errors).mean()
    rmse = np.sqrt((errors ** 2).mean())

    print(f"{name}: MAE=${mae:,.2f} | RMSE=${rmse:,.2f}")


print("\nBaseline values learned from training:")
print(f"Median load price: ${median_rate:,.2f}")
print(f"Median rate per mile: ${median_rate_per_mile:.4f}")

print("\nHoldout baseline results:")
evaluate_predictions("Constant median", y_true, constant_predictions)
evaluate_predictions("Distance baseline", y_true, distance_predictions)

# Keep each prediction next to its actual shipment
baseline_errors = holdout[
    ["load_id", "date", "equipment", "distance", "posted_rate"]
].copy()

baseline_errors["predicted_rate"] = distance_predictions
baseline_errors["error"] = (
    baseline_errors["predicted_rate"] - baseline_errors["posted_rate"]
)
baseline_errors["absolute_error"] = baseline_errors["error"].abs()

print("\nLargest baseline errors:")
print(
    baseline_errors.nlargest(10, "absolute_error")
    .round(2)
    .to_string(index=False)
)

print("\nBaseline performance by equipment:")
print(
    baseline_errors.groupby("equipment").agg(
        loads=("absolute_error", "size"),
        mae=("absolute_error", "mean"),
        mean_error=("error", "mean"),
    ).round(2).to_string()
)

print("\nBaseline performance by month:")
print(
    baseline_errors.groupby(
        baseline_errors["date"].dt.to_period("M")
    ).agg(
        loads=("absolute_error", "size"),
        mae=("absolute_error", "mean"),
        mean_error=("error", "mean"),
    ).round(2).to_string()
)

# Learn a separate median rate per mile for each equipment type
equipment_rates = (
    train.assign(
        rate_per_mile=train["posted_rate"] / train["distance"]
    )
    .groupby("equipment")["rate_per_mile"]
    .median()
)

print("\nTraining median rate per mile by equipment:")
print(equipment_rates.round(4))

# Match each holdout load to its equipment's training rate
holdout_rates = holdout["equipment"].map(equipment_rates)

# Fall back to the overall training median for unseen equipment types
holdout_rates = holdout_rates.fillna(median_rate_per_mile)

equipment_predictions = (
    holdout["distance"] * holdout_rates
).to_numpy()

print("\nBaseline comparison:")
evaluate_predictions("Distance only", y_true, distance_predictions)
evaluate_predictions(
    "Distance + equipment", y_true, equipment_predictions
)

# Select model inputs and the target
features = ["distance", "weight", "equipment"]

X_train = train[features]
y_train = train["posted_rate"]

X_holdout = holdout[features]

# Keep numeric columns and encode the equipment category
preprocessor = ColumnTransformer(
    transformers=[
        ("numeric", "passthrough", ["distance", "weight"]),
        (
            "equipment",
            OneHotEncoder(
                categories=[["Dry Van", "Flatbed", "Reefer"]],
                drop="first",
                handle_unknown="ignore",
                sparse_output=False,
            ),
            ["equipment"],
        ),
    ]
)

# Combine preprocessing and regression in one object
linear_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression()),
    ]
)

# Learn only from the training period
linear_model.fit(X_train, y_train)

# Predict prices for the later evaluation period
linear_predictions = linear_model.predict(X_holdout)

print("\nModel comparison:")
evaluate_predictions(
    "Distance + equipment baseline", y_true, equipment_predictions
)
evaluate_predictions(
    "Linear regression", y_true, linear_predictions
)

print("\nLearned coefficients:")
feature_names = (
    linear_model.named_steps["preprocessor"].get_feature_names_out()
)
coefficients = linear_model.named_steps["regressor"].coef_

print(pd.Series(coefficients, index=feature_names).round(4))
print(
    "Intercept:",
    round(linear_model.named_steps["regressor"].intercept_, 2),
)

print("\nNon-positive linear predictions:")
print((linear_predictions <= 0).sum())

# Compare performance on seen and unseen periods
linear_train_predictions = linear_model.predict(X_train)

print("\nLinear regression: train vs holdout:")
evaluate_predictions(
    "Train",
    y_train.to_numpy(),
    linear_train_predictions,
)
evaluate_predictions(
    "Holdout",
    y_true,
    linear_predictions,
)

# Inspect how large the typical and extreme holdout errors are
absolute_errors = np.abs(y_true - linear_predictions)

print("\nHoldout absolute error percentiles:")
for percentile in [50, 90, 95, 99]:
    value = np.percentile(absolute_errors, percentile)
    print(f"P{percentile}: ${value:,.2f}")
    


X_train_interactions = make_interaction_features(train)
X_holdout_interactions = make_interaction_features(holdout)

interaction_model = LinearRegression()
interaction_model.fit(X_train_interactions, y_train)

interaction_predictions = interaction_model.predict(
    X_holdout_interactions
)

print("\nInteraction model comparison:")
evaluate_predictions(
    "Original linear model", y_true, linear_predictions
)
evaluate_predictions(
    "Linear model with interactions", y_true, interaction_predictions
)

print("\nInteraction coefficients:")
print(
    pd.Series(
        interaction_model.coef_,
        index=X_train_interactions.columns,
    ).round(4)
)

print("\nNon-positive interaction predictions:")
print((interaction_predictions <= 0).sum())

def make_market_features(frame):
    features = make_interaction_features(frame)

    # Center the index around 1 for easier interpretation
    market_change = frame["market_index"] - 1.0

    features["market_change"] = market_change
    features["distance_market"] = (
        frame["distance"] * market_change
    )
    features["market_index_missing"] = frame["market_index_missing"]

    return features


X_train_market = make_market_features(train)
X_holdout_market = make_market_features(holdout)

market_model = LinearRegression()
market_model.fit(X_train_market, y_train)

market_predictions = market_model.predict(X_holdout_market)

print("\nMarket feature comparison:")
evaluate_predictions(
    "Interactions only", y_true, interaction_predictions
)
evaluate_predictions(
    "Interactions + market", y_true, market_predictions
)

print("\nMarket model train performance:")
evaluate_predictions(
    "Train",
    y_train.to_numpy(),
    market_model.predict(X_train_market),
)

print("\nMarket coefficients:")
print(
    pd.Series(
        market_model.coef_,
        index=X_train_market.columns,
    ).round(4)
)

print("\nNon-positive market predictions:")
print((market_predictions <= 0).sum())

# Compare market-index distributions before imputation
print("\nOriginal market index by month:")
print(
    df.groupby(df["date"].dt.to_period("M"))["market_index"]
    .agg(["count", "mean", "median", "min", "max"])
    .round(3)
    .to_string()
)

# Compare both models on exactly the same evaluation rows
market_comparison = holdout[["date"]].copy()

market_comparison["interaction_error"] = (
    interaction_predictions - y_true
)
market_comparison["market_error"] = (
    market_predictions - y_true
)

market_comparison["interaction_abs_error"] = (
    market_comparison["interaction_error"].abs()
)
market_comparison["market_abs_error"] = (
    market_comparison["market_error"].abs()
)

print("\nModel comparison by month:")
print(
    market_comparison.groupby(
        market_comparison["date"].dt.to_period("M")
    ).agg(
        interaction_mae=("interaction_abs_error", "mean"),
        market_mae=("market_abs_error", "mean"),
        interaction_bias=("interaction_error", "mean"),
        market_bias=("market_error", "mean"),
    ).round(2).to_string()
)

# Scale features, then fit a regression less sensitive to large residuals
huber_model = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "regressor",
            HuberRegressor(
                epsilon=1.35,
                max_iter=1000,
            ),
        ),
    ]
)

huber_model.fit(X_train_interactions, y_train)

huber_predictions = huber_model.predict(X_holdout_interactions)

print("\nOrdinary vs Huber regression:")
evaluate_predictions(
    "Linear interactions", y_true, interaction_predictions
)
evaluate_predictions(
    "Huber interactions", y_true, huber_predictions
)

print("\nHuber training performance:")
evaluate_predictions(
    "Train",
    y_train.to_numpy(),
    huber_model.predict(X_train_interactions),
)

print("\nHuber holdout error percentiles:")
huber_errors = np.abs(y_true - huber_predictions)

for percentile in [50, 90, 95, 99]:
    print(
        f"P{percentile}: "
        f"${np.percentile(huber_errors, percentile):,.2f}"
    )

print("\nNon-positive Huber predictions:")
print((huber_predictions <= 0).sum())

windows = [
    ("2025-05-01", "2025-07-01"),
    ("2025-07-01", "2025-09-01"),
    ("2025-09-01", "2025-11-01"),
]

results = []

for start, end in windows:
    # Start from the original data, before imputation
    fold_train = df.loc[df["date"] < start].copy()
    fold_eval = df.loc[
        (df["date"] >= start) & (df["date"] < end)
    ].copy()

    # Clean invalid weights in both parts
    for subset in (fold_train, fold_eval):
        subset["weight"] = subset["weight"].mask(
            subset["weight"] <= 0
        )

    # Learn imputation separately inside each training window
    weight_median = fold_train["weight"].median()

    for subset in (fold_train, fold_eval):
        subset["weight"] = subset["weight"].fillna(weight_median)

    X_fit = make_interaction_features(fold_train)
    X_eval = make_interaction_features(fold_eval)
    y_fit = fold_train["posted_rate"]
    y_eval = fold_eval["posted_rate"].to_numpy()

    models = {
        "Linear": LinearRegression(),
        "Huber": clone(huber_model),
    }

    for name, model in models.items():
        model.fit(X_fit, y_fit)
        predictions = model.predict(X_eval)
        errors = predictions - y_eval

        results.append({
            "eval_start": start,
            "model": name,
            "train_rows": len(fold_train),
            "eval_rows": len(fold_eval),
            "mae": np.abs(errors).mean(),
            "rmse": np.sqrt((errors ** 2).mean()),
        })

results_df = pd.DataFrame(results)

print("\nTemporal validation results:")
print(results_df.round(2).to_string(index=False))