import numpy as np
import pandas as pd


def make_interaction_features(frame):
    required = ["distance", "weight", "equipment"]
    missing_columns = set(required) - set(frame.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {sorted(missing_columns)}"
        )

    valid_equipment = ["Dry Van", "Flatbed", "Reefer"]
    invalid_equipment = ~frame["equipment"].isin(valid_equipment)

    if invalid_equipment.any():
        raise ValueError("Unknown or missing equipment type")

    features = frame[["distance", "weight"]].copy()

    # Convert invalid weights to missing values
    features["weight"] = features["weight"].mask(
        features["weight"] <= 0
    )

    # Dry Van is the reference category
    features["is_flatbed"] = (
        frame["equipment"] == "Flatbed"
    ).astype(int)

    features["is_reefer"] = (
        frame["equipment"] == "Reefer"
    ).astype(int)

    # Allow the distance coefficient to vary by equipment
    features["distance_flatbed"] = (
        features["distance"] * features["is_flatbed"]
    )

    features["distance_reefer"] = (
        features["distance"] * features["is_reefer"]
    )

    return features


def make_calendar_features(frame):
    features = make_interaction_features(frame)

    dates = pd.to_datetime(frame["date"], errors="raise")

    if dates.isna().any():
        raise ValueError("Missing dates found")

    # Monday = 0, Sunday = 6
    weekday = dates.dt.dayofweek

    # Monday is the reference category
    for day in range(1, 7):
        features[f"weekday_{day}"] = (
            weekday == day
        ).astype(int)

    return features


def make_seasonal_features(frame):
    dates = pd.to_datetime(frame["date"], errors="raise")

    if dates.isna().any():
        raise ValueError("Missing dates found")

    # Account for leap years
    days_in_year = np.where(
        dates.dt.is_leap_year,
        366.0,
        365.0,
    )

    # Represent the position within the annual cycle
    phase = (
        2 * np.pi * (dates.dt.dayofyear - 1) / days_in_year
    )

    features = pd.DataFrame(index=frame.index)
    features["year_sin"] = np.sin(phase)
    features["year_cos"] = np.cos(phase)

    return features