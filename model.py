from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.ensemble import HistGradientBoostingRegressor
from features import make_seasonal_features
from features import make_interaction_features, make_calendar_features


def build_model(use_calendar=False):
    feature_function = (
        make_calendar_features
        if use_calendar
        else make_interaction_features
    )

    return Pipeline(
        steps=[
            (
                "features",
                FunctionTransformer(feature_function),
            ),
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "regressor",
                LinearRegression(),
            ),
        ]
    )


def build_location_model():
    numeric_features = Pipeline(
        steps=[
            (
                "features",
                FunctionTransformer(make_interaction_features),
            ),
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_features,
                ["distance", "weight", "equipment"],
            ),
            (
                "locations",
                OneHotEncoder(
                    drop="first",
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                ["pickup", "delivery"],
            ),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", LinearRegression()),
        ]
    )
def build_boosting_model():
    model = build_location_model()

    model.set_params(
        regressor=HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_iter=200,
            max_leaf_nodes=15,
            min_samples_leaf=30,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=42,
        )
    )

    return model
def build_signal_model():
    model = build_boosting_model()

    preprocessor = model.named_steps["preprocessor"]

    signal_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=True,
                ),
            ),
        ]
    )

    transformers = list(preprocessor.transformers)
    transformers.append(
        ("signal", signal_pipeline, ["quote_signal"])
    )

    preprocessor.set_params(transformers=transformers)

    return model
def build_seasonal_model():
    model = build_location_model()
    preprocessor = model.named_steps["preprocessor"]

    transformers = list(preprocessor.transformers)
    transformers.append(
        (
            "season",
            FunctionTransformer(make_seasonal_features),
            ["date"],
        )
    )

    preprocessor.set_params(transformers=transformers)
    return model