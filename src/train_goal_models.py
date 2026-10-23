from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ==========================================
# PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent.parent

FEATURE_FILE = BASE_DIR / "data" / "ucl_ml_dataset.csv"
MATCH_FILE = BASE_DIR / "data" / "ucl_matches_all.csv"

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True)


# ==========================================
# LOAD DATA
# ==========================================

features_df = pd.read_csv(FEATURE_FILE)
matches_df = pd.read_csv(MATCH_FILE)

features_df["Date"] = pd.to_datetime(features_df["Date"])
matches_df["Date"] = pd.to_datetime(matches_df["Date"])


# ==========================================
# MERGE ACTUAL GOALS
# ==========================================

key_columns = [
    "Season",
    "Date",
    "HomeTeam",
    "AwayTeam"
]

goal_columns = [
    "HomeGoals",
    "AwayGoals"
]

goal_data = matches_df[
    key_columns + goal_columns
].copy()


df = features_df.merge(
    goal_data,
    on=key_columns,
    how="left",
    validate="one_to_one"
)


# ==========================================
# VALIDATION
# ==========================================

if df["HomeGoals"].isna().any():

    missing = df["HomeGoals"].isna().sum()

    raise ValueError(
        f"{missing} matches could not be matched "
        "with the original goal data."
    )


if df["AwayGoals"].isna().any():

    missing = df["AwayGoals"].isna().sum()

    raise ValueError(
        f"{missing} matches could not be matched "
        "with the original goal data."
    )


df = df.sort_values("Date").reset_index(drop=True)


print("\n===================================")
print("UCL GOAL PREDICTION MODEL")
print("===================================")

print(
    f"\nTotal matches: {len(df)}"
)


# ==========================================
# FEATURES
# ==========================================

exclude_columns = [
    "Season",
    "Date",
    "HomeTeam",
    "AwayTeam",
    "Result",
    "HomeGoals",
    "AwayGoals"
]

feature_columns = [
    column
    for column in df.columns
    if column not in exclude_columns
]


X = df[feature_columns]

y_home = df["HomeGoals"]
y_away = df["AwayGoals"]


print(
    f"Features used: {len(feature_columns)}"
)


# ==========================================
# CHRONOLOGICAL SPLIT
# ==========================================

unique_dates = sorted(
    df["Date"].unique()
)

split_index = int(
    len(unique_dates) * 0.80
)

train_dates = set(
    unique_dates[:split_index]
)

test_dates = set(
    unique_dates[split_index:]
)


train_mask = df["Date"].isin(train_dates)
test_mask = df["Date"].isin(test_dates)


X_train = X[train_mask]
X_test = X[test_mask]

y_home_train = y_home[train_mask]
y_home_test = y_home[test_mask]

y_away_train = y_away[train_mask]
y_away_test = y_away[test_mask]


print(
    f"\nTraining matches: {len(X_train)}"
)

print(
    f"Testing matches : {len(X_test)}"
)

print(
    f"\nTraining period:"
    f" {df.loc[train_mask, 'Date'].min().date()}"
    f" → "
    f"{df.loc[train_mask, 'Date'].max().date()}"
)

print(
    f"Testing period:"
    f" {df.loc[test_mask, 'Date'].min().date()}"
    f" → "
    f"{df.loc[test_mask, 'Date'].max().date()}"
)


# ==========================================
# HOME GOAL MODEL
# ==========================================

print("\nTraining Home Goals model...")

home_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)

home_model.fit(
    X_train,
    y_home_train
)


# ==========================================
# AWAY GOAL MODEL
# ==========================================

print("Training Away Goals model...")

away_model = RandomForestRegressor(
    n_estimators=300,
    max_depth=12,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1
)

away_model.fit(
    X_train,
    y_away_train
)


# ==========================================
# PREDICTIONS
# ==========================================

home_predictions = home_model.predict(X_test)
away_predictions = away_model.predict(X_test)


# ==========================================
# METRICS
# ==========================================

home_mae = mean_absolute_error(
    y_home_test,
    home_predictions
)

home_rmse = mean_squared_error(
    y_home_test,
    home_predictions
) ** 0.5


away_mae = mean_absolute_error(
    y_away_test,
    away_predictions
)

away_rmse = mean_squared_error(
    y_away_test,
    away_predictions
) ** 0.5


# ==========================================
# BASELINE
# ==========================================

home_baseline = y_home_train.mean()
away_baseline = y_away_train.mean()


home_baseline_predictions = [
    home_baseline
] * len(y_home_test)

away_baseline_predictions = [
    away_baseline
] * len(y_away_test)


baseline_home_mae = mean_absolute_error(
    y_home_test,
    home_baseline_predictions
)

baseline_away_mae = mean_absolute_error(
    y_away_test,
    away_baseline_predictions
)


# ==========================================
# RESULTS
# ==========================================

print("\n===================================")
print("MODEL PERFORMANCE")
print("===================================")

print("\nHome Goals Model")

print(
    f"MAE  : {home_mae:.4f}"
)

print(
    f"RMSE : {home_rmse:.4f}"
)

print(
    f"Baseline MAE : {baseline_home_mae:.4f}"
)


print("\nAway Goals Model")

print(
    f"MAE  : {away_mae:.4f}"
)

print(
    f"RMSE : {away_rmse:.4f}"
)

print(
    f"Baseline MAE : {baseline_away_mae:.4f}"
)


# ==========================================
# MODEL QUALITY CHECK
# ==========================================

print("\n===================================")
print("BASELINE COMPARISON")
print("===================================")


if home_mae < baseline_home_mae:

    print(
        "Home goal model: BETTER than baseline"
    )

else:

    print(
        "Home goal model: NOT better than baseline"
    )


if away_mae < baseline_away_mae:

    print(
        "Away goal model: BETTER than baseline"
    )

else:

    print(
        "Away goal model: NOT better than baseline"
    )


# ==========================================
# SAVE MODELS
# ==========================================

joblib.dump(
    home_model,
    MODEL_DIR / "home_goal_model.pkl"
)

joblib.dump(
    away_model,
    MODEL_DIR / "away_goal_model.pkl"
)

joblib.dump(
    feature_columns,
    MODEL_DIR / "goal_feature_columns.pkl"
)


print("\n===================================")
print("GOAL MODELS SAVED")
print("===================================")

print(
    "models/home_goal_model.pkl"
)

print(
    "models/away_goal_model.pkl"
)

print(
    "models/goal_feature_columns.pkl"
)