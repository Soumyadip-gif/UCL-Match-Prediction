import os
import joblib
import pandas as pd
import numpy as np


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(BASE_DIR, "data", "ucl_matches_all.csv")

RESULT_MODEL_PATH = os.path.join(
    BASE_DIR, "models", "ucl_match_model.pkl"
)

RESULT_FEATURE_PATH = os.path.join(
    BASE_DIR, "models", "feature_columns.pkl"
)

HOME_GOAL_MODEL_PATH = os.path.join(
    BASE_DIR, "models", "home_goal_model.pkl"
)

AWAY_GOAL_MODEL_PATH = os.path.join(
    BASE_DIR, "models", "away_goal_model.pkl"
)

GOAL_FEATURE_PATH = os.path.join(
    BASE_DIR, "models", "goal_feature_columns.pkl"
)


# ============================================================
# LOAD DATA AND MODELS
# ============================================================

matches = pd.read_csv(DATA_PATH)
matches["Date"] = pd.to_datetime(matches["Date"])

result_model = joblib.load(RESULT_MODEL_PATH)
result_features = joblib.load(RESULT_FEATURE_PATH)

home_goal_model = joblib.load(HOME_GOAL_MODEL_PATH)
away_goal_model = joblib.load(AWAY_GOAL_MODEL_PATH)
goal_features = joblib.load(GOAL_FEATURE_PATH)


# ============================================================
# BUILD TEAM HISTORY
# ============================================================

def build_team_history():

    history = {}

    for team in set(matches["HomeTeam"]).union(set(matches["AwayTeam"])):

        history[team] = {
            "matches": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "goals_for": 0,
            "goals_against": 0,

            "home_matches": 0,
            "home_wins": 0,
            "home_goals_for": 0,
            "home_goals_against": 0,

            "away_matches": 0,
            "away_wins": 0,
            "away_goals_for": 0,
            "away_goals_against": 0,

            "recent_points": [],
            "recent_goal_diff": []
        }

    return history


def safe_rate(value, total):
    if total == 0:
        return 0.0
    return value / total


def update_history(history, home, away, home_goals, away_goals):

    # HOME TEAM
    h = history[home]

    h["matches"] += 1
    h["goals_for"] += home_goals
    h["goals_against"] += away_goals

    h["home_matches"] += 1
    h["home_goals_for"] += home_goals
    h["home_goals_against"] += away_goals

    if home_goals > away_goals:
        h["wins"] += 1
        h["home_wins"] += 1
        points = 3
    elif home_goals == away_goals:
        h["draws"] += 1
        points = 1
    else:
        h["losses"] += 1
        points = 0

    h["recent_points"].append(points)
    h["recent_goal_diff"].append(home_goals - away_goals)

    h["recent_points"] = h["recent_points"][-10:]
    h["recent_goal_diff"] = h["recent_goal_diff"][-10:]

    # AWAY TEAM
    a = history[away]

    a["matches"] += 1
    a["goals_for"] += away_goals
    a["goals_against"] += home_goals

    a["away_matches"] += 1
    a["away_goals_for"] += away_goals
    a["away_goals_against"] += home_goals

    if away_goals > home_goals:
        a["wins"] += 1
        a["away_wins"] += 1
        points = 3
    elif away_goals == home_goals:
        a["draws"] += 1
        points = 1
    else:
        a["losses"] += 1
        points = 0

    a["recent_points"].append(points)
    a["recent_goal_diff"].append(away_goals - home_goals)

    a["recent_points"] = a["recent_points"][-10:]
    a["recent_goal_diff"] = a["recent_goal_diff"][-10:]


# ============================================================
# BUILD FEATURES
# ============================================================

def build_prediction_features(home_team, away_team):

    history = build_team_history()

    for _, row in matches.sort_values("Date").iterrows():

        update_history(
            history,
            row["HomeTeam"],
            row["AwayTeam"],
            int(row["HomeGoals"]),
            int(row["AwayGoals"])
        )

    home = history[home_team]
    away = history[away_team]

    features = {

        "HomeMatches": home["matches"],
        "HomeWinRate": safe_rate(home["wins"], home["matches"]),
        "HomeAvgGoalsFor": safe_rate(
            home["goals_for"], home["matches"]
        ),
        "HomeAvgGoalsAgainst": safe_rate(
            home["goals_against"], home["matches"]
        ),

        "HomeWinRateAtHome": safe_rate(
            home["home_wins"], home["home_matches"]
        ),

        "HomeAvgGoalsAtHome": safe_rate(
            home["home_goals_for"], home["home_matches"]
        ),

        "HomeAvgConcededAtHome": safe_rate(
            home["home_goals_against"], home["home_matches"]
        ),

        "HomeRecentPoints": (
            np.mean(home["recent_points"])
            if home["recent_points"] else 0
        ),

        "HomeRecentGoalDiff": (
            np.mean(home["recent_goal_diff"])
            if home["recent_goal_diff"] else 0
        ),

        "AwayMatches": away["matches"],
        "AwayWinRate": safe_rate(
            away["wins"], away["matches"]
        ),

        "AwayAvgGoalsFor": safe_rate(
            away["goals_for"], away["matches"]
        ),

        "AwayAvgGoalsAgainst": safe_rate(
            away["goals_against"], away["matches"]
        ),

        "AwayWinRateAway": safe_rate(
            away["away_wins"], away["away_matches"]
        ),

        "AwayAvgGoalsAway": safe_rate(
            away["away_goals_for"], away["away_matches"]
        ),

        "AwayAvgConcededAway": safe_rate(
            away["away_goals_against"], away["away_matches"]
        ),

        "AwayRecentPoints": (
            np.mean(away["recent_points"])
            if away["recent_points"] else 0
        ),

        "AwayRecentGoalDiff": (
            np.mean(away["recent_goal_diff"])
            if away["recent_goal_diff"] else 0
        )
    }

    # --------------------------------------------------------
    # Strength differences
    # --------------------------------------------------------

    features["OverallWinRateDiff"] = (
        features["HomeWinRate"] -
        features["AwayWinRate"]
    )

    features["OverallGoalForDiff"] = (
        features["HomeAvgGoalsFor"] -
        features["AwayAvgGoalsFor"]
    )

    features["OverallGoalAgainstDiff"] = (
        features["HomeAvgGoalsAgainst"] -
        features["AwayAvgGoalsAgainst"]
    )

    features["HomeAwayWinRateDiff"] = (
        features["HomeWinRateAtHome"] -
        features["AwayWinRateAway"]
    )

    features["HomeAwayGoalForDiff"] = (
        features["HomeAvgGoalsAtHome"] -
        features["AwayAvgGoalsAway"]
    )

    features["HomeAwayGoalAgainstDiff"] = (
        features["HomeAvgConcededAtHome"] -
        features["AwayAvgConcededAway"]
    )

    features["RecentPointsDiff"] = (
        features["HomeRecentPoints"] -
        features["AwayRecentPoints"]
    )

    features["RecentGoalDiff"] = (
        features["HomeRecentGoalDiff"] -
        features["AwayRecentGoalDiff"]
    )

    # Missing richer features default to zero.
    for column in result_features:
        if column not in features:
            features[column] = 0

    return pd.DataFrame([features])


# ============================================================
# PREDICTION
# ============================================================

def predict_match(home_team, away_team):

    if home_team not in set(matches["HomeTeam"]).union(
        set(matches["AwayTeam"])
    ):
        raise ValueError(f"Unknown home team: {home_team}")

    if away_team not in set(matches["HomeTeam"]).union(
        set(matches["AwayTeam"])
    ):
        raise ValueError(f"Unknown away team: {away_team}")

    if home_team == away_team:
        raise ValueError("Home and away teams must be different.")

    feature_data = build_prediction_features(
        home_team,
        away_team
    )

    # --------------------------------------------------------
    # Match result prediction
    # --------------------------------------------------------

    result_input = feature_data[result_features]

    result_probabilities = result_model.predict_proba(
        result_input
    )[0]

    classes = list(result_model.classes_)

    probabilities = {
        "H": 0.0,
        "D": 0.0,
        "A": 0.0
    }

    for class_name, probability in zip(
        classes,
        result_probabilities
    ):
        probabilities[class_name] = float(probability)

    prediction = max(
        probabilities,
        key=probabilities.get
    )

    # --------------------------------------------------------
    # Goal prediction
    # --------------------------------------------------------

    goal_input = feature_data.copy()

    for column in goal_features:

        if column not in goal_input:
            goal_input[column] = 0

    goal_input = goal_input[goal_features]

    predicted_home_goals = float(
        home_goal_model.predict(goal_input)[0]
    )

    predicted_away_goals = float(
        away_goal_model.predict(goal_input)[0]
    )

    # Goals cannot be negative
    predicted_home_goals = max(
        0.0,
        predicted_home_goals
    )

    predicted_away_goals = max(
        0.0,
        predicted_away_goals
    )

    predicted_total_goals = (
        predicted_home_goals +
        predicted_away_goals
    )

    # Rounded expected score for display
    predicted_home_score = int(
        round(predicted_home_goals)
    )

    predicted_away_score = int(
        round(predicted_away_goals)
    )

    return {

        "home_team": home_team,
        "away_team": away_team,

        "prediction": prediction,

        "home_win_probability":
            probabilities["H"],

        "draw_probability":
            probabilities["D"],

        "away_win_probability":
            probabilities["A"],

        "predicted_home_goals":
            predicted_home_goals,

        "predicted_away_goals":
            predicted_away_goals,

        "predicted_total_goals":
            predicted_total_goals,

        "predicted_score":
            f"{predicted_home_score}-{predicted_away_score}"
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    home = "FC Barcelona (ESP)"
    away = "Real Madrid (ESP)"

    try:

        result = predict_match(home, away)

        print("\n===================================")
        print("UCL MATCH + GOAL PREDICTION")
        print("===================================")

        print(
            f"\n{home} vs {away}"
        )

        print(
            f"\nMatch Prediction: "
            f"{result['prediction']}"
        )

        print(
            f"Home Win: "
            f"{result['home_win_probability']:.2%}"
        )

        print(
            f"Draw: "
            f"{result['draw_probability']:.2%}"
        )

        print(
            f"Away Win: "
            f"{result['away_win_probability']:.2%}"
        )

        print("\nGoal Prediction")

        print(
            f"Home Goals: "
            f"{result['predicted_home_goals']:.2f}"
        )

        print(
            f"Away Goals: "
            f"{result['predicted_away_goals']:.2f}"
        )

        print(
            f"Total Goals: "
            f"{result['predicted_total_goals']:.2f}"
        )

        print(
            f"Predicted Score: "
            f"{result['predicted_score']}"
        )

    except Exception as e:

        print(f"\nERROR: {e}")