from pathlib import Path
from collections import defaultdict, deque

import pandas as pd


# ==============================
# PATHS
# ==============================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = BASE_DIR / "data" / "ucl_matches_all.csv"
OUTPUT_FILE = BASE_DIR / "data" / "ucl_ml_dataset.csv"


# ==============================
# LOAD DATA
# ==============================

df = pd.read_csv(INPUT_FILE)

df["Date"] = pd.to_datetime(df["Date"])

# Stable chronological order
df = df.sort_values(
    ["Date", "Season"]
).reset_index(drop=True)


# ==============================
# TEAM HISTORY
# ==============================

def create_team():
    return {
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

        "points_5": deque(maxlen=5),
        "points_10": deque(maxlen=10),

        "goal_diff_5": deque(maxlen=5),
        "goal_diff_10": deque(maxlen=10),

        "goals_for_5": deque(maxlen=5),
        "goals_for_10": deque(maxlen=10),

        "goals_against_5": deque(maxlen=5),
        "goals_against_10": deque(maxlen=10),
    }


teams = defaultdict(create_team)


# ==============================
# HEAD-TO-HEAD HISTORY
# ==============================

# Key = sorted pair of teams
# Stores previous H2H results from the perspective
# of the first team in the key.
h2h = defaultdict(
    lambda: {
        "matches": 0,
        "team1_wins": 0,
        "team2_wins": 0,
        "draws": 0,
    }
)


# ==============================
# HELPER
# ==============================

def average(values, default):
    if not values:
        return default

    return sum(values) / len(values)


def get_features(team):

    stats = teams[team]

    if stats["matches"] > 0:
        win_rate = stats["wins"] / stats["matches"]
        avg_goals_for = (
            stats["goals_for"] / stats["matches"]
        )
        avg_goals_against = (
            stats["goals_against"] / stats["matches"]
        )
    else:
        win_rate = 0.5
        avg_goals_for = 1.5
        avg_goals_against = 1.5

    if stats["home_matches"] > 0:
        home_win_rate = (
            stats["home_wins"] /
            stats["home_matches"]
        )

        home_avg_goals_for = (
            stats["home_goals_for"] /
            stats["home_matches"]
        )

        home_avg_goals_against = (
            stats["home_goals_against"] /
            stats["home_matches"]
        )
    else:
        home_win_rate = 0.5
        home_avg_goals_for = 1.5
        home_avg_goals_against = 1.5

    if stats["away_matches"] > 0:
        away_win_rate = (
            stats["away_wins"] /
            stats["away_matches"]
        )

        away_avg_goals_for = (
            stats["away_goals_for"] /
            stats["away_matches"]
        )

        away_avg_goals_against = (
            stats["away_goals_against"] /
            stats["away_matches"]
        )
    else:
        away_win_rate = 0.5
        away_avg_goals_for = 1.5
        away_avg_goals_against = 1.5

    return {
        "matches": stats["matches"],
        "win_rate": win_rate,
        "avg_goals_for": avg_goals_for,
        "avg_goals_against": avg_goals_against,

        "home_win_rate": home_win_rate,
        "home_avg_goals_for": home_avg_goals_for,
        "home_avg_goals_against": home_avg_goals_against,

        "away_win_rate": away_win_rate,
        "away_avg_goals_for": away_avg_goals_for,
        "away_avg_goals_against": away_avg_goals_against,

        "recent_points_5": average(
            stats["points_5"], 1.5
        ),

        "recent_points_10": average(
            stats["points_10"], 1.5
        ),

        "recent_goal_diff_5": average(
            stats["goal_diff_5"], 0.0
        ),

        "recent_goal_diff_10": average(
            stats["goal_diff_10"], 0.0
        ),

        "recent_goals_for_5": average(
            stats["goals_for_5"], 1.5
        ),

        "recent_goals_for_10": average(
            stats["goals_for_10"], 1.5
        ),

        "recent_goals_against_5": average(
            stats["goals_against_5"], 1.5
        ),

        "recent_goals_against_10": average(
            stats["goals_against_10"], 1.5
        ),
    }


# ==============================
# BUILD DATASET
# ==============================

feature_rows = []


for _, match in df.iterrows():

    home_team = match["HomeTeam"]
    away_team = match["AwayTeam"]

    home = get_features(home_team)
    away = get_features(away_team)

    # --------------------------
    # HEAD-TO-HEAD
    # --------------------------

    team1, team2 = sorted(
        [home_team, away_team]
    )

    h2h_key = (team1, team2)

    h2h_stats = h2h[h2h_key]

    h2h_matches = h2h_stats["matches"]

    if h2h_matches > 0:

        if home_team == team1:
            h2h_home_win_rate = (
                h2h_stats["team1_wins"] /
                h2h_matches
            )

            h2h_away_win_rate = (
                h2h_stats["team2_wins"] /
                h2h_matches
            )

        else:
            h2h_home_win_rate = (
                h2h_stats["team2_wins"] /
                h2h_matches
            )

            h2h_away_win_rate = (
                h2h_stats["team1_wins"] /
                h2h_matches
            )

        h2h_draw_rate = (
            h2h_stats["draws"] /
            h2h_matches
        )

    else:
        h2h_home_win_rate = 0.5
        h2h_away_win_rate = 0.5
        h2h_draw_rate = 0.0


    # ==============================
    # PRE-MATCH FEATURES
    # ==============================

    row = {

        "Season": match["Season"],
        "Date": match["Date"],

        "HomeTeam": home_team,
        "AwayTeam": away_team,

        # Home team
        "HomeMatches": home["matches"],
        "HomeWinRate": home["win_rate"],
        "HomeAvgGoalsFor": home["avg_goals_for"],
        "HomeAvgGoalsAgainst": home["avg_goals_against"],

        "HomeWinRateAtHome": home["home_win_rate"],
        "HomeAvgGoalsAtHome": home["home_avg_goals_for"],
        "HomeAvgConcededAtHome": home["home_avg_goals_against"],

        "HomeRecentPoints5": home["recent_points_5"],
        "HomeRecentPoints10": home["recent_points_10"],

        "HomeRecentGoalDiff5": home["recent_goal_diff_5"],
        "HomeRecentGoalDiff10": home["recent_goal_diff_10"],

        "HomeRecentGoalsFor5": home["recent_goals_for_5"],
        "HomeRecentGoalsFor10": home["recent_goals_for_10"],

        "HomeRecentGoalsAgainst5": home[
            "recent_goals_against_5"
        ],

        "HomeRecentGoalsAgainst10": home[
            "recent_goals_against_10"
        ],


        # Away team
        "AwayMatches": away["matches"],
        "AwayWinRate": away["win_rate"],
        "AwayAvgGoalsFor": away["avg_goals_for"],
        "AwayAvgGoalsAgainst": away["avg_goals_against"],

        "AwayWinRateAway": away["away_win_rate"],
        "AwayAvgGoalsAway": away["away_avg_goals_for"],
        "AwayAvgConcededAway": away[
            "away_avg_goals_against"
        ],

        "AwayRecentPoints5": away["recent_points_5"],
        "AwayRecentPoints10": away["recent_points_10"],

        "AwayRecentGoalDiff5": away["recent_goal_diff_5"],
        "AwayRecentGoalDiff10": away["recent_goal_diff_10"],

        "AwayRecentGoalsFor5": away["recent_goals_for_5"],
        "AwayRecentGoalsFor10": away["recent_goals_for_10"],

        "AwayRecentGoalsAgainst5": away[
            "recent_goals_against_5"
        ],

        "AwayRecentGoalsAgainst10": away[
            "recent_goals_against_10"
        ],


        # Strength differences
        "WinRateDifference": (
            home["win_rate"] -
            away["win_rate"]
        ),

        "GoalScoringDifference": (
            home["avg_goals_for"] -
            away["avg_goals_for"]
        ),

        "GoalConcedingDifference": (
            away["avg_goals_against"] -
            home["avg_goals_against"]
        ),

        "HomeAdvantageStrength": (
            home["home_win_rate"] -
            away["away_win_rate"]
        ),

        "RecentFormDifference5": (
            home["recent_points_5"] -
            away["recent_points_5"]
        ),

        "RecentFormDifference10": (
            home["recent_points_10"] -
            away["recent_points_10"]
        ),

        "RecentGoalDiffDifference5": (
            home["recent_goal_diff_5"] -
            away["recent_goal_diff_5"]
        ),

        "RecentGoalDiffDifference10": (
            home["recent_goal_diff_10"] -
            away["recent_goal_diff_10"]
        ),

        # Head-to-head
        "H2HMatches": h2h_matches,
        "H2HHomeWinRate": h2h_home_win_rate,
        "H2HAwayWinRate": h2h_away_win_rate,
        "H2HDrawRate": h2h_draw_rate,

        # Target
        "Result": match["Result"],
    }

    feature_rows.append(row)


    # ==============================
    # UPDATE TEAM HISTORY
    # ==============================

    home_goals = int(match["HomeGoals"])
    away_goals = int(match["AwayGoals"])

    if home_goals > away_goals:
        home_points = 3
        away_points = 0
        result = "H"

    elif home_goals < away_goals:
        home_points = 0
        away_points = 3
        result = "A"

    else:
        home_points = 1
        away_points = 1
        result = "D"


    # HOME TEAM

    teams[home_team]["matches"] += 1

    if result == "H":
        teams[home_team]["wins"] += 1
    elif result == "D":
        teams[home_team]["draws"] += 1
    else:
        teams[home_team]["losses"] += 1

    teams[home_team]["goals_for"] += home_goals
    teams[home_team]["goals_against"] += away_goals

    teams[home_team]["home_matches"] += 1

    if result == "H":
        teams[home_team]["home_wins"] += 1

    teams[home_team]["home_goals_for"] += home_goals
    teams[home_team]["home_goals_against"] += away_goals

    teams[home_team]["points_5"].append(home_points)
    teams[home_team]["points_10"].append(home_points)

    teams[home_team]["goal_diff_5"].append(
        home_goals - away_goals
    )
    teams[home_team]["goal_diff_10"].append(
        home_goals - away_goals
    )

    teams[home_team]["goals_for_5"].append(home_goals)
    teams[home_team]["goals_for_10"].append(home_goals)

    teams[home_team]["goals_against_5"].append(
        away_goals
    )
    teams[home_team]["goals_against_10"].append(
        away_goals
    )


    # AWAY TEAM

    teams[away_team]["matches"] += 1

    if result == "A":
        teams[away_team]["wins"] += 1
    elif result == "D":
        teams[away_team]["draws"] += 1
    else:
        teams[away_team]["losses"] += 1

    teams[away_team]["goals_for"] += away_goals
    teams[away_team]["goals_against"] += home_goals

    teams[away_team]["away_matches"] += 1

    if result == "A":
        teams[away_team]["away_wins"] += 1

    teams[away_team]["away_goals_for"] += away_goals
    teams[away_team]["away_goals_against"] += home_goals

    teams[away_team]["points_5"].append(away_points)
    teams[away_team]["points_10"].append(away_points)

    teams[away_team]["goal_diff_5"].append(
        away_goals - home_goals
    )
    teams[away_team]["goal_diff_10"].append(
        away_goals - home_goals
    )

    teams[away_team]["goals_for_5"].append(away_goals)
    teams[away_team]["goals_for_10"].append(away_goals)

    teams[away_team]["goals_against_5"].append(
        home_goals
    )
    teams[away_team]["goals_against_10"].append(
        home_goals
    )


    # ==============================
    # UPDATE H2H
    # ==============================

    if home_team == team1:

        if result == "H":
            h2h_stats["team1_wins"] += 1
        elif result == "A":
            h2h_stats["team2_wins"] += 1
        else:
            h2h_stats["draws"] += 1

    else:

        if result == "H":
            h2h_stats["team2_wins"] += 1
        elif result == "A":
            h2h_stats["team1_wins"] += 1
        else:
            h2h_stats["draws"] += 1

    h2h_stats["matches"] += 1


# ==============================
# SAVE
# ==============================

ml_df = pd.DataFrame(feature_rows)

ml_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)


# ==============================
# VALIDATION
# ==============================

print("\n===================================")
print("IMPROVED ML DATASET CREATED")
print("===================================")

print("Rows:", len(ml_df))
print("Columns:", len(ml_df.columns))

print("\nOutput:")
print(OUTPUT_FILE)

print("\nResult distribution:")
print(ml_df["Result"].value_counts())

print("\nMissing values:")
print(ml_df.isnull().sum().sum())

print("\nFirst 5 rows:")
print(
    ml_df.head().to_string(index=False)
)

print("\n===================================")
print("FEATURE ENGINEERING COMPLETED")
print("===================================")