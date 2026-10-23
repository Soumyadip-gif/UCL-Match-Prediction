from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "ucl_matches_all.csv"
OUTPUT_DIR = BASE_DIR / "data" / "eda"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_FILE)
df["Date"] = pd.to_datetime(df["Date"])


print("\n===================================")
print("UCL EXPLORATORY DATA ANALYSIS")
print("===================================")


# ==============================
# 1. RESULT DISTRIBUTION
# ==============================

result_counts = df["Result"].value_counts()

print("\n1. RESULT DISTRIBUTION")
print("----------------------")
print(result_counts)

result_labels = {
    "H": "Home Win",
    "D": "Draw",
    "A": "Away Win"
}

print("\nReadable format:")
for result, count in result_counts.items():
    print(f"{result_labels[result]}: {count}")


# ==============================
# 2. GOAL STATISTICS
# ==============================

df["TotalGoals"] = df["HomeGoals"] + df["AwayGoals"]

print("\n2. GOAL STATISTICS")
print("------------------")
print(f"Average home goals: {df['HomeGoals'].mean():.2f}")
print(f"Average away goals: {df['AwayGoals'].mean():.2f}")
print(f"Average total goals: {df['TotalGoals'].mean():.2f}")
print(f"Maximum goals in one match: {df['TotalGoals'].max()}")


# ==============================
# 3. HOME ADVANTAGE
# ==============================

home_wins = (df["Result"] == "H").sum()
draws = (df["Result"] == "D").sum()
away_wins = (df["Result"] == "A").sum()

total_matches = len(df)

print("\n3. MATCH RESULT PERCENTAGES")
print("---------------------------")
print(f"Home wins : {home_wins / total_matches * 100:.2f}%")
print(f"Draws     : {draws / total_matches * 100:.2f}%")
print(f"Away wins : {away_wins / total_matches * 100:.2f}%")


# ==============================
# 4. TEAM STATISTICS
# ==============================

home_team_stats = (
    df.groupby("HomeTeam")
    .agg(
        Matches=("HomeTeam", "count"),
        Wins=("Result", lambda x: (x == "H").sum()),
        Goals=("HomeGoals", "sum")
    )
)

away_team_stats = (
    df.groupby("AwayTeam")
    .agg(
        Matches=("AwayTeam", "count"),
        Wins=("Result", lambda x: (x == "A").sum()),
        Goals=("AwayGoals", "sum")
    )
)
all_teams = sorted(
    set(home_team_stats.index) | set(away_team_stats.index)
)

team_stats = pd.DataFrame(
    index=all_teams)

team_stats["Matches"] = (
    home_team_stats["Matches"].fillna(0)
    + away_team_stats["Matches"].fillna(0)
)

team_stats["Wins"] = (
    home_team_stats["Wins"].fillna(0)
    + away_team_stats["Wins"].fillna(0)
)

team_stats["Goals"] = (
    home_team_stats["Goals"].fillna(0)
    + away_team_stats["Goals"].fillna(0)
)

team_stats = team_stats.sort_values(
    "Wins",
    ascending=False
)

print("\n4. TOP 10 TEAMS BY WINS")
print("-----------------------")
print(team_stats.head(10))


# ==============================
# 5. SEASON STATISTICS
# ==============================

season_stats = (
    df.groupby("Season")
    .agg(
        Matches=("Season", "count"),
        Goals=("TotalGoals", "sum"),
        AverageGoals=("TotalGoals", "mean")
    )
)

print("\n5. SEASON STATISTICS")
print("--------------------")
print(season_stats)


# ==============================
# 6. SAVE ANALYSIS DATA
# ==============================

team_stats.to_csv(
    OUTPUT_DIR / "team_statistics.csv"
)

season_stats.to_csv(
    OUTPUT_DIR / "season_statistics.csv"
)

result_counts.to_csv(
    OUTPUT_DIR / "result_distribution.csv"
)


# ==============================
# 7. RESULT DISTRIBUTION CHART
# ==============================

plt.figure(figsize=(8, 5))

plt.bar(
    ["Home Win", "Draw", "Away Win"],
    [
        result_counts.get("H", 0),
        result_counts.get("D", 0),
        result_counts.get("A", 0)
    ]
)

plt.title("UCL Match Result Distribution")
plt.xlabel("Result")
plt.ylabel("Number of Matches")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "result_distribution.png",
    dpi=150
)

plt.close()


# ==============================
# 8. GOAL DISTRIBUTION
# ==============================

plt.figure(figsize=(8, 5))

plt.hist(
    df["TotalGoals"],
    bins=range(
        int(df["TotalGoals"].max()) + 2
    ),
    edgecolor="black"
)

plt.title("UCL Total Goals per Match")
plt.xlabel("Total Goals")
plt.ylabel("Number of Matches")
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "goal_distribution.png",
    dpi=150
)

plt.close()


# ==============================
# 9. SEASON GOALS
# ==============================

plt.figure(figsize=(12, 5))

plt.plot(
    season_stats.index,
    season_stats["AverageGoals"],
    marker="o"
)

plt.title("Average Goals per UCL Match by Season")
plt.xlabel("Season")
plt.ylabel("Average Goals")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "season_average_goals.png",
    dpi=150
)

plt.close()


print("\n===================================")
print("EDA COMPLETED SUCCESSFULLY")
print("===================================")

print(f"Analysis files saved to:")
print(OUTPUT_DIR)