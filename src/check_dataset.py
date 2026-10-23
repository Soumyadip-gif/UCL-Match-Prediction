from pathlib import Path
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "ucl_matches_all.csv"

df = pd.read_csv(DATA_FILE)


print("\n===================================")
print("UCL DATASET QUALITY CHECK")
print("===================================")


# 1. Basic information
print("\n1. BASIC INFORMATION")
print("--------------------")
print("Rows:", len(df))
print("Columns:", len(df.columns))
print("Seasons:", df["Season"].nunique())


# 2. Missing values
print("\n2. MISSING VALUES")
print("-----------------")
missing = df.isnull().sum()
print(missing)

if missing.sum() == 0:
    print("STATUS: PASS")
else:
    print("STATUS: CHECK REQUIRED")


# 3. Duplicate matches
print("\n3. DUPLICATE MATCHES")
print("--------------------")

duplicates = df.duplicated(
    subset=[
        "Season",
        "Date",
        "HomeTeam",
        "AwayTeam"
    ]
).sum()

print("Duplicate matches:", duplicates)

if duplicates == 0:
    print("STATUS: PASS")
else:
    print("STATUS: CHECK REQUIRED")


# 4. Invalid scores
print("\n4. SCORE VALIDATION")
print("-------------------")

invalid_scores = (
    (df["HomeGoals"] < 0) |
    (df["AwayGoals"] < 0)
).sum()

print("Invalid scores:", invalid_scores)

if invalid_scores == 0:
    print("STATUS: PASS")
else:
    print("STATUS: CHECK REQUIRED")


# 5. Result validation
print("\n5. RESULT VALIDATION")
print("--------------------")

valid_results = {"H", "D", "A"}

invalid_results = (~df["Result"].isin(valid_results)).sum()

print("Invalid results:", invalid_results)

if invalid_results == 0:
    print("STATUS: PASS")
else:
    print("STATUS: CHECK REQUIRED")


# 6. Check whether Result matches scores
print("\n6. RESULT vs SCORE")
print("------------------")

calculated_result = df.apply(
    lambda row:
        "H" if row["HomeGoals"] > row["AwayGoals"]
        else "A" if row["HomeGoals"] < row["AwayGoals"]
        else "D",
    axis=1
)

result_mismatch = (
    calculated_result != df["Result"]
).sum()

print("Result mismatches:", result_mismatch)

if result_mismatch == 0:
    print("STATUS: PASS")
else:
    print("STATUS: CHECK REQUIRED")


# 7. Season validation
print("\n7. SEASON VALIDATION")
print("--------------------")

invalid_seasons = df[
    ~df["Season"].str.match(r"^\d{4}-\d{2}$")
]

print("Invalid season values:", len(invalid_seasons))

if len(invalid_seasons) == 0:
    print("STATUS: PASS")
else:
    print("STATUS: CHECK REQUIRED")


# 8. Date validation
print("\n8. DATE VALIDATION")
print("------------------")

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

invalid_dates = df["Date"].isna().sum()

print("Invalid dates:", invalid_dates)

if invalid_dates == 0:
    print("STATUS: PASS")
else:
    print("STATUS: CHECK REQUIRED")


# 9. Final summary
print("\n===================================")
print("QUALITY CHECK COMPLETED")
print("===================================")

if (
    missing.sum() == 0
    and duplicates == 0
    and invalid_scores == 0
    and invalid_results == 0
    and result_mismatch == 0
    and len(invalid_seasons) == 0
    and invalid_dates == 0
):
    print("FINAL STATUS: DATASET READY")
else:
    print("FINAL STATUS: ISSUES FOUND")

print("===================================")