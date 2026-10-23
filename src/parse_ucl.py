import re
from pathlib import Path
import pandas as pd
from datetime import datetime


# ==============================
# PATHS
# ==============================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "champions-league-master"
OUTPUT_FILE = BASE_DIR / "data" / "ucl_matches_all.csv"


# ==============================
# DATE REGEX
# ==============================

DATE_PATTERN = re.compile(
    r"^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+"
    r"([A-Z][a-z]{2})\s+(\d{1,2})"
    r"(?:\s+(\d{4}))?$"
)

# Match line:
# Team A v Team B 2-1
MATCH_PATTERN = re.compile(
    r"^\s*(?:\d{1,2}:\d{2}\s+)?"
    r"(.+?)\s+v\s+(.+?)\s+"
    r"(\d+)-(\d+)(.*)$"
)


MONTHS = {
    "Jan": 1,
    "Feb": 2,
    "Mar": 3,
    "Apr": 4,
    "May": 5,
    "Jun": 6,
    "Jul": 7,
    "Aug": 8,
    "Sep": 9,
    "Oct": 10,
    "Nov": 11,
    "Dec": 12,
}


# ==============================
# DATE CONVERSION
# ==============================

def parse_date(line, season_start_year):
    match = DATE_PATTERN.match(line.strip())

    if not match:
        return None

    _, month_name, day, explicit_year = match.groups()

    month = MONTHS[month_name]
    day = int(day)

    if explicit_year:
        year = int(explicit_year)
    else:
        # UCL season:
        # Sep-Dec -> starting year
        # Jan-May -> next year
        if month >= 8:
            year = season_start_year
        else:
            year = season_start_year + 1

    return datetime(year, month, day).date()


# ==============================
# RESULT EXTRACTION
# ==============================

def extract_match(line, current_date, season):
    match = MATCH_PATTERN.match(line)

    if not match or current_date is None:
        return None

    home_team = match.group(1).strip()
    away_team = match.group(2).strip()

    home_goals = int(match.group(3))
    away_goals = int(match.group(4))

    extra = match.group(5)

    # For penalty shootouts:
    # Example:
    # 4-3 pen. 1-1 a.e.t.
    #
    # We use the score after extra time,
    # not the penalty shootout score.
    if "pen." in extra:
        et_match = re.search(
            r"pen\.\s+(\d+)-(\d+)\s+a\.e\.t\.",
            extra
        )

        if et_match:
            home_goals = int(et_match.group(1))
            away_goals = int(et_match.group(2))

    if home_goals > away_goals:
        result = "H"
    elif home_goals < away_goals:
        result = "A"
    else:
        result = "D"

    return {
        "Season": season,
        "Date": current_date,
        "HomeTeam": home_team,
        "AwayTeam": away_team,
        "HomeGoals": home_goals,
        "AwayGoals": away_goals,
        "Result": result,
    }


# ==============================
# MAIN PARSER
# ==============================

all_matches = []

season_folders = sorted(
    [
        folder
        for folder in DATA_DIR.iterdir()
        if folder.is_dir() and re.match(r"^\d{4}-\d{2}$", folder.name)
    ]
)

for season_folder in season_folders:

    season = season_folder.name
    season_start_year = int(season[:4])

    cl_file = season_folder / "cl.txt"

    if not cl_file.exists():
        print(f"Skipping {season}: cl.txt not found")
        continue

    current_date = None

    with open(cl_file, "r", encoding="utf-8") as file:

        for line in file:
            line = line.strip()

            if not line:
                continue

            # Check for date heading
            parsed_date = parse_date(line, season_start_year)

            if parsed_date:
                current_date = parsed_date
                continue

            # Check for match
            match_data = extract_match(
                line,
                current_date,
                season
            )

            if match_data:
                all_matches.append(match_data)


# ==============================
# CREATE DATAFRAME
# ==============================

df = pd.DataFrame(all_matches)

df["Date"] = pd.to_datetime(df["Date"])

df = df.sort_values(
    ["Date", "Season", "HomeTeam"]
).reset_index(drop=True)


# ==============================
# SAVE DATASET
# ==============================

df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8"
)


# ==============================
# VALIDATION
# ==============================

print("\n===================================")
print("UCL MASTER DATASET CREATED")
print("===================================")

print(f"Total matches : {len(df)}")
print(f"Total seasons : {df['Season'].nunique()}")
print(f"Output        : {OUTPUT_FILE}")

print("\nMatches by season:")
print(df["Season"].value_counts().sort_index())

print("\nDate range:")
print(df["Date"].min().date(), "to", df["Date"].max().date())

print("\nFirst 10 matches:")
print(
    df[
        [
            "Season",
            "Date",
            "HomeTeam",
            "AwayTeam",
            "HomeGoals",
            "AwayGoals",
            "Result",
        ]
    ].head(10).to_string(index=False)
)

print("\nDataset columns:")
print(list(df.columns))

print("\nParser completed successfully.")