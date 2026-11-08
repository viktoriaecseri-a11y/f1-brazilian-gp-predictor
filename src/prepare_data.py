
import pandas as pd
from pathlib import Path

# STEP 1: Load combined F1 data
df = pd.read_csv("data/f1_combined.csv")

# Sort race results chronologically
df = df.sort_values(
    ["Date", "Round", "DriverID"]
).reset_index(drop=True)

# FEATURE 1: Starting grid position
# Replace pit-lane starts (0) with 21 for now
df["GridPosition"] = df["GridPosition"].replace(0, 21)

# FEATURE 2: Driver's previous 5 finishing positions
df["RecentForm"] = (
    df.groupby("DriverID")["FinishPosition"]
    .transform(
        lambda x: x.shift(1).rolling(
            5, min_periods=5
        ).mean()
    )
)

# STEP 2: Calculate team results per race
team_races = (
    df.groupby(
        ["Year", "Round", "Date", "Team"],
        as_index=False
    )
    .agg(
        TeamFinish=("FinishPosition", "mean"),
        TeamPoints=("Points", "sum")
    )
)

# Sort team races chronologically
team_races = team_races.sort_values(
    ["Team", "Date", "Round"]
)

# FEATURE 3: Team's previous 5 finishing positions
team_races["TeamForm"] = (
    team_races.groupby("Team")["TeamFinish"]
    .transform(
        lambda x: x.shift(1).rolling(
            5, min_periods=5
        ).mean()
    )
)

# FEATURE 8: Average team points in previous 5 races
team_races["RecentTeamPoints"] = (
    team_races.groupby("Team")["TeamPoints"]
    .transform(
        lambda x: x.shift(1).rolling(
            5, min_periods=5
        ).mean()
    )
)

# Add both team features to main dataset
df = df.merge(
    team_races[
        ["Year", "Round", "Team",
         "TeamForm", "RecentTeamPoints"]
    ],
    on=["Year", "Round", "Team"],
    how="left",
    validate="many_to_one"
)

# FEATURE 4: Wins before the current race
df["SeasonWins"] = (
    df.groupby(["Year", "DriverID"])["Won"]
    .cumsum() - df["Won"]
).astype(int)

# FEATURE 5: Season win rate
df["PreviousStarts"] = (
    df.groupby(["Year", "DriverID"]).cumcount()
)

df["SeasonWinRate"] = (
    df["SeasonWins"]
    / df["PreviousStarts"].replace(0, float("nan"))
).fillna(0)

# FEATURE 6: Average points in driver's previous 5 races
df["RecentPoints"] = (
    df.groupby("DriverID")["Points"]
    .transform(
        lambda x: x.shift(1).rolling(
            5, min_periods=5
        ).mean()
    )
)

# FEATURE 7: Average qualifying position
# in driver's previous 5 races
qualifying = pd.read_csv(
    "data/f1_qualifying_combined.csv"
)

qualifying = qualifying[
    ["Year", "Round", "DriverID", "QualifyingPosition"]
].drop_duplicates(
    subset=["Year", "Round", "DriverID"]
)

df = df.merge(
    qualifying,
    on=["Year", "Round", "DriverID"],
    how="left",
    validate="one_to_one"
)

df = df.sort_values(
    ["Date", "Round", "DriverID"]
).reset_index(drop=True)

df["RecentQualifying"] = (
    df.groupby("DriverID")["QualifyingPosition"]
    .transform(
        lambda x: x.shift(1).rolling(
            5, min_periods=5
        ).mean()
    )
)

# STEP 3: Save prepared data
Path("data").mkdir(exist_ok=True)

df.to_csv(
    "data/f1_prepared.csv",
    index=False
)

# STEP 4: Check the new feature
print("\n--- NEW TEAM PERFORMANCE FEATURE ---")

print(
    df[
        ["Year", "Race", "Team",
         "TeamForm", "RecentTeamPoints"]
    ].tail(10).to_string(index=False)
)

print("\nPrepared dataset saved successfully!")

print("Total rows:", len(df))

print(
    "RecentTeamPoints values available:",
    df["RecentTeamPoints"].notna().sum()
)
