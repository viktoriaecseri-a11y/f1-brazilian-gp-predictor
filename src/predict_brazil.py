
import pandas as pd
import requests
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# STEP 1: Load historical datasets
history = pd.read_csv("data/f1_combined.csv")
prepared = pd.read_csv("data/f1_prepared.csv")

# The target race must not be in our training data
race_date = "2026-11-08"

history = history[history["Date"] < race_date].copy()
prepared = prepared[
    (prepared["Year"] < 2026) |
    ((prepared["Year"] == 2026) & (prepared["Round"] <= 19))
].copy()

features = [
    "GridPosition",
    "RecentForm",
    "TeamForm",
    "SeasonWins",
    "SeasonWinRate",
    "RecentPoints"
]

# STEP 2: Train Random Forest
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    ))
])

model.fit(prepared[features], prepared["Won"])

print("Random Forest trained successfully!")

# STEP 3: Download actual 2026 Brazil qualifying results
url = (
    "https://api.jolpi.ca/ergast/f1/"
    "2026/circuits/interlagos/qualifying.json"
)

response = requests.get(
    url,
    params={"limit": 100},
    timeout=30
)
response.raise_for_status()

races = response.json()["MRData"]["RaceTable"]["Races"]

if not races:
    raise SystemExit("2026 qualifying data not available.")

brazil = races[-1]
qualifying = brazil["QualifyingResults"]

# STEP 4: Adjust for grid penalties
# Mapping: qualifying position -> starting position
# Based on the FIA provisional starting grid
grid_map = {
    1: 1,
    2: 7,
    3: 2,
    4: 14,
    5: 3,
    6: 4,
    7: 5,
    8: 6,
    9: 8,
    10: 9,
    11: 20,
    12: 10,
    13: 11,
    14: 12,
    15: 13,
    16: 21,
    17: 15,
    18: 16,
    19: 17,
    20: 18,
    21: 22,
    22: 19
}

if len(qualifying) != 22:
    raise SystemExit(
        f"Expected 22 drivers, found {len(qualifying)}."
    )

predictions = []

# STEP 5: Calculate current features for every driver
for result in qualifying:

    driver_id = result["Driver"]["driverId"]
    name = (
        result["Driver"]["givenName"] + " " +
        result["Driver"]["familyName"]
    )

    q_position = int(result["position"])
    grid_position = grid_map[q_position]

    driver_history = history[
        history["DriverID"] == driver_id
    ].sort_values(["Date", "Round"])

    # Latest known team, or qualifying team for rookies
    if not driver_history.empty:
        team = driver_history.iloc[-1]["Team"]
    else:
        team = result["Constructor"]["name"]

    # Driver performance over previous five races
    recent = driver_history.tail(5)

    recent_form = (
        recent["FinishPosition"].mean()
        if len(recent) == 5 else float("nan")
    )

    recent_points = (
        recent["Points"].mean()
        if len(recent) == 5 else float("nan")
    )

    # Team performance over previous five races
    team_history = history[history["Team"] == team]

    team_races = (
        team_history.groupby(
            ["Year", "Round", "Date"], as_index=False
        )["FinishPosition"]
        .mean()
        .sort_values(["Date", "Round"])
        .tail(5)
    )

    team_form = (
        team_races["FinishPosition"].mean()
        if len(team_races) == 5 else float("nan")
    )

    # Driver wins before Brazil in 2026
    season = driver_history[
        driver_history["Year"] == 2026
    ]

    season_wins = int(season["Won"].sum())
    starts = len(season)

    season_win_rate = (
        season_wins / starts if starts else 0
    )

    predictions.append({
        "Driver": name,
        "Team": team,
        "QualifyingPosition": q_position,
        "GridPosition": grid_position,
        "RecentForm": recent_form,
        "TeamForm": team_form,
        "SeasonWins": season_wins,
        "SeasonWinRate": season_win_rate,
        "RecentPoints": recent_points
    })

# STEP 6: Predict winning probabilities
df = pd.DataFrame(predictions)

if df["GridPosition"].nunique() != 22:
    raise SystemExit("Starting grid contains duplicate positions.")

df["RawProbability"] = model.predict_proba(
    df[features]
)[:, 1]

total = df["RawProbability"].sum()

if total <= 0:
    raise SystemExit("No usable probability estimates.")

# Normalise the scores so they total 100%
df["WinProbability"] = (
    df["RawProbability"] / total * 100
)

df = df.sort_values(
    "WinProbability",
    ascending=False
)

# STEP 7: Display our predictions
print("\n--- BRAZILIAN GP 2026 PREDICTION ---")

print(
    df[
        [
            "Driver",
            "Team",
            "GridPosition",
            "WinProbability"
        ]
    ].round(2).to_string(index=False)
)

winner = df.iloc[0]

print("\nPREDICTED WINNER:", winner["Driver"])
print(
    "Estimated winning chance:",
    round(winner["WinProbability"], 2), "%"
)

# STEP 8: Save results for GitHub
Path("results").mkdir(exist_ok=True)

df.to_csv(
    "results/brazil_2026_post_qualifying.csv",
    index=False
)

print("\nPrediction saved successfully!")
