
import pandas as pd
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# STEP 1: Load prepared race data
df = pd.read_csv("data/f1_prepared.csv")

# STEP 2: Define our feature sets
original_features = [
    "GridPosition",
    "RecentForm",
    "TeamForm",
    "SeasonWins",
    "SeasonWinRate",
    "RecentPoints"
]

qualifying_features = original_features + [
    "RecentQualifying"
]

team_points_features = original_features + [
    "RecentTeamPoints"
]

feature_sets = {
    "Original": original_features,
    "Qualifying": qualifying_features,
    "Team Points": team_points_features
}

# STEP 3: Define validation seasons
validation_years = [2021, 2022, 2023, 2024]


# STEP 4: Create Random Forest model
def create_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            max_depth=8,
            min_samples_leaf=5,
            random_state=42,
            n_jobs=-1
        ))
    ])


# STEP 5: Evaluate predictions
def evaluate(model, data, features):
    results = data.copy()

    results["WinProbability"] = model.predict_proba(
        results[features]
    )[:, 1]

    # Choose the driver with highest probability per race
    predictions = (
        results.sort_values(
            "WinProbability",
            ascending=False
        )
        .drop_duplicates(["Year", "Round"])
    )

    correct = int(predictions["Won"].sum())
    total = len(predictions)

    return correct, total


# STEP 6: Walk-forward validation
all_results = []

for year in validation_years:

    train = df[df["Year"] < year].copy()
    validation = df[df["Year"] == year].copy()

    print(f"Evaluating {year}...")

    for model_name, features in feature_sets.items():

        model = create_model()

        model.fit(
            train[features],
            train["Won"]
        )

        correct, total = evaluate(
            model,
            validation,
            features
        )

        all_results.append({
            "Year": year,
            "Model": model_name,
            "Correct": correct,
            "Total": total,
            "Accuracy": correct / total * 100
        })

    # Baseline: always select pole sitter
    pole = validation[
        validation["GridPosition"] == 1
    ]

    total = validation[
        ["Year", "Round"]
    ].drop_duplicates().shape[0]

    all_results.append({
        "Year": year,
        "Model": "Pole baseline",
        "Correct": int(pole["Won"].sum()),
        "Total": total,
        "Accuracy": pole["Won"].sum() / total * 100
    })

# STEP 7: Show accuracy by season
results_df = pd.DataFrame(all_results)

print("\n--- ACCURACY BY SEASON ---")
print(
    results_df.pivot(
        index="Year",
        columns="Model",
        values="Accuracy"
    ).round(2).to_string()
)

# STEP 8: Show overall accuracy
summary = (
    results_df.groupby("Model")[["Correct", "Total"]]
    .sum()
)

summary["OverallAccuracy"] = (
    summary["Correct"] / summary["Total"] * 100
)

print("\n--- OVERALL RESULTS ---")
print(summary.round(2).to_string())

# STEP 9: Save results
Path("results").mkdir(exist_ok=True)

results_df.to_csv(
    "results/team_points_experiment.csv",
    index=False
)

print("\nExperiment completed successfully!")
