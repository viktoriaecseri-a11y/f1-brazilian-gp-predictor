
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss, brier_score_loss

# STEP 1: Load our data
df = pd.read_csv("data/f1_prepared.csv")

features = [
    "GridPosition",
    "RecentForm",
    "TeamForm",
    "SeasonWins",
    "SeasonWinRate",
    "RecentPoints"
]

# STEP 2: Create our Random Forest
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

# STEP 3: Walk-forward predictions
all_predictions = []

for year in [2021, 2022, 2023, 2024]:

    train = df[df["Year"] < year]
    test = df[df["Year"] == year].copy()

    model = create_model()
    model.fit(train[features], train["Won"])

    test["RawProbability"] = model.predict_proba(
        test[features]
    )[:, 1]

    # Normalise across each race
    test["RaceProbability"] = (
        test["RawProbability"]
        / test.groupby(["Year", "Round"])["RawProbability"]
          .transform("sum")
    )

    all_predictions.append(test)

predictions = pd.concat(
    all_predictions,
    ignore_index=True
)

# STEP 4: Evaluate raw driver-level probabilities
raw_log_loss = log_loss(
    predictions["Won"],
    predictions["RawProbability"].clip(1e-10, 1 - 1e-10)
)

raw_brier = brier_score_loss(
    predictions["Won"],
    predictions["RawProbability"]
)

# STEP 5: Evaluate normalised race probabilities
race_log_loss = -np.log(
    predictions.loc[
        predictions["Won"] == 1,
        "RaceProbability"
    ].clip(lower=1e-10)
).mean()

# STEP 6: Calculate Top-1 accuracy
top_predictions = (
    predictions.sort_values(
        "RaceProbability",
        ascending=False
    )
    .drop_duplicates(["Year", "Round"])
)

top1_accuracy = top_predictions["Won"].mean() * 100

# STEP 7: Display results
print("\n--- PROBABILITY EVALUATION ---")
print(f"Raw driver-level log loss: {raw_log_loss:.4f}")
print(f"Raw driver-level Brier score: {raw_brier:.4f}")
print(f"Race-level winner log loss: {race_log_loss:.4f}")
print(f"Top-1 accuracy: {top1_accuracy:.2f}%")

# STEP 8: Save predictions
Path("results").mkdir(exist_ok=True)

predictions[
    [
        "Year", "Round", "Race", "Driver",
        "Won", "RawProbability", "RaceProbability"
    ]
].to_csv(
    "results/probability_predictions.csv",
    index=False
)

print("\nPredictions saved successfully!")
