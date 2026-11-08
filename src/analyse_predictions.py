
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# STEP 1: Load prepared F1 data
df = pd.read_csv("data/f1_prepared.csv")

features = [
    "GridPosition",
    "RecentForm",
    "TeamForm",
    "SeasonWins"
]

# STEP 2: Split historical and test races
train = df[df["Year"] <= 2024].copy()
test = df[df["Year"] >= 2025].copy()

# STEP 3: Create and train Random Forest
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        min_samples_leaf=5
    ))
])

model.fit(train[features], train["Won"])

# STEP 4: Predict each driver's win probability
test["WinProbability"] = model.predict_proba(
    test[features]
)[:, 1]

# STEP 5: Select highest-ranked driver per race
predictions = (
    test.sort_values(
        "WinProbability",
        ascending=False
    )
    .drop_duplicates(
        subset=["Year", "Round"]
    )
    .sort_values(["Year", "Round"])
)

# STEP 6: Get actual winners
actual_winners = test[test["Won"] == 1][
    ["Year", "Round", "Driver"]
].rename(columns={"Driver": "ActualWinner"})

# STEP 7: Compare predictions to actual results
comparison = predictions[
    ["Year", "Round", "Race", "Driver",
     "GridPosition", "WinProbability", "Won"]
].rename(columns={"Driver": "PredictedWinner"})

comparison = comparison.merge(
    actual_winners,
    on=["Year", "Round"],
    how="left"
)

comparison["Correct"] = (
    comparison["PredictedWinner"]
    == comparison["ActualWinner"]
)

# STEP 8: Display incorrect predictions
mistakes = comparison[~comparison["Correct"]]

print("\n--- INCORRECT PREDICTIONS ---")
print(
    mistakes[
        ["Year", "Race", "PredictedWinner",
         "ActualWinner", "GridPosition"]
    ].to_string(index=False)
)

print("\nTotal mistakes:", len(mistakes))

# STEP 9: Display feature importance
forest = model.named_steps["classifier"]

importance = pd.DataFrame({
    "Feature": features,
    "Importance": forest.feature_importances_
}).sort_values("Importance", ascending=False)

print("\n--- FEATURE IMPORTANCE ---")
print(importance.to_string(index=False))
