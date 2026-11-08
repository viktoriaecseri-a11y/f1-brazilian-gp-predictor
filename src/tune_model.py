
import pandas as pd
from itertools import product
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# STEP 1: Load our prepared dataset
df = pd.read_csv("data/f1_prepared.csv")

# STEP 2: Define the two feature sets
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

# STEP 3: Split data chronologically
train = df[df["Year"] <= 2023].copy()
validation = df[df["Year"] == 2024].copy()

# STEP 4: Calculate race winner accuracy
def race_accuracy(model, data, features):

    results = data.copy()

    results["WinProbability"] = (
        model.predict_proba(results[features])[:, 1]
    )

    predictions = (
        results.sort_values(
            "WinProbability",
            ascending=False
        )
        .drop_duplicates(["Year", "Round"])
    )

    return predictions["Won"].mean() * 100


# STEP 5: Test different Random Forest settings
parameters = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5, 8, None],
    "min_samples_leaf": [3, 5, 10]
}

experiments = []

for model_name, features in [
    ("Six features", original_features),
    ("Seven features", qualifying_features)
]:

    for trees, depth, leaf in product(
        parameters["n_estimators"],
        parameters["max_depth"],
        parameters["min_samples_leaf"]
    ):

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", RandomForestClassifier(
                n_estimators=trees,
                max_depth=depth,
                min_samples_leaf=leaf,
                random_state=42,
                n_jobs=-1
            ))
        ])

        model.fit(train[features], train["Won"])

        accuracy = race_accuracy(
            model, validation, features
        )

        experiments.append({
            "Model": model_name,
            "Trees": trees,
            "MaxDepth": str(depth),
            "MinLeaf": leaf,
            "Accuracy": accuracy
        })

# STEP 6: Compare results
results_df = pd.DataFrame(experiments)

results_df = results_df.sort_values(
    "Accuracy",
    ascending=False
)

print("\n--- TOP 10 CONFIGURATIONS ---")
print(results_df.head(10).to_string(index=False))

# STEP 7: Show best result for each feature set
print("\n--- BEST MODEL COMPARISON ---")

best_by_model = (
    results_df.groupby("Model")["Accuracy"]
    .max()
    .sort_values(ascending=False)
)

print(best_by_model)

# STEP 8: Calculate pole-position baseline
pole = validation[
    validation["GridPosition"] == 1
]

pole_accuracy = pole["Won"].mean() * 100

print("\nPole-position baseline:", round(pole_accuracy, 2), "%")

# STEP 9: Save results for GitHub
Path("results").mkdir(exist_ok=True)

results_df.to_csv(
    "results/qualifying_feature_experiment.csv",
    index=False
)

print("\nExperiment results saved successfully!")
