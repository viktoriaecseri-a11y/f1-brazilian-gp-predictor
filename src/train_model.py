
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# STEP 1: Load prepared data
df = pd.read_csv("data/f1_prepared.csv")

# STEP 2: Define our feature sets

original_features = [
    "GridPosition",
    "RecentForm",
    "TeamForm",
    "SeasonWins"
]

improved_features = [
    "GridPosition",
    "RecentForm",
    "TeamForm",
    "SeasonWins",
    "SeasonWinRate",
    "RecentPoints"
]

# STEP 3: Split data by time
train = df[df["Year"] <= 2023].copy()
validation = df[df["Year"] == 2024].copy()
test = df[df["Year"] >= 2025].copy()


# STEP 4: Define our Random Forest
def create_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            min_samples_leaf=5,
            random_state=42
        ))
    ])


# STEP 5: Evaluate the winner of each race
def evaluate(model, data, features):

    results = data.copy()

    results["WinProbability"] = (
        model.predict_proba(results[features])[:, 1]
    )

    # Select the highest probability driver per race
    predictions = (
        results.sort_values(
            "WinProbability", ascending=False
        )
        .drop_duplicates(subset=["Year", "Round"])
    )

    correct = predictions["Won"].sum()
    total = len(predictions)

    return 100 * correct / total


# STEP 6: Compare models on 2024
feature_sets = {
    "Original": original_features,
    "Improved": improved_features
}

validation_scores = {}

for name, features in feature_sets.items():

    model = create_model()
    model.fit(train[features], train["Won"])

    score = evaluate(model, validation, features)
    validation_scores[name] = score

    print(f"{name} validation accuracy: {score:.2f}%")


# STEP 7: Pick the better feature set
best_name = max(validation_scores, key=validation_scores.get)
best_features = feature_sets[best_name]

print("\nSelected model:", best_name)

# STEP 8: Train on all data through 2024
final_train = df[df["Year"] <= 2024].copy()

final_model = create_model()

final_model.fit(
    final_train[best_features],
    final_train["Won"]
)

# STEP 9: Evaluate on 2025-2026
test_accuracy = evaluate(
    final_model, test, best_features
)

# Pole-position baseline on same test races
pole = test[test["GridPosition"] == 1]
pole_accuracy = pole["Won"].mean() * 100

print("\n--- FINAL MODEL COMPARISON ---")

print(f"Random Forest: {test_accuracy:.2f}%")
print(f"Pole position: {pole_accuracy:.2f}%")

print(
    f"Difference: "
    f"{test_accuracy - pole_accuracy:.2f} percentage points"
)

# STEP 10: Feature importance
forest = final_model.named_steps["classifier"]

importance = pd.DataFrame({
    "Feature": best_features,
    "Importance": forest.feature_importances_
}).sort_values("Importance", ascending=False)

print("\n--- FEATURE IMPORTANCE ---")
print(importance.to_string(index=False))

print("\nTraining completed successfully!")
