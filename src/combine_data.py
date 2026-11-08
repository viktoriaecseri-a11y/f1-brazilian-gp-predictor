
import pandas as pd
from pathlib import Path

# Load historical results
historical = pd.read_csv(
    "data/f1_results_2018_2025.csv"
)

# Load completed 2026 races
current = pd.read_csv(
    "data/f1_2026.csv"
)

# Combine the two datasets
combined = pd.concat(
    [historical, current],
    ignore_index=True
)

# Sort races chronologically
combined = combined.sort_values(
    ["Date", "Round"]
)

# Remove any duplicate driver-race records
combined = combined.drop_duplicates(
    subset=["Year", "Round", "DriverID"]
)

# Save our combined dataset
Path("data").mkdir(exist_ok=True)

combined.to_csv(
    "data/f1_combined.csv",
    index=False
)

# Check the results
print("Total driver results:", len(combined))

print(
    "Total races:",
    combined[["Year", "Round"]]
    .drop_duplicates()
    .shape[0]
)

print("Years included:", sorted(combined["Year"].unique()))

print("\nLatest races:")
print(
    combined[["Year", "Round", "Race", "Date"]]
    .drop_duplicates()
    .tail(5)
    .to_string(index=False)
)

print("\nCombined dataset saved successfully!")
