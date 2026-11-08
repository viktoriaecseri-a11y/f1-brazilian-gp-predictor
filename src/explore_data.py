
import matplotlib.pyplot as plt
from pathlib import Path

import pandas as pd

# Load our historical Formula 1 dataset
df = pd.read_csv("data/f1_results_2018_2025.csv")

# Display the first 5 rows
print(df.head())

# Check the number of rows and columns
print("Dataset shape:", df.shape)

# Show all column names
print("Columns:", df.columns.tolist())

# Find the winners of each race
winners = df[df["Won"] == 1]

# Count the total number of races
total_races = len(winners)

# Find winners who started from pole position
pole_winners = winners[winners["GridPosition"] == 1]

# Count how many won from pole
pole_wins = len(pole_winners)

# Calculate the percentage
pole_win_percentage = (pole_wins / total_races) * 100

# Display the results
print("Total races:", total_races)
print("Races won from pole:", pole_wins)
print("Pole position win percentage:", round(pole_win_percentage, 2), "%")

# Calculate win rate for each starting position
grid_analysis = df.groupby("GridPosition")["Won"].agg(
    ["sum", "count", "mean"]
)

# Convert win rate into a percentage
grid_analysis["WinPercentage"] = (
    grid_analysis["mean"] * 100
)

# Show starting positions 1 to 10
print("\nWin rates by starting position:")
print(
    grid_analysis.loc[1:10, [
        "sum", "count", "WinPercentage"
    ]].round(2)
)

# Create a chart of win probability by grid position
positions = grid_analysis.loc[1:10]

plt.figure(figsize=(10, 6))

plt.bar(
    positions.index,
    positions["WinPercentage"],
    color="darkgreen"
)

plt.title("F1 Win Probability by Starting Position (2018-2025)")
plt.xlabel("Starting Grid Position")
plt.ylabel("Win Percentage (%)")

plt.xticks(range(1, 11))
plt.grid(axis="y", alpha=0.3)

# Save chart for our GitHub project
Path("results").mkdir(exist_ok=True)

plt.savefig(
    "results/grid_position_win_probability.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# Calculate total wins for each driver
driver_wins = df.groupby("Driver")["Won"].sum()

# Sort drivers by number of wins
driver_wins = driver_wins.sort_values(ascending=False)

# Show the top 10 drivers
print("\nTop 10 drivers by race wins:")
print(driver_wins.head(10))

# Sort race results into chronological order
df = df.sort_values(["Date", "Round"])

# Calculate average finish over previous 5 races
df["RecentForm"] = (
    df.groupby("DriverID")["FinishPosition"]
    .transform(
        lambda x: x.shift(1).rolling(5, min_periods=5).mean()
    )
)

# Show Max Verstappen's latest results
max_results = df[df["DriverID"] == "max_verstappen"]

print("\nMax Verstappen's recent form:")
print(
    max_results[
        ["Date", "Race", "FinishPosition", "RecentForm"]
    ].tail(10).to_string(index=False)
)
