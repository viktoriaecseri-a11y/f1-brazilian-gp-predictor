
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# STEP 1: Load predictions from our model
df = pd.read_csv(
    "results/probability_predictions.csv"
)

# STEP 2: Create probability groups
# Example: 0-10%, 10-20%, 20-30%, etc.
bins = np.linspace(0, 1, 11)

df["ProbabilityGroup"] = pd.cut(
    df["RaceProbability"],
    bins=bins,
    include_lowest=True
)

# STEP 3: Calculate actual win rate in each group
calibration = (
    df.groupby(
        "ProbabilityGroup",
        observed=True
    )
    .agg(
        PredictedProbability=("RaceProbability", "mean"),
        ActualWinRate=("Won", "mean"),
        NumberOfPredictions=("Won", "count")
    )
    .reset_index()
)

# STEP 4: Display the results
print("\n--- CALIBRATION RESULTS ---")

print(
    calibration[
        [
            "PredictedProbability",
            "ActualWinRate",
            "NumberOfPredictions"
        ]
    ].round(3).to_string(index=False)
)

# STEP 5: Create calibration chart
plt.figure(figsize=(8, 6))

# Perfect calibration reference
plt.plot(
    [0, 1],
    [0, 1],
    "--",
    label="Perfect calibration"
)

# Our model's actual calibration
plt.plot(
    calibration["PredictedProbability"],
    calibration["ActualWinRate"],
    marker="o",
    label="Random Forest"
)

plt.xlabel("Predicted win probability")
plt.ylabel("Actual winning frequency")
plt.title("F1 Random Forest Probability Calibration")

plt.xlim(0, 1)
plt.ylim(0, 1)
plt.grid(alpha=0.3)
plt.legend()

# STEP 6: Save chart for GitHub
Path("results").mkdir(exist_ok=True)

plt.savefig(
    "results/calibration_chart.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nCalibration chart saved successfully!")
