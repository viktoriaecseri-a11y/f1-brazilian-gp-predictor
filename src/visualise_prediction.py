import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load the prediction results
df = pd.read_csv("results/brazil_2026_post_qualifying.csv")

# Keep only the top 10 predicted drivers
top10 = df.head(10).copy()

# Reverse order so the highest probability appears at the top
top10 = top10.iloc[::-1]

# Create output folder if needed
Path("results").mkdir(exist_ok=True)

# Create chart
plt.figure(figsize=(10, 6))
plt.barh(top10["Driver"], top10["WinProbability"])

# Add labels to bars
for i, value in enumerate(top10["WinProbability"]):
    plt.text(value + 0.3, i, f"{value:.2f}%", va="center")

plt.title("2026 Brazilian Grand Prix Prediction - Top 10 Drivers")
plt.xlabel("Estimated Win Probability (%)")
plt.ylabel("Driver")
plt.grid(axis="x", alpha=0.3)

plt.tight_layout()

# Save chart
plt.savefig(
    "results/brazil_2026_prediction_chart.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Chart saved successfully!")