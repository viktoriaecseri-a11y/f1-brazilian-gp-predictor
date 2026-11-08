
import requests
import pandas as pd
import time
from pathlib import Path

# Collect all completed races before Brazil
url = "https://api.jolpi.ca/ergast/f1/2026/results.json"

results = []
offset = 0
limit = 100

while True:
    print(f"Downloading results from offset {offset}...")

    response = requests.get(
        url,
        params={"limit": limit, "offset": offset},
        headers={"User-Agent": "F1BrazilPredictor/1.0"},
        timeout=30
    )

    # Handle API rate limits
    if response.status_code == 429:
        print("Rate limit reached. Try again after it resets.")
        raise SystemExit(1)

    response.raise_for_status()

    data = response.json()
    races = data["MRData"]["RaceTable"]["Races"]

    # Extract each driver's results
    for race in races:

        # Brazil is round 20 in 2026
        if int(race["round"]) >= 20:
            continue

        for result in race["Results"]:
            results.append({
                "Year": int(race["season"]),
                "Round": int(race["round"]),
                "Race": race["raceName"],
                "Date": race["date"],
                "Circuit": race["Circuit"]["circuitId"],
                "DriverID": result["Driver"]["driverId"],
                "Driver": (
                    result["Driver"]["givenName"] + " " +
                    result["Driver"]["familyName"]
                ),
                "Team": result["Constructor"]["name"],
                "GridPosition": int(result["grid"]),
                "FinishPosition": int(result["position"]),
                "Points": float(result["points"]),
                "Status": result["status"],
                "Won": int(result["position"] == "1")
            })

    # Move to the next page
    offset += limit

    # Stop when all available pages are downloaded
    total = int(data["MRData"]["total"])

    if offset >= total:
        break

    time.sleep(1)

# Convert to a pandas DataFrame
df = pd.DataFrame(results)

# Save the complete dataset
Path("data").mkdir(exist_ok=True)
df.to_csv("data/f1_2026.csv", index=False)

# Display the number of races
print("\nTotal 2026 races:", df["Round"].nunique())
print("Total driver results:", len(df))

print("\nRaces collected:")
print(
    df[["Round", "Race"]]
    .drop_duplicates()
    .sort_values("Round")
    .to_string(index=False)
)

print("\n2026 dataset saved successfully!")
