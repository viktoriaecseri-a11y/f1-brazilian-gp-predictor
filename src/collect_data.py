
import requests
import pandas as pd
import time
from pathlib import Path

# Create a folder for our datasets
data_folder = Path("data")
data_folder.mkdir(exist_ok=True)

# Collect seasons from 2018 to 2025
years = range(2018, 2026)

for year in years:

    filename = data_folder / f"f1_{year}.csv"

    # Skip seasons already downloaded
    if filename.exists():
        print(f"{year} already downloaded. Skipping...")
        continue

    print(f"Downloading {year} season...")

    season_results = []
    offset = 0

    while True:

        url = f"https://api.jolpi.ca/ergast/f1/{year}/results.json"

        response = requests.get(
            url,
            params={"limit": 100, "offset": offset},
            headers={"User-Agent": "F1BrazilPredictor/1.0"},
            timeout=30
        )

        # Stop safely if the API rate limit is reached
        if response.status_code == 429:
            print("API rate limit reached.")
            print("Previously saved seasons are safe.")
            print("Run the script again after the limit resets.")
            raise SystemExit(0)

        response.raise_for_status()

        data = response.json()
        races = data["MRData"]["RaceTable"]["Races"]

        for race in races:
            for result in race["Results"]:

                season_results.append({
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

        # Check if we've downloaded every result
        total = int(data["MRData"]["total"])
        offset += 100

        if offset >= total:
            break

        time.sleep(1)

    # Save this season immediately
    df = pd.DataFrame(season_results)
    df.to_csv(filename, index=False)

    print(f"Saved {year}: {len(df)} driver results")

    time.sleep(2)

# Combine all seasons when they're available
files = [data_folder / f"f1_{year}.csv" for year in years]

if all(file.exists() for file in files):

    datasets = [pd.read_csv(file) for file in files]
    combined = pd.concat(datasets, ignore_index=True)

    combined.to_csv(
        data_folder / "f1_results_2018_2025.csv",
        index=False
    )

    print("All seasons downloaded successfully!")
    print("Total records:", len(combined))
    print(combined.head())
