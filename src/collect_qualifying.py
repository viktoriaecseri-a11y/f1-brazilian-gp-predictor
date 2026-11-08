
import requests
import pandas as pd
import time
from pathlib import Path

# STEP 1: Create our data folder
data_folder = Path("data")
data_folder.mkdir(exist_ok=True)

# STEP 2: Collect qualifying from 2018 to 2026
years = range(2018, 2027)

for year in years:

    filename = data_folder / f"qualifying_{year}.csv"

    # Don't download the same season twice
    if filename.exists():
        print(f"{year} already saved. Skipping...")
        continue

    print(f"Downloading {year} qualifying results...")

    all_results = []
    offset = 0

    while True:

        url = (
            f"https://api.jolpi.ca/ergast/f1/"
            f"{year}/qualifying.json"
        )

        response = requests.get(
            url,
            params={"limit": 100, "offset": offset},
            headers={"User-Agent": "F1BrazilPredictor/1.0"},
            timeout=30
        )

        # Stop safely if we hit the rate limit
        if response.status_code == 429:
            print("API rate limit reached.")
            print("Previously saved seasons are safe.")
            print("Run again after the limit resets.")
            raise SystemExit(1)

        response.raise_for_status()

        data = response.json()
        races = data["MRData"]["RaceTable"]["Races"]

        for race in races:

            round_number = int(race["round"])

            # For 2026, use only races before Brazil
            if year == 2026 and round_number >= 20:
                continue

            for result in race["QualifyingResults"]:

                all_results.append({
                    "Year": year,
                    "Round": round_number,
                    "Race": race["raceName"],
                    "DriverID": result["Driver"]["driverId"],
                    "Driver": (
                        result["Driver"]["givenName"] + " " +
                        result["Driver"]["familyName"]
                    ),
                    "QualifyingPosition": int(
                        result["position"]
                    ),
                    "Q1": result.get("Q1"),
                    "Q2": result.get("Q2"),
                    "Q3": result.get("Q3")
                })

        # Move to the next page
        offset += int(data["MRData"]["limit"])
        total = int(data["MRData"]["total"])

        if offset >= total:
            break

        time.sleep(1)

    # Save each season as soon as it's complete
    df = pd.DataFrame(all_results)
    df.to_csv(filename, index=False)

    print(
        f"Saved {year}: "
        f"{df['Round'].nunique()} qualifying sessions"
    )

    time.sleep(2)

# STEP 3: Combine all qualifying seasons
files = [
    data_folder / f"qualifying_{year}.csv"
    for year in years
]

if all(file.exists() for file in files):

    datasets = [pd.read_csv(file) for file in files]

    combined = pd.concat(
        datasets,
        ignore_index=True
    )

    combined = combined.drop_duplicates(
        subset=["Year", "Round", "DriverID"]
    )

    combined.to_csv(
        data_folder / "f1_qualifying_combined.csv",
        index=False
    )

    print("\nAll qualifying data saved!")
    print("Total records:", len(combined))
    print(
        "Total qualifying sessions:",
        combined[["Year", "Round"]]
        .drop_duplicates()
        .shape[0]
    )

    print("\nSample data:")
    print(combined.tail(10).to_string(index=False))
