
import pandas as pd
from pathlib import Path

# Project folder
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset path
CSV_FILE = BASE_DIR / "map_data" / "north_roads_cleaned.csv"

# Check file
if not CSV_FILE.exists():
    print("ERROR: CSV file not found!")
    print("Expected location:", CSV_FILE)
    raise SystemExit

# Read data
df = pd.read_csv(CSV_FILE)

# Clean road names
df["road_name"] = df["road_name"].fillna("").astype(str).str.strip()

# Select named roads
named_df = df[
    (df["road_name"] != "") &
    (df["road_name"].str.lower() != "nan") &
    (df["road_name"].str.lower() != "unnamed road")
].copy()

# Count points for each road
road_counts = (
    named_df.groupby("road_name")
    .size()
    .reset_index(name="point_count")
    .sort_values("point_count", ascending=False)
)

# Display results
print("TOTAL DATA POINTS:", len(df))
print("NAMED DATA POINTS:", len(named_df))
print("UNIQUE NAMED ROADS:", len(road_counts))

print("\nTOP 20 ROAD NAMES:")
print(road_counts.head(20).to_string(index=False))

# Save report
OUTPUT_FILE = BASE_DIR / "map_data" / "road_name_counts.csv"
road_counts.to_csv(OUTPUT_FILE, index=False)

print("\nReport saved successfully!")
print("Location:", OUTPUT_FILE)