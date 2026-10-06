
import joblib
import pandas as pd
import math

# STEP 1: Load the trained model
model = joblib.load("waymate_model.pkl")
print("WAYMATE model loaded successfully!")

# STEP 2: Load cleaned pickup locations
locations = pd.read_csv("map_data/cleaned_pickup_points.csv")

print("\nAVAILABLE PICKUP LOCATIONS:")
for i, row in locations.iterrows():
    print(
        f"{i}: {row['area']} | {row['pickup_name']} | "
        f"Lat: {row['latitude']} | Lon: {row['longitude']}"
    )

# STEP 3: Select two pickup locations
rider_index = int(input("\nEnter rider pickup index: "))
passenger_index = int(input("Enter passenger pickup index: "))

if not (0 <= rider_index < len(locations)):
    raise ValueError("Invalid rider pickup index")

if not (0 <= passenger_index < len(locations)):
    raise ValueError("Invalid passenger pickup index")

rider = locations.iloc[rider_index]
passenger = locations.iloc[passenger_index]

# STEP 4: Calculate straight-line distance in km
lat1 = math.radians(float(rider["latitude"]))
lon1 = math.radians(float(rider["longitude"]))
lat2 = math.radians(float(passenger["latitude"]))
lon2 = math.radians(float(passenger["longitude"]))

dlat = lat2 - lat1
dlon = lon2 - lon1

a = (
    math.sin(dlat / 2) ** 2
    + math.cos(lat1) * math.cos(lat2)
    * math.sin(dlon / 2) ** 2
)

c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
pickup_distance_km = 6371 * c

print(f"\nPickup distance: {pickup_distance_km:.4f} km")

# STEP 5: Prepare model input
new_data = pd.DataFrame([{
    "route_overlap_percent": 80,
    "pickup_distance_km": pickup_distance_km,
    "drop_distance_km": 3.0,
    "detour_km": 1.5,
    "time_difference_min": 10,
    "trip_duration_min": 35,
    "rider_rating": 4.5,
    "acceptance_rate_percent": 90,
    "cancellation_rate_percent": 5,
    "verification_status": 1,
    "completed_shared_trips_history": 20
}])

# STEP 6: Predict match success
print("\nInput data:")
print(new_data.to_string(index=False))

prediction = model.predict(new_data)
probability = model.predict_proba(new_data)

print("\nPrediction:", prediction[0])

if prediction[0] == 1:
    print("Match successful")
else:
    print("Match not successful")

print("\nPrediction probabilities:")
for i, class_label in enumerate(model.classes_):
    print(f"Class {class_label}: {probability[0][i] * 100:.2f}%")