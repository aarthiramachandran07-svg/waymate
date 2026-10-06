
import streamlit as st
import pandas as pd
import joblib
import folium
import math
from streamlit_folium import st_folium
from pathlib import Path

# ---------------------------------
# WAYMATE - Pickup Matching App
# ---------------------------------

st.set_page_config(
    page_title="WAYMATE - Ride Matching",
    page_icon="🚗",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "waymate_model.pkl"
CSV_PATH = BASE_DIR / "map_data" / "cleaned_pickup_points.csv"

# ---------------------------------
# Load model and locations
# ---------------------------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_locations():
    return pd.read_csv(CSV_PATH)

try:
    model = load_model()
    locations = load_locations()
except Exception as e:
    st.error(f"Could not load model or location data: {e}")
    st.stop()

# Find location name, latitude and longitude columns
def find_column(df, choices):
    for col in choices:
        if col in df.columns:
            return col
    return None

name_col = find_column(
    locations,
   ["pickup_name", "location", "name", "pickup_location", "place", "area"]
)
lat_col = find_column(
    locations,
    ["latitude", "lat", "Latitude", "LATITUDE"]
)
lon_col = find_column(
    locations,
    ["longitude", "lng", "lon", "Longitude", "LONGITUDE"]
)

if lat_col is None or lon_col is None:
    st.error(
        "Latitude/longitude columns were not found in "
        "cleaned_pickup_points.csv. Please check the column names."
    )
    st.write("Available columns:", list(locations.columns))
    st.stop()

if name_col is None:
    locations = locations.copy()
    locations["location_name"] = [
        f"Pickup {i + 1}" for i in range(len(locations))
    ]
    name_col = "location_name"

locations = locations.copy()
locations[lat_col] = pd.to_numeric(
    locations[lat_col], errors="coerce"
)
locations[lon_col] = pd.to_numeric(
    locations[lon_col], errors="coerce"
)
locations = locations.dropna(
    subset=[lat_col, lon_col]
).reset_index(drop=True)

if len(locations) == 0:
    st.error("No valid pickup coordinates were found.")
    st.stop()

# ---------------------------------
# Distance calculation
# ---------------------------------

def calculate_distance(lat1, lon1, lat2, lon2):
    earth_radius_km = 6371.0

    lat1 = math.radians(float(lat1))
    lon1 = math.radians(float(lon1))
    lat2 = math.radians(float(lat2))
    lon2 = math.radians(float(lon2))

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(dlon / 2) ** 2
    )

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return earth_radius_km * c

# ---------------------------------
# Session state
# Apply pending map selections BEFORE
# creating the selectbox widgets.
# ---------------------------------

if "pending_rider_index" in st.session_state:
    st.session_state["rider_index"] = st.session_state.pop(
        "pending_rider_index"
    )

if "pending_passenger_index" in st.session_state:
    st.session_state["passenger_index"] = st.session_state.pop(
        "pending_passenger_index"
    )

if "rider_index" not in st.session_state:
    st.session_state["rider_index"] = 0

if "passenger_index" not in st.session_state:
    st.session_state["passenger_index"] = (
        1 if len(locations) > 1 else 0
    )

# Keep indices valid if the CSV changes
st.session_state["rider_index"] = min(
    st.session_state["rider_index"], len(locations) - 1
)
st.session_state["passenger_index"] = min(
    st.session_state["passenger_index"], len(locations) - 1
)

# ---------------------------------
# App heading
# ---------------------------------

st.title("🚗 WAYMATE")
st.subheader("Smart Ride Matching System")
st.write(
    "Choose two pickup points or click on the map "
    "to select the nearest available pickup."
)

# ---------------------------------
# Pickup dropdowns
# ---------------------------------

location_options = list(range(len(locations)))

def format_location(index):
    return str(locations.iloc[index][name_col])

col1, col2 = st.columns(2)

with col1:
    st.selectbox(
        "Rider pickup location",
        options=location_options,
        format_func=format_location,
        key="rider_index"
    )

with col2:
    st.selectbox(
        "Passenger pickup location",
        options=location_options,
        format_func=format_location,
        key="passenger_index"
    )

rider_index = st.session_state["rider_index"]
passenger_index = st.session_state["passenger_index"]

rider = locations.iloc[rider_index]
passenger = locations.iloc[passenger_index]

rider_lat = float(rider[lat_col])
rider_lon = float(rider[lon_col])
passenger_lat = float(passenger[lat_col])
passenger_lon = float(passenger[lon_col])

pickup_distance = calculate_distance(
    rider_lat, rider_lon, passenger_lat, passenger_lon
)

# ---------------------------------
# Map
# ---------------------------------

st.markdown("### 📍 Select a pickup by clicking the map")

select_for = st.radio(
    "Which pickup do you want to set?",
    ["Rider", "Passenger"],
    horizontal=True,
    key="select_for"
)

center_lat = (rider_lat + passenger_lat) / 2
center_lon = (rider_lon + passenger_lon) / 2

m = folium.Map(
    location=[center_lat, center_lon],
    zoom_start=12,
    tiles=None
)

folium.TileLayer(
    tiles=(
        "https://server.arcgisonline.com/ArcGIS/rest/"
        "services/World_Street_Map/MapServer/tile/{z}/{y}/{x}"
    ),
    attr="Esri",
    name="Street Map"
).add_to(m)

# Show all available pickup points
for idx, row in locations.iterrows():
    lat = float(row[lat_col])
    lon = float(row[lon_col])
    label = str(row[name_col])

    if idx == rider_index:
        color = "green"
        popup_text = f"Rider: {label}"
    elif idx == passenger_index:
        color = "blue"
        popup_text = f"Passenger: {label}"
    else:
        color = "gray"
        popup_text = label

    folium.CircleMarker(
        location=[lat, lon],
        radius=7 if idx in [rider_index, passenger_index] else 4,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.9,
        tooltip=popup_text
    ).add_to(m)

# Draw a line between selected points
folium.PolyLine(
    locations=[
        [rider_lat, rider_lon],
        [passenger_lat, passenger_lon]
    ],
    color="red",
    weight=3,
    tooltip="Selected pickup points"
).add_to(m)

map_data = st_folium(
    m,
    width=None,
    height=500,
    returned_objects=["last_clicked"],
    key="waymate_map"
)

# ---------------------------------
# Process map click
# Save as pending selection.
# Do not modify an already-created widget.
# ---------------------------------

if map_data and map_data.get("last_clicked"):
    clicked = map_data["last_clicked"]
    click_key = (
        round(clicked["lat"], 6),
        round(clicked["lng"], 6),
        select_for
    )

    if click_key != st.session_state.get(
        "last_processed_click"
    ):
        st.session_state["last_processed_click"] = click_key

        distances = locations.apply(
            lambda row: calculate_distance(
                clicked["lat"],
                clicked["lng"],
                row[lat_col],
                row[lon_col]
            ),
            axis=1
        )

        nearest_index = int(distances.idxmin())

        if select_for == "Rider":
            st.session_state["pending_rider_index"] = (
                nearest_index
            )
        else:
            st.session_state["pending_passenger_index"] = (
                nearest_index
            )

        st.rerun()
# ---------------------------------
# Destination selection
# ---------------------------------

st.markdown("### 🏁 Select Destinations")

col3, col4 = st.columns(2)

with col3:
    rider_destination_index = st.selectbox(
        "Rider destination",
        options=location_options,
        format_func=format_location,
        key="rider_destination_index"
    )

with col4:
    passenger_destination_index = st.selectbox(
        "Passenger destination",
        options=location_options,
        format_func=format_location,
        key="passenger_destination_index"
    )

rider_destination = locations.iloc[
    rider_destination_index
]
passenger_destination = locations.iloc[
    passenger_destination_index
]

rider_drop_distance = calculate_distance(
    rider_lat,
    rider_lon,
    float(rider_destination[lat_col]),
    float(rider_destination[lon_col])
)

passenger_drop_distance = calculate_distance(
    passenger_lat,
    passenger_lon,
    float(passenger_destination[lat_col]),
    float(passenger_destination[lon_col])
)

st.write(
    "Rider drop distance:",
    f"{rider_drop_distance:.3f} km"
)

st.write(
    "Passenger drop distance:",
    f"{passenger_drop_distance:.3f} km"
)
drop_distance = calculate_distance(
    float(rider_destination[lat_col]),
    float(rider_destination[lon_col]),
    float(passenger_destination[lat_col]),
    float(passenger_destination[lon_col])
)
# ---------------------------------
# Distance display
# ---------------------------------

st.markdown("### 📏 Pickup distance")

metric1, metric2, metric3 = st.columns(3)

with metric1:
    st.metric("Rider", str(rider[name_col]))

with metric2:
    st.metric("Passenger", str(passenger[name_col]))

with metric3:
    st.metric(
        "Straight-line distance",
        f"{pickup_distance:.3f} km"
    )

st.caption(
    "Distance shown is straight-line distance, "
    "not actual road distance."
)

# ---------------------------------
# ML prediction
# ---------------------------------

st.markdown("### 🤖 Ride match prediction")

# These are sample/default values for features
# that are not yet calculated from real trip data.
# Only pickup_distance_km is taken from the map.

input_data = pd.DataFrame([{
    "route_overlap_percent": 70.0,
    "pickup_distance_km": pickup_distance,
    "drop_distance_km": drop_distance,
    "detour_km": 1.0,
    "time_difference_min": 5.0,
    "trip_duration_min": 30.0,
    "rider_rating": 4.5,
    "acceptance_rate_percent": 80.0,
    "cancellation_rate_percent": 10.0,
    "verification_status": 1,
    "completed_shared_trips_history": 5
}])
st.markdown("### 🧭 Trip Summary")

summary_col1, summary_col2 = st.columns(2)

with summary_col1:
    st.write("**Rider Trip**")
    st.write("Pickup:", rider[name_col])
    st.write("Destination:", rider_destination[name_col])
    st.write(
        "Trip distance:",
        f"{rider_drop_distance:.3f} km"
    )

with summary_col2:
    st.write("**Passenger Trip**")
    st.write("Pickup:", passenger[name_col])
    st.write("Destination:", passenger_destination[name_col])
    st.write(
        "Trip distance:",
        f"{passenger_drop_distance:.3f} km"
    )
feature_order = [
    "route_overlap_percent",
    "pickup_distance_km",
    "drop_distance_km",
    "detour_km",
    "time_difference_min",
    "trip_duration_min",
    "rider_rating",
    "acceptance_rate_percent",
    "cancellation_rate_percent",
    "verification_status",
    "completed_shared_trips_history"
]

input_data = input_data[feature_order]

try:
    prediction = model.predict(input_data)[0]

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_data)[0]
        classes = list(model.classes_)

        if 1 in classes:
            match_probability = (
                probabilities[classes.index(1)] * 100
            )
        else:
            match_probability = 0.0
    else:
        match_probability = None

    if int(prediction) == 1:
        st.success("Potential match: Yes")
    else:
        st.warning("Potential match: No")

    if match_probability is not None:
        st.metric(
            "Model class 1 probability",
            f"{match_probability:.2f}%"
        )

    with st.expander("View model input values"):
        st.dataframe(input_data, use_container_width=True)

    st.caption(
        "Prototype prediction only. Several model inputs "
        "are sample values, not live route or trip data. "
        "The probability is not a guarantee of a successful match."
    )

except Exception as e:
    st.error(f"Prediction error: {e}")