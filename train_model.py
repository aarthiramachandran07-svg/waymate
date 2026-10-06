
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# STEP 1: Load dataset
df = pd.read_excel("dataset chennai.xlsx")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)


# STEP 2: Select features
X = df[
    [
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
]


# STEP 3: Select target
y = df["match_success"]


# STEP 4: Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data:", X_train.shape)
print("Testing data:", X_test.shape)


# STEP 5: Create model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)


# STEP 6: Train model
print("\nTraining the model...")
model.fit(X_train, y_train)

print("Model training completed!")


# STEP 7: Test model
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))


# STEP 8: Save model
joblib.dump(model, "waymate_model.pkl")

print("\nModel saved successfully!")