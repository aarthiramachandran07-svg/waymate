import joblib

print("Checking model...")

model = joblib.load("waymate_model.pkl")

print("MODEL TYPE:")
print(type(model))

print("\nNUMBER OF FEATURES:")
print(getattr(model, "n_features_in_", "Not available"))

print("\nFEATURE NAMES:")
print(getattr(model, "feature_names_in_", "Not available"))

print("\nPIPELINE STEPS:")
if hasattr(model, "steps"):
    for name, step in model.steps:
        print(name, ":", type(step))
else:
    print("This is not a pipeline.")
