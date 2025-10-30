import joblib

# Load your trained regression model pipeline
lgbm_pipeline = joblib.load("models/best_classification_pipeline.pkl")

# Print all expected feature names
print("Expected features:", lgbm_pipeline.feature_names_in_)
print("Total features:", len(lgbm_pipeline.feature_names_in_))