import joblib

# Load your trained regression model pipeline
reg_pipeline = joblib.load("models/best_regression_pipeline.pkl")

# Print all expected feature names
print("Expected features:", reg_pipeline.feature_names_in_)
print("Total features:", len(reg_pipeline.feature_names_in_))