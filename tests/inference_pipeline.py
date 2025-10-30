import pandas as pd
import logging
from src.modeling.inference_pipeline import InferencePipeline
import numpy as np
# =====================
# Logging Setup
# =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# =====================
# Inference Execution
# =====================
if __name__ == "__main__":
    try:
        # Initialize inference pipeline
        inference = InferencePipeline(
            reg_path="models/best_regression_pipeline.pkl",
            clf_path="models/best_classification_pipeline.pkl"
        )

        # Sample input data (mock new orders)
        df_new = pd.DataFrame([
            {
                'order_id': 1003,
    'distance_km': 2.0,
    'relative_humidity_2m (%)': 45,
    'cloud_cover (%)': 20,
    'wind_speed_10m (km/h)': 5.0,
    'precipitation (mm)': 0.0,
    'accept_hour_sin': np.sin(2 * np.pi * 10/24),  # 10:00 AM (non-peak)
    'accept_hour_cos': np.cos(2 * np.pi * 10/24),
    'accept_dow_sin': np.sin(2 * np.pi * 3/7),     # Thursday (or any normal weekday)
    'accept_dow_cos': np.cos(2 * np.pi * 3/7),
    'Weather_Label': 'Clear',
    'Traffic_Label': 'Low',
    'city': 'yt',
    'aoi_type': 1  # Residential
}
        ])

        logger.info(f"📦 Loaded sample data with shape: {df_new.shape}")

        # Run predictions
        preds = inference.predict(df_new)

        # Combine predictions with input
        df_results = df_new.copy()
        df_results["Predicted_ETA"] = preds["ETA_Prediction"]
        df_results["Predicted_Delay"] = preds["Delay_Prediction"]

        # Show results
        print("\n===== Inference Results =====")
        print(df_results[["order_id", "city", "Predicted_ETA", "Predicted_Delay"]])
        

    except Exception as e:
        logger.exception(f"❌ Inference failed: {e}")
