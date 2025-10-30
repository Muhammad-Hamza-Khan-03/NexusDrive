# tests/utils/sample_data.py
import pandas as pd
import numpy as np

def make_sample_input():
    """Generate mock order data with minimal features for inference."""
    return pd.DataFrame([
        {
            "order_id": 1003,
            "region_id": 1,
            "city": "yt",
            "courier_id": 99,
            "lng": 67.001,
            "lat": 24.861,
            "aoi_id": 10,
            "aoi_type": 1,
            "accept_time": "2025-10-30 10:00:00",
            "accept_gps_time": "2025-10-30 10:00:10",
            "accept_gps_lng": 67.001,
            "accept_gps_lat": 24.861,
            "delivery_time": "2025-10-30 10:30:00",
            "delivery_gps_time": "2025-10-30 10:30:10",
            "delivery_gps_lng": 67.003,
            "delivery_gps_lat": 24.863,
            "ds": "2025-10-30",
            "time": 10.0,
            "relative_humidity_2m (%)": 45,
            "cloud_cover (%)": 20,
            "wind_speed_10m (km/h)": 5.0,
            "precipitation (mm)": 0.0,
            "is_day ()": 1,
            "Weather_Label": "Clear",
            "Traffic_Label": "Low",
            "pickup_time": 0.0,
            "distance_km": 2.0,
            "accept_hour": 10,
            "accept_day_of_week": 3,
            "accept_month": 10,
            "is_weekend": 0,
            "accept_hour_sin": np.sin(2 * np.pi * 10 / 24),
            "accept_hour_cos": np.cos(2 * np.pi * 10 / 24),
            "accept_dow_sin": np.sin(2 * np.pi * 3 / 7),
            "accept_dow_cos": np.cos(2 * np.pi * 3 / 7),
            "avg_speed_kmh": 25.0,
            "distance_bin": "short",
            "delay_threshold": 5.0,
            "accept_day": 30,
            "accept_dayofweek": 3
        }
    ])
