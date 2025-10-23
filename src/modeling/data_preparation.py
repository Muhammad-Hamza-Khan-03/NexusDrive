# src/modeling/data_preparation.py
import pandas as pd
import numpy as np

class DataPreparator:
    """Handles data preparation and feature generation for modeling."""

    def __init__(self):
        pass

    def prepare_features(self, df: pd.DataFrame):
        """Generate and return numerical + categorical feature lists and targets."""
        df_model = df.copy()

        datetime_cols = ['accept_time', 'delivery_time', 'pickup_time']
        for col in datetime_cols:
            if col in df_model.columns:
                df_model[col] = pd.to_datetime(df_model[col])

        # Time-based features
        df_model['accept_hour'] = df_model['accept_time'].dt.hour
        df_model['accept_day'] = df_model['accept_time'].dt.day
        df_model['accept_dayofweek'] = df_model['accept_time'].dt.dayofweek
        df_model['accept_month'] = df_model['accept_time'].dt.month

        # Cyclical encoding
        df_model['accept_hour_sin'] = np.sin(2 * np.pi * df_model['accept_hour'] / 24)
        df_model['accept_hour_cos'] = np.cos(2 * np.pi * df_model['accept_hour'] / 24)
        df_model['accept_dow_sin'] = np.sin(2 * np.pi * df_model['accept_dayofweek'] / 7)
        df_model['accept_dow_cos'] = np.cos(2 * np.pi * df_model['accept_dayofweek'] / 7)

        numerical_features = [
            'distance_km', 'relative_humidity_2m (%)', 'cloud_cover (%)',
            'wind_speed_10m (km/h)', 'precipitation (mm)',
            'accept_hour_sin', 'accept_hour_cos', 'accept_dow_sin', 'accept_dow_cos'
        ]

        categorical_features = ['Weather_Label', 'Traffic_Label', 'city', 'aoi_type']

        y_reg = df_model['ETA_target']
        y_clf = df_model['is_delayed']

        return df_model, numerical_features, categorical_features, y_reg, y_clf

    def time_based_split(self, df: pd.DataFrame, test_size=0.2):
        """Perform time-based split to prevent lookahead bias."""
        df_sorted = df.sort_values('accept_time').reset_index(drop=True)
        split_idx = int(len(df_sorted) * (1 - test_size))
        train_df = df_sorted.iloc[:split_idx]
        test_df = df_sorted.iloc[split_idx:]
        return train_df, test_df
