# src/feature_engineering.py
"""
Feature Engineering Module
--------------------------
This module performs advanced feature engineering on delivery datasets,
including geospatial, temporal, and derived metrics such as speed and delay classification.
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
from haversine import haversine


# =====================
# Abstract Base Class
# =====================
class BaseFeatureEngineer(ABC):
    """Abstract base class for all feature engineering strategies."""

    @abstractmethod
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply feature engineering transformations to the dataframe."""
        pass


# =====================
# Concrete Implementation
# =====================
class DeliveryFeatureEngineer(BaseFeatureEngineer):
    """Feature engineering pipeline for delivery datasets."""

    def __init__(self, speed_min=1, speed_max=150, distance_bins=20):
        self.speed_min = speed_min
        self.speed_max = speed_max
        self.distance_bins = distance_bins

    @staticmethod
    def _calculate_haversine_distance(row) -> float:
        """Calculate distance between pickup and delivery points in kilometers."""
        pickup = (row["accept_gps_lat"], row["accept_gps_lng"])
        delivery = (row["delivery_gps_lat"], row["delivery_gps_lng"])
        return haversine(pickup, delivery)

    def _add_distance_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        df["distance_km"] = df.apply(self._calculate_haversine_distance, axis=1)
        return df

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        if not np.issubdtype(df["accept_time"].dtype, np.datetime64):
            df["accept_time"] = pd.to_datetime(df["accept_time"])

        df["accept_hour"] = df["accept_time"].dt.hour
        df["accept_day_of_week"] = df["accept_time"].dt.dayofweek
        df["accept_month"] = df["accept_time"].dt.month
        df["is_weekend"] = (df["accept_day_of_week"] >= 5).astype(int)

        # Cyclical encoding
        df["accept_hour_sin"] = np.sin(2 * np.pi * df["accept_hour"] / 24)
        df["accept_hour_cos"] = np.cos(2 * np.pi * df["accept_hour"] / 24)
        df["accept_dow_sin"] = np.sin(2 * np.pi * df["accept_day_of_week"] / 7)
        df["accept_dow_cos"] = np.cos(2 * np.pi * df["accept_day_of_week"] / 7)
        return df

    def _add_speed_feature(self, df: pd.DataFrame) -> pd.DataFrame:
        df["avg_speed_kmh"] = df["distance_km"] / df["ETA_target"]
        df = df[(df["avg_speed_kmh"] > self.speed_min) & (df["avg_speed_kmh"] < self.speed_max)]
        return df

    def _add_delay_label(self, df: pd.DataFrame) -> pd.DataFrame:
        df["distance_bin"] = pd.cut(df["distance_km"], bins=self.distance_bins, labels=False)
        delay_thresholds = df.groupby("distance_bin")["ETA_target"].quantile(0.75).to_dict()
        df["delay_threshold"] = df["distance_bin"].map(delay_thresholds)
        df["is_delayed"] = (df["ETA_target"] > df["delay_threshold"]).astype(int)
        return df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply all feature engineering steps sequentially."""
        df_eng = df.copy()
        df_eng = self._add_distance_feature(df_eng)
        df_eng = self._add_time_features(df_eng)
        df_eng = self._add_speed_feature(df_eng)
        df_eng = self._add_delay_label(df_eng)
        return df_eng


# =====================
# Utility Runner
# =====================
def run_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience function for quick feature engineering."""
    engineer = DeliveryFeatureEngineer()
    return engineer.transform(df)
