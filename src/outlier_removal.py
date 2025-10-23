# src/outlier_removal.py
"""
Outlier Removal Module
----------------------
Implements IQR-based outlier detection and removal in a reusable, 
object-oriented manner for any DataFrame column.
"""

from abc import ABC, abstractmethod
import pandas as pd


# =====================
# Abstract Base Class
# =====================
class BaseOutlierRemover(ABC):
    """Abstract base class for outlier removal strategies."""

    @abstractmethod
    def remove_outliers(self, df: pd.DataFrame, column: str):
        """
        Remove outliers from a dataframe column.
        Returns (clean_df, outlier_df)
        """
        pass


# =====================
# Concrete Implementation (IQR Method)
# =====================
class IQROutlierRemover(BaseOutlierRemover):
    """Removes outliers using the Interquartile Range (IQR) method."""

    def __init__(self, multiplier: float = 1.5):
        self.multiplier = multiplier

    def remove_outliers(self, df: pd.DataFrame, column: str):
        df_clean = df.copy()

        # Calculate IQR
        Q1 = df_clean[column].quantile(0.25)
        Q3 = df_clean[column].quantile(0.75)
        IQR = Q3 - Q1

        # Boundaries
        lower_bound = Q1 - self.multiplier * IQR
        upper_bound = Q3 + self.multiplier * IQR

        # Outlier mask
        outlier_mask = (df_clean[column] < lower_bound) | (df_clean[column] > upper_bound)

        outliers = df_clean[outlier_mask]
        df_clean = df_clean[~outlier_mask]

        # Log summary
        print("====== Outlier Removal Summary ======")
        print(f"Column processed: {column}")
        print(f"Original dataset shape: {df.shape}")
        print(f"Number of outliers removed: {outlier_mask.sum()}")
        print(f"Clean dataset shape: {df_clean.shape}")
        print(f"Percentage of data removed: {outlier_mask.sum() / len(df) * 100:.2f}%")
        print(f"Lower bound: {lower_bound:.2f}, Upper bound: {upper_bound:.2f}")
        print("=====================================")

        return df_clean, outliers


# =====================
# Utility Function
# =====================
def remove_outliers_iqr(df: pd.DataFrame, column: str, multiplier: float = 1.5):
    """
    Convenience wrapper for IQR-based outlier removal.
    """
    remover = IQROutlierRemover(multiplier)
    return remover.remove_outliers(df, column)
