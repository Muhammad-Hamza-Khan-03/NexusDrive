import pandas as pd
from abc import ABC, abstractmethod

class AbstractDataAlign(ABC):
    @abstractmethod
    def transform_delivery_dataset(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def transform_amazon_dataset(self) -> pd.DataFrame:
        pass

    @abstractmethod
    def align(self) -> pd.DataFrame:
        pass


class DataAligner(AbstractDataAlign):
    def __init__(self, enriched_df: pd.DataFrame, amazon_df: pd.DataFrame):
        self.enriched_df = enriched_df
        self.amazon_df = amazon_df

    def transform_delivery_dataset(self) -> pd.DataFrame:
        """Reshape the enriched delivery dataset into common schema"""
        df1 = self.enriched_df.copy()

        # ensure datetime parsing
        df1["accept_time"] = pd.to_datetime(df1["time"], format="%m-%d %H:%M:%S", errors="coerce")
        pickup_datetime1 = df1["time"]

        df1_reshaped = pd.DataFrame({
            "order_id": df1["order_id"],
            "Date": pickup_datetime1.dt.date,
            "pickup_time": pickup_datetime1,
            "delivery_time": pd.to_datetime(df1["delivery_time"], errors="coerce"),
            "pickup_lat": df1["accept_gps_lat"].fillna(df1["lat"]),
            "pickup_lng": df1["accept_gps_lng"].fillna(df1["lng"]),
            "drop_lat": df1["delivery_gps_lat"],
            "drop_lng": df1["delivery_gps_lng"],
            "weather": df1["Weather_Label"],
            "traffic": df1["Traffic_Label"]
        })

        df1_reshaped["ETA_target"] = (
            (df1_reshaped["delivery_time"] - df1_reshaped["pickup_time"])
            .dt.total_seconds() / 60
        )
        return df1_reshaped

    def transform_amazon_dataset(self) -> pd.DataFrame:
        """Reshape the amazon dataset into common schema"""
        df2 = self.amazon_df.copy()

        pickup_datetime = pd.to_datetime(
            df2["Order_Date"].astype(str) + " " + df2["Order_Time"].fillna("00:00:00"),
            errors="coerce"
        )
        delivery_datetime = pickup_datetime + pd.to_timedelta(df2["Delivery_Time"], unit="m")

        df2_reshaped = pd.DataFrame({
            "order_id": df2["Order_ID"],
            "Date": df2["Order_Date"],
            "pickup_time": pickup_datetime,
            "delivery_time": delivery_datetime,
            "pickup_lat": df2["Store_Latitude"],
            "pickup_lng": df2["Store_Longitude"],
            "drop_lat": df2["Drop_Latitude"],
            "drop_lng": df2["Drop_Longitude"],
            "traffic": df2["Traffic"],
        })

        df2_reshaped["ETA_target"] = (
            (df2_reshaped["delivery_time"] - df2_reshaped["pickup_time"])
            .dt.total_seconds() / 60.0
        )
        return df2_reshaped

    def align(self) -> pd.DataFrame:
        """Align both datasets into a single final dataset"""
        df1 = self.transform_delivery_dataset()
        df2 = self.transform_amazon_dataset()
        final_df = pd.concat([df1, df2], ignore_index=True)
        return final_df
