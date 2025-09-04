import os
import pandas as pd
from abc import ABC, abstractmethod
from mock.weather_generator import WeatherMockGenerator
from mock.traffic_generator import TrafficMockGenerator

class AbstractDataIngest(ABC):
    @abstractmethod
    def load_data(self):
        pass

    @abstractmethod
    def enrich_with_weather(self):
        pass

    @abstractmethod
    def enrich_with_traffic_and_vehicles(self):
        pass


class DataIngest(AbstractDataIngest):
    def __init__(self, delivery_file: str, weather_file: str):
        self.delivery_file = delivery_file
        self.weather_file = weather_file
        self.delivery_df = None
        self.weather_df = None
        self.load_data()

    def load_data(self):
        # check files exist
        if not os.path.exists(self.delivery_file):
            raise FileNotFoundError(f"❌ Delivery file not found: {self.delivery_file}")
        if not os.path.exists(self.weather_file):
            raise FileNotFoundError(f"❌ Weather file not found: {self.weather_file}")

        # load csv
        self.delivery_df = pd.read_csv(self.delivery_file)
        self.weather_df = pd.read_csv(self.weather_file)

        # ensure datetime conversion
        self.delivery_df["delivery_time"] = pd.to_datetime(
                "2023-" + self.delivery_df["delivery_time"].astype(str),
                format="%Y-%m-%d %H:%M:%S",
                errors="coerce"
            )

        self.weather_df["time"] = pd.to_datetime(self.weather_df["time"])

    def enrich_with_weather(self) -> pd.DataFrame:
        """
        Align delivery times with nearest earlier weather hour (using merge_asof).
        Then generate weather labels with WeatherMockGenerator.
        """
        merged = pd.merge_asof(
            self.delivery_df.sort_values("delivery_time"),
            self.weather_df.sort_values("time"),
            left_on="delivery_time",
            right_on="time",
            direction="backward",          # take earlier weather record
            tolerance=pd.Timedelta("1h")   # only allow max 1-hour difference
        )

        generator = WeatherMockGenerator()
        merged["Weather_Label"] = merged.apply(generator.generate_label, axis=1)

        self.delivery_df = merged
        return self.delivery_df

    def enrich_with_traffic_and_vehicles(self) -> pd.DataFrame:
        generator = TrafficMockGenerator()
        self.delivery_df = generator.generate(self.delivery_df)
        return self.delivery_df

