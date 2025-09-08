import os
import pandas as pd
from src.data_ingest import DataIngest
from src.data_transformation import DataAligner


class DataExtraction:
    def __init__(self, city_file_pairs: list, main_data_folder: str, output_folder: str = "extracted_data"):
        """
        city_file_pairs: list of [delivery_file, weather_file] for each city
        main_data_folder: base folder for data
        output_folder: where processed datasets will be saved
        """
        self.city_file_pairs = city_file_pairs
        self.main_data_folder = main_data_folder
        self.output_folder = output_folder
        os.makedirs(self.output_folder, exist_ok=True)

    def process_city_datasets(self) -> pd.DataFrame:
        """Process each city (delivery + weather) and combine into one enriched dataset"""
        enriched_datasets = []

        for delivery_file, weather_file in self.city_file_pairs:
            loader = DataIngest(delivery_file, weather_file)
            enriched_df = loader.enrich_with_weather()
            enriched_df = loader.enrich_with_traffic_and_vehicles()

            # add city column automatically from filename
            city_name = delivery_file.split("/")[-1].replace("delivery_", "").replace(".csv", "")
            enriched_df["city"] = city_name

            enriched_df["accept_time"] = pd.to_datetime(enriched_df["time"], format="%m-%d %H:%M:%S", errors="coerce")
            enriched_df["pickup_time"] = enriched_df["time"]
            enriched_df["delivery_time"] = pd.to_datetime(enriched_df["delivery_time"], errors="coerce")
            
            enriched_df["ETA_target"] = (
                (enriched_df["delivery_time"] - enriched_df["pickup_time"])
                .dt.total_seconds() / 60
            )

            enriched_datasets.append(enriched_df)
            print(f"✅ Processed {city_name} dataset")

        combined_enriched_df = pd.concat(enriched_datasets, ignore_index=True)
        print("✅ Combined enriched dataset shape:", combined_enriched_df.shape)

        # Save combined enriched data
        combined_enriched_path = os.path.join(self.output_folder, "combined_enriched.csv")
        combined_enriched_df.to_csv(combined_enriched_path, index=False)
        print(f"💾 Saved combined enriched dataset to {combined_enriched_path}")

        return combined_enriched_df

    def align_with_amazon(self, combined_enriched_df: pd.DataFrame) -> pd.DataFrame:
        """Align the enriched dataset with amazon.csv and save final dataset"""
        amazon_file = os.path.join("Pickup_and_delivery_data/delivery/amazon_delivery.csv")
        amazon_df = pd.read_csv(amazon_file)

        aligner = DataAligner(combined_enriched_df, amazon_df)
        final_df = aligner.align()

        # Save final aligned dataset
        final_path = os.path.join(self.output_folder, "final_aligned.csv")
        final_df.to_csv(final_path, index=False)

        print("✅ Final aligned dataset:")
        print(final_df.head())
        print(final_df.shape)
        print(f"💾 Saved final aligned dataset to {final_path}")

        return final_df

    def run(self):
        """Full pipeline: process cities → align with amazon → return final dataset"""
        combined_enriched_df = self.process_city_datasets()
        final_df = self.align_with_amazon(combined_enriched_df)
        return final_df
