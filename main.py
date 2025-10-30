from src.modeling.modeling_pipeline import ModelingPipeline
from src.data_extraction import DataExtraction
from src.feature_engineering import run_feature_engineering
from src.outlier_removal import remove_outliers_iqr
import os
import pandas as pd


if __name__ == "__main__":
    # === Data Extraction ===
    if not os.path.exists("extracted_data/combined_enriched.csv"):
        main_data_folder = "Pickup_and_delivery_data"

        city_file_pairs = [
            [f"{main_data_folder}/delivery/delivery_yt.csv", f"{main_data_folder}/weather/yt_weather.csv"],
            [f"{main_data_folder}/delivery/delivery_cq.csv", f"{main_data_folder}/weather/cq_weather.csv"],
            [f"{main_data_folder}/delivery/delivery_hz.csv", f"{main_data_folder}/weather/hz_weather.csv"],
            [f"{main_data_folder}/delivery/delivery_jl.csv", f"{main_data_folder}/weather/jl_weather.csv"],
            [f"{main_data_folder}/delivery/delivery_sh.csv", f"{main_data_folder}/weather/sh_weather.csv"],
        ]

        extractor = DataExtraction(city_file_pairs, main_data_folder, output_folder="extracted_data")
        final_df = extractor.run()
    else:
        final_df = pd.read_csv("extracted_data/combined_enriched.csv")

    print("✅ Data loaded. Shape:", final_df.shape)

    # === Feature Engineering ===
    df_eng = run_feature_engineering(final_df)
    print("✅ Feature engineering complete. Shape:", df_eng.shape)

    # === Outlier Removal ===
    df_clean, outliers = remove_outliers_iqr(df_eng, column="ETA_target", multiplier=1.5)
    print("✅ Outlier removal complete. Clean shape:", df_clean.shape)

    # === Modeling Pipeline ===
    pipeline = ModelingPipeline()
    best_reg_model, best_clf_model = pipeline.run(df_clean)


