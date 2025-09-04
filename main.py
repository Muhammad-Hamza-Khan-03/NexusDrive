from src.data_extraction import DataExtraction

if __name__ == "__main__":
    main_data_folder = "Pickup_and_delivery_data"

    city_file_pairs = [
        [main_data_folder + "/delivery/delivery_yt.csv", main_data_folder + "/weather/yt_weather.csv"],
        [main_data_folder + "/delivery/delivery_cq.csv", main_data_folder + "/weather/cq_weather.csv"],
        [main_data_folder + "/delivery/delivery_hz.csv", main_data_folder + "/weather/hz_weather.csv"],
        [main_data_folder + "/delivery/delivery_jl.csv", main_data_folder + "/weather/jl_weather.csv"],
        [main_data_folder + "/delivery/delivery_sh.csv", main_data_folder + "/weather/sh_weather.csv"],
    ]

    extractor = DataExtraction(city_file_pairs, main_data_folder, output_folder="extracted_data")
    final_df = extractor.run()

