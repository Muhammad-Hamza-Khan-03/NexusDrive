# NexusDrive
 Real-Time Delivery ETA Prediction and Delay Risk Analytics

Dataset links:
LaDe: https://huggingface.co/datasets/Cainiao-AI/LaDe
Amazon delievery: https://www.kaggle.com/datasets/sujalsuthar/amazon-delivery-dataset

API:
openmateo historical api: https://open-meteo.com/en/docs/historical-weather-api 

Rules for weather labeling:
1. Fog

Needs high humidity, low visibility, low wind.

Variables:

Relative Humidity (2 m) → > 90%

Cloud cover Low → > 80%

Wind Speed (10 m) → < 2 m/s

Optional: Weather code if it has fog indicator.

2. Stormy

Strong winds + precipitation.

Variables:

Wind Gusts (10 m) or Wind Speed (10 m) → > 12 m/s

Precipitation (rain + snow) → > 2 mm/h

Weather code if thunderstorm available.

3. Cloudy

High cloud cover, no heavy rain.

Variables:

Cloud cover Total → > 70%

Precipitation → < 1 mm/h

4. Sandstorms

Strong winds, dry conditions, low precipitation.

Variables:

Wind Speed (10 m) → > 8–10 m/s

Precipitation → < 0.1 mm/h

Relative Humidity → < 40%

(This will be region-specific: desert areas → higher chance.)

5. Windy

Strong winds, but no storm-level precipitation.

Variables:

Wind Speed (10 m) → 6–12 m/s

Precipitation → < 1 mm/h

6. Sunny

High solar radiation, low cloud cover.

Variables:

Cloud cover Total → < 30%

Shortwave Solar Radiation → high

Is Day or Night → must be day

test_inference_pipeline: python -m tests.inference_pipeline

All tests: pytest -v

Entry file to run all : python train_model.py


RUN FAST API server: uvicorn main:app --reload

RUN ML FLOW: mlflow server --host 127.0.0.1 --port 8080

pull docker image : hamzakhan03/nexusdrive_fastapi:latest