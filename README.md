# 🚚 NexusDrive
### Real-Time Delivery ETA Prediction and Delay Risk Analytics

NexusDrive is a **machine learning–driven analytics system** that predicts **delivery Estimated Time of Arrival (ETA)** and classifies **delay risk** in real time.  
It integrates weather, traffic, and logistics data, using an optimized ML pipeline served via a **FastAPI + Dockerized microservice**, with **Redis caching** for fast inference and **MLflow** for experiment tracking.

---

## 🧠 Project Overview

**Core Features:**
- Predict real-time delivery ETA using regression models.
- Classify delivery delay risk with logistic models.
- Cache repeated inferences using Redis.
- Track and compare model performance with MLflow.
- Containerized for seamless deployment (FastAPI + Redis).
- Includes full testing suite for inference and training validation.

---

## 🧩 Datasets

| Dataset | Source | Description |
|----------|---------|-------------|
| **LaDe** | [Hugging Face - Cainiao-AI/LaDe](https://huggingface.co/datasets/Cainiao-AI/LaDe) | Real-world logistics delivery dataset. |
| **Amazon Delivery Dataset** | [Kaggle](https://www.kaggle.com/datasets/sujalsuthar/amazon-delivery-dataset) | Delivery time and shipment delay data from Amazon. |

---

## 🌦️ External API Used

**Historical Weather Data:**  
[Open-Meteo API](https://open-meteo.com/en/docs/historical-weather-api)  
Used to enrich dataset features with historical weather metrics.

---

## ☁️ Weather Labeling Rules

### 1. Fog
| Variable | Threshold |
|-----------|------------|
| Relative Humidity (2m) | > 90% |
| Cloud Cover Low | > 80% |
| Wind Speed (10m) | < 2 m/s |

---

### 2. Stormy
| Variable | Threshold |
|-----------|------------|
| Wind Gusts / Speed | > 12 m/s |
| Precipitation | > 2 mm/h |

---

### 3. Cloudy
| Variable | Threshold |
|-----------|------------|
| Cloud Cover | > 70% |
| Precipitation | < 1 mm/h |

---

### 4. Sandstorms
| Variable | Threshold |
|-----------|------------|
| Wind Speed | > 8–10 m/s |
| Precipitation | < 0.1 mm/h |
| Relative Humidity | < 40% |

---

### 5. Windy
| Variable | Threshold |
|-----------|------------|
| Wind Speed | 6–12 m/s |
| Precipitation | < 1 mm/h |

---

### 6. Sunny
| Variable | Threshold |
|-----------|------------|
| Cloud Cover | < 30% |
| Shortwave Radiation | High |
| Is Day | True |

---

## ⚙️ Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/NexusDrive.git
cd NexusDrive
```

### 2. Create Virtual Environment (optional)

```bash
python -m venv venv
source venv/bin/activate  # for Linux/Mac
venv\Scripts\activate     # for Windows
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

---

## 🧪 Testing

### Run All Tests

```bash
pytest -v
```

### Test Only the Inference Pipeline

```bash
python -m tests.inference_pipeline
```

---

## 🚀 Running the Application

### Run the FastAPI Server (Local)

```bash
uvicorn main:app --reload
```

### Run MLflow for Experiment Tracking

```bash
mlflow server --host 127.0.0.1 --port 8080
```

---

## 🐳 Docker Deployment

### 1. Build Docker Image

```bash
docker build -t nexusdrive_fastapi .
```

### 2. Run with Docker Compose

```bash
docker-compose up --build
```

### 3. Verify Running Containers

```bash
docker ps
```

Expected:

```
CONTAINER ID   IMAGE                COMMAND                  STATUS          PORTS
2e17d678e179   nexusdrive_fastapi   "uvicorn main:app --…"   Up 8 minutes    0.0.0.0:8000->8000/tcp
92aff2e9a8ab   redis:7              "docker-entrypoint.s…"   Up 11 minutes   0.0.0.0:6379->6379/tcp
```

### 4. Access the API

```bash
http://localhost:8000/docs
```

---

## ☁️ Docker Hub Image

You can directly pull the image:

```bash
docker pull hamzakhan03/nexusdrive_fastapi:latest
```

Run it:

```bash
docker run -d -p 8000:8000 hamzakhan03/nexusdrive_fastapi:latest
```

---

## 🧰 Tech Stack

| Component               | Technology                      |
| ----------------------- | ------------------------------- |
| **Backend API**         | FastAPI                         |
| **Model Serving**       | Pickle + Scikit-learn pipelines |
| **Caching Layer**       | Redis                           |
| **Containerization**    | Docker, Docker Compose          |
| **Experiment Tracking** | MLflow                          |
| **Testing**             | Pytest                          |
| **Dataset Handling**    | Pandas, NumPy                   |
| **Logging**             | Python logging module           |

---

## 📂 Project Structure

```
NexusDrive/
│
├── main.py                     # FastAPI entrypoint
├── train_model.py              # Model training script
├── inference_pipeline.py       # Inference class
├── tests/
│   ├── inference_pipeline.py   # Unit tests
│   └── ...
├── models/
│   ├── best_regression_pipeline.pkl
│   ├── best_classification_pipeline.pkl
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 🧾 Example Endpoints

| Method | Endpoint          | Description                      |
| ------ | ----------------- | -------------------------------- |
| `POST` | `/predict_eta`    | Predicts estimated delivery time |
| `POST` | `/classify_delay` | Predicts delay risk category     |
| `GET`  | `/health`         | Returns API health status        |

---

## 🧠 Model Lifecycle

1. Data preprocessing → feature engineering (time, location, weather)
2. Training using regression/classification pipelines
3. Model evaluation & tracking with MLflow
4. Pickle-based model export → `/models`
5. Loaded by FastAPI inference service
6. Caching frequent requests via Redis

---

## 🧮 Example Commands Summary

| Task                   | Command                                             |
| ---------------------- | --------------------------------------------------- |
| Train model            | `python train_model.py`                             |
| Run FastAPI            | `uvicorn main:app --reload`                         |
| Run MLflow             | `mlflow server --host 127.0.0.1 --port 8080`        |
| Run Tests              | `pytest -v`                                         |
| Run via Docker Compose | `docker-compose up --build`                         |
| Pull from Docker Hub   | `docker pull hamzakhan03/nexusdrive_fastapi:latest` |

---

## 🏁 Next Steps

* [ ] Deploy containerized stack on Render or AWS ECS
* [ ] Integrate live weather and GPS data streams
* [ ] Add Grafana dashboard for real-time analytics
* [ ] Enable model retraining API endpoint

---

## 👨‍💻 Author

**Hamza Khan**  
AI Engineer & Full-Stack Developer  
📧 [Contact](mailto:your-email@example.com) | 🌐 [LinkedIn](https://linkedin.com/in/your-profile)

---

**⭐ Star this repository if you find it helpful!**