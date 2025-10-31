from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import pandas as pd
import logging
import json
import hashlib
import redis.asyncio as redis
import os
from src.modeling.inference_pipeline import InferencePipeline
from contextlib import asynccontextmanager

# =====================
# Logging Setup
# =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# =====================
# Globals
# =====================
inference_pipeline: Optional[InferencePipeline] = None
redis_client: Optional[redis.Redis] = None


# =====================
# Startup and Shutdown
# =====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    global inference_pipeline, redis_client

    # --- Load ML Models ---
    try:
        inference_pipeline = InferencePipeline(
            reg_path="models/best_regression_pipeline.pkl",
            clf_path="models/best_classification_pipeline.pkl"
        )
        logger.info("✅ Models loaded successfully.")
    except Exception as e:
        logger.error(f"❌ Failed to load models: {e}")
        inference_pipeline = None

    # --- Redis Connection ---
    redis_host = os.getenv("REDIS_HOST", "redis-server")  # service name in Docker
    redis_port = int(os.getenv("REDIS_PORT", 6379))

    try:
        redis_client = redis.Redis(host=redis_host, port=redis_port, db=0)
        await redis_client.ping()
        logger.info(f"🔗 Connected to Redis at {redis_host}:{redis_port}")
    except Exception as e:
        logger.warning(f"⚠️ Redis not available: {e}")
        redis_client = None

    # --- Yield Control to FastAPI ---
    yield

    # --- Cleanup Section (on shutdown) ---
    if redis_client:
        await redis_client.close()
        logger.info("🧹 Redis connection closed.")

# =====================
# FastAPI App
# =====================
app = FastAPI(title="NexusDrive Inference API", version="2.0", lifespan=lifespan)

# =====================
# Pydantic Schemas
# =====================
class InferenceRequest(BaseModel):
    order_id: int
    distance_km: float
    relative_humidity_2m: float = Field(..., alias="relative_humidity_2m (%)")
    cloud_cover: float = Field(..., alias="cloud_cover (%)")
    wind_speed_10m: float = Field(..., alias="wind_speed_10m (km/h)")
    precipitation: float = Field(..., alias="precipitation (mm)")
    accept_hour_sin: float
    accept_hour_cos: float
    accept_dow_sin: float
    accept_dow_cos: float
    Weather_Label: str
    Traffic_Label: str
    city: str
    aoi_type: int


class InferenceResponse(BaseModel):
    order_id: int
    city: str
    Predicted_ETA: float
    Predicted_Delay: int


# =====================
# Health Check
# =====================
@app.get("/health")
async def health_check():
    return {"status": "ok", "message": "Async Inference API running 🚀"}


# =====================
# Helper: Generate Cache Key
# =====================
def generate_cache_key(data: dict) -> str:
    """Create a hash key from the request payload."""
    json_str = json.dumps(data, sort_keys=True)
    return hashlib.sha256(json_str.encode()).hexdigest()


# =====================
# Inference Endpoint (Async + Cache)
# =====================
@app.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest):
    global inference_pipeline, redis_client

    try:
        df = pd.DataFrame([request.model_dump(by_alias=True)])
        cache_key = f"inference:{generate_cache_key(request.model_dump(by_alias=True))}"

        # === Check Redis Cache ===
        if redis_client:
            cached = await redis_client.get(cache_key)
            if cached:
                logger.info(f"⚡ Cache hit for {cache_key}")
                return json.loads(cached)

        logger.info(f"📦 Cache miss → running inference for {request.order_id}")
        preds = inference_pipeline.predict(df)

        # Extract predictions
        eta = float(preds["ETA_Prediction"][0]) if isinstance(preds, dict) else float(preds[0])
        delay = int(preds["Delay_Prediction"][0]) if isinstance(preds, dict) else int(preds[1])


        response = {
            "order_id": request.order_id,
            "city": request.city,
            "Predicted_ETA": eta,
            "Predicted_Delay": delay
        }

        # === Store in Redis Cache ===
        if redis_client:
            await redis_client.setex(cache_key, 300, json.dumps(response))  # TTL = 5 min

        logger.info(f"✅ Inference successful for order_id={request.order_id}")
        return response

    except Exception as e:
        logger.exception("❌ Inference failed.")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {"message": "NexusDrive Inference API is running!"}

@app.get("/cache/health")
async def cache_health():
    if not redis_client:
        raise HTTPException(status_code=503, detail="Redis not initialized")
    try:
        pong = await redis_client.ping()
        return {"redis": "connected" if pong else "unreachable"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
