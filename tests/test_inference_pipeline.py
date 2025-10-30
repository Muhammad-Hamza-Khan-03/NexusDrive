import logging
from src.modeling.inference_pipeline import InferencePipeline
from tests.utils.sample_data import make_sample_input

logger = logging.getLogger(__name__)

def test_inference(model_paths):
    """Test model inference on sample data."""
    try:
        pipeline = InferencePipeline(**model_paths)
        df_new = make_sample_input()
        logger.info(f"Loaded sample input shape: {df_new.shape}")

        preds = pipeline.predict(df_new)

        # ✅ Fix: preds is a dict, not a DataFrame
        assert isinstance(preds, dict)
        assert "ETA_Prediction" in preds
        assert "Delay_Prediction" in preds

        logger.info("✅ Inference pipeline executed successfully")

        print("\n===== Inference Output =====")
        print(preds)

    except Exception as e:
        logger.exception(f"❌ Inference test failed: {e}")
        raise
