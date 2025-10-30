# tests/conftest.py
import os
import sys
import logging
import pytest

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler()]
)

@pytest.fixture(scope="session")
def model_paths():
    """Fixture to provide common model paths"""
    return {
        "reg_path": "models/best_regression_pipeline.pkl",
        "clf_path": "models/best_classification_pipeline.pkl"
    }
