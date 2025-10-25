import os
import logging
import mlflow
import mlflow.sklearn

from src.modeling.data_preparation import DataPreparator
from src.modeling.preprocessing import PreprocessorFactory
from src.modeling.regression_models import RegressionTrainer
from src.modeling.classification_models import ClassificationTrainer
import pandas as pd


# =====================
# Logger Setup
# =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("logs/modeling_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# =====================
# Modeling Pipeline
# =====================
class ModelingPipeline:
    """End-to-end ML modeling pipeline with MLflow tracking."""

    def __init__(self, experiment_name="ETA_Delay_Prediction"):
        self.preparator = DataPreparator()
        self.reg_trainer = RegressionTrainer()
        self.clf_trainer = ClassificationTrainer()

        # Initialize MLflow
        mlflow.set_tracking_uri("file:./mlruns")  # local MLflow storage
        mlflow.set_experiment(experiment_name)

    def run(self, df_clean: pd.DataFrame):
        logger.info("🚀 Starting Modeling Pipeline...")

        # === Step 1: Prepare Features ===
        df_model, num_features, cat_features, y_reg, y_clf = self.preparator.prepare_features(df_clean)
        logger.info(f"Feature preparation complete. Shape: {df_model.shape}")

        # === Step 2: Time-based Split ===
        train_df, test_df = self.preparator.time_based_split(df_model)
        X_train = train_df.drop(['ETA_target', 'is_delayed'], axis=1)
        X_test = test_df.drop(['ETA_target', 'is_delayed'], axis=1)
        y_reg_train, y_reg_test = train_df['ETA_target'], test_df['ETA_target']
        y_clf_train, y_clf_test = train_df['is_delayed'], test_df['is_delayed']
        logger.info("Data split complete. Training size: %s, Test size: %s", X_train.shape, X_test.shape)

        # === Step 3: Preprocessing ===
        preprocessor = PreprocessorFactory.create_preprocessor(num_features, cat_features)
        X_train_proc = preprocessor.fit_transform(X_train)
        X_test_proc = preprocessor.transform(X_test)
        logger.info("Preprocessing pipeline fitted successfully.")

        # === Step 4: Model Training (MLflow Tracking) ===
        with mlflow.start_run(run_name="Regression_and_Classification_Pipeline"):
            logger.info("MLflow tracking started.")

            # Regression
            reg_results, best_reg_model = self.reg_trainer.train_models(
                X_train_proc, y_reg_train, X_test_proc, y_reg_test
            )
            best_reg = list(reg_results.keys())[0]
            mlflow.log_param("best_regression_model", best_reg)
            mlflow.log_metric("reg_RMSE", reg_results[best_reg]['rmse'])
            mlflow.log_metric("reg_MAE", reg_results[best_reg]['mae'])
            mlflow.sklearn.log_model(best_reg_model, "best_regression_model")

            # Classification
            clf_results, best_clf_model = self.clf_trainer.train_models(
                X_train_proc, y_clf_train, X_test_proc, y_clf_test
            )
            best_clf = list(clf_results.keys())[0]
            mlflow.log_param("best_classification_model", best_clf)
            mlflow.log_metric("clf_ROC_AUC", clf_results[best_clf]['roc_auc'])
            mlflow.log_metric("clf_Accuracy", clf_results[best_clf]['accuracy'])
            mlflow.sklearn.log_model(best_clf_model, "best_classification_model")

            logger.info("Models and metrics logged to MLflow successfully.")

            # Visualize
            mlflow.log_artifact("logs/modeling_pipeline.log")

        logger.info("✅ Modeling Pipeline Completed Successfully.")
        return best_reg_model, best_clf_model
