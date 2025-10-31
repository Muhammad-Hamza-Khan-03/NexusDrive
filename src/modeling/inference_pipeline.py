import joblib
import pandas as pd

class InferencePipeline:
    """Loads trained pipeline and predicts on new data."""

    def __init__(self, reg_path="models/best_regression_pipeline.pkl",
                 clf_path="models/best_classification_pipeline.pkl"):
        self.reg_pipeline = joblib.load(reg_path)
        self.clf_pipeline = joblib.load(clf_path)

    def predict(self, df_new: pd.DataFrame):
        reg_pred = self.reg_pipeline.predict(df_new)
        clf_pred = self.clf_pipeline.predict(df_new)
        return {
            "ETA_Prediction": reg_pred,
            "Delay_Prediction": clf_pred
        }
