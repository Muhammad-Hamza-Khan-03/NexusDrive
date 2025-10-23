# src/modeling/regression_models.py
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import lightgbm as lgb

class RegressionTrainer:
    """Train and evaluate regression models."""

    def __init__(self):
        self.models = {
            'XGBoost': xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42),
            'LightGBM': lgb.LGBMRegressor(n_estimators=100, random_state=42)
        }

    def train_models(self, X_train, y_train, X_test, y_test):
        results = {}
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            results[name] = {'model': model, 'mae': mae, 'rmse': rmse, 'r2': r2, 'predictions': y_pred}
            print(f"{name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, R²: {r2:.4f}")

        best_model_name = min(results, key=lambda x: results[x]['rmse'])
        print(f"✅ Best Regression Model: {best_model_name}")
        return results, results[best_model_name]['model']
