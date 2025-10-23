# src/modeling/classification_models.py
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
import lightgbm as lgb

class ClassificationTrainer:
    """Train and evaluate classification models."""

    def __init__(self):
        self.models = {
            'XGBoost': xgb.XGBClassifier(objective='binary:logistic', n_estimators=100, random_state=42),
            'LightGBM': lgb.LGBMClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        }

    def train_models(self, X_train, y_train, X_test, y_test):
        results = {}
        for name, model in self.models.items():
            print(f"Training {name}...")
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            acc = model.score(X_test, y_test)
            roc = roc_auc_score(y_test, y_pred_proba)
            results[name] = {'model': model, 'accuracy': acc, 'roc_auc': roc}
            print(f"{name} - Accuracy: {acc:.4f}, ROC AUC: {roc:.4f}")
            if name == 'XGBoost':
                print("\nClassification Report:\n", classification_report(y_test, y_pred))

        best_model_name = max(results, key=lambda x: results[x]['roc_auc'])
        print(f"✅ Best Classification Model: {best_model_name}")
        return results, results[best_model_name]['model']
