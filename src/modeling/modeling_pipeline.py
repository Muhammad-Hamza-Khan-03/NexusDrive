# src/modeling/modeling_pipeline.py
from src.modeling.data_preparation import DataPreparator
from src.modeling.preprocessing import PreprocessorFactory
from src.modeling.regression_models import RegressionTrainer
from src.modeling.classification_models import ClassificationTrainer
import pandas as pd

class ModelingPipeline:
    """End-to-end ML modeling pipeline for ETA prediction and delay classification."""

    def __init__(self):
        self.preparator = DataPreparator()
        self.reg_trainer = RegressionTrainer()
        self.clf_trainer = ClassificationTrainer()

    def run(self, df_clean: pd.DataFrame):
        """Run full modeling process."""
        print("\n🚀 Starting Modeling Pipeline...")

        # === Step 1: Prepare Features ===
        df_model, num_features, cat_features, y_reg, y_clf = self.preparator.prepare_features(df_clean)

        # === Step 2: Time-based Split ===
        train_df, test_df = self.preparator.time_based_split(df_model)

        X_train = train_df.drop(['ETA_target', 'is_delayed'], axis=1)
        X_test = test_df.drop(['ETA_target', 'is_delayed'], axis=1)
        y_reg_train, y_reg_test = train_df['ETA_target'], test_df['ETA_target']
        y_clf_train, y_clf_test = train_df['is_delayed'], test_df['is_delayed']

        # === Step 3: Preprocessing ===
        preprocessor = PreprocessorFactory.create_preprocessor(num_features, cat_features)
        X_train_proc = preprocessor.fit_transform(X_train)
        X_test_proc = preprocessor.transform(X_test)

        # === Step 4: Regression Models ===
        reg_results, best_reg_model = self.reg_trainer.train_models(
            X_train_proc, y_reg_train, X_test_proc, y_reg_test
        )

        # === Step 5: Classification Models ===
        clf_results, best_clf_model = self.clf_trainer.train_models(
            X_train_proc, y_clf_train, X_test_proc, y_clf_test
        )


        print("\n✅ Modeling Pipeline Completed Successfully.")
        return best_reg_model, best_clf_model
