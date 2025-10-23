# src/modeling/preprocessing.py
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

class PreprocessorFactory:
    """Factory for creating preprocessing pipelines."""

    @staticmethod
    def create_preprocessor(numerical_features, categorical_features):
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_features),
                ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
            ]
        )
        return preprocessor
