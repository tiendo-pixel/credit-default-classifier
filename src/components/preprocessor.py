import os
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler, OneHotEncoder


class OutlierCapper(BaseEstimator, TransformerMixin):
    # Winsorize: clip values outside [p1, p99] to reduce outlier noise
    ...


class DataPreprocessor:
    def __init__(self, num_cols: list, cat_cols: list):
        # Build sklearn pipeline for numeric and categorical features
        ...

    def fit_transform(self, X_train: pd.DataFrame):
        # Fit on train set and transform
        ...

    def transform(self, X: pd.DataFrame):
        # Transform only (never refit on test set — avoids data leakage)
        ...

    def save(self, filepath: str = "models/preprocessor.joblib"):
        # Save fitted pipeline to disk
        ...

    @classmethod
    def load(cls, filepath: str = "models/preprocessor.joblib"):
        # Load pipeline from disk for inference
        ...
