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
    def __init__(self, lower_p: float = 0.25, upper_p: float = 0.75):
        # Initialize with desired percentile bounds (default 25th and 75th)
        self.lower_p = lower_p
        self.upper_p = upper_p
        self.lower_bound_ = None
        self.upper_bound_ = None

    def fit(self, X, y=None):
        # Compute bounds from training data (never fit on test data)
        self.lower_bound_ = np.percentile(X, self.lower_p * 100)
        self.upper_bound_ = np.percentile(X, self.upper_p * 100)
        return self

    def transform(self, X):
        # Clip values to [lower_bound, upper_bound]
        return np.clip(
            X, 
            a_min=self.lower_bound_, 
            a_max=self.upper_bound_
        )

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
