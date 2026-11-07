import os
from typing import List, Optional, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler


class OutlierCapper(BaseEstimator, TransformerMixin):
    """
    Percentile-based outlier capper for continuous features.
    Clips extreme values outside [lower_p, upper_p] quantiles fitted strictly on training data.
    """

    def __init__(self, lower_p: float = 0.01, upper_p: float = 0.99):
        self.lower_p = lower_p
        self.upper_p = upper_p
        self.lower_bound_ = None
        self.upper_bound_ = None

    def fit(self, X: Union[np.ndarray, pd.DataFrame], y=None):
        x_arr = np.asarray(X)
        self.lower_bound_ = np.nanpercentile(x_arr, self.lower_p * 100, axis=0)
        self.upper_bound_ = np.nanpercentile(x_arr, self.upper_p * 100, axis=0)
        return self

    def transform(self, X: Union[np.ndarray, pd.DataFrame]):
        x_arr = np.asarray(X).copy()
        return np.clip(x_arr, a_min=self.lower_bound_, a_max=self.upper_bound_)


class DataPreprocessor:
    """
    Production-grade DataPreprocessor for credit risk underwriting modeling.
    
    Implements banking-standard three-tier imputation & transformation:
    1. Zero-imputation + RobustScaler: For credit card & facility features where missing = no debt/facility.
    2. Median-imputation + RobustScaler: For behavioral financial features (income, ratios, credit scores).
    3. Categorical 'UNKNOWN' + OneHotEncoder: For demographic and categorical variables.
    """

    DEFAULT_ZERO_COLS: List[str] = [
        "cic_cc_util",
        "cic_fin_ratio",
        "cic_balance_trend",
        "cic_util_trend",
    ]

    def __init__(
        self,
        zero_cols: Optional[List[str]] = None,
        median_cols: Optional[List[str]] = None,
        cat_cols: Optional[List[str]] = None,
    ):
        self.zero_cols = zero_cols or self.DEFAULT_ZERO_COLS
        self.median_cols = median_cols
        self.cat_cols = cat_cols
        self.pipeline: Optional[ColumnTransformer] = None
        self.feature_names_in_: List[str] = []

    def _resolve_column_groups(self, df: pd.DataFrame):
        """Automatically identify and partition numeric and categorical columns if not explicitly provided."""
        all_cols = list(df.columns)
        self.feature_names_in_ = all_cols

        # Filter zero-impute columns present in input
        resolved_zero = [c for c in self.zero_cols if c in all_cols]

        # Categorical columns
        if self.cat_cols is None:
            resolved_cat = df.select_dtypes(include=["object", "category"]).columns.tolist()
        else:
            resolved_cat = [c for c in self.cat_cols if c in all_cols]

        # Median numeric columns (all remaining numeric columns)
        if self.median_cols is None:
            resolved_median = [
                c
                for c in df.select_dtypes(include=[np.number]).columns
                if c not in resolved_zero
            ]
        else:
            resolved_median = [c for c in self.median_cols if c in all_cols]

        self.zero_cols = resolved_zero
        self.cat_cols = resolved_cat
        self.median_cols = resolved_median

    def build_pipeline(self) -> ColumnTransformer:
        """Construct Scikit-Learn ColumnTransformer pipeline with RobustScalers."""
        zero_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value=0.0)),
            ("scaler", RobustScaler()),
        ])

        median_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ])

        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        transformer = ColumnTransformer(
            transformers=[
                ("num_zero", zero_pipe, self.zero_cols),
                ("num_median", median_pipe, self.median_cols),
                ("cat", cat_pipe, self.cat_cols),
            ],
            remainder="passthrough",
        )
        return transformer

    def fit(self, X_train: pd.DataFrame, y=None):
        """Fit preprocessing transformers strictly on training partition to eliminate data leakage."""
        self._resolve_column_groups(X_train)
        self.pipeline = self.build_pipeline()
        self.pipeline.fit(X_train)
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform inference/test partition using fitted statistics."""
        if self.pipeline is None:
            raise RuntimeError("DataPreprocessor must be fitted before calling transform().")
        
        # Ensure all expected columns are present
        X_aligned = X.copy()
        for col in self.feature_names_in_:
            if col not in X_aligned.columns:
                X_aligned[col] = np.nan
        
        # Reorder to match training schema
        X_aligned = X_aligned[self.feature_names_in_]
        return self.pipeline.transform(X_aligned)

    def fit_transform(self, X_train: pd.DataFrame, y=None) -> np.ndarray:
        """Fit on training data and return transformed numeric matrix."""
        return self.fit(X_train, y).transform(X_train)

    def get_feature_names_out(self) -> List[str]:
        """Return list of transformed feature names after one-hot encoding."""
        if self.pipeline is None:
            raise RuntimeError("DataPreprocessor must be fitted to get feature names.")
        try:
            return list(self.pipeline.get_feature_names_out())
        except Exception:
            return []

    def save(self, filepath: str = "models/preprocessor.joblib"):
        """Serialize preprocessor ColumnTransformer to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.pipeline, filepath)
        print(f"[OK] Preprocessor pipeline saved to: {filepath}")

    @classmethod
    def load(cls, filepath: str = "models/preprocessor.joblib") -> "DataPreprocessor":
        """Load serialized ColumnTransformer artifact from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Preprocessor file not found at: {filepath}")
        
        preprocessor = cls()
        preprocessor.pipeline = joblib.load(filepath)
        return preprocessor
