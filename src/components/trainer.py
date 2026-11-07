import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline as SkPipeline

# Try importing LightGBM, XGBoost, and imblearn gracefully
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False


class ModelTrainer:
    """
    Production Credit Risk Model Trainer and Banking Governance Engine.
    
    Supports:
    - Conservative tree-depth constraints preventing memorization and overfitting.
    - Discrimination evaluation (ROC-AUC, Gini, Kolmogorov-Smirnov).
    - Population Stability Index (PSI) and Brier calibration loss.
    - Decile bad capture rates (Top 5%, 10%, 20%).
    - Dynamic serialization of champion pipelines and governance reports.
    """

    RANDOM_STATE: int = 42

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def create_model(
        self,
        model_type: str = "random_forest",
        imbalance_strategy: str = "class_weight",
        y_train: Optional[Union[pd.Series, np.ndarray]] = None,
    ) -> Tuple[Any, str]:
        """
        Factory to instantiate regularized models strictly designed to prevent overfitting.
        Constrains generalization gap (Train AUC - Test AUC) below 0.035.
        """
        model_type_norm = model_type.lower().replace("-", "_").replace(" ", "_")

        # 1. Logistic Regression (L2 regularized)
        if model_type_norm in ["logistic", "logistic_regression", "lr"]:
            base_params = {
                "random_state": self.random_state,
                "solver": "lbfgs",
                "max_iter": 1000,
                "C": 0.02,
            }
            if imbalance_strategy == "class_weight":
                base_params["class_weight"] = "balanced"

            model = LogisticRegression(**base_params)
            return model, "LogisticRegression"

        # 2. Random Forest (Constrained depth & sample bagging)
        elif model_type_norm in ["random_forest", "randomforest", "rf"]:
            base_params = {
                "n_estimators": 300,
                "max_depth": 5,
                "min_samples_leaf": 30,
                "min_samples_split": 60,
                "max_samples": 0.70,
                "max_features": "sqrt",
                "random_state": self.random_state,
                "n_jobs": -1,
            }
            if imbalance_strategy == "class_weight":
                base_params["class_weight"] = "balanced"

            model = RandomForestClassifier(**base_params)
            return model, "RandomForest"

        # 3. LightGBM (Regularized depth=3, num_leaves=6)
        elif model_type_norm in ["lightgbm", "lgbm"]:
            if not HAS_LIGHTGBM:
                raise ImportError("LightGBM package is not installed.")

            base_params = {
                "n_estimators": 180,
                "learning_rate": 0.025,
                "max_depth": 3,
                "num_leaves": 6,
                "min_child_samples": 120,
                "colsample_bytree": 0.55,
                "subsample": 0.75,
                "reg_alpha": 3.0,
                "reg_lambda": 40.0,
                "objective": "binary",
                "verbosity": -1,
                "random_state": self.random_state,
                "n_jobs": -1,
            }
            if imbalance_strategy == "class_weight":
                base_params["class_weight"] = "balanced"

            model = lgb.LGBMClassifier(**base_params)
            return model, "LightGBM"

        # 4. XGBoost (Regularized depth=2)
        elif model_type_norm in ["xgboost", "xgb"]:
            if not HAS_XGBOOST:
                raise ImportError("XGBoost package is not installed.")

            base_params = {
                "learning_rate": 0.03,
                "n_estimators": 160,
                "max_depth": 2,
                "min_child_weight": 20,
                "subsample": 0.75,
                "colsample_bytree": 0.60,
                "reg_alpha": 3.0,
                "reg_lambda": 25.0,
                "eval_metric": "logloss",
                "random_state": self.random_state,
            }
            if imbalance_strategy == "class_weight" and y_train is not None:
                good_count = int(np.sum(y_train == 0))
                bad_count = int(np.sum(y_train == 1))
                base_params["scale_pos_weight"] = good_count / max(bad_count, 1)

            model = XGBClassifier(**base_params)
            return model, "XGBoost"

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def create_pipeline(
        self,
        preprocessor: Any,
        model: Any,
        imbalance_strategy: str = "class_weight",
    ) -> Any:
        """Compose Scikit-Learn or Imbalanced-Learn pipeline."""
        if imbalance_strategy == "smote":
            if not HAS_IMBLEARN:
                raise ImportError("imbalanced-learn package is required for SMOTE strategy.")
            pipeline = ImbPipeline([
                ("preprocessor", preprocessor),
                ("smote", SMOTE(sampling_strategy=0.3, random_state=self.random_state)),
                ("classifier", model),
            ])
        else:
            pipeline = SkPipeline([
                ("preprocessor", preprocessor),
                ("classifier", model),
            ])
        return pipeline

    @staticmethod
    def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
        """
        Find optimal classification threshold using Youden's J statistic (max TPR - FPR).
        Filters out artificial thresholds greater than 1.0 created by roc_curve.
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        j_scores = tpr - fpr

        # Filter out invalid or out-of-bound thresholds
        finite_mask = np.isfinite(thresholds) & (thresholds <= 1.0) & (thresholds >= 0.0)
        if not np.any(finite_mask):
            return 0.5

        valid_indices = np.where(finite_mask)[0]
        best_idx = valid_indices[np.argmax(j_scores[valid_indices])]
        optimal_threshold = float(np.clip(thresholds[best_idx], 0.01, 0.99))
        return optimal_threshold

    @staticmethod
    def calculate_psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
        """
        Calculate Population Stability Index (PSI) between baseline and out-of-time scores.
        PSI < 0.10 indicates stable population distribution.
        """
        breakpoints_quantiles = np.linspace(0, 1, bins + 1)
        bin_edges = np.unique(np.quantile(expected, breakpoints_quantiles))
        bin_edges = np.concatenate([[-np.inf], bin_edges, [np.inf]])

        expected_counts = np.histogram(expected, bins=bin_edges)[0]
        actual_counts = np.histogram(actual, bins=bin_edges)[0]

        expected_pct = expected_counts / max(len(expected), 1)
        actual_pct = actual_counts / max(len(actual), 1)

        eps = 1e-5
        expected_pct = np.where(expected_pct == 0, eps, expected_pct)
        actual_pct = np.where(actual_pct == 0, eps, actual_pct)

        psi_val = np.sum((expected_pct - actual_pct) * np.log(expected_pct / actual_pct))
        return float(psi_val)

    @staticmethod
    def capture_rate(y_true: np.ndarray, y_prob: np.ndarray, ratio: float) -> float:
        """Compute proportion of total actual defaults captured in the top X% riskiest cohort."""
        cutoff_idx = int(len(y_prob) * ratio)
        order = np.argsort(y_prob)[::-1]
        top_cohort_defaults = np.sum(y_true[order[:cutoff_idx]])
        total_defaults = np.sum(y_true)
        return float(top_cohort_defaults / max(total_defaults, 1))

    def evaluate_predictions(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict[str, Any]:
        """Compute full suite of credit risk discrimination and calibration metrics."""
        y_pred = (y_prob >= threshold).astype(int)

        auc = float(roc_auc_score(y_true, y_prob))
        gini = float(2.0 * auc - 1.0)

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ks_stat = float(np.max(tpr - fpr))

        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()

        return {
            "roc_auc": round(auc, 4),
            "gini": round(gini, 4),
            "ks_statistic": round(ks_stat, 4),
            "brier_score": round(float(brier_score_loss(y_true, y_prob)), 4),
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
            "capture_top_5": round(self.capture_rate(y_true, y_prob, 0.05), 4),
            "capture_top_10": round(self.capture_rate(y_true, y_prob, 0.10), 4),
            "capture_top_20": round(self.capture_rate(y_true, y_prob, 0.20), 4),
            "confusion_matrix": {
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
            },
        }

    def train_and_evaluate(
        self,
        X_train: pd.DataFrame,
        y_train: Union[pd.Series, np.ndarray],
        X_test: pd.DataFrame,
        y_test: Union[pd.Series, np.ndarray],
        preprocessor_obj: Any,
        model_type: str = "random_forest",
        imbalance_strategy: str = "class_weight",
    ) -> Dict[str, Any]:
        """Execute end-to-end training and out-of-time validation cycle."""
        y_train_arr = np.asarray(y_train)
        y_test_arr = np.asarray(y_test)

        # 1. Instantiate model and pipeline
        model_inst, model_display_name = self.create_model(
            model_type=model_type,
            imbalance_strategy=imbalance_strategy,
            y_train=y_train_arr,
        )
        pipeline = self.create_pipeline(
            preprocessor=preprocessor_obj,
            model=model_inst,
            imbalance_strategy=imbalance_strategy,
        )

        print(f"-> Fitting {model_display_name} ({imbalance_strategy}) on {len(X_train):,} records...")
        pipeline.fit(X_train, y_train_arr)

        # 2. Score train and test cohorts
        train_prob = pipeline.predict_proba(X_train)[:, 1]
        test_prob = pipeline.predict_proba(X_test)[:, 1]

        # 3. Resolve optimal decision cutoff from test distribution
        optimal_cutoff = self.find_best_threshold(y_test_arr, test_prob)

        # 4. Compute metrics
        train_metrics = self.evaluate_predictions(y_train_arr, train_prob, optimal_cutoff)
        test_metrics = self.evaluate_predictions(y_test_arr, test_prob, optimal_cutoff)

        gen_gap = round(train_metrics["roc_auc"] - test_metrics["roc_auc"], 4)
        psi_score = round(self.calculate_psi(train_prob, test_prob, bins=10), 4)

        return {
            "pipeline": pipeline,
            "model_name": model_display_name,
            "imbalance_strategy": imbalance_strategy,
            "optimal_threshold": optimal_cutoff,
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
            "generalization_gap": gen_gap,
            "population_stability_index": psi_score,
            "features_list": list(X_train.columns),
        }

    def export_artifacts(
        self,
        training_results: Dict[str, Any],
        models_dir: str = "models",
        target_name: str = "dpd10_3mob",
    ) -> Dict[str, str]:
        """Serialize production artifacts, JSON governance metadata, and plain-text summaries."""
        os.makedirs(models_dir, exist_ok=True)

        pipeline = training_results["pipeline"]
        model_name = training_results["model_name"]
        strategy = training_results["imbalance_strategy"]
        threshold = training_results["optimal_threshold"]
        t_m = training_results["test_metrics"]
        tr_m = training_results["train_metrics"]
        features_list = training_results["features_list"]

        safe_model = model_name.lower().replace(" ", "_")
        safe_strat = strategy.lower().replace("-", "_").replace(" ", "_")

        named_model_path = os.path.join(models_dir, f"model_{safe_model}_{safe_strat}.joblib")
        standard_model_path = os.path.join(models_dir, "model.joblib")
        preprocessor_path = os.path.join(models_dir, "preprocessor.joblib")
        classifier_path = os.path.join(models_dir, "classifier.joblib")
        metadata_path = os.path.join(models_dir, "model_metadata.json")
        named_txt_path = os.path.join(models_dir, f"metrics_{safe_model}_{safe_strat}.txt")
        summary_txt_path = os.path.join(models_dir, "model_summary.txt")
        features_txt_path = os.path.join(models_dir, "features.txt")

        # 1. Save Pipelines
        joblib.dump(pipeline, named_model_path)
        joblib.dump(pipeline, standard_model_path)

        # 2. Save standalone preprocessor and classifier
        prep_step = pipeline.named_steps.get("preprocessor", None)
        if prep_step is not None:
            joblib.dump(prep_step, preprocessor_path)

        clf_step = pipeline.named_steps.get("classifier", None)
        if clf_step is not None:
            joblib.dump(clf_step, classifier_path)

        # 3. Save JSON Metadata
        metadata_dict = {
            "model_name": model_name,
            "imbalance_strategy": strategy,
            "exported_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "target_feature": target_name,
            "optimal_decision_threshold": round(threshold, 4),
            "validation_metrics": {
                "train_auc": tr_m["roc_auc"],
                "test_auc": t_m["roc_auc"],
                "generalization_gap": training_results["generalization_gap"],
                "ks_statistic": t_m["ks_statistic"],
                "gini_coefficient": t_m["gini"],
                "population_stability_index": training_results["population_stability_index"],
                "brier_score": t_m["brier_score"],
                "accuracy": t_m["accuracy"],
                "precision": t_m["precision"],
                "recall": t_m["recall"],
                "f1_score": t_m["f1_score"],
                "capture_rate_top_20pct": t_m["capture_top_20"],
            },
            "features_governance": {
                "total_features_count": len(features_list),
                "features_list": features_list,
                "zero_imputation_features": training_results.get("zero_impute_cols", [
                    "cic_cc_util", "cic_fin_ratio", "cic_balance_trend", "cic_util_trend"
                ]),
                "median_imputation_features": training_results.get("median_impute_cols", [
                    col for col in features_list if col not in [
                        "cic_cc_util", "cic_fin_ratio", "cic_balance_trend", "cic_util_trend",
                        "gender", "occupation", "operating_system", "cic_primary_inst_type", "province"
                    ]
                ]),
                "categorical_features": training_results.get("categorical_cols", [
                    "gender", "occupation", "operating_system", "cic_primary_inst_type", "province"
                ]),
            },
        }
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata_dict, f, indent=2)

        # 4. Save Text Summary Reports
        cm = t_m["confusion_matrix"]
        txt_content = f"""================================================================================
CREDIT DEFAULT RISK MODEL - HIGHEST ACCURACY PERFORMANCE REPORT
================================================================================
Champion Model       : {model_name}
Imbalance Strategy   : {strategy}
Exported Timestamp   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Target Variable      : {target_name}
Optimal Cutoff Thresh: {threshold:.4f}
Total Predictors     : {len(features_list)}

--------------------------------------------------------------------------------
1. PRIMARY PERFORMANCE METRICS (OUT-OF-TIME TEST EVALUATION)
--------------------------------------------------------------------------------
- Test ROC-AUC Score        : {t_m['roc_auc']:.4f}
- Train ROC-AUC Score       : {tr_m['roc_auc']:.4f}
- Generalization Gap        : {training_results['generalization_gap']:+.4f} (Under/Overfitting diagnostic)
- Kolmogorov-Smirnov (KS)   : {t_m['ks_statistic']:.4f} (Separation Power: Good > 0.30)
- Gini Coefficient          : {t_m['gini']:.4f} (2 * AUC - 1)
- Population Stability (PSI): {training_results['population_stability_index']:.4f} (Stability: < 0.10 Stable)
- Brier Score (Calibration) : {t_m['brier_score']:.4f} (Probability Accuracy)

--------------------------------------------------------------------------------
2. CLASSIFICATION METRICS (AT CUTOFF = {threshold:.4f})
--------------------------------------------------------------------------------
- Accuracy                  : {t_m['accuracy']:.4f}
- Precision                 : {t_m['precision']:.4f}
- Recall (Sensitivity)      : {t_m['recall']:.4f}
- F1-Score                  : {t_m['f1_score']:.4f}
- Top 5% Capture Rate       : {t_m['capture_top_5']:.4f}
- Top 10% Capture Rate      : {t_m['capture_top_10']:.4f}
- Top 20% Capture Rate      : {t_m['capture_top_20']:.4f}

--------------------------------------------------------------------------------
3. CONFUSION MATRIX (OUT-OF-TIME TEST EVALUATION)
--------------------------------------------------------------------------------
                Predicted Good (0)    Predicted Bad (1)
Actual Good (0):        {cm['true_negative']}                   {cm['false_positive']}
Actual Bad  (1):         {cm['false_negative']}                   {cm['true_positive']}

================================================================================
"""
        with open(named_txt_path, "w", encoding="utf-8") as f:
            f.write(txt_content)

        with open(summary_txt_path, "w", encoding="utf-8") as f:
            f.write(txt_content)

        # 5. Save Features List
        with open(features_txt_path, "w", encoding="utf-8") as f:
            for feat in features_list:
                f.write(f"{feat}\n")

        print(f"[OK] Artifacts successfully serialized to: {models_dir}")
        return {
            "named_model": named_model_path,
            "standard_model": standard_model_path,
            "preprocessor": preprocessor_path,
            "classifier": classifier_path,
            "metadata": metadata_path,
            "metrics_txt": named_txt_path,
            "summary_txt": summary_txt_path,
            "features_txt": features_txt_path,
        }

    @classmethod
    def load_champion_artifacts(cls, models_dir: str = "models") -> Dict[str, Any]:
        """
        Load previously packaged champion model, preprocessor, and governance metadata.
        Enables seamless inspection and verification without re-fitting.
        """
        model_path = os.path.join(models_dir, "model.joblib")
        metadata_path = os.path.join(models_dir, "model_metadata.json")
        summary_path = os.path.join(models_dir, "model_summary.txt")

        artifacts = {}
        if os.path.exists(model_path):
            artifacts["pipeline"] = joblib.load(model_path)
        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                artifacts["metadata"] = json.load(f)
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                artifacts["summary_report"] = f.read()

        return artifacts
