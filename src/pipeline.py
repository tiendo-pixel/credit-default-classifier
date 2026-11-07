import argparse
import os
import sys
from datetime import datetime
import numpy as np
import pandas as pd

from src.components.data_loader import DataLoader
from src.components.preprocessor import DataPreprocessor
from src.components.trainer import ModelTrainer


def run_pipeline(
    data_path: str = "data/raw/Professional Test.xlsx",
    lookup_path: str = "data/raw/Institutions code (1).xlsx",
    model_type: str = "random_forest",
    imbalance_strategy: str = "class_weight",
    models_dir: str = "models",
    target_col: str = "dpd10_3mob",
    train_ratio: float = 0.80,
):
    """
    Execute end-to-end retraining and governance pipeline from command line.
    
    Workflow:
    1. Ingest raw loan originations & parse CIC institution mappings via DataLoader.
    2. Temporal split strictly on disbursement date to eliminate lookahead bias.
    3. Fit domain-specific DataPreprocessor on train partition.
    4. Fit and evaluate ModelTrainer against Out-Of-Time (OOT) test cohort.
    5. Export serialized model artifacts, JSON metadata, and plain-text summaries.
    """
    print("=" * 80)
    print("STARTING CREDIT RISK MODEL TRAINING & GOVERNANCE PIPELINE")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Model Type: {model_type} | Strategy: {imbalance_strategy}")
    print("=" * 80)

    # 1. Data Ingestion & CIC Feature Extraction
    print("\n[STEP 1/5] Ingesting and mapping raw credit datasets...")
    loader = DataLoader(data_path=data_path, lookup_path=lookup_path)
    df = loader.load_and_map_data()
    print(f"-> Successfully loaded {len(df):,} records with {df.shape[1]} raw attributes.")

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset.")

    # 2. Temporal Partitioning (Zero Lookahead Leakage)
    print("\n[STEP 2/5] Performing Out-of-Time temporal split...")
    if "disbursement_date" in df.columns:
        df["disbursement_date"] = pd.to_datetime(df["disbursement_date"])
        cutoff_date = df["disbursement_date"].quantile(train_ratio)
        train_mask = df["disbursement_date"] <= cutoff_date
        test_mask = df["disbursement_date"] > cutoff_date
        print(f"-> Cutoff Date: {cutoff_date.strftime('%Y-%m-%d')}")
    else:
        print("-> Warning: 'disbursement_date' not found. Falling back to index-based 80/20 split.")
        split_idx = int(len(df) * train_ratio)
        train_mask = df.index < split_idx
        test_mask = df.index >= split_idx

    # Exclude non-feature columns
    excluded_cols = [target_col, "disbursement_date", "contract_number", "customer_id", "app_id"]
    feature_cols = [c for c in df.columns if c not in excluded_cols]

    X_train = df.loc[train_mask, feature_cols].copy()
    y_train = df.loc[train_mask, target_col].copy()
    X_test = df.loc[test_mask, feature_cols].copy()
    y_test = df.loc[test_mask, target_col].copy()

    print(f"-> Train Dev Cohort: {len(X_train):,} records | Bad Rate: {y_train.mean():.2%}")
    print(f"-> Test OOT Cohort : {len(X_test):,} records | Bad Rate: {y_test.mean():.2%}")

    # 3. Preprocessing Transformation
    print("\n[STEP 3/5] Fitting DataPreprocessor ColumnTransformer...")
    preprocessor = DataPreprocessor()
    preprocessor.fit(X_train)
    fitted_transformer = preprocessor.build_pipeline()

    # 4. Model Training & OOT Evaluation
    print(f"\n[STEP 4/5] Training {model_type} with {imbalance_strategy} imbalance mitigation...")
    trainer = ModelTrainer(random_state=42)
    training_results = trainer.train_and_evaluate(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        preprocessor_obj=fitted_transformer,
        model_type=model_type,
        imbalance_strategy=imbalance_strategy,
    )

    # 5. Production Artifact Serialization
    print("\n[STEP 5/5] Exporting production artifacts and governance records...")
    exported_files = trainer.export_artifacts(
        training_results=training_results,
        models_dir=models_dir,
        target_name=target_col,
    )

    t_m = training_results["test_metrics"]
    tr_m = training_results["train_metrics"]

    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Champion Model            : {training_results['model_name']} ({imbalance_strategy})")
    print(f"Train ROC-AUC             : {tr_m['roc_auc']:.4f}")
    print(f"Out-of-Time Test ROC-AUC  : {t_m['roc_auc']:.4f}")
    print(f"Generalization Gap (AUC)  : {training_results['generalization_gap']:+.4f}")
    print(f"Kolmogorov-Smirnov (KS)   : {t_m['ks_statistic']:.4f}")
    print(f"Gini Coefficient          : {t_m['gini']:.4f}")
    print(f"Population Stability (PSI): {training_results['population_stability_index']:.4f}")
    print(f"Optimal Decision Cutoff   : {training_results['optimal_threshold']:.4f}")
    print(f"Top 20% Bad Capture Rate  : {t_m['capture_top_20']:.4f}")
    print(f"Artifacts Saved To        : {os.path.abspath(models_dir)}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Credit Risk Model Retraining Pipeline CLI")
    parser.add_argument("--data_path", type=str, default="data/raw/Professional Test.xlsx", help="Raw dataset path")
    parser.add_argument("--lookup_path", type=str, default="data/raw/Institutions code (1).xlsx", help="Lookup table")
    parser.add_argument("--model_type", type=str, default="random_forest", choices=["random_forest", "LightGBM", "xgboost", "logistic"], help="Algorithm type")
    parser.add_argument("--imbalance_strategy", type=str, default="class_weight", choices=["class_weight", "normal", "smote"], help="Imbalance handling")
    parser.add_argument("--models_dir", type=str, default="models", help="Destination folder for artifacts")
    parser.add_argument("--target", type=str, default="dpd10_3mob", help="Target risk feature")

    args = parser.parse_args()

    run_pipeline(
        data_path=args.data_path,
        lookup_path=args.lookup_path,
        model_type=args.model_type,
        imbalance_strategy=args.imbalance_strategy,
        models_dir=args.models_dir,
        target_col=args.target,
    )
