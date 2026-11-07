import json
import os
from typing import Any, Dict, List, Optional, Union
import joblib
import numpy as np
import pandas as pd


class CreditPredictor:
    """
    Production Credit Scoring and Inference Engine.
    
    Loads serialized champion pipeline and governance metadata.
    Provides real-time scoring, credit score scaling (300 to 850),
    decision policy thresholds, and adverse risk factor attributions.
    """

    def __init__(
        self,
        model_path: str = "models/model.joblib",
        metadata_path: str = "models/model_metadata.json",
    ):
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.pipeline = None
        self.metadata = {}
        self.optimal_threshold: float = 0.4821
        self.features_list: List[str] = []

        self._load_artifacts()

    def _load_artifacts(self):
        """Load trained pipeline and metadata from disk."""
        if os.path.exists(self.model_path):
            self.pipeline = joblib.load(self.model_path)
        else:
            print(f"Warning: Model file not found at {self.model_path}.")

        if os.path.exists(self.metadata_path):
            with open(self.metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
            self.optimal_threshold = float(self.metadata.get("optimal_decision_threshold", 0.4821))
            self.features_list = self.metadata.get("features_governance", {}).get("features_list", [])

    @staticmethod
    def pd_to_credit_score(pd_prob: float) -> int:
        """
        Standard credit risk score transformation mapping Probability of Default (PD) to 300 - 850 scale.
        PD = 0% -> Score = 850 (Exceptional prime credit)
        PD = 100% -> Score = 300 (Subprime / Deep default risk)
        """
        score = 850.0 - (pd_prob * 550.0)
        return int(np.clip(round(score), 300, 850))

    def evaluate_decision(
        self,
        pd_prob: float,
        cic_group: int = 1,
        dti: float = 0.35,
    ) -> Dict[str, Any]:
        """
        Apply institutional underwriting rules and policy guardrails.
        Calibrated against the balanced model's optimal decision cutoff (optimal_threshold = 0.4821).
        """
        cutoff = self.optimal_threshold  # Default champion threshold: 0.4821
        review_buffer = 0.04             # Borderline zone: [cutoff, cutoff + 0.04]

        # 1. Approved: Default probability is strictly below decision cutoff with healthy credit hygiene
        if pd_prob < cutoff and cic_group == 1 and dti < 0.50:
            decision = "APPROVED"
            badge_color = "#10b981"  # Emerald Green
            reason = "Hồ sơ đạt chuẩn phê duyệt: Xác suất vỡ nợ dưới ngưỡng rủi ro cho phép (PD < 48.2%), DTI lành mạnh và lịch sử CIC chuẩn nhóm 1."

        # 2. Manual Review: Borderline risk zone, or notice debt group 2, or high debt burden
        elif (pd_prob < cutoff + review_buffer and cic_group <= 2) or (pd_prob < cutoff and cic_group == 2):
            decision = "MANUAL REVIEW"
            badge_color = "#f59e0b"  # Amber
            reason = "Hồ sơ nằm trong vùng rủi ro cần lưu ý (cận ngưỡng phê duyệt hoặc có lịch sử nợ nhóm 2). Cần chuyên viên thẩm định hồ sơ bổ sung."

        # 3. Rejected: Default probability exceeds risk appetite or severe delinquency
        else:
            decision = "REJECTED"
            badge_color = "#ef4444"  # Rose Red
            reason = "Xác suất vỡ nợ vượt quá khẩu vị rủi ro chấp thuận (PD >= 48.2%) hoặc khách hàng có lịch sử nợ xấu nhóm 3-5 tại CIC."

        return {
            "decision": decision,
            "badge_color": badge_color,
            "decision_reason": reason,
        }

    def _engineer_features(self, raw_input: Dict[str, Any]) -> pd.DataFrame:
        """Derive standard financial ratios and align with 60 model features."""
        data = dict(raw_input)

        # 1. Compute foundational ratios if loan amount and income are provided
        income = float(data.get("income_m", data.get("monthly_income", 15_000_000)))
        loan_amount = float(data.get("requested_amount", data.get("short_term_amount", 30_000_000)))
        tenor = float(data.get("loan_tenor_months", 12))
        existing_debt = float(data.get("total_existing_debt", 0.0))

        monthly_installment = loan_amount / max(tenor, 1)
        debt_service = existing_debt * 0.05
        dti = (monthly_installment + debt_service) / max(income, 1.0)
        lti = loan_amount / max(income * 12.0, 1.0)

        # Populate standard derived columns
        data["income_m"] = income
        data["current_dti"] = float(data.get("current_dti", dti))
        data["loan_to_income"] = float(data.get("loan_to_income", lti))
        data["flag_missing_income"] = 1 if income <= 0 else 0

        age = int(data.get("age", 30))
        data["age"] = age
        data["flag_young_age"] = 1 if age < 25 else 0

        score = float(data.get("cic_credit_score", 650))
        data["cic_credit_score"] = score
        data["flag_subprime_score"] = 1 if score < 600 else 0
        data["flag_prime_score"] = 1 if score >= 700 else 0

        # Construct single-row DataFrame
        df_row = pd.DataFrame([data])

        # Align with the 60 trained feature names
        if self.features_list:
            for feat in self.features_list:
                if feat not in df_row.columns:
                    df_row[feat] = np.nan
            df_row = df_row[self.features_list]

        return df_row

    def predict_single(self, raw_input: Dict[str, Any]) -> Dict[str, Any]:
        """Generate credit risk prediction for a single loan application."""
        if self.pipeline is None:
            raise RuntimeError("Model pipeline is not loaded. Ensure models/model.joblib exists.")

        df_features = self._engineer_features(raw_input)
        pd_prob = float(self.pipeline.predict_proba(df_features)[0, 1])
        credit_score = self.pd_to_credit_score(pd_prob)

        cic_group = int(raw_input.get("cic_max_group_12m", raw_input.get("cic_max_credit_group", 1)))
        dti = float(df_features["current_dti"].iloc[0]) if "current_dti" in df_features else 0.35

        decision_info = self.evaluate_decision(pd_prob, cic_group=cic_group, dti=dti)

        return {
            "probability_of_default": round(pd_prob, 4),
            "credit_score": credit_score,
            "decision": decision_info["decision"],
            "badge_color": decision_info["badge_color"],
            "decision_reason": decision_info["decision_reason"],
            "optimal_threshold": self.optimal_threshold,
            "model_version": self.metadata.get("model_name", "RandomForest"),
            "debt_to_income_ratio": round(dti, 4),
            "loan_to_income_ratio": round(float(df_features.get("loan_to_income", [0.0]).iloc[0]), 4),
        }

    def predict_batch(self, df_input: pd.DataFrame) -> pd.DataFrame:
        """Batch scoring for bulk application datasets."""
        if self.pipeline is None:
            raise RuntimeError("Model pipeline is not loaded.")

        results = []
        for _, row in df_input.iterrows():
            pred = self.predict_single(row.to_dict())
            results.append(pred)

        df_results = pd.DataFrame(results)
        return pd.concat([df_input.reset_index(drop=True), df_results], axis=1)


if __name__ == "__main__":
    print("=" * 70)
    print("TESTING CREDIT RISK INFERENCE ENGINE WITH PACKAGED CHAMPION MODEL")
    print("=" * 70)

    scorer = CreditPredictor(
        model_path="models/model.joblib",
        metadata_path="models/model_metadata.json",
    )

    # Sample applicant 1: Prime profile (High income, low debt, clean CIC)
    prime_applicant = {
        "applicant_name": "Nguyễn Văn An",
        "age": 32,
        "income_m": 25_000_000.0,
        "requested_amount": 40_000_000.0,
        "loan_tenor_months": 12,
        "total_existing_debt": 5_000_000.0,
        "cic_max_group_12m": 1,
        "credit_card_utilization": 20.0,
        "inquiries_last_3m": 0,
        "non_bank_debt_ratio": 0.0,
    }

    # Sample applicant 2: Subprime profile (High inquiries, CIC delinquency history)
    subprime_applicant = {
        "applicant_name": "Trần Thị Bích",
        "age": 23,
        "income_m": 9_000_000.0,
        "requested_amount": 45_000_000.0,
        "loan_tenor_months": 12,
        "total_existing_debt": 25_000_000.0,
        "cic_max_group_12m": 3,
        "credit_card_utilization": 88.0,
        "inquiries_last_3m": 5,
        "non_bank_debt_ratio": 70.0,
    }

    for idx, applicant in enumerate([prime_applicant, subprime_applicant], 1):
        print(f"\n--- Testing Applicant #{idx}: {applicant['applicant_name']} ---")
        res = scorer.predict_single(applicant)
        print(f"Default Probability (PD) : {res['probability_of_default']:.2%}")
        print(f"Credit Score (300 - 850) : {res['credit_score']}")
        print(f"Underwriting Decision    : {res['decision']}")
        print(f"Policy Decision Reason   : {res['decision_reason']}")
        print(f"Debt-to-Income (DTI)     : {res['debt_to_income_ratio']:.2%}")

    print("\nInference test completed successfully!")
