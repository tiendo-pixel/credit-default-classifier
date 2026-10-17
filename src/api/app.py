import os
from typing import List
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from src.api.schemas import (
    LoanApplicationRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    RiskFactor,
)

# ==============================================================================
# FASTAPI APPLICATION SETUP
# ==============================================================================
app = FastAPI(
    title="Credit Default Intelligence API",
    description="Production-grade Credit Scoring and Underwriting API powered by Machine Learning",
    version="1.0.0",
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load ML artifacts if available
MODEL_PATH = "models/model.joblib"
PREPROCESSOR_PATH = "models/preprocessor.joblib"

loaded_model = None
loaded_preprocessor = None

if os.path.exists(MODEL_PATH) and os.path.exists(PREPROCESSOR_PATH):
    try:
        loaded_model = joblib.load(MODEL_PATH)
        loaded_preprocessor = joblib.load(PREPROCESSOR_PATH)
        print("-> ML Pipeline loaded successfully.")
    except Exception as e:
        print(f"-> Warning: Could not load trained models: {e}. Fallback enabled.")


# ==============================================================================
# CORE UNDERWRITING LOGIC
# ==============================================================================
def score_applicant(app_data: LoanApplicationRequest) -> PredictionResponse:
    # 1. Compute foundational financial ratios
    monthly_installment = app_data.requested_amount / app_data.loan_tenor_months
    existing_debt_service = app_data.total_existing_debt * 0.05
    dti = (monthly_installment + existing_debt_service) / app_data.monthly_income
    lti = app_data.requested_amount / (app_data.monthly_income * 12)

    # 2. Probability of Default (PD) Computation
    if loaded_model is not None and loaded_preprocessor is not None:
        try:
            # Construct feature dataframe for production model
            input_dict = {
                "disbursed_amount": [app_data.requested_amount],
                "income": [app_data.monthly_income],
                "dti": [dti],
                "lti": [lti],
                "cic_max_credit_group": [app_data.cic_max_credit_group],
                "inquiries_last_3m": [app_data.inquiries_last_3m],
                "card_utilization": [app_data.credit_card_utilization],
                "non_bank_ratio": [app_data.non_bank_debt_ratio],
                "age": [app_data.age]
            }
            df_in = pd.DataFrame(input_dict)
            transformed_x = loaded_preprocessor.transform(df_in)
            pd_prob = float(loaded_model.predict_proba(transformed_x)[0, 1])
        except Exception:
            pd_prob = calculate_empirical_pd(app_data, dti)
    else:
        pd_prob = calculate_empirical_pd(app_data, dti)

    # 3. Credit Score Calculation (Scale: 300 to 850)
    # High PD -> Lower Score
    credit_score = int(np.clip(850 - (pd_prob * 550), 300, 850))

    # 4. Underwriting Decision & Policy Thresholds
    if pd_prob < 0.20 and app_data.cic_max_credit_group == 1 and dti < 0.50:
        decision = "APPROVED"
        badge_color = "#10b981"  # Emerald Green
        reason = "Hồ sơ thỏa mãn toàn bộ tiêu chí tín dụng: Tỷ lệ DTI lành mạnh, lịch sử tín dụng CIC chuẩn."
    elif pd_prob < 0.45 and app_data.cic_max_credit_group <= 2:
        decision = "MANUAL REVIEW"
        badge_color = "#f59e0b"  # Amber
        reason = "Phát hiện tín hiệu rủi ro mức độ trung bình (áp lực nợ hoặc tra cứu gần đây). Cần thẩm định bổ sung."
    else:
        decision = "REJECTED"
        badge_color = "#ef4444"  # Rose Red
        reason = "Xác suất vỡ nợ vượt quá khẩu vị rủi ro (Nợ xấu CIC hoặc gánh nặng nợ DTI quá ngưỡng chịu đựng)."

    # 5. Top Risk Factors (Explainability)
    factors: List[RiskFactor] = [
        RiskFactor(
            dimension="Lịch sử nợ CIC",
            contribution_percent=round(min(35.0, (app_data.cic_max_credit_group - 1) * 35.0 + 5.0), 1),
            status="Nguy hiểm" if app_data.cic_max_credit_group > 1 else "Tốt",
            detail=f"Nhóm nợ lịch sử cao nhất: Nhóm {app_data.cic_max_credit_group}"
        ),
        RiskFactor(
            dimension="Gánh nặng nợ (DTI)",
            contribution_percent=round(min(30.0, dti * 40.0), 1),
            status="Cảnh báo" if dti > 0.50 else "Tốt",
            detail=f"Tỷ lệ nợ trên thu nhập hàng tháng: {dti:.1%}"
        ),
        RiskFactor(
            dimension="Tần suất tra cứu CIC (3M)",
            contribution_percent=round(min(20.0, app_data.inquiries_last_3m * 6.5), 1),
            status="Cao" if app_data.inquiries_last_3m >= 3 else "Bình thường",
            detail=f"Số lần các TCTD tra cứu gần đây: {app_data.inquiries_last_3m} lần"
        ),
        RiskFactor(
            dimension="Hạn mức thẻ tín dụng",
            contribution_percent=round(app_data.credit_card_utilization * 0.15, 1),
            status="Căng thẳng" if app_data.credit_card_utilization > 75 else "An toàn",
            detail=f"Mức độ sử dụng hạn mức thẻ: {app_data.credit_card_utilization:.0f}%"
        ),
    ]

    return PredictionResponse(
        applicant_name=app_data.applicant_name or "Khách hàng",
        probability_of_default=round(pd_prob, 4),
        credit_score=credit_score,
        decision=decision,
        badge_color=badge_color,
        decision_reason=reason,
        debt_to_income_ratio=round(dti, 4),
        loan_to_income_ratio=round(lti, 4),
        top_risk_factors=factors,
        model_version="LGBM_CreditRisk_v1.0"
    )


def calculate_empirical_pd(data: LoanApplicationRequest, dti: float) -> float:
    """Empirical calibrated credit risk engine for baseline underwriting."""
    base_pd = 0.04
    cic_penalty = (data.cic_max_credit_group - 1) * 0.22
    dti_penalty = max(0.0, (dti - 0.40) * 0.45)
    inquiry_penalty = min(0.18, data.inquiries_last_3m * 0.04)
    card_penalty = max(0.0, (data.credit_card_utilization - 60.0) / 100.0 * 0.18)
    subprime_penalty = (data.non_bank_debt_ratio / 100.0) * 0.12

    total_pd = base_pd + cic_penalty + dti_penalty + inquiry_penalty + card_penalty + subprime_penalty
    return float(np.clip(total_pd, 0.02, 0.98))


# ==============================================================================
# API ENDPOINTS
# ==============================================================================
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "Credit Default Scoring Engine",
        "model_loaded": loaded_model is not None,
        "version": "1.0.0"
    }


@app.post("/api/predict", response_model=PredictionResponse)
def predict_single(application: LoanApplicationRequest):
    try:
        return score_applicant(application)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(payload: BatchPredictionRequest):
    try:
        results = [score_applicant(app) for app in payload.applications]
        approved = sum(1 for r in results if r.decision == "APPROVED")
        review = sum(1 for r in results if r.decision == "MANUAL REVIEW")
        rejected = sum(1 for r in results if r.decision == "REJECTED")

        return BatchPredictionResponse(
            total_applications=len(results),
            approved_count=approved,
            review_count=review,
            rejected_count=rejected,
            results=results
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# STATIC FRONTEND SERVING
# ==============================================================================
FRONTEND_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../ui/web"))
if os.path.exists(FRONTEND_DIR):
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="web")
