import os
from typing import List
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
from src.predict import CreditPredictor

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

# Initialize production scoring engine
predictor = CreditPredictor(
    model_path="models/model.joblib",
    metadata_path="models/model_metadata.json",
)


# ==============================================================================
# CORE UNDERWRITING LOGIC
# ==============================================================================
def score_applicant(app_data: LoanApplicationRequest) -> PredictionResponse:
    # 1. Compute foundational financial ratios
    monthly_installment = app_data.requested_amount / app_data.loan_tenor_months
    existing_debt_service = app_data.total_existing_debt * 0.05
    dti = (monthly_installment + existing_debt_service) / app_data.monthly_income
    lti = app_data.requested_amount / (app_data.monthly_income * 12)

    # 2. Probability of Default (PD) Computation via ML Engine
    if predictor.pipeline is not None:
        try:
            # Map input payload to model feature inputs
            app_dict = {
                "age": app_data.age,
                "income_m": app_data.monthly_income,
                "requested_amount": app_data.requested_amount,
                "loan_tenor_months": app_data.loan_tenor_months,
                "total_existing_debt": app_data.total_existing_debt,
                "cic_max_group_12m": app_data.cic_max_credit_group,
                "cic_cc_util": app_data.credit_card_utilization / 100.0,
                "enq_3m": app_data.inquiries_last_3m,
                "cic_inquiry_3m": app_data.inquiries_last_3m,
                "non_bank_loan_ratio": app_data.non_bank_debt_ratio / 100.0,
                "current_dti": dti,
                "loan_to_income": lti,
            }
            pred_res = predictor.predict_single(app_dict)
            pd_prob = pred_res["probability_of_default"]
            credit_score = pred_res["credit_score"]
            decision = pred_res["decision"]
            badge_color = pred_res["badge_color"]
            reason = pred_res["decision_reason"]
            model_ver = pred_res["model_version"]
        except Exception:
            pd_prob = calculate_empirical_pd(app_data, dti)
            credit_score = predictor.pd_to_credit_score(pd_prob)
            dec_info = predictor.evaluate_decision(pd_prob, app_data.cic_max_credit_group, dti)
            decision = dec_info["decision"]
            badge_color = dec_info["badge_color"]
            reason = dec_info["decision_reason"]
            model_ver = "Empirical_RuleEngine_v1"
    else:
        pd_prob = calculate_empirical_pd(app_data, dti)
        credit_score = predictor.pd_to_credit_score(pd_prob)
        dec_info = predictor.evaluate_decision(pd_prob, app_data.cic_max_credit_group, dti)
        decision = dec_info["decision"]
        badge_color = dec_info["badge_color"]
        reason = dec_info["decision_reason"]
        model_ver = "Empirical_RuleEngine_v1"

    # 3. Top Risk Factors (Explainability)
    factors: List[RiskFactor] = [
        RiskFactor(
            dimension="Lịch sử nợ CIC",
            contribution_percent=round(min(35.0, (app_data.cic_max_credit_group - 1) * 35.0 + 5.0), 1),
            status="Nguy hiểm" if app_data.cic_max_credit_group > 1 else "Tốt",
            detail=f"Nhóm nợ lịch sử cao nhất: Nhóm {app_data.cic_max_credit_group}",
        ),
        RiskFactor(
            dimension="Gánh nặng nợ (DTI)",
            contribution_percent=round(min(30.0, dti * 40.0), 1),
            status="Cảnh báo" if dti > 0.50 else "Tốt",
            detail=f"Tỷ lệ nợ trên thu nhập hàng tháng: {dti:.1%}",
        ),
        RiskFactor(
            dimension="Tần suất tra cứu CIC (3M)",
            contribution_percent=round(min(20.0, app_data.inquiries_last_3m * 6.5), 1),
            status="Cao" if app_data.inquiries_last_3m >= 3 else "Bình thường",
            detail=f"Số lần các TCTD tra cứu gần đây: {app_data.inquiries_last_3m} lần",
        ),
        RiskFactor(
            dimension="Hạn mức thẻ tín dụng",
            contribution_percent=round(app_data.credit_card_utilization * 0.15, 1),
            status="Căng thẳng" if app_data.credit_card_utilization > 75 else "An toàn",
            detail=f"Mức độ sử dụng hạn mức thẻ: {app_data.credit_card_utilization:.0f}%",
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
        model_version=model_ver,
    )


def calculate_empirical_pd(data: LoanApplicationRequest, dti: float) -> float:
    """Empirical calibrated credit risk engine for baseline underwriting fallback."""
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
        "model_loaded": predictor.pipeline is not None,
        "model_name": predictor.metadata.get("model_name", "RandomForest"),
        "version": "1.0.0",
    }


@app.post("/api/predict", response_model=PredictionResponse)
def predict_single(application: LoanApplicationRequest):
    try:
        return score_applicant(application)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(batch_req: BatchPredictionRequest):
    try:
        results = [score_applicant(app) for app in batch_req.applications]
        approved = sum(1 for r in results if r.decision == "APPROVED")
        review = sum(1 for r in results if r.decision == "MANUAL REVIEW")
        rejected = sum(1 for r in results if r.decision == "REJECTED")

        return BatchPredictionResponse(
            total_applications=len(results),
            approved_count=approved,
            review_count=review,
            rejected_count=rejected,
            results=results,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==============================================================================
# STATIC FRONTEND WEB DASHBOARD MOUNTING
# ==============================================================================
WEB_DIR = "ui/web"
if os.path.exists(WEB_DIR):
    @app.get("/")
    def serve_dashboard():
        """Serve the CreditGuard AI interactive underwriting web dashboard."""
        return FileResponse(os.path.join(WEB_DIR, "index.html"))

    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web_ui")
