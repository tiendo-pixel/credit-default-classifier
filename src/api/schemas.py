from typing import List, Optional
from pydantic import BaseModel, Field


class LoanApplicationRequest(BaseModel):
    """Input payload for single applicant risk scoring."""
    applicant_name: Optional[str] = Field(default="Nguyễn Văn A", description="Applicant Name")
    age: int = Field(default=30, ge=18, le=75, description="Applicant Age")
    monthly_income: float = Field(default=15000000.0, gt=0, description="Monthly Net Income in VND")
    requested_amount: float = Field(default=30000000.0, gt=0, description="Requested Loan Amount in VND")
    loan_tenor_months: int = Field(default=12, ge=3, le=60, description="Loan Term in Months")
    total_existing_debt: float = Field(default=10000000.0, ge=0, description="Total Current Outstanding Debt in VND")
    cic_max_credit_group: int = Field(default=1, ge=1, le=5, description="Max CIC Historical Delinquency Group (1-5)")
    credit_card_utilization: float = Field(default=35.0, ge=0, le=100, description="Credit Card Limit Utilization (%)")
    inquiries_last_3m: int = Field(default=1, ge=0, le=30, description="CIC Credit Inquiries in Last 3 Months")
    non_bank_debt_ratio: float = Field(default=20.0, ge=0, le=100, description="Share of Debt at Finance/Subprime Companies (%)")

    class Config:
        json_schema_extra = {
            "example": {
                "applicant_name": "Trần Văn Bình",
                "age": 28,
                "monthly_income": 18000000.0,
                "requested_amount": 40000000.0,
                "loan_tenor_months": 12,
                "total_existing_debt": 15000000.0,
                "cic_max_credit_group": 1,
                "credit_card_utilization": 42.0,
                "inquiries_last_3m": 2,
                "non_bank_debt_ratio": 25.0
            }
        }


class RiskFactor(BaseModel):
    dimension: str
    contribution_percent: float
    status: str
    detail: str


class PredictionResponse(BaseModel):
    """Output prediction result for credit underwriting."""
    applicant_name: str
    probability_of_default: float
    credit_score: int
    decision: str  # APPROVED | MANUAL REVIEW | REJECTED
    badge_color: str
    decision_reason: str
    debt_to_income_ratio: float
    loan_to_income_ratio: float
    top_risk_factors: List[RiskFactor]
    model_version: str


class BatchPredictionRequest(BaseModel):
    applications: List[LoanApplicationRequest]


class BatchPredictionResponse(BaseModel):
    total_applications: int
    approved_count: int
    review_count: int
    rejected_count: int
    results: List[PredictionResponse]
