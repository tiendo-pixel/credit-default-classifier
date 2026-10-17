# Model Design Specification & Validation Document (MDD)

## Alternative Credit Scoring & Underwriting Engine (FIN-04)

---

### 1. Executive Overview & Champion Model Architecture

#### 1.1 Objective
To construct a deterministic, highly discriminative, and fully auditable credit risk scorecard predicting the likelihood of early default (`DPD10_3MOB`) for underbanked loan applicants.

#### 1.2 Model Selection Rationale
| Architecture | Role | Key Rationale & Trade-offs |
| :--- | :---: | :--- |
| **LightGBM (Champion)** | **Primary Scoring Engine** | • Superior handling of complex non-linear interactions (e.g., Debt-to-Income combined with revolving credit utilization).<br/>• Fast histogram-based gradient boosting suited for real-time low-latency serving ($< 50\text{ms}$).<br/>• Robust to outliers and native support for monotonicity constraints.<br/>• Direct TreeExplainer compatibility for SHAP local attribution. |
| **Logistic Regression (Challenger)** | **Regulatory Baseline** | • Serves as the traditional banking benchmark scorecard.<br/>• Verifies that tree-based gains ($+4\text{–}6\%$ AUC) justify production complexity. |

---

### 2. Data Partitioning & Temporal Validation (OOT Split)

#### 2.1 Why Random K-Fold Is Prohibited
In credit risk modeling, cross-sectional random splitting causes **lookahead bias** and **temporal leakage**:
* Future macroeconomic conditions and portfolio vintages leak into the training partition.
* A model evaluated via random cross-validation reports artificially inflated AUCs but degrades sharply in production.

#### 2.2 Out-Of-Time (OOT) Temporal Partitioning Strategy
The dataset of 14,000 observations is ordered chronologically by `disbursement_date` (April 8, 2024 to June 30, 2024) and partitioned into three disjoint cohorts:

```
Timeline: [2024-04-08 ────────────────────────── 2024-06-13] [2024-06-14 ── 2024-06-21] [2024-06-22 ── 2024-06-30]
Partition:        TRAINING SET (70%)                            VALIDATION (15%)              TEST / OOT (15%)
Samples:          10,022 records                                1,899 records                 2,079 records
Bad Rate:         ~20.1%                                        ~18.9%                        ~19.4%
Purpose:          Model parameter estimation                    Early stopping & tuning       Final unbiased audit
```

---

### 3. Feature Engineering & Preprocessing Pipeline

#### 3.1 Zero Data Leakage Contract
All preprocessing artifacts—including median statistics, outlier bounds, and categorical label mappings—are computed **strictly on the Training partition** and frozen. Test and production records are transformed using fitted training parameters without refitting.

#### 3.2 Domain-Specific Imputation Rules
| Feature Family | Imputation Value | Banking Domain Justification |
| :--- | :---: | :--- |
| **CIC Balances & Card Trends** (`cic_cc_util`, `cic_fin_ratio`, `cic_util_trend`, etc.) | **`0.0`** | Missing values reflect the **absence of an active credit line** with that specific institution, not missing measurement data. |
| **Declared Income & Bureau Scores** (`month_incom`, `cic_credit_score`, `cic_max_group_12m`) | **`Train Median`** | Unobserved values for existing borrowers are imputed using central tendency to prevent extreme skew or division-by-zero. |
| **Categorical Attributes** (`occupation`, `cic_primary_inst_type`, `province`) | **`'UNKNOWN'`** | Preserves missingness as an explicit risk category rather than guessing the modal class. |

#### 3.3 Outlier Handling: `OutlierCapper`
To protect gradient computations against extreme leverage anomalies (e.g., reported income typos or anomalous corporate debt), numerical features are bounded using training percentile caps ($1^{\text{st}}$ and $99^{\text{th}}$ percentiles).

---

### 4. Class Imbalance Mitigation

The portfolio exhibits an empirical bad rate of $\approx 19.84\%$ (a class imbalance ratio of approximately $4:1$). 

* **Strategy:** Cost-sensitive learning via LightGBM's `class_weight='balanced'`.
* **Formula:** Penalizes false negatives (failing to detect a defaulting borrower) $4\times$ more heavily than false positives (declining a good borrower), directly mirroring the financial asymmetry of credit loss vs missed interest margin.

---

### 5. Hyperparameter Configuration

```python
lgbm_params = {
    'objective': 'binary',
    'metric': 'auc',
    'boosting_type': 'gbdt',
    'n_estimators': 300,
    'learning_rate': 0.03,
    'num_leaves': 31,
    'max_depth': 6,
    'min_child_samples': 50,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'class_weight': 'balanced',
    'random_state': 42,
    'verbose': -1
}
```

* **Early Stopping:** Monitored against the Validation set with a 30-round tolerance to prevent over-fitting.

---

### 6. Quantitative Evaluation & Banking Benchmarks

#### 6.1 Performance Summary Table (OOT Test Set)

| Metric | Minimum Banking Threshold | Model Score (OOT Test) | Governance Result |
| :--- | :---: | :---: | :---: |
| **ROC-AUC** | $\ge 0.70$ | **`0.7546`** | **PASSED (Strong)** |
| **Gini Coefficient** ($2 \times \text{AUC} - 1$) | $\ge 0.40$ | **`0.5093`** | **PASSED (Production Grade)** |
| **Kolmogorov-Smirnov (KS)** | $\ge 0.30$ | **`0.3978`** | **PASSED (High Discrimination)** |
| **Brier Score Loss** | $\le 0.20$ | **`0.1412`** | **PASSED (Well-Calibrated)** |

#### 6.2 Key Metric Definitions
1. **Gini Coefficient ($0.5093$):** Measures the discriminatory power between good and bad risk profiles across all cutoff points. Banking scorecards with Gini $> 0.40$ are approved for live underwriting.
2. **KS Statistic ($0.3978$):** Identifies the maximum vertical separation between the cumulative distribution function of defaults vs non-defaults:
   $$KS = \max |F_{\text{bad}}(s) - F_{\text{good}}(s)|$$
   A KS of $39.8\%$ demonstrates that the model effectively separates safe applicants from high-risk defaults.

---

### 7. Score Calibration & Scaling Formula

The raw probability of default ($PD \in [0, 1]$) is converted into a standard commercial credit score ($300\text{–}850$):

$$\text{Credit Score} = \text{round}\Big(850 - (PD \times 550)\Big)$$

#### Risk Banding Decision Rules:
* **Score $\ge 750$ ($PD < 5\%$):** Prime — Automated instant disbursal up to 100M VND.
* **Score $650 - 749$ ($PD \in [5\%, 15\%)$):** Near-Prime — Standard approval up to 50M VND.
* **Score $550 - 649$ ($PD \in [15\%, 30\%)$):** Gray Zone — Enhanced due diligence, capped at 20M VND.
* **Score $< 550$ ($PD \ge 30\%$):** High Risk — Auto-decline with Adverse Action Notice.

---

### 8. Explainability (XAI) & Compliance Guardrails

#### 8.1 SHAP TreeExplainer
For every loan application scored, the SHAP engine calculates exact Shapley values $\phi_i$ satisfying the local efficiency property:
$$f(x) = \phi_0 + \sum_{i=1}^{M} \phi_i(x)$$

#### 8.2 Adverse Action Code Generation
When an application is declined or granted a reduced credit limit, the system extracts the top 3 negative contributors ($\phi_i > 0$ pushing up default probability):
1. **Adverse Reason 1:** High Debt-to-Income ratio (`feat_dti`).
2. **Adverse Reason 2:** Elevated inquiry velocity within past 3 months (`feat_enq_velocity`).
3. **Adverse Reason 3:** High dependency on non-bank finance companies (`feat_non_bank_loan_ratio`).

#### 8.3 Anti-Bias & Fairness Audit
In compliance with SBV and PDPA anti-discrimination principles:
* `gender` and `province` are evaluated via **Disparate Impact Ratio (DIR)**:
  $$\text{DIR} = \frac{P(\text{Approved} \mid \text{Protected Group})}{P(\text{Approved} \mid \text{Reference Group})} \ge 0.80$$
* If DIR falls below the $80\%$ threshold (Four-Fifths Rule), the model is flagged for bias mitigation.

---

### 9. Production Serving & Retraining Protocol

* **Artifacts Persisted:**
  * `models/model.joblib`: Serialized LightGBM booster.
  * `models/preprocessor.joblib`: Fitted pipeline and imputation dictionaries.
  * `models/metrics.json`: Versioned benchmark scores.
* **Inference Latency Target:** $P99 < 50\text{ms}$ on standard single-instance API server.
* **Drift Monitoring (PSI):**
  $$\text{PSI} = \sum \Big((Actual\% - Expected\%) \times \ln(Actual\% / Expected\%)\Big)$$
  * $\text{PSI} < 0.10$: Model stable; continue monitoring.
  * $0.10 \le \text{PSI} < 0.25$: Moderate drift; alert credit risk team.
  * $\text{PSI} \ge 0.25$: Significant population drift; trigger automated retraining.
