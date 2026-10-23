# Data Dictionary & Feature Catalog

## Alternative Credit Scoring & Underwriting System (FIN-04)

---

### 1. Overview & Data Provenance
* **Dataset Shape:** 14,000 historical loan applications × 88 base features + 11 engineered risk indicators.
* **Extraction Period:** April 8, 2024 to June 30, 2024.
* **Target Population:** Underbanked, gig-economy freelancers, students, and micro-merchants applying for unsecured digital consumer facilities.
* **Data Sources:** Core loan origination system, alternative behavioral proxies, e-wallet transaction metadata, and Credit Information Center (CIC) lookup extracts.

---

### 2. Target Variable & Temporal Identification

| Column Name | Data Type | Missing Count | Description & Risk Interpretation |
| :--- | :---: | :---: | :--- |
| `disbursement_date` | `datetime64[ns]` | 0 (0.0%) | Date facility was disbursed. Used **strictly** for temporal splitting to eliminate lookahead bias. Never used as an ML input feature. |
| `dpd10_3mob` | `int64` / `float64` | 0 (0.0%) | **Target Variable:** Binary flag indicating whether borrower accumulated ≥ 10 Days Past Due (DPD) within the first 3 Months On Books (3MOB). `1` = Default (Bad), `0` = Non-default (Good). Default rate: ~19.84%. |

---

### 3. Borrower Demographics & Declared Financials

| Column Name | Data Type | Missing Count | Business Definition & Imputation Strategy |
| :--- | :---: | :---: | :--- |
| `age` | `int64` | 0 (0.0%) | Chronological age of applicant in years. Policy knock-out range: 18 ≤ Age ≤ 65. |
| `gender` | `object` | 0 (0.0%) | Borrower gender (`MALE`, `FEMALE`). Protected attribute monitored for fair lending compliance. |
| `occupation` | `object` | 1 (0.01%) | Employment classification (e.g., `OFFICE_WORKER`, `FREELANCER`, `STUDENT`). Imputed with `'UNKNOWN'`. |
| `month_incom` | `float64` | 1 (0.01%) | Declared monthly net income in VND. Imputed with **training median** (never 0 to prevent DTI division-by-zero). |
| `operating_system`| `object` | 0 (0.0%) | Mobile device OS (`ANDROID`, `IOS`). Behavioral proxy for device tier and digital engagement. |
| `province` | `object` | 0 (0.0%) | Geographic province / municipality in Vietnam. Protected attribute monitored for regional bias. |

---

### 4. Credit Facility & Debt Horizon (Tenure & Lender Type)

| Column Name | Data Type | Description |
| :--- | :---: | :--- |
| `short_term_amount` | `float64` | Outstanding principal amount on short-term facilities ($< 12$ months) in VND. |
| `mid_term_amount` | `float64` | Outstanding principal amount on medium-term facilities ($12 - 36$ months) in VND. |
| `long_term_amount` | `float64` | Outstanding principal amount on long-term facilities ($> 36$ months, e.g., mortgages) in VND. |
| `short_term_count` | `int64` | Number of active short-term loan facilities across all institutions. |
| `mid_term_count` | `int64` | Number of active medium-term loan facilities across all institutions. |
| `long_term_count` | `int64` | Number of active long-term loan facilities across all institutions. |
| `short_term_count_bank` | `int64` | Active short-term loans financed by commercial banks. |
| `mid_term_count_bank` | `int64` | Active medium-term loans financed by commercial banks. |
| `long_term_count_bank` | `int64` | Active long-term loans financed by commercial banks. |
| `short_term_count_non_bank` | `int64` | Active short-term loans financed by non-bank financial companies (subprime proxy). |
| `mid_term_count_non_bank` | `int64` | Active medium-term loans financed by non-bank financial companies. |
| `long_term_count_non_bank` | `int64` | Active long-term loans financed by non-bank financial companies. |
| `num_loans` | `int64` | Total active credit facilities across all financial institutions. |
| `num_loans_bank` | `int64` | Total active loan contracts with commercial banks. |
| `num_loans_non_bank` | `int64` | Total active loan contracts with finance companies / micro-lenders. |
| `num_cc` | `int64` | Total active credit card accounts held by applicant. |
| `num_cc_bank` | `int64` | Credit card accounts issued by commercial banks. |
| `num_cc_non_bank` | `int64` | Credit cards / revolving lines issued by non-bank finance companies. |
| `num_relationship` | `int64` | Total count of distinct financial institutions with whom borrower has an active relationship. |

---

### 5. Historical Longitudinal Balances (12-Month Performance)

*Fields `M1` represent the most recent billing cycle prior to application; `M12` represents the position 12 months prior.*

| Feature Family | Format | Count | Description |
| :--- | :--- | :---: | :--- |
| `outs_bal_loan_m1` to `m12` | `float64` | 12 | Monthly aggregate outstanding loan balances for cycles 1 through 12. |
| `outs_bal_cc_m1` to `m12` | `float64` | 12 | Monthly aggregate revolving credit card outstanding balances for cycles 1 through 12. |
| `outs_bal_all_m1` to `m12` | `float64` | 12 | Total aggregate debt balance (Loan + Card) for cycles 1 through 12. |
| `cc_limit_total` | `float64` | 1 | Total approved revolving credit card limit across all issuers. |
| `outs_bal_cc_current` | `float64` | 1 | Current revolving balance utilized on credit cards at time of underwriting. |

---

### 6. Inquiry Velocity & Credit Hunger Signals

| Column Name | Data Type | Description & Risk Interpretation |
| :--- | :---: | :--- |
| `enq_3m` | `int64` | Hard credit inquiries within past 3 months. Elevated inquiries signal immediate liquidity search. |
| `enq_6m` | `int64` | Hard credit inquiries within past 6 months. |
| `enq_9m` | `int64` | Hard credit inquiries within past 9 months. |
| `enq_12m` | `int64` | Hard credit inquiries within past 12 months. |
| `num_new_loan_3m` | `int64` | New credit lines opened in the past 3 months. |
| `num_new_loan_6m` | `int64` | New credit lines opened in the past 6 months. |
| `num_new_loan_9m` | `int64` | New credit lines opened in the past 9 months. |
| `num_new_loan_12m` | `int64` | New credit lines opened in the past 12 months. |

---

### 7. Credit Bureau (CIC) & Alternative Risk Proxies

| Column Name | Data Type | Missing Count | Meaning & Domain Imputation Policy |
| :--- | :---: | :---: | :--- |
| `has_cic` | `int64` | 0 (0.0%) | Binary flag indicating if borrower has an existing credit record at CIC (1=Yes, 0=Thin-file). |
| `cic_credit_score` | `float64` | 1 (0.01%) | Official bureau credit score (range 300–850). Imputed with **training median**. |
| `cic_max_group_12m` | `float64` | 1 (0.01%) | Maximum debt classification group in past 12 months (1=Standard, 2=Special Mention, 3-5=Bad Debt). Imputed with **training median**. |
| `cic_total_balance` | `float64` | 0 (0.0%) | Aggregate debt balance registered at CIC. |
| `cic_bank_balance` | `float64` | 0 (0.0%) | Total debt balance owed to commercial banks. |
| `cic_fin_balance` | `float64` | 0 (0.0%) | Total debt balance owed to consumer finance companies. |
| `cic_num_loans` | `int64` | 0 (0.0%) | Total number of loans reported to bureau. |
| `cic_num_fin_loans` | `int64` | 0 (0.0%) | Number of loans with consumer finance companies. |
| `cic_has_fin_loan` | `int64` | 0 (0.0%) | Indicator flag if applicant holds loans with consumer finance companies. |
| `cic_primary_inst_type`| `object` | 3,717 (26.6%) | Primary institution type (`BANK`, `FINANCE_COMPANY`). Missing indicates no history → Imputed with `'UNKNOWN'`. |
| `cic_cc_util` | `float64` | 3,782 (27.0%) | Bureau credit card limit utilization ratio. Missing indicates no active credit card → Imputed with `0.0`. |
| `cic_inquiry_3m` | `int64` | 0 (0.0%) | Inquiries recorded by bureau within past 3 months. |
| `cic_util_trend` | `float64` | 3,746 (26.8%) | Trend slope in credit card utilization over time. Missing → Imputed with `0.0`. |
| `cic_balance_trend` | `float64` | 3,746 (26.8%) | Trend slope in total outstanding balance. Missing → Imputed with `0.0`. |
| `cic_fin_ratio` | `float64` | 3,717 (26.6%) | Ratio of finance company debt to total debt. Missing → Imputed with `0.0`. |

---

### 8. Engineered Domain Features & Mathematical Formulations

#### 8.1 Feature Catalog Summary Table

| Feature Name | Type | Code Reference | Business Interpretation & Purpose |
| :--- | :---: | :--- | :--- |
| `feat_dti` | `float64` | `clip(total_loan_amount / month_incom, 0, 50)` | **Debt-to-Income:** Overall debt obligations relative to declared monthly income. |
| `feat_bal_to_income` | `float64` | `clip(outs_bal_all_m1 / month_incom, 0, 50)` | **Balance-to-Income:** Evaluates immediate short-term leverage pressure. |
| `feat_non_bank_loan_ratio` | `float64` | `clip(num_loans_non_bank / max(num_loans, 1), 0, 1)` | **Subprime Dependency:** Share of credit financed by consumer finance companies. |
| `feat_cc_util_calc` | `float64` | `clip(outs_bal_cc_current / max(cc_limit_total, 1), 0, 5)` | **Revolving Stress:** Utilization rate of active credit cards. |
| `feat_enq_velocity` | `float64` | `clip(enq_3m / max(enq_12m / 4.0, 1.0), 0, 10)` | **Inquiry Acceleration:** Surge in credit search over the preceding quarter. |
| `feat_new_loan_velocity` | `float64` | `clip(num_new_loan_3m / max(num_new_loan_12m / 4.0, 1.0), 0, 10)` | **Origination Velocity:** Rapid stacking of newly opened credit facilities. |
| `feat_bal_trend_m1_m3` | `float64` | `clip((outs_bal_all_m1 - outs_bal_all_m3) / (outs_bal_all_m3 + 1), -10, 10)` | **Debt Trajectory:** 3-month longitudinal balance accumulation trend. |
| `feat_high_risk_cic` | `int64` | `(cic_max_group_12m >= 3).astype(int)` | **Severe Bureau Flag:** Historical classified non-performing debt (Group 3–5). |
| `flag_missing_income` | `int64` | `month_incom.isna().astype(int)` | **Income Missingness:** Identifies unbanked gig-workers lacking formal payroll. |
| `flag_missing_cic_util` | `int64` | `cic_cc_util.isna().astype(int)` | **Revolving Absence:** Identifies borrowers with zero credit card footprint. |
| `flag_thin_file` | `int64` | `(has_cic == 0).astype(int)` | **Thin-File Segmentation:** Distinguishes unbanked borrowers with zero prior bureau files. |

---

#### 8.2 Formal Mathematical Specifications (LaTeX)

##### 1. Debt-to-Income Ratio ($\text{DTI}$)
Quantifies the borrower's total contractual loan exposure against their verifiable monthly earning power:

$$
\text{DTI} = \operatorname{clip}\left( \frac{\text{Principal}_{\text{short}} + \text{Principal}_{\text{mid}} + \text{Principal}_{\text{long}}}{\text{Income}_{\text{monthly}}}, 0, 50 \right)
$$

##### 2. Balance-to-Income Ratio ($\text{BTI}$)
Evaluates immediate liquidity pressure by measuring current cycle ($M_1$) outstanding balance against monthly cash flow:

$$
\text{BTI} = \operatorname{clip}\left( \frac{\text{Balance}_{M1}}{\text{Income}_{\text{monthly}}}, 0, 50 \right)
$$

##### 3. Subprime & Non-Bank Dependency Ratio
Measures the concentration of debt originating from non-bank consumer finance companies:

$$
\text{Ratio}_{\text{non-bank}} = \operatorname{clip}\left( \frac{N_{\text{loans, non-bank}}}{\max(N_{\text{loans, total}}, 1)}, 0, 1 \right)
$$

##### 4. Revolving Credit Card Utilization ($\text{Util}_{\text{card}}$)
Identifies liquidity distress when revolving card balances approach or exceed credit limits:

$$
\text{Util}_{\text{card}} = \operatorname{clip}\left( \frac{\text{Balance}_{\text{card, current}}}{\max(\text{Limit}_{\text{card, total}}, 1)}, 0, 5 \right)
$$

##### 5. Credit Inquiry Velocity ($\text{Velocity}_{\text{inquiry}}$)
Detects urgent credit hunger by comparing short-term inquiry velocity (past 3 months) to the historical quarterly baseline:

$$
\text{Velocity}_{\text{inquiry}} = \operatorname{clip}\left( \frac{\text{Inquiries}_{3M}}{\max(\text{Inquiries}_{12M} / 4.0, 1.0)}, 0, 10 \right)
$$

##### 6. Quarterly Debt Trajectory ($\Delta\text{Balance}_{3M}$)
Evaluates whether borrower indebtedness is escalating or de-leveraging over a 90-day window:

$$
\Delta\text{Balance}_{3M} = \operatorname{clip}\left( \frac{\text{Balance}_{M1} - \text{Balance}_{M3}}{\text{Balance}_{M3} + 1.0}, -10, 10 \right)
$$

##### 7. Historical Delinquency Segmentation Flag
Formalizes regulatory knock-outs based on Central Credit Information Center (CIC) debt classification:

$$
\text{Flag}_{\text{bad-debt}} = \begin{cases} 1 & \text{if } \text{CIC Group}_{12M} \ge 2 \text{ (Special Mention / Default)} \\ 0 & \text{otherwise (Standard performing)} \end{cases}
$$

##### 8. Thin-File Unbanked Segmentation Flag
Distinguishes conventional "thick-file" applicants from unbanked individuals assessed via alternative behavioral proxies:

$$
\text{Flag}_{\text{thin-file}} = \begin{cases} 1 & \text{if applicant has zero credit bureau footprint} \\ 0 & \text{otherwise} \end{cases}
$$


