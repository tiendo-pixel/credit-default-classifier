# Data Dictionary & Feature Catalog

## Alternative Credit Scoring & Underwriting System (FIN-04)

---

### 1. Overview & Data Provenance
* **Dataset Shape:** 14,000 historical loan applications $\times$ 88 base features + 8 engineered risk indicators.
* **Extraction Period:** April 8, 2024 to June 30, 2024.
* **Target Population:** Underbanked, gig-economy freelancers, students, and micro-merchants applying for unsecured digital consumer facilities.
* **Data Sources:** Core loan origination system, alternative behavioral proxies, e-wallet transaction metadata, and Credit Information Center (CIC) lookup extracts.

---

### 2. Target Variable & Temporal Identification

| Column Name | Data Type | Missing Count | Description & Risk Interpretation |
| :--- | :---: | :---: | :--- |
| `disbursement_date` | `datetime64[ns]` | 0 (0.0%) | Date facility was disbursed. Used **strictly** for temporal splitting to eliminate lookahead bias. Never used as an ML input feature. |
| `dpd10_3mob` | `int64` / `float64` | 0 (0.0%) | **Target Variable:** Binary flag indicating whether borrower accumulated $\ge 10$ Days Past Due (DPD) within the first 3 Months On Books (3MOB). `1` = Default (Bad), `0` = Non-default (Good). Default rate: ~19.84%. |

---

### 3. Borrower Demographics & Declared Financials

| Column Name | Data Type | Missing Count | Business Definition & Imputation Strategy |
| :--- | :---: | :---: | :--- |
| `age` | `int64` | 0 (0.0%) | Chronological age of applicant in years. Policy knock-out range: $18 \le \text{Age} \le 65$. |
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
| `cic_primary_inst_type`| `object` | 3,717 (26.6%) | Primary institution type (`BANK`, `FINANCE_COMPANY`). Missing indicates no history $\rightarrow$ Imputed with `'UNKNOWN'`. |
| `cic_cc_util` | `float64` | 3,782 (27.0%) | Bureau credit card limit utilization ratio. Missing indicates no active credit card $\rightarrow$ Imputed with `0.0`. |
| `cic_inquiry_3m` | `int64` | 0 (0.0%) | Inquiries recorded by bureau within past 3 months. |
| `cic_util_trend` | `float64` | 3,746 (26.8%) | Trend slope in credit card utilization over time. Missing $\rightarrow$ Imputed with `0.0`. |
| `cic_balance_trend` | `float64` | 3,746 (26.8%) | Trend slope in total outstanding balance. Missing $\rightarrow$ Imputed with `0.0`. |
| `cic_fin_ratio` | `float64` | 3,717 (26.6%) | Ratio of finance company debt to total debt. Missing $\rightarrow$ Imputed with `0.0`. |

---

### 8. Engineered Domain Features (Derived Features)

| Feature Name | Computation Formula | Business Rationale |
| :--- | :--- | :--- |
| `feat_dti` | $\text{clip}\left(\frac{\text{Total Loan Amount}}{\text{Declared Monthly Income}}, 0, 50\right)$ | **Debt-to-Income:** Measures overall debt load against earning power. Values $> 5$ indicate high distress. |
| `feat_bal_to_income` | $\text{clip}\left(\frac{\text{outs\_bal\_all\_m1}}{\text{Declared Monthly Income}}, 0, 50\right)$ | **Balance-to-Income:** Evaluates immediate short-term balance strain vs current income. |
| `feat_non_bank_loan_ratio`| $\text{clip}\left(\frac{\text{num\_loans\_non\_bank}}{\text{num\_loans} + 1e-5}, 0, 1\right)$ | **Subprime Dependency:** High dependency on finance companies indicates rejection by traditional banks. |
| `feat_cc_util_calc` | $\text{clip}\left(\frac{\text{outs\_bal\_cc\_current}}{\text{cc\_limit\_total} + 1e-5}, 0, 5\right)$ | **Revolving Stress:** Card utilization $> 80\%$ reflects acute liquidity needs. |
| `feat_enq_velocity` | $\text{clip}\left(\frac{\text{enq\_3m}}{\text{enq\_12m} / 4.0 + 1e-5}, 0, 10\right)$ | **Inquiry Acceleration:** Ratio $> 1.0$ indicates search for credit has accelerated recently. |
| `feat_new_loan_velocity` | $\text{clip}\left(\frac{\text{num\_new\_loan\_3m}}{\text{num\_new\_loan\_12m} / 4.0 + 1e-5}, 0, 10\right)$ | **Loan Origination Velocity:** Rapid loan accumulation in short duration precedes default. |
| `feat_bal_trend_m1_m3` | $\text{clip}\left(\frac{\text{outs\_bal\_all\_m1} - \text{outs\_bal\_all\_m3}}{\text{outs\_bal\_all\_m3} + 1e-5}, -10, 10\right)$| **Debt Growth Rate:** Positive growth indicates escalating leverage over the prior quarter. |
| `feat_high_risk_cic` | $(\text{cic\_max\_group\_12m} \ge 3)\text{.astype(int)}$ | **Bureau Flag:** Direct binary indicator for historical sub-prime or default status. |
