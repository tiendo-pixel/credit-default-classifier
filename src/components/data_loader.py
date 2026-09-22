import json
import os
import re
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import train_test_split


class DataLoader:
    def __init__(
        self,
        data_path: str = "data/raw/Professional Test.xlsx",
        lookup_path: str = "data/raw/Institutions code (1).xlsx",
        output_path: str = "data/processed",
        sheet_name: str = "Dataset",
    ):
        # Initialize file paths and output directory
        self.data_path = data_path
        self.lookup_path = lookup_path
        self.output_path = output_path
        self.sheet_name = sheet_name

        # Create output directory if it does not exist
        os.makedirs(self.output_path, exist_ok=True)

    # ------------------------------------------------------------------
    # Private helpers: safe type conversion (never raises exceptions)
    # ------------------------------------------------------------------
    @staticmethod
    def _safe_int(x) -> int:
        # Convert value to int safely, return np.nan on failure
        try:
            return int(x)
        except (TypeError, ValueError):
            return np.nan

    @staticmethod
    def _safe_float(x) -> float:
        # Convert value to float safely, return np.nan on failure
        try:
            return float(x)
        except (TypeError, ValueError):
            return np.nan

    # ------------------------------------------------------------------
    # Step 1: Load institution lookup table
    # Map first 5 chars of SBV code -> institution type (Bank / Finance / Other)
    # ------------------------------------------------------------------
    def load_institution_mapping(self) -> dict:
        # Read the Excel lookup table and return a SBV_CODE[:5] -> Type mapping dict
        if not os.path.exists(self.lookup_path):
            raise FileNotFoundError(f"Lookup file not found: {self.lookup_path}")

        lookup_df = pd.read_excel(self.lookup_path)

        # Validate required columns exist
        required_cols = {"SBV code", "Type"}
        missing = required_cols - set(lookup_df.columns)
        if missing:
            raise KeyError(f"Lookup file is missing columns: {missing}")

        # Use first 5 characters of SBV code as the key (matches MATCTD in cic_data)
        code_to_type = {
            str(row["SBV code"])[:5]: str(row["Type"]).strip() if pd.notna(row["Type"]) and str(row["Type"]).strip() not in ("", "nan", "None") else "Other"
            for _, row in lookup_df.iterrows()
        }
        print(f"  -> Loaded {len(code_to_type):,} institution mappings")
        return code_to_type

    # ------------------------------------------------------------------
    # Step 2: Parse a single cic_data JSON string into numeric features
    # Based on CIC JSON structure with Vietnamese keys (NOIDUNG, QHTDHT, etc.)
    # ------------------------------------------------------------------
    def parse_cic_data(self, cic_json, inst_map: dict, application_date=None) -> dict:
        # Default result for customers with no CIC record (new-to-credit / thin-file)
        result = {
            "has_cic": 0,
            "cic_credit_score": np.nan,
            "cic_max_group_12m": np.nan,
            "cic_total_balance": 0.0,
            "cic_bank_balance": 0.0,
            "cic_fin_balance": 0.0,
            "cic_num_loans": 0,
            "cic_num_fin_loans": 0,
            "cic_has_fin_loan": 0,
            "cic_primary_inst_type": "None",
            "cic_cc_util": np.nan,
            "cic_inquiry_3m": 0,
            "cic_delinquency_count_12m": 0,
            "cic_recent_delinquency": 0,
            "cic_util_trend": np.nan,
            "cic_balance_trend": np.nan,
            "cic_fin_ratio": np.nan,
        }

        # Skip null, empty, or placeholder values
        if pd.isna(cic_json) or str(cic_json).strip() in ("", "[]", "null", "{}"):
            return result

        try:
            # Parse JSON string -> Python dict
            data = json.loads(cic_json) if isinstance(cic_json, str) else cic_json
            nd = data.get("NOIDUNG", {})
            result["has_cic"] = 1

            # Credit score
            score = self._safe_float(nd.get("DIEMTD", {}).get("DIEM"))
            result["cic_credit_score"] = score

            # Active loans (QHTDHT -> QHTD -> DONG)
            loans = nd.get("QHTDHT", {}).get("QHTD", {}).get("DONG", []) or []

            max_group = 0
            delinquency_count = 0

            for loan in loans:
                amount = self._safe_float(loan.get("TONG_VND")) or 0.0

                result["cic_total_balance"] += amount
                result["cic_num_loans"] += 1

                # Debt group from CTLOAIVAY.DONG list
                group_list = loan.get("CTLOAIVAY", {}).get("DONG", []) or []
                if isinstance(group_list, list) and group_list:
                    group = self._safe_float(group_list[0].get("NHOMNO"))
                    if not np.isnan(group):
                        max_group = max(max_group, int(group))
                        if group >= 2:
                            delinquency_count += 1

                # Institution type via SBV code (first 5 chars)
                inst_code = str(loan.get("MATCTD", ""))[:5]
                inst_type = inst_map.get(inst_code, "Other")
                inst_type_lower = str(inst_type).lower()

                if "bank" in inst_type_lower and "non-bank" not in inst_type_lower:
                    result["cic_bank_balance"] += amount
                elif "finan" in inst_type_lower or "leasing" in inst_type_lower:
                    result["cic_fin_balance"] += amount
                    result["cic_num_fin_loans"] += 1

            result["cic_max_group_12m"] = max_group if max_group > 0 else np.nan
            result["cic_delinquency_count_12m"] = delinquency_count
            result["cic_recent_delinquency"] = 1 if max_group >= 2 else 0

            # ----------------------------------------------------------
            # Credit card utilization (DUNO_THETD)
            # ----------------------------------------------------------
            cards = nd.get("QHTDHT", {}).get("DUNO_THETD", {}).get("DONG", []) or []

            cc_balance = sum(self._safe_float(c.get("SOTIEN_PHAI_TT")) or 0 for c in cards)
            cc_limit = sum(self._safe_float(c.get("HANMUC_THETD")) or 0 for c in cards)

            if cc_limit > 0:
                result["cic_cc_util"] = cc_balance / cc_limit

            # ----------------------------------------------------------
            # Inquiry count in last 90 days (LS_TRACUU_12THANG)
            # ----------------------------------------------------------
            inquiries = nd.get("LS_TRACUU_12THANG", {}).get("DONG", []) or []
            recent_3m = 0

            if application_date is not None:
                # Normalize application_date to datetime object
                if not isinstance(application_date, datetime):
                    try:
                        application_date = pd.to_datetime(application_date)
                    except Exception:
                        application_date = None

            if application_date is not None:
                for inq in inquiries:
                    date_str = str(inq.get("NGAYTRACUU", ""))[:8]
                    if date_str:
                        try:
                            inq_date = datetime.strptime(date_str, "%Y%m%d")
                            if (application_date - inq_date).days <= 90:
                                recent_3m += 1
                        except (ValueError, TypeError):
                            pass

            result["cic_inquiry_3m"] = recent_3m

            # ----------------------------------------------------------
            # 12-month balance history trend (LSQHTD -> DUNO_12THANG)
            # ----------------------------------------------------------
            history = nd.get("LSQHTD", {}).get("DUNO_12THANG", {}).get("DONG", []) or []

            hist_data = []
            for entry in history:
                month = entry.get("THANG", "")
                bal = self._safe_float(entry.get("TONGDUNO"))
                if month and not np.isnan(bal):
                    hist_data.append((month, bal))

            # Sort chronologically by month string (YYYYMM format)
            hist_data = sorted(hist_data, key=lambda x: x[0])

            if len(hist_data) >= 2:
                balances = [x[1] for x in hist_data]
                result["cic_balance_trend"] = balances[-1] - balances[0]
                result["cic_util_trend"] = float(np.std(balances))

            # ----------------------------------------------------------
            # Derived: Finance debt ratio, flag, and primary institution
            # ----------------------------------------------------------
            result["cic_has_fin_loan"] = 1 if (result["cic_num_fin_loans"] > 0 or result["cic_fin_balance"] > 0) else 0

            if result["cic_total_balance"] > 0:
                result["cic_fin_ratio"] = result["cic_fin_balance"] / result["cic_total_balance"]

                if result["cic_fin_balance"] > result["cic_bank_balance"]:
                    result["cic_primary_inst_type"] = "Finance"
                elif result["cic_bank_balance"] > 0:
                    result["cic_primary_inst_type"] = "Bank"
                else:
                    result["cic_primary_inst_type"] = "Other"
            elif result["cic_num_loans"] > 0:
                result["cic_primary_inst_type"] = "Other"
            else:
                result["cic_primary_inst_type"] = "None"

        except Exception:
            # If JSON is malformed or structure is unexpected, mark as no-CIC
            result["has_cic"] = 0

        return result

    # ------------------------------------------------------------------
    # Step 3: Load main dataset, extract CIC features, and merge
    # ------------------------------------------------------------------
    def load_and_map_data(self, date_col: str = "disbursement_date") -> pd.DataFrame:
        # Validate main data file exists
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        print(f"Loading sheet '{self.sheet_name}' from {self.data_path} ...")
        df = pd.read_excel(self.data_path, sheet_name=self.sheet_name)
        print(f"  -> Loaded {len(df):,} rows x {len(df.columns)} columns")

        # Load institution mapping dictionary
        inst_map = self.load_institution_mapping()

        # Detect the cic_data column (case-insensitive search)
        cic_col = next((col for col in df.columns if "cic" in col.lower()), None)

        if cic_col is None:
            print("  [WARNING] No cic_data column found. Skipping CIC feature extraction.")
            return df

        print(f"  -> Extracting CIC features from column '{cic_col}' ...")

        results = []
        total = len(df)

        for idx, row in df.iterrows():
            cic_val = row[cic_col]
            app_date = row.get(date_col) if date_col in df.columns else None

            if pd.notna(cic_val) and str(cic_val).strip() not in ("", "[]"):
                res = self.parse_cic_data(cic_val, inst_map, application_date=app_date)
            else:
                res = {"has_cic": 0}

            results.append(res)

            if idx > 0 and idx % 1000 == 0:
                print(f"  Processed {idx:,} / {total:,} rows ...")

        cic_df = pd.DataFrame(results, index=df.index)

        # Drop raw JSON column and append extracted numeric features
        df = pd.concat([df.drop(columns=[cic_col]), cic_df], axis=1)

        # Remove any duplicated columns (keep last occurrence)
        df = df.loc[:, ~df.columns.duplicated(keep="last")]

        # --------------------------------------------------------------
        # Extract Province / City from address and drop raw full address
        # --------------------------------------------------------------
        addr_cols = [c for c in df.columns if any(k in c.lower() for k in ["address", "dia_chi", "diachi", "location", "province"])]
        for col in addr_cols:
            if col.lower() != "province":
                print(f"  -> Extracting clean 'province' from '{col}' and dropping raw address ...")
                df["province"] = df[col].apply(self.extract_province)
                df = df.drop(columns=[col])
                print(f"  -> Extracted {df['province'].nunique():,} unique provinces")
                break
            else:
                df["province"] = df["province"].apply(self.extract_province)

        # --------------------------------------------------------------
        # Drop data leakage, IDs, and constant columns
        # --------------------------------------------------------------
        df = self.drop_leakage_and_metadata(df, target_col="DPD10_3MOB")

        print(f"  -> Added {len(cic_df.columns)} CIC feature columns")
        print(f"  -> CIC coverage: {cic_df['has_cic'].sum():,} / {total:,} records have CIC data")

        return df

    # ------------------------------------------------------------------
    # Step 4: Drop data leakage features, PII IDs, and constant columns
    # - Drops post-disbursement repayment proxies: FPD10+, FPD30, etc.
    # - Drops customer/application IDs: customer_id, contract_id, cif, etc.
    # - Drops single-value (zero-variance) columns
    # ------------------------------------------------------------------
    @staticmethod
    def drop_leakage_and_metadata(
        df: pd.DataFrame,
        target_col: str = "DPD10_3MOB",
        additional_drop: list = None,
    ) -> pd.DataFrame:
        cols_to_drop = set()

        # 1. Post-disbursement performance / Data leakage patterns
        leakage_patterns = ["fpd", "collection", "recovery", "writeoff", "chargeoff", "overdue_days"]
        for col in df.columns:
            if col == target_col:
                continue
            col_lower = col.lower().strip()
            # Match any FPD variations (e.g., FPD10+, fpd10, fpd30, FPD_flag)
            if any(p in col_lower for p in leakage_patterns):
                cols_to_drop.add(col)
            # Secondary DPD target proxies (if different from primary target DPD10_3MOB)
            elif "dpd" in col_lower and col != target_col:
                cols_to_drop.add(col)

        # 2. PII identifiers & Metadata keys
        id_patterns = [
            "customer_id", "application_id", "contract_id", "cif", "id_number",
            "id_card", "phone_number", "phone", "email", "full_name", "cust_id", "app_id"
        ]
        for col in df.columns:
            if col == target_col:
                continue
            col_lower = col.lower().strip()
            if any(p in col_lower for p in id_patterns) or col_lower in ["id", "cif_no", "app_id", "cust_id"]:
                cols_to_drop.add(col)

        # 3. User-specified columns to drop
        if additional_drop:
            for col in additional_drop:
                if col in df.columns and col != target_col:
                    cols_to_drop.add(col)

        # 4. Zero-variance / constant columns (no predictive power)
        for col in df.columns:
            if col != target_col and df[col].nunique(dropna=False) <= 1:
                cols_to_drop.add(col)

        cols_to_drop_list = [c for c in df.columns if c in cols_to_drop]
        if cols_to_drop_list:
            print(f"  -> Dropped {len(cols_to_drop_list)} leakage/ID/constant columns:")
            for col in cols_to_drop_list:
                print(f"     - {col}")
            df = df.drop(columns=cols_to_drop_list)

        return df

    # ------------------------------------------------------------------
    # Step 5: Helper to extract & group Vietnam Province into 4 major tiers:
    # 'Hanoi', 'Danang', 'Saigon', and 'Others'
    # ------------------------------------------------------------------
    @staticmethod
    def extract_province(addr) -> str:
        if pd.isna(addr) or not str(addr).strip():
            return "Others"

        addr_str = str(addr).strip().lower()

        # Check for Saigon / Ho Chi Minh City
        if any(k in addr_str for k in ["hồ chí minh", "ho chi minh", "hcm", "sài gòn", "saigon", "tp.hcm", "tphcm"]):
            return "Saigon"

        # Check for Hanoi
        if any(k in addr_str for k in ["hà nội", "ha noi", "hn", "tp.hn"]):
            return "Hanoi"

        # Check for Danang
        if any(k in addr_str for k in ["đà nẵng", "da nang", "đn"]):
            return "Danang"

        # All other provinces / unknown
        return "Others"

    # ------------------------------------------------------------------
    # Step 6: Sanitize DataFrame dtypes and normalize categoricals
    # - Maps '0', '0.0', missing values in 'operating_system' to 'Unknown' / 'Other'
    # - Standardizes 'IOS' -> 'iOS', 'ANDROID' -> 'Android'
    # - Converts mixed object columns to consistent string dtypes for Parquet
    # ------------------------------------------------------------------
    @staticmethod
    def _sanitize_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
        df_clean = df.copy()

        # Specific normalization for operating_system if present
        os_col = next((c for c in df_clean.columns if c.lower() in ("operating_system", "os")), None)
        if os_col is not None:
            def _normalize_os(val):
                if pd.isna(val):
                    return "Unknown"
                s = str(val).strip().lower()
                if s in ("", "0", "0.0", "-1", "none", "nan", "null", "unknown", "other"):
                    return "Unknown"
                if "ios" in s or "iphone" in s:
                    return "iOS"
                if "android" in s:
                    return "Android"
                return str(val).strip().title()

            df_clean[os_col] = df_clean[os_col].apply(_normalize_os)

        # General object column sanitization to prevent pyarrow type inference errors
        for col in df_clean.columns:
            if df_clean[col].dtype == "object":
                df_clean[col] = df_clean[col].apply(
                    lambda x: str(x).strip() if pd.notna(x) and str(x).strip() not in ("", "nan", "None", "<NA>") else None
                )

        return df_clean

    # ------------------------------------------------------------------
    # Step 7: Save fully extracted dataset as a single file (before train/test split)
    # ------------------------------------------------------------------
    def save_extracted(self, df: pd.DataFrame, filename: str = "data_extracted.parquet") -> str:
        # Save the complete processed DataFrame to data/processed/ for inspection and EDA
        output_path = os.path.join(self.output_path, filename)
        df_clean = self._sanitize_for_parquet(df)
        df_clean.to_parquet(output_path, index=False)
        print(f"Saved extracted dataset -> {output_path}  ({len(df_clean):,} rows x {len(df_clean.columns)} cols)")
        return output_path

    # ------------------------------------------------------------------
    # Step 8: Stratified train/test split and save to parquet (run after EDA is done)
    # ------------------------------------------------------------------
    def split_and_save(
        self,
        df: pd.DataFrame,
        target_col: str = "DPD10_3MOB",
        test_size: float = 0.2,
        random_state: int = 42,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        # Use stratified split to preserve bad rate in both train and test sets
        stratify = df[target_col] if target_col in df.columns else None
        if stratify is None:
            print(f"  [WARNING] Target column '{target_col}' not found. Using random split.")

        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )

        # Save to parquet format (lightweight, type-safe, fast I/O)
        train_path = os.path.join(self.output_path, "train.parquet")
        test_path = os.path.join(self.output_path, "test.parquet")

        self._sanitize_for_parquet(train_df).to_parquet(train_path, index=False)
        self._sanitize_for_parquet(test_df).to_parquet(test_path, index=False)

        print(f"Saved:")
        print(f"  Train -> {train_path}  ({len(train_df):,} rows)")
        print(f"  Test  -> {test_path}  ({len(test_df):,} rows)")

        return train_df, test_df


if __name__ == "__main__":
    # Initialize DataLoader with default paths
    loader = DataLoader()

    print("=" * 60)
    print("STEP 1: Testing Institution Lookup Mapping")
    print("=" * 60)
    try:
        inst_map = loader.load_institution_mapping()
        print(f"Sample mappings (first 5): {dict(list(inst_map.items())[:5])}\n")
    except Exception as e:
        print(f"Lookup mapping failed: {e}\n")

    print("=" * 60)
    print("STEP 2: Testing Main Data Loading & CIC Feature Extraction")
    print("=" * 60)
    try:
        df_extracted = loader.load_and_map_data()
        print(f"\nExtracted DataFrame shape: {df_extracted.shape}")

        # Display extracted CIC features
        cic_features = [col for col in df_extracted.columns if col.startswith("cic_") or col == "has_cic"]
        print(f"\nExtracted CIC Features ({len(cic_features)}):")
        for i, feat in enumerate(cic_features, 1):
            print(f"  {i:02d}. {feat}")

        # Display all columns
        print(f"\nAll Features / Columns in Dataset ({len(df_extracted.columns)}):")
        for i, col in enumerate(df_extracted.columns, 1):
            print(f"  {i:02d}. {col:<30} (dtype: {df_extracted[col].dtype})")

        # Save extracted dataset
        print("\n" + "=" * 60)
        print("STEP 3: Saving Extracted Dataset")
        print("=" * 60)
        loader.save_extracted(df_extracted)
    except Exception as e:
        print(f"Data loading failed: {e}")

