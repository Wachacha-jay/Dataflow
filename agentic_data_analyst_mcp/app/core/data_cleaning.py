"""
Data cleaning functionality for preprocessing datasets.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional

from app.utils.logger import LoggerMixin


class DataCleaner(LoggerMixin):
    """
    Data cleaning engine to preprocess datasets before analysis.
    Handles missing values, duplicates, outliers, and type coercion.
    """

    def __init__(self, df: pd.DataFrame, dataset_name: str = "dataset"):
        """Initialize data cleaner with DataFrame."""
        try:
            if df is None or df.empty:
                raise ValueError("Empty DataFrame provided for cleaning.")

            self.df = df.copy()
            self.dataset_name = dataset_name
            self.cleaning_report: Dict[str, Any] = {
                "missing_values": {},
                "duplicates": {},
                "outliers": {},
                "string_normalization": {},
                "type_coercion": {}
            }

            self.logger.info(f"DataCleaner initialized for {dataset_name}")

        except Exception as e:
            self.logger.error(f"Error initializing DataCleaner: {e}", exc_info=True)
            raise

    def handle_missing(self, strategy: str = "mean", fill_value: Optional[Any] = None) -> pd.DataFrame:
        """
        Handle missing values using various strategies.

        Args:
            strategy: Method to handle missing values
                     ('drop', 'mean', 'median', 'mode', 'constant', 'ffill', 'bfill')
            fill_value: Value to use when strategy is 'constant'

        Returns:
            DataFrame with handled missing values
        """
        try:
            missing_summary = self.df.isnull().sum().to_dict()
            self.logger.info(f"Handling missing values with strategy: {strategy}")

            if strategy == "drop":
                self.df = self.df.dropna()
            elif strategy == "mean":
                self.df = self.df.fillna(self.df.mean(numeric_only=True))
            elif strategy == "median":
                self.df = self.df.fillna(self.df.median(numeric_only=True))
            elif strategy == "mode":
                for col in self.df.columns:
                    if self.df[col].isnull().any():
                        mode_val = self.df[col].mode(dropna=True)
                        if not mode_val.empty:
                            self.df[col] = self.df[col].fillna(mode_val[0])
            elif strategy == "constant":
                if fill_value is None:
                    raise ValueError("fill_value must be provided when strategy='constant'")
                self.df = self.df.fillna(fill_value)
            elif strategy in ["ffill", "bfill"]:
                self.df = self.df.fillna(method=strategy)
            else:
                raise ValueError(f"Unsupported missing value strategy: {strategy}")

            self.cleaning_report["missing_values"] = {
                "strategy": strategy,
                "initial_missing": missing_summary,
                "remaining_missing": self.df.isnull().sum().to_dict()
            }

            self.logger.info("Missing values handled successfully")
            return self.df

        except Exception as e:
            self.logger.error(f"Error handling missing values: {e}", exc_info=True)
            raise

    def remove_duplicates(self) -> pd.DataFrame:
        """
        Remove duplicate rows from the DataFrame.

        Returns:
            DataFrame with duplicates removed
        """
        try:
            initial_count = len(self.df)
            self.df = self.df.drop_duplicates()
            final_count = len(self.df)

            removed = initial_count - final_count
            self.cleaning_report["duplicates"] = {
                "initial_rows": initial_count,
                "final_rows": final_count,
                "removed": removed
            }

            self.logger.info(f"Removed {removed} duplicate rows")
            return self.df

        except Exception as e:
            self.logger.error(f"Error removing duplicates: {e}", exc_info=True)
            raise

    def handle_outliers(self, method: str = "zscore", threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect outliers in numeric columns.

        Args:
            method: Method to detect outliers ('zscore' or 'iqr')
            threshold: Z-score threshold or IQR multiplier

        Returns:
            DataFrame with outlier information
        """
        try:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            outlier_report = {}

            for col in numeric_cols:
                col_data = self.df[col].dropna()
                if method == "zscore":
                    z_scores = (col_data - col_data.mean()) / col_data.std()
                    outliers = col_data[abs(z_scores) > threshold]
                elif method == "iqr":
                    Q1, Q3 = col_data.quantile([0.25, 0.75])
                    IQR = Q3 - Q1
                    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
                    outliers = col_data[(col_data < lower) | (col_data > upper)]
                else:
                    raise ValueError(f"Unsupported outlier method: {method}")

                outlier_report[col] = {
                    "num_outliers": int(len(outliers)),
                    "outlier_indices": outliers.index.tolist()
                }

            self.cleaning_report["outliers"] = outlier_report
            self.logger.info(f"Outlier detection completed using {method}")
            return self.df

        except Exception as e:
            self.logger.error(f"Error handling outliers: {e}", exc_info=True)
            raise

    def normalize_strings(self) -> pd.DataFrame:
        """
        Normalize string columns by converting to lowercase, removing special characters,
        and stripping whitespace.

        Returns:
            DataFrame with normalized strings
        """
        try:
            str_cols = self.df.select_dtypes(include=["object"]).columns
            for col in str_cols:
                self.df[col] = (
                    self.df[col]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .str.replace(r"[^a-zA-Z0-9\s]", "", regex=True)
                )

            self.cleaning_report["string_normalization"] = {
                "normalized_columns": str_cols.tolist()
            }

            self.logger.info(f"String normalization applied to {len(str_cols)} columns")
            return self.df

        except Exception as e:
            self.logger.error(f"Error normalizing strings: {e}", exc_info=True)
            raise

    def coerce_types(self) -> pd.DataFrame:
        """
        Attempt to coerce columns to appropriate types (datetime, numeric).

        Returns:
            DataFrame with coerced types
        """
        try:
            initial_types = self.df.dtypes.astype(str).to_dict()
            
            for col in self.df.columns:
                # Try datetime
                if self.df[col].dtype == object:
                    try:
                        self.df[col] = pd.to_datetime(self.df[col], errors="ignore")
                    except Exception:
                        pass
                # Try numeric
                if self.df[col].dtype == object:
                    self.df[col] = pd.to_numeric(self.df[col], errors="ignore")

            final_types = self.df.dtypes.astype(str).to_dict()
            self.cleaning_report["type_coercion"] = {
                "initial_types": initial_types,
                "final_types": final_types
            }

            self.logger.info("Type coercion completed")
            return self.df

        except Exception as e:
            self.logger.error(f"Error coercing types: {e}", exc_info=True)
            raise

    def get_cleaning_report(self) -> Dict[str, Any]:
        """Get the cleaning operations report."""
        return self.cleaning_report

    # ------------------
    # Missing Values
    # ------------------
    def handle_missing(self, strategy: str = "mean", fill_value: Optional[Any] = None) -> pd.DataFrame:
        """
        Handle missing values using various strategies:
        - drop
        - mean, median, mode
        - constant (requires fill_value)
        - ffill / bfill
        """
        try:
            missing_summary = self.df.isnull().sum().to_dict()

            if strategy == "drop":
                self.df = self.df.dropna()
            elif strategy == "mean":
                self.df = self.df.fillna(self.df.mean(numeric_only=True))
            elif strategy == "median":
                self.df = self.df.fillna(self.df.median(numeric_only=True))
            elif strategy == "mode":
                for col in self.df.columns:
                    if self.df[col].isnull().any():
                        mode_val = self.df[col].mode(dropna=True)
                        if not mode_val.empty:
                            self.df[col] = self.df[col].fillna(mode_val[0])
            elif strategy == "constant":
                if fill_value is None:
                    raise ValueError("fill_value must be provided when strategy='constant'")
                self.df = self.df.fillna(fill_value)
            elif strategy in ["ffill", "bfill"]:
                self.df = self.df.fillna(method=strategy)
            else:
                raise ValueError(f"Unsupported missing value strategy: {strategy}")

            self.cleaning_report["missing_values"] = {
                "strategy": strategy,
                "initial_missing": missing_summary,
                "remaining_missing": self.df.isnull().sum().to_dict()
            }

            logger.info(f"✅ Missing values handled with strategy: {strategy}")
            return self.df

        except Exception as e:
            raise DataCleaningError(f"Missing value handling failed: {e}", error_detail=sys)

    # ------------------
    # Duplicates
    # ------------------
    def remove_duplicates(self) -> pd.DataFrame:
        try:
            initial_count = len(self.df)
            self.df = self.df.drop_duplicates()
            final_count = len(self.df)

            self.cleaning_report["duplicates"] = {
                "initial_rows": initial_count,
                "final_rows": final_count,
                "removed": initial_count - final_count
            }

            logger.info(f"✅ Removed {initial_count - final_count} duplicate rows")
            return self.df

        except Exception as e:
            raise DataCleaningError(f"Duplicate removal failed: {e}", error_detail=sys)

    # ------------------
    # Outliers
    # ------------------
    def handle_outliers(self, method: str = "zscore", threshold: float = 3.0) -> pd.DataFrame:
        """
        Detect and optionally remove outliers.
        - zscore: marks values with |z| > threshold
        - iqr: marks values outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
        """
        try:
            numeric_cols = self.df.select_dtypes(include=[np.number]).columns
            outlier_report = {}

            for col in numeric_cols:
                col_data = self.df[col].dropna()
                if method == "zscore":
                    z_scores = (col_data - col_data.mean()) / col_data.std()
                    outliers = col_data[abs(z_scores) > threshold]
                elif method == "iqr":
                    Q1, Q3 = col_data.quantile([0.25, 0.75])
                    IQR = Q3 - Q1
                    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
                    outliers = col_data[(col_data < lower) | (col_data > upper)]
                else:
                    raise ValueError(f"Unsupported outlier method: {method}")

                outlier_report[col] = {
                    "num_outliers": int(len(outliers)),
                    "outlier_indices": outliers.index.tolist()
                }

            self.cleaning_report["outliers"] = outlier_report
            logger.info(f"✅ Outlier detection complete using {method}")
            return self.df

        except Exception as e:
            raise DataCleaningError(f"Outlier handling failed: {e}", error_detail=sys)

    # ------------------
    # String Normalization
    # ------------------
    def normalize_strings(self) -> pd.DataFrame:
        try:
            str_cols = self.df.select_dtypes(include=["object"]).columns
            for col in str_cols:
                self.df[col] = (
                    self.df[col]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .str.replace(r"[^a-zA-Z0-9\s]", "", regex=True)
                )

            self.cleaning_report["string_normalization"] = {
                "normalized_columns": str_cols.tolist()
            }

            logger.info("✅ String normalization applied")
            return self.df

        except Exception as e:
            raise DataCleaningError(f"String normalization failed: {e}", error_detail=sys)

    # ------------------
    # Type Coercion
    # ------------------
    def coerce_types(self) -> pd.DataFrame:
        try:
            for col in self.df.columns:
                # Try datetime
                if self.df[col].dtype == object:
                    try:
                        self.df[col] = pd.to_datetime(self.df[col], errors="ignore")
                    except Exception:
                        pass
                # Try numeric
                if self.df[col].dtype == object:
                    self.df[col] = pd.to_numeric(self.df[col], errors="ignore")

            self.cleaning_report["type_coercion"] = {
                "coerced_columns": self.df.dtypes.astype(str).to_dict()
            }

            logger.info("✅ Type coercion applied where possible")
            return self.df

        except Exception as e:
            raise DataCleaningError(f"Type coercion failed: {e}", error_detail=sys)

    # ------------------
    # Report
    # ------------------
    def get_cleaning_report(self) -> Dict[str, Any]:
        return self.cleaning_report
