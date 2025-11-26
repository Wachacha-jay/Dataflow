"""
Data validation and quality checking utilities.
"""

from typing import Dict, List, Any

import numpy as np
import pandas as pd
from app.config import settings
from app.utils.logger import LoggerMixin


class DataValidator(LoggerMixin):
    """Utility class for data validation and quality checks."""

    def __init__(self, df: pd.DataFrame):
        """
        Initialize validator with a DataFrame.

        Args:
            df: DataFrame to validate
        """
        self.df = df
        self.logger.info(f"Initialized DataValidator with DataFrame of shape {df.shape}")

    def check_missing_values(self) -> Dict[str, Any]:
        """
        Analyze missing values in the DataFrame.

        Returns:
            Dictionary with missing value statistics
        """
        try:
            missing_counts = self.df.isnull().sum()
            missing_percentages = (missing_counts / len(self.df)) * 100

            result = {
                "total_cells": self.df.size,
                "missing_cells": missing_counts.sum(),
                "missing_percentage": round((missing_counts.sum() / self.df.size) * 100, 2),
                "columns_with_missing": missing_counts[missing_counts > 0].to_dict(),
                "missing_percentages": {
                    col: round(pct, 2)
                    for col, pct in missing_percentages[missing_percentages > 0].items()
                },
            }

            self.logger.info(
                f"Missing value check complete: {result['missing_percentage']}% missing"
            )
            return result

        except Exception as e:
            self.logger.error(f"Error checking missing values: {e}", exc_info=True)
            raise

    def check_duplicates(self, subset: List[str] = None) -> Dict[str, Any]:
        """
        Check for duplicate rows.

        Args:
            subset: List of columns to check for duplicates. None checks all columns.

        Returns:
            Dictionary with duplicate statistics
        """
        try:
            duplicates = self.df.duplicated(subset=subset, keep=False)
            duplicate_count = duplicates.sum()

            result = {
                "total_rows": len(self.df),
                "duplicate_rows": int(duplicate_count),
                "duplicate_percentage": round((duplicate_count / len(self.df)) * 100, 2),
                "checked_columns": subset or "all",
            }

            self.logger.info(
                f"Duplicate check complete: {result['duplicate_rows']} duplicates found"
            )
            return result

        except Exception as e:
            self.logger.error(f"Error checking duplicates: {e}", exc_info=True)
            raise

    def detect_outliers(self, column: str, method: str = "iqr") -> Dict[str, Any]:
        """
        Detect outliers in a numeric column.

        Args:
            column: Column name to check
            method: Method to use ('iqr' or 'zscore')

        Returns:
            Dictionary with outlier information
        """
        try:
            if column not in self.df.columns:
                raise ValueError(f"Column '{column}' not found in DataFrame")

            if not pd.api.types.is_numeric_dtype(self.df[column]):
                raise ValueError(f"Column '{column}' is not numeric")

            data = self.df[column].dropna()

            if method == "iqr":
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = (data < lower_bound) | (data > upper_bound)

            elif method == "zscore":
                z_scores = np.abs((data - data.mean()) / data.std())
                outliers = z_scores > settings.outlier_std_threshold

            else:
                raise ValueError(f"Unknown method: {method}")

            outlier_count = outliers.sum()

            result = {
                "column": column,
                "method": method,
                "outlier_count": int(outlier_count),
                "outlier_percentage": round((outlier_count / len(data)) * 100, 2),
                "outlier_values": data[outliers].tolist()[:100],  # Limit to 100
            }

            if method == "iqr":
                result["bounds"] = {
                    "lower": float(lower_bound),
                    "upper": float(upper_bound),
                }

            self.logger.info(
                f"Outlier detection for '{column}': {outlier_count} outliers found"
            )
            return result

        except Exception as e:
            self.logger.error(f"Error detecting outliers: {e}", exc_info=True)
            raise

    def validate_data_types(self) -> Dict[str, Any]:
        """
        Validate and report on data types.

        Returns:
            Dictionary with data type information
        """
        try:
            type_counts = self.df.dtypes.value_counts().to_dict()
            type_counts = {str(k): int(v) for k, v in type_counts.items()}

            column_types = {col: str(dtype) for col, dtype in self.df.dtypes.items()}

            result = {
                "type_distribution": type_counts,
                "column_types": column_types,
                "numeric_columns": self.df.select_dtypes(include=[np.number]).columns.tolist(),
                "categorical_columns": self.df.select_dtypes(
                    include=["object", "category"]
                ).columns.tolist(),
                "datetime_columns": self.df.select_dtypes(include=["datetime"]).columns.tolist(),
            }

            self.logger.info("Data type validation complete")
            return result

        except Exception as e:
            self.logger.error(f"Error validating data types: {e}", exc_info=True)
            raise

    def get_data_quality_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive data quality report.

        Returns:
            Dictionary with complete quality assessment
        """
        try:
            self.logger.info("Generating comprehensive data quality report")

            report = {
                "shape": {"rows": len(self.df), "columns": len(self.df.columns)},
                "missing_values": self.check_missing_values(),
                "duplicates": self.check_duplicates(),
                "data_types": self.validate_data_types(),
                "memory_usage_mb": round(self.df.memory_usage(deep=True).sum() / (1024 * 1024), 2),
            }

            self.logger.info("Data quality report generated successfully")
            return report

        except Exception as e:
            self.logger.error(f"Error generating quality report: {e}", exc_info=True)
            raise