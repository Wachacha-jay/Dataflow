"""
Schema discovery and inference logic.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from app.config import settings
from app.models.base_models import ColumnSchema, SchemaDiscoveryOutput
from app.utils.data_checks import DataValidator
from app.utils.io_utils import IOUtils
from app.utils.logger import LoggerMixin


class SchemaDiscoverer(LoggerMixin):
    """Discover and infer schema from data files."""

    def __init__(self):
        """Initialize schema discoverer."""
        self.io_utils = IOUtils()

    def discover_schema(
        self,
        file_path: str,
        sample_size: Optional[int] = None,
        detect_relationships: bool = True,
    ) -> SchemaDiscoveryOutput:
        """
        Discover schema from a data file.

        Args:
            file_path: Path to the data file
            sample_size: Number of rows to sample
            detect_relationships: Whether to detect column relationships

        Returns:
            SchemaDiscoveryOutput with discovered schema
        """
        try:
            self.logger.info(f"Starting schema discovery for {file_path}")

            # Load data
            df = self.io_utils.load_data(file_path)

            # Sample if requested
            if sample_size and sample_size < len(df):
                df = df.sample(n=sample_size, random_state=42)
                self.logger.info(f"Sampled {sample_size} rows for analysis")

            # Analyze each column
            columns = []
            for col in df.columns:
                column_schema = self._analyze_column(df, col)
                columns.append(column_schema)

            # Calculate data quality score
            validator = DataValidator(df)
            quality_report = validator.get_data_quality_report()
            quality_score = self._calculate_quality_score(quality_report)

            # Generate recommendations
            recommendations = self._generate_recommendations(df, quality_report)

            result = SchemaDiscoveryOutput(
                file_path=file_path,
                total_rows=len(df),
                total_columns=len(df.columns),
                columns=columns,
                data_quality_score=quality_score,
                recommendations=recommendations,
            )

            self.logger.info("Schema discovery completed successfully")
            return result

        except Exception as e:
            self.logger.error(f"Error during schema discovery: {e}", exc_info=True)
            raise

    def _analyze_column(self, df: pd.DataFrame, column: str) -> ColumnSchema:
        """
        Analyze a single column and extract schema information.

        Args:
            df: DataFrame containing the column
            column: Column name to analyze

        Returns:
            ColumnSchema with column information
        """
        try:
            col_data = df[column]
            null_count = col_data.isnull().sum()
            total_rows = len(col_data)

            # Get sample values
            non_null_values = col_data.dropna()
            sample_values = []
            if len(non_null_values) > 0:
                sample_size = min(5, len(non_null_values))
                sample_values = non_null_values.sample(n=sample_size, random_state=42).tolist()

            # Infer semantic type
            inferred_type = self._infer_semantic_type(col_data)

            # Extract constraints
            constraints = self._extract_constraints(col_data)

            return ColumnSchema(
                name=column,
                data_type=str(col_data.dtype),
                null_count=int(null_count),
                null_percentage=round((null_count / total_rows) * 100, 2),
                unique_count=int(col_data.nunique()),
                unique_percentage=round((col_data.nunique() / total_rows) * 100, 2),
                sample_values=sample_values,
                inferred_type=inferred_type,
                constraints=constraints,
            )

        except Exception as e:
            self.logger.error(f"Error analyzing column {column}: {e}", exc_info=True)
            raise

    def _infer_semantic_type(self, series: pd.Series) -> str:
        """
        Infer semantic type of a column.

        Args:
            series: Pandas Series to analyze

        Returns:
            Inferred semantic type
        """
        # Numeric types
        if pd.api.types.is_numeric_dtype(series):
            if pd.api.types.is_integer_dtype(series):
                # Check if it's an ID (high uniqueness)
                if series.nunique() / len(series) > 0.95:
                    return "identifier"
                return "integer"
            return "numeric"

        # DateTime types
        if pd.api.types.is_datetime64_any_dtype(series):
            return "datetime"

        # Boolean types
        if pd.api.types.is_bool_dtype(series):
            return "boolean"

        # String types
        if pd.api.types.is_object_dtype(series):
            non_null = series.dropna()
            if len(non_null) == 0:
                return "text"

            # Check for categorical (low cardinality)
            cardinality_ratio = series.nunique() / len(series)
            if cardinality_ratio < 0.05:
                return "categorical"

            # Check for email pattern
            if non_null.astype(str).str.contains("@", regex=False).mean() > 0.8:
                return "email"

            # Check for URL pattern
            if non_null.astype(str).str.contains("http", regex=False).mean() > 0.8:
                return "url"

            return "text"

        return "unknown"

    def _extract_constraints(self, series: pd.Series) -> Dict[str, Any]:
        """
        Extract constraints and metadata from a column.

        Args:
            series: Pandas Series to analyze

        Returns:
            Dictionary of constraints
        """
        constraints = {}

        if pd.api.types.is_numeric_dtype(series):
            non_null = series.dropna()
            if len(non_null) > 0:
                constraints["min"] = float(non_null.min())
                constraints["max"] = float(non_null.max())
                constraints["mean"] = float(non_null.mean())
                constraints["median"] = float(non_null.median())
                constraints["std"] = float(non_null.std())

        elif pd.api.types.is_object_dtype(series):
            non_null = series.dropna()
            if len(non_null) > 0:
                lengths = non_null.astype(str).str.len()
                constraints["min_length"] = int(lengths.min())
                constraints["max_length"] = int(lengths.max())
                constraints["avg_length"] = float(lengths.mean())

        return constraints

    def _calculate_quality_score(self, quality_report: Dict[str, Any]) -> float:
        """
        Calculate overall data quality score.

        Args:
            quality_report: Quality report from DataValidator

        Returns:
            Quality score between 0 and 100
        """
        score = 100.0

        # Penalize for missing values
        missing_pct = quality_report["missing_values"]["missing_percentage"]
        score -= min(missing_pct, 30)

        # Penalize for duplicates
        duplicate_pct = quality_report["duplicates"]["duplicate_percentage"]
        score -= min(duplicate_pct * 0.5, 20)

        return round(max(score, 0), 2)

    def _generate_recommendations(
        self, df: pd.DataFrame, quality_report: Dict[str, Any]
    ) -> List[str]:
        """
        Generate recommendations based on schema analysis.

        Args:
            df: DataFrame being analyzed
            quality_report: Quality report

        Returns:
            List of recommendation strings
        """
        recommendations = []

        # Missing values
        missing_pct = quality_report["missing_values"]["missing_percentage"]
        if missing_pct > settings.missing_value_threshold * 100:
            recommendations.append(
                f"High percentage of missing values ({missing_pct}%). "
                "Consider imputation or removal strategies."
            )

        # Duplicates
        duplicate_pct = quality_report["duplicates"]["duplicate_percentage"]
        if duplicate_pct > 5:
            recommendations.append(
                f"Found {duplicate_pct}% duplicate rows. Consider deduplication."
            )

        # Column types
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 10:
            recommendations.append(
                "Large number of numeric columns. Consider dimensionality reduction or feature selection."
            )

        return recommendations