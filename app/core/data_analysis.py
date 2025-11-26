"""
Comprehensive data analysis logic.
"""

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats
from app.models.base_models import DataAnalysisOutput, AnalysisType
from app.utils.data_checks import DataValidator
from app.utils.io_utils import IOUtils
from app.utils.logger import LoggerMixin


class DataAnalyzer(LoggerMixin):
    """Perform comprehensive data analysis."""

    def __init__(self):
        """Initialize data analyzer."""
        self.io_utils = IOUtils()

    def analyze_data(
        self,
        file_path: str,
        analysis_type: AnalysisType = AnalysisType.COMPREHENSIVE,
        columns: Optional[List[str]] = None,
        groupby_column: Optional[str] = None,
    ) -> DataAnalysisOutput:
        """
        Perform data analysis.

        Args:
            file_path: Path to the data file
            analysis_type: Type of analysis to perform
            columns: Specific columns to analyze
            groupby_column: Column to group by for aggregated analysis

        Returns:
            DataAnalysisOutput with analysis results
        """
        try:
            self.logger.info(f"Starting {analysis_type.value} analysis for {file_path}")

            # Load data
            df = self.io_utils.load_data(file_path)

            # Filter columns if specified
            if columns:
                df = df[columns]

            # Perform analysis based on type
            if analysis_type == AnalysisType.DESCRIPTIVE:
                summary_stats = self._descriptive_analysis(df)
            elif analysis_type == AnalysisType.STATISTICAL:
                summary_stats = self._statistical_analysis(df)
            else:  # COMPREHENSIVE
                summary_stats = self._comprehensive_analysis(df, groupby_column)

            # Generate insights
            insights = self._generate_insights(df, summary_stats)

            # Generate warnings
            warnings = self._generate_warnings(df)

            # Generate recommendations
            recommendations = self._generate_recommendations(df, summary_stats)

            result = DataAnalysisOutput(
                file_path=file_path,
                analysis_type=analysis_type.value,
                summary_statistics=summary_stats,
                insights=insights,
                warnings=warnings,
                recommendations=recommendations,
            )

            self.logger.info("Data analysis completed successfully")
            return result

        except Exception as e:
            self.logger.error(f"Error during data analysis: {e}", exc_info=True)
            raise

    def _descriptive_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform basic descriptive analysis."""
        try:
            analysis = {
                "shape": {"rows": len(df), "columns": len(df.columns)},
                "column_names": df.columns.tolist(),
                "data_types": df.dtypes.astype(str).to_dict(),
            }

            # Numeric columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                analysis["numeric_summary"] = df[numeric_cols].describe().to_dict()

            # Categorical columns
            categorical_cols = df.select_dtypes(include=["object", "category"]).columns
            if len(categorical_cols) > 0:
                analysis["categorical_summary"] = {}
                for col in categorical_cols:
                    value_counts = df[col].value_counts().head(10).to_dict()
                    analysis["categorical_summary"][col] = {
                        "unique_values": int(df[col].nunique()),
                        "top_values": value_counts,
                    }

            return analysis

        except Exception as e:
            self.logger.error(f"Error in descriptive analysis: {e}", exc_info=True)
            raise

    def _statistical_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform statistical analysis."""
        try:
            analysis = self._descriptive_analysis(df)

            numeric_cols = df.select_dtypes(include=[np.number]).columns

            if len(numeric_cols) > 0:
                analysis["statistical_tests"] = {}

                for col in numeric_cols:
                    data = df[col].dropna()
                    if len(data) > 3:
                        # Normality test
                        _, p_value_normality = stats.normaltest(data)

                        # Skewness and kurtosis
                        skewness = stats.skew(data)
                        kurtosis = stats.kurtosis(data)

                        analysis["statistical_tests"][col] = {
                            "normality_p_value": round(float(p_value_normality), 6),
                            "is_normal": bool(p_value_normality > 0.05),
                            "skewness": round(float(skewness), 4),
                            "kurtosis": round(float(kurtosis), 4),
                            "range": {
                                "min": float(data.min()),
                                "max": float(data.max()),
                                "range": float(data.max() - data.min()),
                            },
                            "percentiles": {
                                "25th": float(data.quantile(0.25)),
                                "50th": float(data.quantile(0.50)),
                                "75th": float(data.quantile(0.75)),
                                "95th": float(data.quantile(0.95)),
                            },
                        }

            return analysis

        except Exception as e:
            self.logger.error(f"Error in statistical analysis: {e}", exc_info=True)
            raise

    def _comprehensive_analysis(
        self, df: pd.DataFrame, groupby_column: Optional[str] = None
    ) -> Dict[str, Any]:
        """Perform comprehensive analysis."""
        try:
            analysis = self._statistical_analysis(df)

            # Data quality
            validator = DataValidator(df)
            analysis["data_quality"] = validator.get_data_quality_report()

            # Correlation analysis
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                analysis["correlations"] = {
                    "matrix": corr_matrix.round(4).to_dict(),
                    "strong_correlations": self._find_strong_correlations(corr_matrix),
                }

            # Group-by analysis
            if groupby_column and groupby_column in df.columns:
                analysis["grouped_analysis"] = self._grouped_analysis(df, groupby_column)

            # Distribution analysis
            analysis["distributions"] = self._analyze_distributions(df)

            return analysis

        except Exception as e:
            self.logger.error(f"Error in comprehensive analysis: {e}", exc_info=True)
            raise

    def _find_strong_correlations(
        self, corr_matrix: pd.DataFrame, threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """Find strong correlations in correlation matrix."""
        strong_corrs = []
        cols = corr_matrix.columns

        for i, col1 in enumerate(cols):
            for col2 in cols[i + 1 :]:
                corr_value = corr_matrix.loc[col1, col2]
                if abs(corr_value) >= threshold:
                    strong_corrs.append(
                        {
                            "column1": col1,
                            "column2": col2,
                            "correlation": round(float(corr_value), 4),
                        }
                    )

        return strong_corrs

    def _grouped_analysis(self, df: pd.DataFrame, groupby_column: str) -> Dict[str, Any]:
        """Perform grouped analysis."""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns

            if len(numeric_cols) == 0:
                return {"message": "No numeric columns for grouped analysis"}

            grouped = df.groupby(groupby_column)
            
            result = {
                "group_column": groupby_column,
                "n_groups": int(grouped.ngroups),
                "group_sizes": grouped.size().to_dict(),
                "aggregations": {},
            }

            for col in numeric_cols:
                agg_stats = grouped[col].agg(["mean", "median", "std", "min", "max"])
                result["aggregations"][col] = agg_stats.round(4).to_dict()

            return result

        except Exception as e:
            self.logger.warning(f"Error in grouped analysis: {e}")
            return {"error": str(e)}

    def _analyze_distributions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze distributions of numeric columns."""
        distributions = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns

        for col in numeric_cols:
            data = df[col].dropna()
            if len(data) > 0:
                distributions[col] = {
                    "mean": round(float(data.mean()), 4),
                    "median": round(float(data.median()), 4),
                    "mode": float(data.mode()[0]) if len(data.mode()) > 0 else None,
                    "std": round(float(data.std()), 4),
                    "variance": round(float(data.var()), 4),
                    "cv": round(float(data.std() / data.mean()), 4) if data.mean() != 0 else None,
                }

        return distributions

    def _generate_insights(self, df: pd.DataFrame, summary_stats: Dict[str, Any]) -> List[str]:
        """Generate insights from analysis."""
        insights = []

        # Dataset size
        insights.append(
            f"Dataset contains {len(df):,} rows and {len(df.columns)} columns"
        )

        # Data types
        numeric_count = len(df.select_dtypes(include=[np.number]).columns)
        categorical_count = len(df.select_dtypes(include=["object", "category"]).columns)
        
        if numeric_count > 0:
            insights.append(f"Found {numeric_count} numeric columns for quantitative analysis")
        if categorical_count > 0:
            insights.append(f"Found {categorical_count} categorical columns for qualitative analysis")

        # Correlations
        if "correlations" in summary_stats and summary_stats["correlations"]["strong_correlations"]:
            strong_corr_count = len(summary_stats["correlations"]["strong_correlations"])
            insights.append(
                f"Identified {strong_corr_count} strong correlations between variables"
            )

        return insights

    def _generate_warnings(self, df: pd.DataFrame) -> List[str]:
        """Generate warnings about data issues."""
        warnings = []
        validator = DataValidator(df)

        # Missing values
        missing_info = validator.check_missing_values()
        if missing_info["missing_percentage"] > 10:
            warnings.append(
                f"High percentage of missing values: {missing_info['missing_percentage']}%"
            )

        # Duplicates
        duplicate_info = validator.check_duplicates()
        if duplicate_info["duplicate_percentage"] > 5:
            warnings.append(
                f"Found {duplicate_info['duplicate_rows']} duplicate rows "
                f"({duplicate_info['duplicate_percentage']}%)"
            )

        # Low variance columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].std() < 0.01:
                warnings.append(f"Column '{col}' has very low variance")

        return warnings

    def _generate_recommendations(
        self, df: pd.DataFrame, summary_stats: Dict[str, Any]
    ) -> List[str]:
        """Generate recommendations based on analysis."""
        recommendations = []

        # Missing values
        if "data_quality" in summary_stats:
            missing_pct = summary_stats["data_quality"]["missing_values"]["missing_percentage"]
            if missing_pct > 5:
                recommendations.append(
                    "Consider handling missing values through imputation or removal"
                )

        # High cardinality categoricals
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns
        for col in categorical_cols:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio > 0.5:
                recommendations.append(
                    f"Column '{col}' has high cardinality. Consider feature engineering or encoding"
                )

        # Skewed distributions
        if "statistical_tests" in summary_stats:
            for col, stats_info in summary_stats["statistical_tests"].items():
                if abs(stats_info["skewness"]) > 2:
                    recommendations.append(
                        f"Column '{col}' is highly skewed. Consider transformation"
                    )

        return recommendations