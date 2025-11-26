"""
Relationship and correlation discovery logic.
"""

from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from app.config import settings
from app.models.base_models import Relationship, RelationshipDiscoveryOutput
from app.utils.io_utils import IOUtils
from app.utils.logger import LoggerMixin


class RelationshipDiscoverer(LoggerMixin):
    """Discover relationships between columns in datasets."""

    def __init__(self):
        """Initialize relationship discoverer."""
        self.io_utils = IOUtils()

    def discover_relationships(
        self,
        file_path: str,
        correlation_threshold: Optional[float] = None,
        max_relationships: int = 50,
    ) -> RelationshipDiscoveryOutput:
        """
        Discover relationships between columns.

        Args:
            file_path: Path to the data file
            correlation_threshold: Minimum correlation threshold
            max_relationships: Maximum number of relationships to return

        Returns:
            RelationshipDiscoveryOutput with discovered relationships
        """
        try:
            self.logger.info(f"Starting relationship discovery for {file_path}")

            # Load data
            df = self.io_utils.load_data(file_path)

            # Set threshold
            threshold = correlation_threshold or settings.correlation_threshold

            # Find relationships
            relationships = []

            # Numeric correlations
            numeric_relationships = self._find_numeric_correlations(df, threshold)
            relationships.extend(numeric_relationships)

            # Categorical associations
            categorical_relationships = self._find_categorical_associations(df)
            relationships.extend(categorical_relationships)

            # Mixed type relationships
            mixed_relationships = self._find_mixed_relationships(df)
            relationships.extend(mixed_relationships)

            # Sort by strength and limit
            relationships.sort(key=lambda x: abs(x.strength), reverse=True)
            relationships = relationships[:max_relationships]

            # Generate correlation matrix for numeric columns
            correlation_matrix = self._generate_correlation_matrix(df)

            # Generate insights
            insights = self._generate_insights(relationships, df)

            result = RelationshipDiscoveryOutput(
                file_path=file_path,
                total_relationships=len(relationships),
                relationships=relationships,
                correlation_matrix=correlation_matrix,
                insights=insights,
            )

            self.logger.info(f"Discovered {len(relationships)} relationships")
            return result

        except Exception as e:
            self.logger.error(f"Error during relationship discovery: {e}", exc_info=True)
            raise

    def _find_numeric_correlations(
        self, df: pd.DataFrame, threshold: float
    ) -> List[Relationship]:
        """
        Find correlations between numeric columns.

        Args:
            df: DataFrame to analyze
            threshold: Minimum correlation threshold

        Returns:
            List of Relationship objects
        """
        relationships = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) < 2:
            return relationships

        try:
            # Calculate correlation matrix
            corr_matrix = df[numeric_cols].corr()

            # Find significant correlations
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i + 1 :]:
                    corr_value = corr_matrix.loc[col1, col2]

                    if abs(corr_value) >= threshold:
                        # Calculate p-value
                        _, p_value = stats.pearsonr(
                            df[col1].dropna(), df[col2].dropna()
                        )

                        relationship_type = self._categorize_correlation(corr_value)

                        relationships.append(
                            Relationship(
                                column1=col1,
                                column2=col2,
                                relationship_type=relationship_type,
                                strength=round(float(corr_value), 4),
                                description=f"{relationship_type.capitalize()} correlation "
                                f"({abs(corr_value):.2f}) between {col1} and {col2}",
                                statistical_metrics={
                                    "correlation": round(float(corr_value), 4),
                                    "p_value": round(float(p_value), 6),
                                    "significance": "significant" if p_value < 0.05 else "not significant",
                                },
                            )
                        )

        except Exception as e:
            self.logger.warning(f"Error finding numeric correlations: {e}")

        return relationships

    def _find_categorical_associations(self, df: pd.DataFrame) -> List[Relationship]:
        """
        Find associations between categorical columns using Chi-square test.

        Args:
            df: DataFrame to analyze

        Returns:
            List of Relationship objects
        """
        relationships = []
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        if len(categorical_cols) < 2:
            return relationships

        try:
            for i, col1 in enumerate(categorical_cols):
                for col2 in categorical_cols[i + 1 :]:
                    # Create contingency table
                    contingency_table = pd.crosstab(df[col1], df[col2])

                    # Perform chi-square test
                    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

                    # Calculate Cramér's V
                    n = contingency_table.sum().sum()
                    min_dim = min(contingency_table.shape[0], contingency_table.shape[1])
                    cramers_v = np.sqrt(chi2 / (n * (min_dim - 1)))

                    if cramers_v >= 0.3 and p_value < 0.05:  # Moderate association
                        relationships.append(
                            Relationship(
                                column1=col1,
                                column2=col2,
                                relationship_type="categorical_association",
                                strength=round(float(cramers_v), 4),
                                description=f"Association between {col1} and {col2} "
                                f"(Cramér's V: {cramers_v:.2f})",
                                statistical_metrics={
                                    "cramers_v": round(float(cramers_v), 4),
                                    "chi_square": round(float(chi2), 4),
                                    "p_value": round(float(p_value), 6),
                                    "degrees_of_freedom": int(dof),
                                },
                            )
                        )

        except Exception as e:
            self.logger.warning(f"Error finding categorical associations: {e}")

        return relationships

    def _find_mixed_relationships(self, df: pd.DataFrame) -> List[Relationship]:
        """
        Find relationships between numeric and categorical columns.

        Args:
            df: DataFrame to analyze

        Returns:
            List of Relationship objects
        """
        relationships = []
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

        if not numeric_cols or not categorical_cols:
            return relationships

        try:
            for num_col in numeric_cols:
                for cat_col in categorical_cols:
                    # Perform ANOVA
                    groups = [
                        group[num_col].dropna()
                        for name, group in df.groupby(cat_col)
                        if len(group[num_col].dropna()) > 0
                    ]

                    if len(groups) < 2:
                        continue

                    f_stat, p_value = stats.f_oneway(*groups)

                    # Calculate effect size (eta-squared)
                    ss_between = sum(
                        len(group) * (group.mean() - df[num_col].mean()) ** 2
                        for group in groups
                    )
                    ss_total = ((df[num_col] - df[num_col].mean()) ** 2).sum()
                    eta_squared = ss_between / ss_total if ss_total > 0 else 0

                    if eta_squared >= 0.06 and p_value < 0.05:  # Medium effect
                        relationships.append(
                            Relationship(
                                column1=num_col,
                                column2=cat_col,
                                relationship_type="numeric_categorical",
                                strength=round(float(eta_squared), 4),
                                description=f"{cat_col} influences {num_col} "
                                f"(η²: {eta_squared:.2f})",
                                statistical_metrics={
                                    "eta_squared": round(float(eta_squared), 4),
                                    "f_statistic": round(float(f_stat), 4),
                                    "p_value": round(float(p_value), 6),
                                },
                            )
                        )

        except Exception as e:
            self.logger.warning(f"Error finding mixed relationships: {e}")

        return relationships

    def _categorize_correlation(self, corr_value: float) -> str:
        """Categorize correlation strength."""
        abs_corr = abs(corr_value)
        direction = "positive" if corr_value > 0 else "negative"

        if abs_corr >= 0.9:
            strength = "very_strong"
        elif abs_corr >= 0.7:
            strength = "strong"
        elif abs_corr >= 0.5:
            strength = "moderate"
        else:
            strength = "weak"

        return f"{direction}_{strength}"

    def _generate_correlation_matrix(self, df: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Generate correlation matrix for numeric columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if len(numeric_cols) < 2:
            return None

        try:
            corr_matrix = df[numeric_cols].corr()
            return {
                "columns": numeric_cols,
                "matrix": corr_matrix.round(4).to_dict(),
            }
        except Exception as e:
            self.logger.warning(f"Error generating correlation matrix: {e}")
            return None

    def _generate_insights(
        self, relationships: List[Relationship], df: pd.DataFrame
    ) -> List[str]:
        """Generate insights from discovered relationships."""
        insights = []

        if not relationships:
            insights.append("No significant relationships found in the data.")
            return insights

        # Strong correlations
        strong_corr = [r for r in relationships if abs(r.strength) >= 0.8]
        if strong_corr:
            insights.append(
                f"Found {len(strong_corr)} strong relationships that may indicate "
                "redundant features or important dependencies."
            )

        # Categorical associations
        cat_assoc = [r for r in relationships if r.relationship_type == "categorical_association"]
        if cat_assoc:
            insights.append(
                f"Identified {len(cat_assoc)} significant associations between categorical variables."
            )

        # Mixed relationships
        mixed_rel = [r for r in relationships if r.relationship_type == "numeric_categorical"]
        if mixed_rel:
            insights.append(
                f"Found {len(mixed_rel)} relationships where categorical variables "
                "significantly influence numeric variables."
            )

        return insights