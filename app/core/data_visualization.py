"""
Data visualization generation logic.
"""

from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from app.config import settings
from app.models.base_models import VisualizationType, VisualizationOutput
from app.utils.io_utils import IOUtils
from app.utils.logger import LoggerMixin


class DataVisualizer(LoggerMixin):
    """Generate data visualizations."""

    def __init__(self):
        """Initialize data visualizer."""
        self.io_utils = IOUtils()
        self._setup_style()

    def _setup_style(self) -> None:
        """Setup visualization style."""
        plt.style.use(settings.plot_style)
        sns.set_palette(settings.color_palette)

    def create_visualization(
        self,
        file_path: str,
        visualization_type: VisualizationType,
        columns: Optional[List[str]] = None,
        title: Optional[str] = None,
        output_path: Optional[str] = None,
    ) -> VisualizationOutput:
        """
        Create a visualization.

        Args:
            file_path: Path to the data file
            visualization_type: Type of visualization
            columns: Columns to visualize
            title: Chart title
            output_path: Output file path

        Returns:
            VisualizationOutput with visualization details
        """
        try:
            self.logger.info(f"Creating {visualization_type.value} visualization")

            # Load data
            df = self.io_utils.load_data(file_path)

            # Generate output path if not provided
            if output_path is None:
                output_path = (
                    settings.visualizations_dir
                    / f"{visualization_type.value}_{Path(file_path).stem}.png"
                )
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Create visualization based on type
            if visualization_type == VisualizationType.HISTOGRAM:
                columns_used = self._create_histogram(df, columns, title, output_path)
            elif visualization_type == VisualizationType.SCATTER:
                columns_used = self._create_scatter(df, columns, title, output_path)
            elif visualization_type == VisualizationType.LINE:
                columns_used = self._create_line(df, columns, title, output_path)
            elif visualization_type == VisualizationType.BAR:
                columns_used = self._create_bar(df, columns, title, output_path)
            elif visualization_type == VisualizationType.BOX:
                columns_used = self._create_box(df, columns, title, output_path)
            elif visualization_type == VisualizationType.HEATMAP:
                columns_used = self._create_heatmap(df, columns, title, output_path)
            elif visualization_type == VisualizationType.CORRELATION:
                columns_used = self._create_correlation_matrix(df, columns, title, output_path)
            elif visualization_type == VisualizationType.DISTRIBUTION:
                columns_used = self._create_distribution(df, columns, title, output_path)
            elif visualization_type == VisualizationType.PAIRPLOT:
                columns_used = self._create_pairplot(df, columns, title, output_path)
            else:
                raise ValueError(f"Unsupported visualization type: {visualization_type}")

            result = VisualizationOutput(
                visualization_type=visualization_type.value,
                output_path=str(output_path),
                columns_used=columns_used,
                description=f"{visualization_type.value} visualization saved to {output_path}",
            )

            self.logger.info(f"Visualization saved to {output_path}")
            return result

        except Exception as e:
            self.logger.error(f"Error creating visualization: {e}", exc_info=True)
            raise

    def _create_histogram(
        self, df: pd.DataFrame, columns: Optional[List[str]], title: str, output_path: Path
    ) -> List[str]:
        """Create histogram visualization."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = columns or numeric_cols[:5]  # Limit to 5 columns

        fig, axes = plt.subplots(
            nrows=(len(columns) + 1) // 2,
            ncols=2,
            figsize=settings.default_figure_size,
        )
        axes = axes.flatten() if len(columns) > 1 else [axes]

        for idx, col in enumerate(columns):
            if col in df.columns:
                df[col].hist(ax=axes[idx], bins=30, edgecolor="black")
                axes[idx].set_title(f"Distribution of {col}")
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel("Frequency")

        # Hide extra subplots
        for idx in range(len(columns), len(axes)):
            axes[idx].set_visible(False)

        if title:
            fig.suptitle(title, fontsize=16, y=1.02)

        plt.tight_layout()
        plt.savefig(output_path, dpi=settings.visualization_dpi, bbox_inches="tight")
        plt.close()

        return columns

    def _create_scatter(
        self, df: pd.DataFrame, columns: Optional[List[str]], title: str, output_path: Path
    ) -> List[str]:
        """Create scatter plot."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if not columns:
            columns = numeric_cols[:2] if len(numeric_cols) >= 2 else numeric_cols

        if len(columns) < 2:
            raise ValueError("Scatter plot requires at least 2 numeric columns")

        plt.figure(figsize=settings.default_figure_size)
        plt.scatter(df[columns[0]], df[columns[1]], alpha=0.6)
        plt.xlabel(columns[0])
        plt.ylabel(columns[1])
        plt.title(title or f"{columns[0]} vs {columns[1]}")
        plt.grid(True, alpha=0.3)

        plt.savefig(output_path, dpi=settings.visualization_dpi, bbox_inches="tight")
        plt.close()

        return columns[:2]

    def _create_line(
        self, df: pd.DataFrame, columns: Optional[List[str]], title: str, output_path: Path
    ) -> List[str]:
        """Create line plot."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = columns or numeric_cols[:5]

        plt.figure(figsize=settings.default_figure_size)
        for col in columns:
            if col in df.columns:
                plt.plot(df.index, df[col], label=col, linewidth=2)

        plt.xlabel("Index")
        plt.ylabel("Value")
        plt.title(title or "Line Plot")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.savefig(output_path, dpi=settings.visualization_dpi, bbox_inches="tight")
        plt.close()

        return columns

    def _create_bar(
        self, df: pd.DataFrame, columns: Optional[List[str]], title: str, output_path: Path
    ) -> List[str]:
        """Create bar chart."""
        if not columns:
            # Use first categorical and first numeric column
            cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            if cat_cols and num_cols:
                columns = [cat_cols[0], num_cols[0]]
            else:
                raise ValueError("Bar chart requires categorical and numeric columns")

        if len(columns) < 2:
            raise ValueError("Bar chart requires at least 2 columns (categorical, numeric)")

        grouped = df.groupby(columns[0])[columns[1]].mean().sort_values(ascending=False).head(20)

        plt.figure(figsize=settings.default_figure_size)
        grouped.plot(kind="bar")
        plt.xlabel(columns[0])
        plt.ylabel(f"Mean {columns[1]}")
        plt.title(title or f"Bar Chart: {columns[0]} vs {columns[1]}")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        plt.savefig(output_path, dpi=settings.visualization_dpi, bbox_inches="tight")
        plt.close()

        return columns[:2]

    def _create_box(
        self, df: pd.DataFrame, columns: Optional[List[str]], title: str, output_path: Path
    ) -> List[str]:
        """Create box plot."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = columns or numeric_cols[:5]

        plt.figure(figsize=settings.default_figure_size)
        df[columns].boxplot()
        plt.ylabel("Value")
        plt.title(title or "Box Plot")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        plt.savefig(output_path, dpi=settings.visualization_dpi, bbox_inches="tight")
        plt.close()

        return columns

    def _create_heatmap(
        self, df: pd.DataFrame, columns: Optional[List[str]], title: str, output_path: Path
    ) -> List[str]:
        """Create heatmap."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = columns or numeric_cols

        if len(columns) == 0:
            raise ValueError("No numeric columns found for heatmap")

        # Calculate correlation or use data directly
        if len(df[columns]) > 100:
            data = df[columns].corr()
        else:
            data = df[columns]

        plt.figure(figsize=settings.default_figure_size)
        sns.heatmap(data, annot=True, cmap="coolwarm", center=0, fmt=".2f")
        plt.title(title or "Heatmap")
        plt.tight_layout()

        plt.savefig(output_path, dpi=settings.visualization_dpi, bbox_inches="tight")
        plt.close()

        return columns

    def _create_correlation_matrix(
        self, df: pd.DataFrame, columns: Optional[List[str]], title: str, output_path: Path
    ) -> List[str]:
        """Create correlation matrix visualization."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = columns or numeric_cols

        if len(columns) < 2:
            raise ValueError("Correlation matrix requires at least 2 numeric columns")

        corr_matrix = df[columns].corr()

        plt.figure(figsize=settings.default_figure_size)
        sns.heatmap(
            corr_matrix,
            annot=True,
            cmap="coolwarm",
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8},
            fmt=".2f",
        )
        plt.title(title or "Correlation Matrix")
        plt.tight_layout()

        plt.savefig(output_path, dpi=settings.visualization_dpi, bbox_inches="tight")
        plt.close()

        return columns

    def _create_distribution(
        self, df: pd.DataFrame, columns: Optional[List[str]], title: str, output_path: Path
    ) -> List[str]:
        """Create distribution plot with KDE."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = columns or numeric_cols[:3]

        fig, axes = plt.subplots(
            nrows=len(columns), ncols=1, figsize=(12, 4 * len(columns))
        )
        if len(columns) == 1:
            axes = [axes]

        for idx, col in enumerate(columns):
            if col in df.columns:
                sns.histplot(df[col], kde=True, ax=axes[idx])
                axes[idx].set_title(f"Distribution of {col}")
                axes[idx].set_xlabel(col)
                axes[idx].set_ylabel("Density")

        if title:
            fig.suptitle(title, fontsize=16, y=1.0)

        plt.tight_layout()
        plt.savefig(output_path, dpi=settings.visualization_dpi, bbox_inches="tight")
        plt.close()

        return columns

    def _create_pairplot(
        self, df: pd.DataFrame, columns: Optional[List[str]], title: str, output_path: Path
    ) -> List[str]:
        """Create pairplot."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        columns = columns or numeric_cols[:4]  # Limit for performance

        if len(columns) < 2:
            raise ValueError("Pairplot requires at least 2 numeric columns")

        pairplot = sns.pairplot(df[columns], diag_kind="kde", plot_kws={"alpha": 0.6})
        if title:
            pairplot.fig.suptitle(title, y=1.0)

        plt.savefig(output_path, dpi=settings.visualization_dpi, bbox_inches="tight")
        plt.close()

        return columns