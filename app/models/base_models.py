"""
Pydantic models for MCP tool inputs and outputs.
"""

from typing import Any, Dict, List, Optional
from enum import Enum

from pydantic import BaseModel, Field


class ReportFormat(str, Enum):
    """Supported report output formats."""
    HTML = "html"
    PDF = "pdf"
    MARKDOWN = "markdown"
    JSON = "json"


class VisualizationType(str, Enum):
    """Supported visualization types."""
    HISTOGRAM = "histogram"
    SCATTER = "scatter"
    LINE = "line"
    BAR = "bar"
    BOX = "box"
    HEATMAP = "heatmap"
    CORRELATION = "correlation"
    DISTRIBUTION = "distribution"
    PAIRPLOT = "pairplot"


class AnalysisType(str, Enum):
    """Types of data analysis."""
    DESCRIPTIVE = "descriptive"
    STATISTICAL = "statistical"
    COMPREHENSIVE = "comprehensive"


# Schema Discovery Models
class SchemaDiscoveryInput(BaseModel):
    """Input model for schema discovery tool."""
    file_path: str = Field(..., description="Path to the data file")
    sample_size: Optional[int] = Field(None, description="Number of rows to sample for analysis")
    detect_relationships: bool = Field(True, description="Whether to detect column relationships")


class ColumnSchema(BaseModel):
    """Schema information for a single column."""
    name: str
    data_type: str
    null_count: int
    null_percentage: float
    unique_count: int
    unique_percentage: float
    sample_values: List[Any]
    inferred_type: Optional[str] = None
    constraints: Dict[str, Any] = Field(default_factory=dict)


class SchemaDiscoveryOutput(BaseModel):
    """Output model for schema discovery tool."""
    file_path: str
    total_rows: int
    total_columns: int
    columns: List[ColumnSchema]
    data_quality_score: float
    recommendations: List[str]


# Relationship Discovery Models
class RelationshipDiscoveryInput(BaseModel):
    """Input model for relationship discovery tool."""
    file_path: str = Field(..., description="Path to the data file")
    correlation_threshold: Optional[float] = Field(None, description="Minimum correlation threshold")
    max_relationships: Optional[int] = Field(50, description="Maximum relationships to return")


class Relationship(BaseModel):
    """A discovered relationship between columns."""
    column1: str
    column2: str
    relationship_type: str
    strength: float
    description: str
    statistical_metrics: Dict[str, Any] = Field(default_factory=dict)


class RelationshipDiscoveryOutput(BaseModel):
    """Output model for relationship discovery tool."""
    file_path: str
    total_relationships: int
    relationships: List[Relationship]
    correlation_matrix: Optional[Dict[str, Any]] = None
    insights: List[str]


# Data Analysis Models
class DataAnalysisInput(BaseModel):
    """Input model for data analysis tool."""
    file_path: str = Field(..., description="Path to the data file")
    analysis_type: AnalysisType = Field(AnalysisType.COMPREHENSIVE, description="Type of analysis")
    columns: Optional[List[str]] = Field(None, description="Specific columns to analyze")
    groupby_column: Optional[str] = Field(None, description="Column to group by")


class DataAnalysisOutput(BaseModel):
    """Output model for data analysis tool."""
    file_path: str
    analysis_type: str
    summary_statistics: Dict[str, Any]
    insights: List[str]
    warnings: List[str]
    recommendations: List[str]


# Visualization Models
class VisualizationInput(BaseModel):
    """Input model for visualization tool."""
    file_path: str = Field(..., description="Path to the data file")
    visualization_type: VisualizationType = Field(..., description="Type of visualization")
    columns: Optional[List[str]] = Field(None, description="Columns to visualize")
    title: Optional[str] = Field(None, description="Chart title")
    output_path: Optional[str] = Field(None, description="Output file path")


class VisualizationOutput(BaseModel):
    """Output model for visualization tool."""
    visualization_type: str
    output_path: str
    columns_used: List[str]
    description: str


# Report Generation Models
class ReportGenerationInput(BaseModel):
    """Input model for report generation tool."""
    file_path: str = Field(..., description="Path to the data file")
    report_type: AnalysisType = Field(
        AnalysisType.COMPREHENSIVE, description="Type of report to generate"
    )
    output_format: ReportFormat = Field(ReportFormat.HTML, description="Output format")
    include_visualizations: bool = Field(True, description="Include visualizations in report")
    include_raw_data: bool = Field(False, description="Include raw data sample in report")
    output_path: Optional[str] = Field(None, description="Custom output path")


class ReportGenerationOutput(BaseModel):
    """Output model for report generation tool."""
    report_path: str
    report_format: str
    file_size_mb: float
    sections_included: List[str]
    visualizations_count: int
    generation_time_seconds: float