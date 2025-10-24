"""Data models for MCP tool inputs and outputs."""

from app.models.base_models import (
    SchemaDiscoveryInput,
    SchemaDiscoveryOutput,
    RelationshipDiscoveryInput,
    RelationshipDiscoveryOutput,
    DataAnalysisInput,
    DataAnalysisOutput,
    VisualizationInput,
    VisualizationOutput,
    ReportGenerationInput,
    ReportGenerationOutput,
)

__all__ = [
    "SchemaDiscoveryInput",
    "SchemaDiscoveryOutput",
    "RelationshipDiscoveryInput",
    "RelationshipDiscoveryOutput",
    "DataAnalysisInput",
    "DataAnalysisOutput",
    "VisualizationInput",
    "VisualizationOutput",
    "ReportGenerationInput",
    "ReportGenerationOutput",
]