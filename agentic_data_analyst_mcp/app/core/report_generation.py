"""
Report generation logic with dashboards and insights.
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import markdown
from jinja2 import Template
from app.config import settings
from app.core.data_analysis import DataAnalyzer
from app.core.data_visualization import DataVisualizer
from app.core.relationship_discovery import RelationshipDiscoverer
from app.core.schema_discovery import SchemaDiscoverer
from app.models.base_models import (
    AnalysisType,
    ReportFormat,
    ReportGenerationOutput,
    VisualizationType,
)
from app.utils.io_utils import IOUtils
from app.utils.logger import LoggerMixin


class ReportGenerator(LoggerMixin):
    """Generate comprehensive reports with visualizations and insights."""

    def __init__(self):
        """Initialize report generator."""
        self.io_utils = IOUtils()
        self.schema_discoverer = SchemaDiscoverer()
        self.relationship_discoverer = RelationshipDiscoverer()
        self.data_analyzer = DataAnalyzer()
        self.data_visualizer = DataVisualizer()

    def generate_report(
        self,
        file_path: str,
        report_type: AnalysisType = AnalysisType.COMPREHENSIVE,
        output_format: ReportFormat = ReportFormat.HTML,
        include_visualizations: bool = True,
        include_raw_data: bool = False,
        output_path: Optional[str] = None,
    ) -> ReportGenerationOutput:
        """
        Generate a comprehensive data analysis report.

        Args:
            file_path: Path to the data file
            report_type: Type of report
            output_format: Output format (HTML, PDF, Markdown)
            include_visualizations: Include visualizations
            include_raw_data: Include raw data sample
            output_path: Custom output path

        Returns:
            ReportGenerationOutput with report details
        """
        start_time = time.time()

        try:
            self.logger.info(f"Generating {report_type.value} report for {file_path}")

            # Load data
            df = self.io_utils.load_data(file_path)

            # Collect all analysis results
            report_data = self._collect_analysis_data(
                file_path, df, report_type, include_visualizations, include_raw_data
            )

            # Generate output path
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"report_{Path(file_path).stem}_{timestamp}.{output_format.value}"
                output_path = settings.reports_dir / filename
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Generate report based on format
            if output_format == ReportFormat.HTML:
                self._generate_html_report(report_data, output_path)
            elif output_format == ReportFormat.MARKDOWN:
                self._generate_markdown_report(report_data, output_path)
            elif output_format == ReportFormat.JSON:
                self._generate_json_report(report_data, output_path)
            elif output_format == ReportFormat.PDF:
                # Generate HTML first, then convert to PDF
                html_path = output_path.with_suffix(".html")
                self._generate_html_report(report_data, html_path)
                self._convert_html_to_pdf(html_path, output_path)
            else:
                raise ValueError(f"Unsupported output format: {output_format}")

            generation_time = time.time() - start_time
            file_size_mb = output_path.stat().st_size / (1024 * 1024)

            result = ReportGenerationOutput(
                report_path=str(output_path),
                report_format=output_format.value,
                file_size_mb=round(file_size_mb, 2),
                sections_included=report_data["sections_included"],
                visualizations_count=report_data["visualizations_count"],
                generation_time_seconds=round(generation_time, 2),
            )

            self.logger.info(f"Report generated successfully in {generation_time:.2f}s")
            return result

        except Exception as e:
            self.logger.error(f"Error generating report: {e}", exc_info=True)
            raise

    def _collect_analysis_data(
        self,
        file_path: str,
        df: pd.DataFrame,
        report_type: AnalysisType,
        include_visualizations: bool,
        include_raw_data: bool,
    ) -> Dict[str, Any]:
        """Collect all analysis data for the report."""
        sections_included = []
        visualizations = []

        # Basic metadata
        report_data = {
            "file_path": file_path,
            "file_name": Path(file_path).name,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "report_type": report_type.value,
        }

        # Schema discovery
        self.logger.info("Running schema discovery...")
        schema_output = self.schema_discoverer.discover_schema(file_path)
        report_data["schema"] = schema_output.model_dump()
        sections_included.append("Schema Discovery")

        # Data analysis
        self.logger.info("Running data analysis...")
        analysis_output = self.data_analyzer.analyze_data(
            file_path, analysis_type=report_type
        )
        report_data["analysis"] = analysis_output.model_dump()
        sections_included.append("Data Analysis")

        # Relationship discovery
        if report_type == AnalysisType.COMPREHENSIVE:
            self.logger.info("Running relationship discovery...")
            relationship_output = self.relationship_discoverer.discover_relationships(
                file_path
            )
            report_data["relationships"] = relationship_output.model_dump()
            sections_included.append("Relationship Discovery")

        # Generate visualizations
        if include_visualizations:
            self.logger.info("Generating visualizations...")
            visualizations = self._generate_visualizations(file_path, df)
            report_data["visualizations"] = visualizations
            sections_included.append("Visualizations")

        # Include raw data sample
        if include_raw_data:
            sample_size = min(100, len(df))
            report_data["raw_data_sample"] = df.head(sample_size).to_dict(orient="records")
            sections_included.append("Raw Data Sample")

        # Add summary
        report_data["sections_included"] = sections_included
        report_data["visualizations_count"] = len(visualizations)

        return report_data

    def _generate_visualizations(
        self, file_path: str, df: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Generate standard set of visualizations."""
        visualizations = []
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

        try:
            # Correlation matrix
            if len(numeric_cols) > 1:
                viz = self.data_visualizer.create_visualization(
                    file_path,
                    VisualizationType.CORRELATION,
                    columns=numeric_cols[:10],
                    title="Correlation Matrix",
                )
                visualizations.append(viz.model_dump())

            # Distribution plots
            if len(numeric_cols) > 0:
                viz = self.data_visualizer.create_visualization(
                    file_path,
                    VisualizationType.DISTRIBUTION,
                    columns=numeric_cols[:3],
                    title="Distribution Analysis",
                )
                visualizations.append(viz.model_dump())

            # Box plots
            if len(numeric_cols) > 0:
                viz = self.data_visualizer.create_visualization(
                    file_path,
                    VisualizationType.BOX,
                    columns=numeric_cols[:5],
                    title="Box Plot Analysis",
                )
                visualizations.append(viz.model_dump())

        except Exception as e:
            self.logger.warning(f"Error generating visualizations: {e}")

        return visualizations

    def _generate_html_report(self, report_data: Dict[str, Any], output_path: Path) -> None:
        """Generate HTML report."""
        html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data Analysis Report - {{ file_name }}</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; line-height: 1.6; color: #333; background: #f5f5f5; }
        .container { max-width: 1200px; margin: 0 auto; padding: 20px; }
        header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 40px 20px; text-align: center; border-radius: 10px; margin-bottom: 30px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        h1 { font-size: 2.5em; margin-bottom: 10px; }
        .meta { font-size: 0.9em; opacity: 0.9; }
        .section { background: white; margin: 20px 0; padding: 30px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        h2 { color: #667eea; border-bottom: 3px solid #667eea; padding-bottom: 10px; margin-bottom: 20px; }
        h3 { color: #764ba2; margin-top: 20px; margin-bottom: 10px; }
        .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }
        .metric-card { background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%); padding: 20px; border-radius: 8px; text-align: center; }
        .metric-value { font-size: 2em; font-weight: bold; color: #667eea; }
        .metric-label { font-size: 0.9em; color: #666; margin-top: 5px; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background-color: #667eea; color: white; font-weight: bold; }
        tr:hover { background-color: #f5f5f5; }
        .insight { background: #e7f3ff; border-left: 4px solid #2196F3; padding: 15px; margin: 10px 0; border-radius: 4px; }
        .warning { background: #fff3cd; border-left: 4px solid #ffc107; padding: 15px; margin: 10px 0; border-radius: 4px; }
        .recommendation { background: #d4edda; border-left: 4px solid #28a745; padding: 15px; margin: 10px 0; border-radius: 4px; }
        .visualization { margin: 20px 0; text-align: center; }
        .visualization img { max-width: 100%; height: auto; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
        .quality-score { font-size: 3em; font-weight: bold; color: {{ quality_color }}; text-align: center; margin: 20px 0; }
        ul { margin-left: 20px; }
        li { margin: 5px 0; }
        .footer { text-align: center; padding: 20px; color: #666; font-size: 0.9em; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>📊 Data Analysis Report</h1>
            <div class="meta">
                <p><strong>File:</strong> {{ file_name }}</p>
                <p><strong>Generated:</strong> {{ generated_at }}</p>
                <p><strong>Report Type:</strong> {{ report_type }}</p>
            </div>
        </header>

        <!-- Overview Section -->
        <div class="section">
            <h2>📋 Dataset Overview</h2>
            <div class="metric-grid">
                <div class="metric-card">
                    <div class="metric-value">{{ schema.total_rows|format_number }}</div>
                    <div class="metric-label">Total Rows</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{{ schema.total_columns }}</div>
                    <div class="metric-label">Total Columns</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">{{ schema.data_quality_score }}%</div>
                    <div class="metric-label">Quality Score</div>
                </div>
            </div>
        </div>

        <!-- Schema Section -->
        <div class="section">
            <h2>🔍 Schema Discovery</h2>
            <table>
                <thead>
                    <tr>
                        <th>Column Name</th>
                        <th>Data Type</th>
                        <th>Inferred Type</th>
                        <th>Null %</th>
                        <th>Unique %</th>
                    </tr>
                </thead>
                <tbody>
                    {% for column in schema.columns %}
                    <tr>
                        <td><strong>{{ column.name }}</strong></td>
                        <td>{{ column.data_type }}</td>
                        <td>{{ column.inferred_type }}</td>
                        <td>{{ column.null_percentage }}%</td>
                        <td>{{ column.unique_percentage }}%</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>

            {% if schema.recommendations %}
            <h3>💡 Recommendations</h3>
            {% for rec in schema.recommendations %}
            <div class="recommendation">{{ rec }}</div>
            {% endfor %}
            {% endif %}
        </div>

        <!-- Analysis Section -->
        <div class="section">
            <h2>📈 Statistical Analysis</h2>
            
            {% if analysis.insights %}
            <h3>🔎 Key Insights</h3>
            {% for insight in analysis.insights %}
            <div class="insight">{{ insight }}</div>
            {% endfor %}
            {% endif %}

            {% if analysis.warnings %}
            <h3>⚠️ Warnings</h3>
            {% for warning in analysis.warnings %}
            <div class="warning">{{ warning }}</div>
            {% endfor %}
            {% endif %}

            {% if analysis.recommendations %}
            <h3>✅ Recommendations</h3>
            {% for rec in analysis.recommendations %}
            <div class="recommendation">{{ rec }}</div>
            {% endfor %}
            {% endif %}
        </div>

        <!-- Relationships Section -->
        {% if relationships %}
        <div class="section">
            <h2>🔗 Relationship Discovery</h2>
            <p>Found <strong>{{ relationships.total_relationships }}</strong> significant relationships.</p>
            
            {% if relationships.insights %}
            <h3>🔎 Insights</h3>
            {% for insight in relationships.insights %}
            <div class="insight">{{ insight }}</div>
            {% endfor %}
            {% endif %}

            <h3>Top Relationships</h3>
            <table>
                <thead>
                    <tr>
                        <th>Column 1</th>
                        <th>Column 2</th>
                        <th>Type</th>
                        <th>Strength</th>
                        <th>Description</th>
                    </tr>
                </thead>
                <tbody>
                    {% for rel in relationships.relationships[:20] %}
                    <tr>
                        <td>{{ rel.column1 }}</td>
                        <td>{{ rel.column2 }}</td>
                        <td>{{ rel.relationship_type }}</td>
                        <td>{{ rel.strength }}</td>
                        <td>{{ rel.description }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        {% endif %}

        <!-- Visualizations Section -->
        {% if visualizations %}
        <div class="section">
            <h2>📊 Visualizations</h2>
            {% for viz in visualizations %}
            <div class="visualization">
                <h3>{{ viz.description }}</h3>
                <img src="file://{{ viz.output_path }}" alt="{{ viz.description }}">
            </div>
            {% endfor %}
        </div>
        {% endif %}

        <div class="footer">
            <p>Generated by Agentic Data Analyst MCP Server</p>
            <p>Report includes {{ sections_included|length }} sections and {{ visualizations_count }} visualizations</p>
        </div>
    </div>
</body>
</html>
        """

        # Prepare template data
        template_data = report_data.copy()
        template_data["quality_color"] = self._get_quality_color(
            report_data["schema"]["data_quality_score"]
        )

        # Render template
        template = Template(html_template)
        template.globals["format_number"] = lambda x: f"{x:,}"
        html_content = template.render(**template_data)

        # Write to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        self.logger.info(f"HTML report saved to {output_path}")

    def _generate_markdown_report(self, report_data: Dict[str, Any], output_path: Path) -> None:
        """Generate Markdown report."""
        md_content = f"""# Data Analysis Report

**File:** {report_data['file_name']}  
**Generated:** {report_data['generated_at']}  
**Report Type:** {report_data['report_type']}

---

## Dataset Overview

- **Total Rows:** {report_data['schema']['total_rows']:,}
- **Total Columns:** {report_data['schema']['total_columns']}
- **Data Quality Score:** {report_data['schema']['data_quality_score']}%

---

## Schema Discovery

"""
        # Add columns table
        md_content += "| Column Name | Data Type | Inferred Type | Null % | Unique % |\n"
        md_content += "|-------------|-----------|---------------|--------|----------|\n"
        for col in report_data['schema']['columns']:
            md_content += f"| {col['name']} | {col['data_type']} | {col['inferred_type']} | {col['null_percentage']}% | {col['unique_percentage']}% |\n"

        # Add insights
        md_content += "\n## Key Insights\n\n"
        for insight in report_data['analysis']['insights']:
            md_content += f"- {insight}\n"

        # Add warnings
        if report_data['analysis']['warnings']:
            md_content += "\n## ⚠️ Warnings\n\n"
            for warning in report_data['analysis']['warnings']:
                md_content += f"- {warning}\n"

        # Add recommendations
        if report_data['analysis']['recommendations']:
            md_content += "\n## ✅ Recommendations\n\n"
            for rec in report_data['analysis']['recommendations']:
                md_content += f"- {rec}\n"

        # Add relationships
        if 'relationships' in report_data:
            md_content += f"\n## Relationship Discovery\n\nFound **{report_data['relationships']['total_relationships']}** significant relationships.\n\n"

        # Write to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md_content)

        self.logger.info(f"Markdown report saved to {output_path}")

    def _generate_json_report(self, report_data: Dict[str, Any], output_path: Path) -> None:
        """Generate JSON report."""
        self.io_utils.save_json(report_data, output_path)
        self.logger.info(f"JSON report saved to {output_path}")

    def _convert_html_to_pdf(self, html_path: Path, pdf_path: Path) -> None:
        """Convert HTML to PDF (requires pdfkit and wkhtmltopdf)."""
        try:
            import pdfkit
            pdfkit.from_file(str(html_path), str(pdf_path))
            self.logger.info(f"PDF report saved to {pdf_path}")
        except Exception as e:
            self.logger.warning(f"PDF conversion failed: {e}. HTML report available at {html_path}")
            raise

    def _get_quality_color(self, score: float) -> str:
        """Get color based on quality score."""
        if score >= 90:
            return "#28a745"  # Green
        elif score >= 70:
            return "#ffc107"  # Yellow
        else:
            return "#dc3545"  # Red