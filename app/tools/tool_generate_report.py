"""Report generation MCP tool."""

from typing import Any, Dict

from app.core.report_generation import ReportGenerator
from app.models.base_models import ReportGenerationInput
from app.utils.logger import get_logger

logger = get_logger(__name__)


async def generate_report_tool(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    MCP tool for generating comprehensive reports.
    
    Args:
        arguments: Tool arguments matching ReportGenerationInput
        
    Returns:
        Dictionary with report generation results
    """
    try:
        # Validate input
        input_data = ReportGenerationInput(**arguments)
        logger.info(f"Report generation requested for: {input_data.file_path}")
        
        # Execute report generation
        generator = ReportGenerator()
        result = generator.generate_report(
            file_path=input_data.file_path,
            report_type=input_data.report_type,
            output_format=input_data.output_format,
            include_visualizations=input_data.include_visualizations,
            include_raw_data=input_data.include_raw_data,
            output_path=input_data.output_path,
        )
        
        logger.info("Report generated successfully")
        return {"success": True, "data": result.model_dump()}
        
    except Exception as e:
        logger.error(f"Error in report generation tool: {e}", exc_info=True)
        return {"success": False, "error": str(e)}