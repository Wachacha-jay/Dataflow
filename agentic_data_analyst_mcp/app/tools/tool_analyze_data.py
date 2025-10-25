"""Data analysis MCP tool."""

from typing import Any, Dict

from app.core.data_analysis import DataAnalyzer
from app.models.base_models import DataAnalysisInput
from app.utils.logger import get_logger

logger = get_logger(__name__)


async def analyze_data_tool(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    MCP tool for performing data analysis.
    
    Args:
        arguments: Tool arguments matching DataAnalysisInput
        
    Returns:
        Dictionary with data analysis results
    """
    try:
        # Validate input
        input_data = DataAnalysisInput(**arguments)
        logger.info(f"Data analysis requested for: {input_data.file_path}")
        
        # Execute data analysis
        analyzer = DataAnalyzer()
        result = analyzer.analyze_data(
            file_path=input_data.file_path,
            analysis_type=input_data.analysis_type,
            columns=input_data.columns,
            groupby_column=input_data.groupby_column,
        )
        
        logger.info("Data analysis completed successfully")
        return {"success": True, "data": result.model_dump()}
        
    except Exception as e:
        logger.error(f"Error in data analysis tool: {e}", exc_info=True)
        return {"success": False, "error": str(e)}