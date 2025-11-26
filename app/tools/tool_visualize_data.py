"""Data visualization MCP tool."""

from typing import Any, Dict

from app.core.data_visualization import DataVisualizer
from app.models.base_models import VisualizationInput
from app.utils.logger import get_logger

logger = get_logger(__name__)


async def visualize_data_tool(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    MCP tool for creating data visualizations.
    
    Args:
        arguments: Tool arguments matching VisualizationInput
        
    Returns:
        Dictionary with visualization results
    """
    try:
        # Validate input
        input_data = VisualizationInput(**arguments)
        logger.info(f"Visualization requested: {input_data.visualization_type}")
        
        # Execute visualization
        visualizer = DataVisualizer()
        result = visualizer.create_visualization(
            file_path=input_data.file_path,
            visualization_type=input_data.visualization_type,
            columns=input_data.columns,
            title=input_data.title,
            output_path=input_data.output_path,
        )
        
        logger.info("Visualization created successfully")
        return {"success": True, "data": result.model_dump()}
        
    except Exception as e:
        logger.error(f"Error in visualization tool: {e}", exc_info=True)
        return {"success": False, "error": str(e)}