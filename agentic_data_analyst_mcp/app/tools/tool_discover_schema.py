"""Schema discovery MCP tool."""

from typing import Any, Dict

from app.core.schema_discovery import SchemaDiscoverer
from app.models.base_models import SchemaDiscoveryInput, SchemaDiscoveryOutput
from app.utils.logger import get_logger

logger = get_logger(__name__)


async def discover_schema_tool(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    MCP tool for discovering schema from data files.
    
    Args:
        arguments: Tool arguments matching SchemaDiscoveryInput
        
    Returns:
        Dictionary with schema discovery results
    """
    try:
        # Validate input
        input_data = SchemaDiscoveryInput(**arguments)
        logger.info(f"Schema discovery requested for: {input_data.file_path}")
        
        # Execute schema discovery
        discoverer = SchemaDiscoverer()
        result = discoverer.discover_schema(
            file_path=input_data.file_path,
            sample_size=input_data.sample_size,
            detect_relationships=input_data.detect_relationships,
        )
        
        logger.info("Schema discovery completed successfully")
        return {"success": True, "data": result.model_dump()}
        
    except Exception as e:
        logger.error(f"Error in schema discovery tool: {e}", exc_info=True)
        return {"success": False, "error": str(e)}