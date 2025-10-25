"""Relationship discovery MCP tool."""

from typing import Any, Dict

from app.core.relationship_discovery import RelationshipDiscoverer
from app.models.base_models import RelationshipDiscoveryInput
from app.utils.logger import get_logger

logger = get_logger(__name__)


async def find_relationships_tool(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    MCP tool for discovering relationships between columns.
    
    Args:
        arguments: Tool arguments matching RelationshipDiscoveryInput
        
    Returns:
        Dictionary with relationship discovery results
    """
    try:
        # Validate input
        input_data = RelationshipDiscoveryInput(**arguments)
        logger.info(f"Relationship discovery requested for: {input_data.file_path}")
        
        # Execute relationship discovery
        discoverer = RelationshipDiscoverer()
        result = discoverer.discover_relationships(
            file_path=input_data.file_path,
            correlation_threshold=input_data.correlation_threshold,
            max_relationships=input_data.max_relationships,
        )
        
        logger.info("Relationship discovery completed successfully")
        return {"success": True, "data": result.model_dump()}
        
    except Exception as e:
        logger.error(f"Error in relationship discovery tool: {e}", exc_info=True)
        return {"success": False, "error": str(e)}