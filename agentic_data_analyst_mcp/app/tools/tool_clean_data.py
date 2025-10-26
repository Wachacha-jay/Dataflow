"""Data cleaning MCP tool."""

from typing import Any, Dict

from app.core.data_cleaning import DataCleaner
from app.utils.io_utils import IOUtils
from app.utils.logger import get_logger

logger = get_logger(__name__)


async def clean_data_tool(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    MCP tool for data cleaning operations.
    
    Args:
        arguments: Tool arguments including:
            - file_path: Path to data file
            - operations: List of cleaning operations to perform
            - parameters: Parameters for each operation
            
    Returns:
        Dictionary with cleaning results
    """
    try:
        # Validate input
        file_path = arguments.get("file_path")
        operations = arguments.get("operations", [])
        parameters = arguments.get("parameters", {})
        
        if not file_path:
            raise ValueError("file_path is required")
            
        logger.info(f"Data cleaning requested for: {file_path}")
        
        # Load data
        io_utils = IOUtils()
        df = io_utils.load_data(file_path)
        
        # Initialize cleaner
        cleaner = DataCleaner(df, dataset_name=file_path)
        
        # Execute cleaning operations
        for operation in operations:
            if operation == "missing_values":
                cleaner.handle_missing(**parameters.get("missing_values", {}))
            elif operation == "duplicates":
                cleaner.remove_duplicates()
            elif operation == "outliers":
                cleaner.handle_outliers(**parameters.get("outliers", {}))
            elif operation == "normalize_strings":
                cleaner.normalize_strings()
            elif operation == "coerce_types":
                cleaner.coerce_types()
        
        # Save cleaned data
        output_path = parameters.get("output_path", file_path.replace(".", "_cleaned."))
        io_utils.save_data(cleaner.df, output_path)
        
        # Get report
        cleaning_report = cleaner.get_cleaning_report()
        cleaning_report["output_path"] = output_path
        cleaning_report["original_rows"] = len(df)
        cleaning_report["cleaned_rows"] = len(cleaner.df)
        
        logger.info("Data cleaning completed successfully")
        return {"success": True, "data": cleaning_report}
        
    except Exception as e:
        logger.error(f"Error in data cleaning tool: {e}", exc_info=True)
        return {"success": False, "error": str(e)}