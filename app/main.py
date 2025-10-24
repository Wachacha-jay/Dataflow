"""
Main MCP server application entry point.
"""

import asyncio
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent

from app.config import settings
from app.utils.logger import get_logger
from app.tools import (
    discover_schema_tool,
    find_relationships_tool,
    analyze_data_tool,
    visualize_data_tool,
    generate_report_tool,
)

logger = get_logger(__name__)

# Initialize MCP server
app = Server("agentic-data-analyst")


# Define available tools
TOOLS = [
    Tool(
        name="discover_schema",
        description="Discover and infer schema from a data file. Analyzes data types, null values, unique values, and provides data quality assessment.",
        inputSchema={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the data file (CSV, Excel, JSON, Parquet)",
                },
                "sample_size": {
                    "type": "integer",
                    "description": "Number of rows to sample for analysis (optional)",
                },
                "detect_relationships": {
                    "type": "boolean",
                    "description": "Whether to detect column relationships",
                    "default": True,
                },
            },
            "required": ["file_path"],
        },
    ),
    Tool(
        name="find_relationships",
        description="Discover relationships and correlations between columns. Finds numeric correlations, categorical associations, and mixed-type relationships.",
        inputSchema={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the data file",
                },
                "correlation_threshold": {
                    "type": "number",
                    "description": "Minimum correlation threshold (0-1)",
                },
                "max_relationships": {
                    "type": "integer",
                    "description": "Maximum number of relationships to return",
                    "default": 50,
                },
            },
            "required": ["file_path"],
        },
    ),
    Tool(
        name="analyze_data",
        description="Perform comprehensive data analysis including descriptive statistics, statistical tests, and insights generation.",
        inputSchema={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the data file",
                },
                "analysis_type": {
                    "type": "string",
                    "enum": ["descriptive", "statistical", "comprehensive"],
                    "description": "Type of analysis to perform",
                    "default": "comprehensive",
                },
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Specific columns to analyze (optional)",
                },
                "groupby_column": {
                    "type": "string",
                    "description": "Column to group by for aggregated analysis (optional)",
                },
            },
            "required": ["file_path"],
        },
    ),
    Tool(
        name="visualize_data",
        description="Create data visualizations including histograms, scatter plots, correlation matrices, and more.",
        inputSchema={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the data file",
                },
                "visualization_type": {
                    "type": "string",
                    "enum": [
                        "histogram",
                        "scatter",
                        "line",
                        "bar",
                        "box",
                        "heatmap",
                        "correlation",
                        "distribution",
                        "pairplot",
                    ],
                    "description": "Type of visualization to create",
                },
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Columns to visualize (optional)",
                },
                "title": {
                    "type": "string",
                    "description": "Chart title (optional)",
                },
                "output_path": {
                    "type": "string",
                    "description": "Custom output path (optional)",
                },
            },
            "required": ["file_path", "visualization_type"],
        },
    ),
    Tool(
        name="generate_report",
        description="Generate comprehensive data analysis reports with visualizations, insights, and recommendations in HTML, PDF, Markdown, or JSON format.",
        inputSchema={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Path to the data file",
                },
                "report_type": {
                    "type": "string",
                    "enum": ["descriptive", "statistical", "comprehensive"],
                    "description": "Type of report to generate",
                    "default": "comprehensive",
                },
                "output_format": {
                    "type": "string",
                    "enum": ["html", "pdf", "markdown", "json"],
                    "description": "Output format for the report",
                    "default": "html",
                },
                "include_visualizations": {
                    "type": "boolean",
                    "description": "Include visualizations in the report",
                    "default": True,
                },
                "include_raw_data": {
                    "type": "boolean",
                    "description": "Include raw data sample in the report",
                    "default": False,
                },
                "output_path": {
                    "type": "string",
                    "description": "Custom output path (optional)",
                },
            },
            "required": ["file_path"],
        },
    ),
]


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    logger.info("Listing available tools")
    return TOOLS


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """
    Handle tool calls.
    
    Args:
        name: Tool name
        arguments: Tool arguments
        
    Returns:
        List of TextContent with results
    """
    logger.info(f"Tool called: {name}")
    
    try:
        if name == "discover_schema":
            result = await discover_schema_tool(arguments)
        elif name == "find_relationships":
            result = await find_relationships_tool(arguments)
        elif name == "analyze_data":
            result = await analyze_data_tool(arguments)
        elif name == "visualize_data":
            result = await visualize_data_tool(arguments)
        elif name == "generate_report":
            result = await generate_report_tool(arguments)
        else:
            logger.error(f"Unknown tool: {name}")
            result = {"success": False, "error": f"Unknown tool: {name}"}
        
        # Format response
        if result["success"]:
            logger.info(f"Tool {name} executed successfully")
            return [TextContent(type="text", text=str(result["data"]))]
        else:
            logger.error(f"Tool {name} failed: {result['error']}")
            return [TextContent(type="text", text=f"Error: {result['error']}")]
            
    except Exception as e:
        logger.error(f"Error executing tool {name}: {e}", exc_info=True)
        return [TextContent(type="text", text=f"Error executing tool: {str(e)}")]


async def main() -> None:
    """Main entry point for the MCP server."""
    logger.info("Starting Agentic Data Analyst MCP Server")
    logger.info(f"Configuration loaded from: {settings.data_dir}")
    logger.info(f"Available tools: {len(TOOLS)}")
    
    async with stdio_server() as (read_stream, write_stream):
        logger.info("MCP server running on stdio")
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server shutting down...")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        raise