"""
Render-safe MCP server entry point with FastAPI app factory.
"""

from typing import Any, Optional
import logging

from mcp.server import Server
from mcp.types import Tool, TextContent

# Attempt FastAPI MCP integration
try:
    from mcp.server.fast import create_fastapi  # ASGI wrapper
    from fastapi import FastAPI
    HAS_FAST_MCP = True
except Exception:
    HAS_FAST_MCP = False
    FastAPI = None  # type: ignore

from app.config import settings
from app.utils.logger import get_logger

from app.tools import (
    discover_schema_tool,
    find_relationships_tool,
    analyze_data_tool,
    visualize_data_tool,
    generate_report_tool,
    clean_data_tool,
)

# -----------------------------------------------------
# Logging
# -----------------------------------------------------
logger = get_logger(__name__)

# -----------------------------------------------------
# TOOL DEFINITIONS
# -----------------------------------------------------

TOOLS: list[Tool] = [
    Tool(
        name="discover_schema",
        description="Discover and infer schema from a data file.",
        inputSchema={
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "sample_size": {"type": "integer"},
                "detect_relationships": {
                    "type": "boolean",
                    "default": True
                },
            },
            "required": ["file_path"],
        },
    ),
    Tool(
        name="find_relationships",
        description="Find correlations and associations.",
        inputSchema={
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "correlation_threshold": {"type": "number"},
                "max_relationships": {"type": "integer", "default": 50},
            },
            "required": ["file_path"],
        },
    ),
    Tool(
        name="analyze_data",
        description="Perform comprehensive data analysis.",
        inputSchema={
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "analysis_type": {
                    "type": "string",
                    "enum": ["descriptive", "statistical", "comprehensive"],
                    "default": "comprehensive",
                },
                "columns": {"type": "array", "items": {"type": "string"}},
                "groupby_column": {"type": "string"},
            },
            "required": ["file_path"],
        },
    ),
    Tool(
        name="visualize_data",
        description="Create visualizations.",
        inputSchema={
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "visualization_type": {"type": "string"},
                "columns": {"type": "array", "items": {"type": "string"}},
                "title": {"type": "string"},
                "output_path": {"type": "string"},
            },
            "required": ["file_path", "visualization_type"],
        },
    ),
    Tool(
        name="generate_report",
        description="Generate a data report.",
        inputSchema={
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "report_type": {"type": "string", "default": "comprehensive"},
                "output_format": {
                    "type": "string",
                    "enum": ["html", "pdf", "markdown", "json"],
                    "default": "html",
                },
                "include_visualizations": {"type": "boolean", "default": True},
                "include_raw_data": {"type": "boolean", "default": False},
                "output_path": {"type": "string"},
            },
            "required": ["file_path"],
        },
    ),
    Tool(
        name="clean_data",
        description="Clean a dataset.",
        inputSchema={
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "operations": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [
                            "missing_values",
                            "duplicates",
                            "outliers",
                            "normalize_strings",
                            "coerce_types",
                        ],
                    },
                },
                "parameters": {"type": "object"},
            },
            "required": ["file_path", "operations"],
        },
    ),
]


# -----------------------------------------------------
# Create MCP Server (no FastAPI yet)
# -----------------------------------------------------

def build_mcp_server() -> Server:
    """
    Construct the MCP server independently from ASGI/FastAPI.
    Safe to call inside create_app().
    """
    server = Server("dataflow")

    @server.list_tools()
    async def _list_tools() -> list[Tool]:
        logger.info("Listing available tools")
        return TOOLS

    @server.call_tool()
    async def _call_tool(name: str, arguments: Any) -> list[TextContent]:
        logger.info(f"Tool called: {name}")

        mapping = {
            "discover_schema": discover_schema_tool,
            "find_relationships": find_relationships_tool,
            "analyze_data": analyze_data_tool,
            "visualize_data": visualize_data_tool,
            "generate_report": generate_report_tool,
            "clean_data": clean_data_tool,
        }

        try:
            if name not in mapping:
                return [TextContent(type="text", text=f"Unknown tool: {name}")]

            handler = mapping[name]
            result = await handler(arguments)

            if result.get("success", False):
                return [TextContent(type="text", text=str(result["data"]))]

            return [TextContent(type="text", text=f"Error: {result.get('error')}")]

        except Exception as e:
            logger.error(f"Tool execution error: {e}", exc_info=True)
            return [TextContent(type="text", text=f"Exception: {str(e)}")]

    return server


# -----------------------------------------------------
# ASGI Factory for Render + Uvicorn
# -----------------------------------------------------

def create_app():
    """
    Render-safe ASGI app factory.
    Always returns a valid FastAPI app.
    """
    logger.info("Initializing Dataflow MCP ASGI wrapper")
    logger.info(f"Config directory: {settings.data_dir}")

    # FAST MCP available → full ASGI MCP server
    if HAS_FAST_MCP:
        mcp_server = build_mcp_server()
        app = create_fastapi(mcp_server)
        return app

    # Fallback → MINIMAL FASTAPI APP
    logger.warning("FAST MCP unavailable → Running STDIO placeholder app")

    app = FastAPI()  # type: ignore

    @app.get("/")
    async def root():
        return {
            "status": "limited",
            "message": "FAST MCP not available. Running in fallback mode.",
        }

    return app


# -----------------------------------------------------
# Local Dev Run
# -----------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:create_app",
        factory=True,
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
