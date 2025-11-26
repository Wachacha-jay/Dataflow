"""
Main MCP server application entry point (Render-safe).
"""

import asyncio
from typing import Any

from mcp.server import Server
from mcp.types import Tool, TextContent

# FastAPI integration (MCP fast) may or may not exist
try:
    from mcp.server.fast import create_fastapi  # type: ignore
    from fastapi import FastAPI
    HAS_FAST_MCP = True
except Exception:
    HAS_FAST_MCP = False
    from mcp.server.stdio import stdio_server  # type: ignore

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

logger = get_logger(__name__)

# MCP internal server (NOT ASGI APP!)
mcp_server = Server("dataflow")


# ------------- TOOL DEFINITIONS -------------------------------------------------

TOOLS = [
    Tool(
        name="discover_schema",
        description="Discover and infer schema from a data file.",
        inputSchema={
            "type": "object",
            "properties": {
                "file_path": {"type": "string"},
                "sample_size": {"type": "integer"},
                "detect_relationships": {"type": "boolean", "default": True},
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


@mcp_server.list_tools()
async def list_tools() -> list[Tool]:
    logger.info("Listing available tools")
    return TOOLS


# Optional fastmcp tool decorators
try:
    from mcp.server.fastmcp import tool as fast_tool  # type: ignore
except Exception:
    fast_tool = None

if fast_tool:
    @fast_tool(name="discover_schema")
    async def _discover_schema(args: Any):
        return await discover_schema_tool(args)

    @fast_tool(name="find_relationships")
    async def _find_relationships(args: Any):
        return await find_relationships_tool(args)

    @fast_tool(name="analyze_data")
    async def _analyze_data(args: Any):
        return await analyze_data_tool(args)

    @fast_tool(name="visualize_data")
    async def _visualize(args: Any):
        return await visualize_data_tool(args)

    @fast_tool(name="generate_report")
    async def _report(args: Any):
        return await generate_report_tool(args)

    @fast_tool(name="clean_data")
    async def _clean(args: Any):
        return await clean_data_tool(args)


@mcp_server.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
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

        result = await mapping[name](arguments)

        if result["success"]:
            return [TextContent(type="text", text=str(result["data"]))]
        else:
            return [TextContent(type="text", text=f"Error: {result['error']}")]

    except Exception as e:
        logger.error(f"Error executing tool {name}: {e}", exc_info=True)
        return [TextContent(type="text", text=f"Exception: {str(e)}")]


# --------------------------------------------------------------------
# ASGI SAFE: THIS is the callable Uvicorn will run on Render
# --------------------------------------------------------------------

def create_app() -> FastAPI:
    """
    ALWAYS return a FastAPI app for Uvicorn.
    MCP server is wrapped internally.
    """
    logger.info("Starting Dataflow MCP Server")
    logger.info(f"Config loaded from: {settings.data_dir}")
    logger.info(f"Tools available: {len(TOOLS)}")

    if not HAS_FAST_MCP:
        # Prevent Render from getting a non-ASGI app
        logger.warning("FAST MCP unavailable; exposing minimal FastAPI placeholder")

        app = FastAPI()

        @app.get("/")
        def root():
            return {"status": "FAST MCP missing", "mode": "stdio"}

        return app

    # Wrap MCP server inside FastAPI
    fastapi_app = create_fastapi(mcp_server)
    return fastapi_app


# --------------------------------------------------------------------
# LOCAL DEV ENTRY POINT
# --------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        create_app(),
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
