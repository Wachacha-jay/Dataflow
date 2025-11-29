"""
STDIO entry point for MCP server.
Use this for Claude Desktop integration.
"""

import asyncio
from mcp.server.stdio import stdio_server
from app.main import build_mcp_server
from app.utils.logger import get_logger

logger = get_logger(__name__)


async def main():
    """Run the MCP server via STDIO."""
    logger.info("Starting Dataflow MCP server via STDIO")
    
    server = build_mcp_server()
    
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
