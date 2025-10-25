"""
Agentic Data Analyst MCP Server

A comprehensive data analysis system with schema discovery, relationship mapping,
and automated report generation.
"""

__version__ = "0.1.0"
__author__ = "James w. Ngaruiya"

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)
logger.info(f"Initializing Agentic Data Analyst MCP Server v{__version__}")

__all__ = ["settings", "get_logger"]