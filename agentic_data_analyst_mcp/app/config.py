"""
Configuration management for the Agentic Data Analyst MCP Server.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load environment variables
load_dotenv()


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/analyst.log"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Directories
    data_dir: Path = Path("data/")
    output_dir: Path = Path("output/")
    reports_dir: Path = Path("output/reports/")
    visualizations_dir: Path = Path("output/visualizations/")
    cache_dir: Path = Path("cache/")
    report_template_dir: Path = Path("templates/")

    # Database (optional)
    db_host: Optional[str] = None
    db_port: Optional[int] = None
    db_name: Optional[str] = None
    db_user: Optional[str] = None
    db_password: Optional[str] = None
    db_driver: str = "postgresql"

    # Analysis settings
    max_rows_preview: int = 1000
    correlation_threshold: float = 0.7
    missing_value_threshold: float = 0.3
    outlier_std_threshold: float = 3.0

    # Visualization settings
    visualization_dpi: int = 300
    plot_style: str = "seaborn"
    default_figure_size: tuple[int, int] = (12, 8)
    color_palette: str = "husl"

    # Report settings
    max_report_size_mb: int = 50
    include_raw_data: bool = False

    # Performance settings
    chunk_size: int = 10000
    max_workers: int = 4
    enable_caching: bool = True

    class Config:
        env_file = ".env"
        case_sensitive = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._create_directories()

    def _create_directories(self) -> None:
        """Create necessary directories if they don't exist."""
        for directory in [
            self.data_dir,
            self.output_dir,
            self.reports_dir,
            self.visualizations_dir,
            self.cache_dir,
            Path(self.log_file).parent,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def db_url(self) -> Optional[str]:
        """Construct database URL if credentials are provided."""
        if all([self.db_host, self.db_name, self.db_user, self.db_password]):
            return (
                f"{self.db_driver}://{self.db_user}:{self.db_password}"
                f"@{self.db_host}:{self.db_port or 5432}/{self.db_name}"
            )
        return None


# Global settings instance
settings = Settings()