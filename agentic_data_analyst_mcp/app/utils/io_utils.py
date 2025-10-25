"""
I/O utilities for reading and writing various data formats.
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd
from app.config import settings
from app.utils.logger import LoggerMixin


class IOUtils(LoggerMixin):
    """Utility class for data input/output operations."""

    @staticmethod
    def load_data(
        file_path: Union[str, Path],
        file_type: Optional[str] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Load data from various file formats.

        Args:
            file_path: Path to the data file
            file_type: File type (csv, excel, json, parquet). Auto-detected if None
            **kwargs: Additional arguments for pandas read functions

        Returns:
            DataFrame containing the loaded data

        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file type is unsupported
        """
        logger = IOUtils().logger
        file_path = Path(file_path)

        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        # Auto-detect file type from extension
        if file_type is None:
            file_type = file_path.suffix.lower().lstrip(".")

        logger.info(f"Loading {file_type} file: {file_path}")

        try:
            if file_type in ["csv", "txt"]:
                df = pd.read_csv(file_path, **kwargs)
            elif file_type in ["xlsx", "xls", "excel"]:
                df = pd.read_excel(file_path, **kwargs)
            elif file_type == "json":
                df = pd.read_json(file_path, **kwargs)
            elif file_type == "parquet":
                df = pd.read_parquet(file_path, **kwargs)
            elif file_type == "feather":
                df = pd.read_feather(file_path, **kwargs)
            elif file_type == "pickle":
                df = pd.read_pickle(file_path, **kwargs)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")

            logger.info(f"Successfully loaded data with shape {df.shape}")
            return df

        except Exception as e:
            logger.error(f"Error loading file: {e}", exc_info=True)
            raise

    @staticmethod
    def save_data(
        df: pd.DataFrame,
        file_path: Union[str, Path],
        file_type: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """
        Save DataFrame to various file formats.

        Args:
            df: DataFrame to save
            file_path: Output file path
            file_type: File type (csv, excel, json, parquet). Auto-detected if None
            **kwargs: Additional arguments for pandas write functions

        Raises:
            ValueError: If file type is unsupported
        """
        logger = IOUtils().logger
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Auto-detect file type from extension
        if file_type is None:
            file_type = file_path.suffix.lower().lstrip(".")

        logger.info(f"Saving data to {file_type} file: {file_path}")

        try:
            if file_type == "csv":
                df.to_csv(file_path, index=False, **kwargs)
            elif file_type in ["xlsx", "excel"]:
                df.to_excel(file_path, index=False, **kwargs)
            elif file_type == "json":
                df.to_json(file_path, **kwargs)
            elif file_type == "parquet":
                df.to_parquet(file_path, index=False, **kwargs)
            elif file_type == "feather":
                df.to_feather(file_path, **kwargs)
            elif file_type == "pickle":
                df.to_pickle(file_path, **kwargs)
            else:
                raise ValueError(f"Unsupported file type: {file_type}")

            logger.info(f"Successfully saved data to {file_path}")

        except Exception as e:
            logger.error(f"Error saving file: {e}", exc_info=True)
            raise

    @staticmethod
    def save_json(data: Dict[str, Any], file_path: Union[str, Path]) -> None:
        """
        Save dictionary to JSON file.

        Args:
            data: Dictionary to save
            file_path: Output file path
        """
        logger = IOUtils().logger
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
            logger.info(f"Saved JSON to {file_path}")
        except Exception as e:
            logger.error(f"Error saving JSON: {e}", exc_info=True)
            raise

    @staticmethod
    def load_json(file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load JSON file.

        Args:
            file_path: Path to JSON file

        Returns:
            Dictionary containing the JSON data
        """
        logger = IOUtils().logger
        file_path = Path(file_path)

        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}")

        try:
            with open(file_path, "r") as f:
                data = json.load(f)
            logger.info(f"Loaded JSON from {file_path}")
            return data
        except Exception as e:
            logger.error(f"Error loading JSON: {e}", exc_info=True)
            raise

    @staticmethod
    def get_file_info(file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Get basic information about a file.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary with file metadata
        """
        file_path = Path(file_path)

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        stats = file_path.stat()
        return {
            "name": file_path.name,
            "path": str(file_path.absolute()),
            "size_bytes": stats.st_size,
            "size_mb": round(stats.st_size / (1024 * 1024), 2),
            "extension": file_path.suffix,
            "modified": stats.st_mtime,
        }