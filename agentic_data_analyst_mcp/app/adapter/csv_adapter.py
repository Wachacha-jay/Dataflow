"""CSV data adapter with advanced reading capabilities."""

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from app.config import settings
from app.utils.logger import LoggerMixin


class CSVAdapter(LoggerMixin):
    """Adapter for reading CSV files with various configurations."""

    def read(
        self,
        file_path: str,
        encoding: str = "utf-8",
        delimiter: str = ",",
        chunk_size: Optional[int] = None,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Read CSV file with error handling and optimization.

        Args:
            file_path: Path to CSV file
            encoding: File encoding
            delimiter: Column delimiter
            chunk_size: Read in chunks for large files
            **kwargs: Additional pandas read_csv arguments

        Returns:
            DataFrame with loaded data
        """
        try:
            self.logger.info(f"Reading CSV file: {file_path}")
            
            file_path = Path(file_path)
            if not file_path.exists():
                raise FileNotFoundError(f"CSV file not found: {file_path}")

            # Read in chunks for large files
            if chunk_size:
                chunks = []
                for chunk in pd.read_csv(
                    file_path,
                    encoding=encoding,
                    delimiter=delimiter,
                    chunksize=chunk_size,
                    **kwargs,
                ):
                    chunks.append(chunk)
                df = pd.concat(chunks, ignore_index=True)
            else:
                df = pd.read_csv(
                    file_path,
                    encoding=encoding,
                    delimiter=delimiter,
                    **kwargs,
                )

            self.logger.info(f"Successfully read CSV with shape {df.shape}")
            return df

        except Exception as e:
            self.logger.error(f"Error reading CSV file: {e}", exc_info=True)
            raise

    def write(
        self,
        df: pd.DataFrame,
        file_path: str,
        encoding: str = "utf-8",
        delimiter: str = ",",
        **kwargs: Any,
    ) -> None:
        """
        Write DataFrame to CSV file.

        Args:
            df: DataFrame to write
            file_path: Output file path
            encoding: File encoding
            delimiter: Column delimiter
            **kwargs: Additional pandas to_csv arguments
        """
        try:
            file_path = Path(file_path)
            file_path.parent.mkdir(parents=True, exist_ok=True)

            self.logger.info(f"Writing CSV file: {file_path}")
            
            df.to_csv(
                file_path,
                encoding=encoding,
                sep=delimiter,
                index=False,
                **kwargs,
            )

            self.logger.info(f"Successfully wrote CSV with {len(df)} rows")

        except Exception as e:
            self.logger.error(f"Error writing CSV file: {e}", exc_info=True)
            raise

    def infer_delimiter(self, file_path: str, sample_lines: int = 5) -> str:
        """
        Infer CSV delimiter from file.

        Args:
            file_path: Path to CSV file
            sample_lines: Number of lines to sample

        Returns:
            Inferred delimiter
        """
        try:
            with open(file_path, "r") as f:
                sample = "".join([f.readline() for _ in range(sample_lines)])

            # Try common delimiters
            delimiters = [",", ";", "\t", "|"]
            delimiter_counts = {d: sample.count(d) for d in delimiters}
            
            inferred = max(delimiter_counts, key=delimiter_counts.get)
            self.logger.info(f"Inferred delimiter: '{inferred}'")
            return inferred

        except Exception as e:
            self.logger.warning(f"Error inferring delimiter: {e}")
            return ","