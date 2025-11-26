"""SQL database adapter for querying and loading data."""

from typing import Any, Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from app.config import settings
from app.utils.logger import LoggerMixin


class SQLAdapter(LoggerMixin):
    """Adapter for SQL database operations."""

    def __init__(self, connection_string: Optional[str] = None):
        """
        Initialize SQL adapter.

        Args:
            connection_string: SQLAlchemy connection string. Uses settings if None.
        """
        self.connection_string = connection_string or settings.db_url
        self.engine: Optional[Engine] = None

    def connect(self) -> Engine:
        """
        Create database engine connection.

        Returns:
            SQLAlchemy engine
        """
        try:
            if not self.connection_string:
                raise ValueError("No database connection string provided")

            self.logger.info("Connecting to database...")
            self.engine = create_engine(self.connection_string)
            
            # Test connection
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            
            self.logger.info("Database connection established")
            return self.engine

        except Exception as e:
            self.logger.error(f"Error connecting to database: {e}", exc_info=True)
            raise

    def read_query(self, query: str, **kwargs: Any) -> pd.DataFrame:
        """
        Execute SQL query and return results as DataFrame.

        Args:
            query: SQL query string
            **kwargs: Additional pandas read_sql arguments

        Returns:
            DataFrame with query results
        """
        try:
            if not self.engine:
                self.connect()

            self.logger.info(f"Executing query: {query[:100]}...")
            
            df = pd.read_sql(query, self.engine, **kwargs)
            
            self.logger.info(f"Query returned {len(df)} rows")
            return df

        except Exception as e:
            self.logger.error(f"Error executing query: {e}", exc_info=True)
            raise

    def read_table(
        self, table_name: str, schema: Optional[str] = None, **kwargs: Any
    ) -> pd.DataFrame:
        """
        Read entire table into DataFrame.

        Args:
            table_name: Name of the table
            schema: Database schema name
            **kwargs: Additional pandas read_sql_table arguments

        Returns:
            DataFrame with table data
        """
        try:
            if not self.engine:
                self.connect()

            self.logger.info(f"Reading table: {table_name}")
            
            df = pd.read_sql_table(table_name, self.engine, schema=schema, **kwargs)
            
            self.logger.info(f"Loaded table with shape {df.shape}")
            return df

        except Exception as e:
            self.logger.error(f"Error reading table: {e}", exc_info=True)
            raise

    def write_table(
        self,
        df: pd.DataFrame,
        table_name: str,
        schema: Optional[str] = None,
        if_exists: str = "replace",
        **kwargs: Any,
    ) -> None:
        """
        Write DataFrame to database table.

        Args:
            df: DataFrame to write
            table_name: Target table name
            schema: Database schema name
            if_exists: How to behave if table exists ('fail', 'replace', 'append')
            **kwargs: Additional pandas to_sql arguments
        """
        try:
            if not self.engine:
                self.connect()

            self.logger.info(f"Writing to table: {table_name}")
            
            df.to_sql(
                table_name,
                self.engine,
                schema=schema,
                if_exists=if_exists,
                index=False,
                **kwargs,
            )
            
            self.logger.info(f"Successfully wrote {len(df)} rows to {table_name}")

        except Exception as e:
            self.logger.error(f"Error writing to table: {e}", exc_info=True)
            raise

    def list_tables(self, schema: Optional[str] = None) -> list[str]:
        """
        List all tables in the database.

        Args:
            schema: Database schema name

        Returns:
            List of table names
        """
        try:
            if not self.engine:
                self.connect()

            from sqlalchemy import inspect
            
            inspector = inspect(self.engine)
            tables = inspector.get_table_names(schema=schema)
            
            self.logger.info(f"Found {len(tables)} tables")
            return tables

        except Exception as e:
            self.logger.error(f"Error listing tables: {e}", exc_info=True)
            raise

    def close(self) -> None:
        """Close database connection."""
        if self.engine:
            self.engine.dispose()
            self.logger.info("Database connection closed")