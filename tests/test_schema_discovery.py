"""Tests for schema discovery functionality."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from app.core.schema_discovery import SchemaDiscoverer


@pytest.fixture
def sample_data():
    """Create sample DataFrame for testing."""
    return pd.DataFrame({
        "id": range(1, 101),
        "name": [f"Person_{i}" for i in range(100)],
        "age": np.random.randint(18, 80, 100),
        "salary": np.random.uniform(30000, 150000, 100),
        "department": np.random.choice(["IT", "HR", "Sales", "Marketing"], 100),
        "email": [f"user{i}@example.com" for i in range(100)],
    })


@pytest.fixture
def temp_csv_file(tmp_path, sample_data):
    """Create temporary CSV file for testing."""
    file_path = tmp_path / "test_data.csv"
    sample_data.to_csv(file_path, index=False)
    return str(file_path)


def test_schema_discovery_basic(temp_csv_file):
    """Test basic schema discovery."""
    discoverer = SchemaDiscoverer()
    result = discoverer.discover_schema(temp_csv_file)
    
    assert result.total_rows == 100
    assert result.total_columns == 6
    assert len(result.columns) == 6
    assert result.data_quality_score > 0


def test_column_analysis(temp_csv_file):
    """Test column-level analysis."""
    discoverer = SchemaDiscoverer()
    result = discoverer.discover_schema(temp_csv_file)
    
    # Check ID column
    id_col = next(c for c in result.columns if c.name == "id")
    assert id_col.inferred_type == "identifier"
    assert id_col.null_count == 0
    
    # Check email column
    email_col = next(c for c in result.columns if c.name == "email")
    assert email_col.inferred_type == "email"


def test_sampling(temp_csv_file):
    """Test schema discovery with sampling."""
    discoverer = SchemaDiscoverer()
    result = discoverer.discover_schema(temp_csv_file, sample_size=50)
    
    assert len(result.columns) == 6
    # Sampling shouldn't affect column count


def test_quality_score_calculation(temp_csv_file):
    """Test data quality score calculation."""
    discoverer = SchemaDiscoverer()
    result = discoverer.discover_schema(temp_csv_file)
    
    # Clean data should have high quality score
    assert result.data_quality_score >= 90


def test_recommendations(temp_csv_file):
    """Test recommendations generation."""
    discoverer = SchemaDiscoverer()
    result = discoverer.discover_schema(temp_csv_file)
    
    assert isinstance(result.recommendations, list)