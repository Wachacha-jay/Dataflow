"""Tests for data analysis functionality."""

import pytest
import pandas as pd
import numpy as np

from app.core.data_analysis import DataAnalyzer
from app.models.base_models import AnalysisType


@pytest.fixture
def sample_data():
    """Create sample DataFrame for testing."""
    np.random.seed(42)
    return pd.DataFrame({
        "numeric1": np.random.randn(100),
        "numeric2": np.random.randn(100) * 10 + 50,
        "category": np.random.choice(["A", "B", "C"], 100),
        "binary": np.random.choice([0, 1], 100),
    })


@pytest.fixture
def temp_csv_file(tmp_path, sample_data):
    """Create temporary CSV file for testing."""
    file_path = tmp_path / "test_data.csv"
    sample_data.to_csv(file_path, index=False)
    return str(file_path)


def test_descriptive_analysis(temp_csv_file):
    """Test descriptive analysis."""
    analyzer = DataAnalyzer()
    result = analyzer.analyze_data(temp_csv_file, analysis_type=AnalysisType.DESCRIPTIVE)
    
    assert result.analysis_type == "descriptive"
    assert "numeric_summary" in result.summary_statistics
    assert "categorical_summary" in result.summary_statistics


def test_statistical_analysis(temp_csv_file):
    """Test statistical analysis."""
    analyzer = DataAnalyzer()
    result = analyzer.analyze_data(temp_csv_file, analysis_type=AnalysisType.STATISTICAL)
    
    assert result.analysis_type == "statistical"
    assert "statistical_tests" in result.summary_statistics


def test_comprehensive_analysis(temp_csv_file):
    """Test comprehensive analysis."""
    analyzer = DataAnalyzer()
    result = analyzer.analyze_data(temp_csv_file, analysis_type=AnalysisType.COMPREHENSIVE)
    
    assert result.analysis_type == "comprehensive"
    assert "data_quality" in result.summary_statistics
    assert "correlations" in result.summary_statistics


def test_insights_generation(temp_csv_file):
    """Test insights generation."""
    analyzer = DataAnalyzer()
    result = analyzer.analyze_data(temp_csv_file)
    
    assert isinstance(result.insights, list)
    assert len(result.insights) > 0


def test_warnings_generation(temp_csv_file):
    """Test warnings generation."""
    analyzer = DataAnalyzer()
    result = analyzer.analyze_data(temp_csv_file)
    
    assert isinstance(result.warnings, list)


def test_recommendations_generation(temp_csv_file):
    """Test recommendations generation."""
    analyzer = DataAnalyzer()
    result = analyzer.analyze_data(temp_csv_file)
    
    assert isinstance(result.recommendations, list)