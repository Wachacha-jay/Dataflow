# Agentic Data Analyst MCP Server

An intelligent Model Context Protocol (MCP) server for automated data analysis, schema discovery, relationship mapping, visualization, and comprehensive reporting.

## Features

- **Schema Discovery**: Automatically infer data types, constraints, and metadata
- **Relationship Discovery**: Find correlations, dependencies, and patterns
- **Data Analysis**: Comprehensive statistical analysis and profiling
- **Visualizations**: Generate interactive charts, plots, and dashboards
- **Report Generation**: Create PDF, HTML, and Markdown reports with insights
- **Exception Handling**: Robust error handling and logging throughout
- **Multiple Data Sources**: Support for CSV, Excel, JSON, and SQL databases

## Installation

### Using Poetry (Recommended)
```bash
poetry install
```

### Using pip
```bash
pip install -r requirements.txt
```

## Configuration

Create a `.env` file in the root directory:

```env
# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/analyst.log

# Data paths
DATA_DIR=data/
OUTPUT_DIR=output/
REPORTS_DIR=output/reports/
VISUALIZATIONS_DIR=output/visualizations/

# Database (optional)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=analytics
DB_USER=user
DB_PASSWORD=password

# Analysis settings
MAX_ROWS_PREVIEW=1000
CORRELATION_THRESHOLD=0.7
VISUALIZATION_DPI=300
```

## Usage

### Starting the MCP Server

```bash
python -m app.main
```

### Available Tools

1. **discover_schema**: Analyze data structure and infer schema
2. **find_relationships**: Discover correlations and patterns
3. **analyze_data**: Perform statistical analysis
4. **visualize_data**: Create charts and visualizations
5. **generate_report**: Produce comprehensive reports with insights

### Example Tool Calls

```python
# Discover schema
{
    "tool": "discover_schema",
    "arguments": {
        "file_path": "data/sales.csv",
        "sample_size": 1000
    }
}

# Generate report
{
    "tool": "generate_report",
    "arguments": {
        "file_path": "data/sales.csv",
        "report_type": "comprehensive",
        "output_format": "html",
        "include_visualizations": true
    }
}
```

## Project Structure

```
agentic_data_analyst_mcp/
├── app/                    # Main application code
│   ├── core/              # Core analytical logic
│   ├── tools/             # MCP tool implementations
│   ├── utils/             # Utilities and helpers
│   ├── models/            # Data models
│   └── adapters/          # Data source connectors
├── tests/                 # Test suite
├── data/                  # Sample data (not in repo)
├── output/                # Generated outputs
│   ├── reports/          # Generated reports
│   └── visualizations/   # Generated charts
└── logs/                  # Log files
```

## Development

### Running Tests

```bash
pytest tests/ -v --cov=app
```

### Code Formatting

```bash
black app/ tests/
ruff check app/ tests/
```

### Type Checking

```bash
mypy app/
```

## License

MIT License