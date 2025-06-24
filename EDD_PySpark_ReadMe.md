# Enhanced EDD System v11

## Overview

The Enhanced Exploratory Data Discovery (EDD) System is a comprehensive PySpark-based tool for automated data profiling and quality assessment. It provides deep statistical analysis, data type classification, and quality insights for both Spark tables and DataFrames.

## Key Features

### 🎯 **Dual Source Support**
- **Spark Tables**: Direct table analysis with optimized TABLESAMPLE
- **DataFrames**: Analyze intermediate results from complex transformations
- **Mixed Batch Processing**: Process tables and DataFrames together

### 📊 **Advanced Column Classification**
- **Numeric**: Statistical analysis with outlier detection and distribution profiling
- **Categorical**: Value frequency analysis with distribution patterns
- **Temporal**: Date/time analysis with pattern recognition
- **Boolean**: Binary distribution analysis
- **Complex**: Array/Map/Struct basic profiling

### ⚡ **Intelligent Sampling Strategy**
- **≤2M rows**: Full dataset analysis
- **2M-20M rows**: Random sampling with configurable size
- **20M+ rows**: TABLESAMPLE optimization for tables, smart sampling for DataFrames

### 🔍 **Comprehensive Statistics**
- Null/blank analysis
- Unique value counting
- Percentile distributions (P1, P5, P25, P50, P75, P95, P99)
- Outlier detection using IQR method
- Distribution shape classification
- Top/bottom value analysis for categorical data

## Installation & Requirements

```python
# Required imports
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.functions import col, count, when, desc, lit, isnan, isnull, approx_count_distinct
from pyspark.sql.types import *
import pandas as pd
import os
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any
import glob
```

## Core Functions

### `generate_edd(source, ...)`

**Primary function for single source analysis**

```python
def generate_edd(source, output_path: str = "/dbfs/tmp/edd", 
                 col_list: Optional[List[str]] = None,
                 filter_condition: Optional[str] = None, 
                 target_sample_size: int = 2_000_000,
                 df_name: Optional[str] = None) -> str
```

**Parameters:**
- `source`: Table name (str) or PySpark DataFrame
- `output_path`: Directory for output files (default: "/dbfs/tmp/edd")
- `col_list`: Optional list of specific columns to analyze
- `filter_condition`: SQL filter (tables only, ignored for DataFrames)
- `target_sample_size`: Target sample size for large datasets
- `df_name`: Custom name for DataFrames (used in logging/filenames)

**Returns:** Path to generated EDD CSV file

### `batch_edd(source_list, ...)`

**Batch processing for multiple sources**

```python
def batch_edd(source_list: List, output_path: str = "/dbfs/tmp/edd", 
              col_list: Optional[List[str]] = None,
              target_sample_size: int = 2_000_000) -> Dict[str, str]
```

**Parameters:**
- `source_list`: List of table names, DataFrames, or (DataFrame, name) tuples
- `output_path`: Directory for output files
- `col_list`: Optional columns to analyze (applied to all sources)
- `target_sample_size`: Target sample size for large datasets

**Returns:** Dictionary mapping source names to file paths or error messages

## Usage Examples

### Table Analysis

```python
# Basic table analysis
filepath = generate_edd("customer_transactions")

# Specific columns with filter
filepath = generate_edd("customer_transactions", 
                       col_list=["customer_id", "amount", "transaction_date"],
                       filter_condition="transaction_date >= '2024-01-01'")

# Large table with custom sampling
filepath = generate_edd("big_table", target_sample_size=5_000_000)
```

### DataFrame Analysis

```python
# Complex transformation pipeline
result_df = (spark.table("sales")
            .join(spark.table("customers"), "customer_id")
            .groupBy("region", "product_category")
            .agg(F.sum("revenue").alias("total_revenue"),
                 F.avg("unit_price").alias("avg_price"),
                 F.count("*").alias("transaction_count"))
            .filter(F.col("total_revenue") > 10000))

# Analyze the result
filepath = generate_edd(result_df, 
                       df_name="regional_sales_analysis",
                       col_list=["region", "total_revenue", "avg_price"])

# Convenience function
filepath = analyze_dataframe(result_df, name="sales_summary")
```

### Batch Processing

```python
# Mixed sources batch
results = batch_edd([
    "raw_customers",                    # Table
    "raw_transactions",                 # Table
    (enriched_df, "enriched_data"),    # (DataFrame, name)
    final_model_df                      # DataFrame (auto-named)
], col_list=["common_columns"])

# Table-only batch (backward compatible)
results = batch_edd(["table1", "table2", "table3"])

# DataFrame-only batch
results = batch_edd([
    (df1, "preprocessing_result"),
    (df2, "feature_engineering"),
    (df3, "model_ready_data")
])
```

## Output Format

### CSV Structure
Each EDD generates a CSV with the following columns:

| Column | Description |
|--------|-------------|
| `Field_Num` | Column sequence number |
| `Field_Name` | Column name |
| `Type` | Data type classification |
| `Total_Rows` | Total row count in source |
| `Sample_Size` | Actual analyzed sample size |
| `Num_Blanks` | Null/blank value count |
| `Num_Entries` | Non-null value count |
| `Num_Unique` | Approximate unique value count |
| `Outlier_Count` | Number of outliers (numeric only) |
| `Distribution_Shape` | Distribution pattern classification |
| `Stddev` | Standard deviation (numeric) |
| `Mean_or_Top1` | Mean (numeric) or top value (categorical) |
| `Min_or_Top2` | Minimum (numeric) or 2nd top value (categorical) |
| `P1_or_Top3` | 1st percentile or 3rd top value |
| `P5_or_Top4` | 5th percentile or 4th top value |
| `P25_or_Top5` | 25th percentile or 5th top value |
| `Median_or_Bot5` | Median or 5th bottom value |
| `P75_or_Bot4` | 75th percentile or 4th bottom value |
| `P95_or_Bot3` | 95th percentile or 3rd bottom value |
| `P99_or_Bot2` | 99th percentile or 2nd bottom value |
| `Max_or_Bot1` | Maximum or bottom value |

### File Naming Convention
- **Tables**: `{table_name}_edd_{timestamp}.csv`
- **Tables with column selection**: `{table_name}_edd_cols_{count}_{timestamp}.csv`
- **DataFrames**: `{df_name}_edd_{timestamp}.csv`
- **DataFrames with column selection**: `{df_name}_edd_cols_{count}_{timestamp}.csv`

## Utility Functions

### `analyze_edd_errors(directory)`
Scans EDD output directory for files with errors and generates analysis report.

### `list_edd_files(output_path)`
Lists all generated EDD files with size information.

### `show_schema_classification(table_name)`
Preview column type classification for a table before running full EDD.

### `view_log_file(output_path)`
Display the latest log file for debugging.

### `analyze_dataframe(df, name, **kwargs)`
Convenience wrapper for DataFrame analysis with descriptive naming.

## Performance Characteristics

### Sampling Strategy
- **Full Analysis**: ≤2M rows (optimal performance)
- **Random Sampling**: 2M-20M rows (balanced speed/accuracy)
- **TABLESAMPLE**: 20M+ rows (maximum efficiency for tables)
- **DataFrame Sampling**: Smart sampling for large DataFrames

### Processing Speed
- **Small datasets** (≤100K rows): ~30 seconds
- **Medium datasets** (100K-2M rows): 1-5 minutes
- **Large datasets** (2M+ rows): 2-10 minutes (depending on sampling)

### Memory Usage
- Cached sample data during processing
- Automatic cleanup after analysis
- Optimized for Spark cluster environments

## Error Handling

### Robust Error Management
- **Column validation**: Checks if specified columns exist
- **Empty dataset protection**: Handles zero-row scenarios
- **Type-specific fallbacks**: Temporal → Categorical fallback
- **Sampling failures**: TABLESAMPLE → Regular sampling fallback
- **Individual column errors**: Continues processing other columns

### Error Output
Failed columns generate error records in the output CSV with descriptive error messages.

## Advanced Features

### Column Type Classification
```python
# Automatic classification based on Spark schema
numeric_types = {IntegerType, LongType, FloatType, DoubleType, DecimalType, ByteType, ShortType}
temporal_types = {DateType, TimestampType}
boolean_types = {BooleanType}
complex_types = {ArrayType, MapType, StructType, NullType}
# Everything else → Categorical
```

### Distribution Analysis
- **Numeric**: Normal vs Skewed based on mean/median comparison
- **Categorical**: Concentrated vs Uniform vs Moderate based on top value percentage
- **Temporal**: Clustered vs Recent_Heavy vs Historical_Heavy vs Uniform
- **Boolean**: Balanced vs Skewed_True vs Skewed_False

### Outlier Detection
Uses IQR (Interquartile Range) method:
- `Q1 - 1.5 * IQR` ← Lower bound
- `Q3 + 1.5 * IQR` ← Upper bound
- Values outside bounds are flagged as outliers

## Best Practices

### When to Use
- **Data Pipeline Validation**: Analyze intermediate transformations
- **Data Quality Assessment**: Profile raw data before processing
- **Model Feature Analysis**: Understand feature distributions
- **Schema Evolution Tracking**: Monitor data changes over time

### Performance Tips
- Use `col_list` to focus on specific columns for faster processing
- For very large datasets, consider increasing `target_sample_size`
- Run batch processing during off-peak hours for large table sets
- Use DataFrame analysis for complex transformation validation

### Column Selection Strategy
```python
# Focus on key business columns
key_columns = ["customer_id", "transaction_amount", "transaction_date", "product_category"]
filepath = generate_edd("transactions", col_list=key_columns)

# Analyze suspect columns identified in previous runs
suspect_columns = ["weird_column", "high_null_column", "potential_pii"]
filepath = generate_edd(my_df, col_list=suspect_columns, df_name="data_quality_check")
```

## Integration Examples

### Data Pipeline Integration
```python
def validate_transformation_step(input_df, step_name):
    """Validate each transformation step"""
    edd_path = analyze_dataframe(input_df, name=f"step_{step_name}")
    print(f"✅ EDD completed for {step_name}: {edd_path}")
    return input_df

# Use in pipeline
result = (raw_data
         .transform(lambda df: validate_transformation_step(df, "raw_data"))
         .filter(F.col("amount") > 0)
         .transform(lambda df: validate_transformation_step(df, "filtered_data"))
         .groupBy("category").agg(F.sum("amount"))
         .transform(lambda df: validate_transformation_step(df, "aggregated_data")))
```

### Scheduled Analysis
```python
def daily_table_health_check():
    """Daily EDD analysis of critical tables"""
    critical_tables = ["customers", "transactions", "products", "orders"]
    
    results = batch_edd(critical_tables, 
                       output_path="/dbfs/daily_edd/",
                       target_sample_size=1_000_000)
    
    # Process results for alerting
    for table, result in results.items():
        if "FAILED" in result:
            send_alert(f"EDD failed for {table}: {result}")
        else:
            print(f"✅ {table} analysis completed: {result}")
```

## Troubleshooting

### Common Issues
1. **Column not found**: Check column names and case sensitivity
2. **TABLESAMPLE errors**: Function falls back to regular sampling automatically
3. **Memory issues**: Reduce `target_sample_size` or focus on specific columns
4. **Empty results**: Check filters and data availability

### Debug Mode
```python
# Enable detailed logging
import logging
logging.getLogger("edd_batch").setLevel(logging.DEBUG)

# View latest log
view_log_file()
```

---

**Version**: 1.0  
**Last Updated**: 2024  
**Compatibility**: PySpark 3.x, Python 3.7+  
**License**: Internal Use
