from pyspark.sql import functions as F
from pyspark.sql.functions import col, count, when, isnan, isnull, regexp_extract, length
import pandas as pd

def analyze_table_issues(table_name: str, sample_size: int = 50000):
    """Deep analysis of table with EDD issues"""
    
    df = spark.table(table_name)
    df_sample = df.sample(fraction=sample_size/df.count(), seed=42) if df.count() > sample_size else df
    
    print(f"Table: {table_name}")
    print(f"Total rows: {df.count():,}, Sample: {df_sample.count():,}")
    print("="*80)
    
    # 1. Analyze the error column: estb_ownership_type_cd
    print("1. ERROR COLUMN ANALYSIS: estb_ownership_type_cd")
    error_col = 'estb_ownership_type_cd'
    
    if error_col in df.columns:
        error_analysis = df_sample.select(
            count(error_col).alias('total'),
            count(when(col(error_col).isNull(), 1)).alias('nulls'),
            count(when(col(error_col) == '', 1)).alias('empty_strings'),
            count(when(col(error_col).rlike(r'^\d+$'), 1)).alias('numeric_strings'),
            count(when(col(error_col).rlike(r'^[a-zA-Z]+$'), 1)).alias('alpha_only'),
            count(when(length(col(error_col)) > 50, 1)).alias('too_long')
        ).collect()[0]
        
        print(f"Nulls: {error_analysis['nulls']:,}, Empty: {error_analysis['empty_strings']:,}")
        print(f"Numeric strings: {error_analysis['numeric_strings']:,}, Alpha only: {error_analysis['alpha_only']:,}")
        print(f"Too long (>50 chars): {error_analysis['too_long']:,}")
        
        # Sample values
        sample_values = df_sample.select(error_col).filter(col(error_col).isNotNull()).limit(10).collect()
        print(f"Sample values: {[row[0] for row in sample_values]}")
        
        # Data type in schema
        schema_type = dict(df.dtypes)[error_col]
        print(f"Schema type: {schema_type}")
    
    print()
    
    # 2. Date field validation
    print("2. DATE FIELD VALIDATION")
    date_candidates = ['start_date', 'term_date', 'created_date', 'last_modified_date', 'ctrl_start_dt', 'ctrl_term_dt']
    
    for col_name in date_candidates:
        if col_name in df.columns:
            # Check if values look like dates
            date_check = df_sample.select(
                count(col_name).alias('total'),
                count(when(col(col_name).rlike(r'^\d{4}-\d{2}-\d{2}'), 1)).alias('yyyy_mm_dd'),
                count(when(col(col_name).rlike(r'^\d{8}'), 1)).alias('yyyymmdd'),
                count(when(col(col_name).rlike(r'^\d{2}/\d{2}/\d{4}'), 1)).alias('mm_dd_yyyy'),
                count(when(col(col_name).isNull(), 1)).alias('nulls')
            ).collect()[0]
            
            total_non_null = date_check['total'] - date_check['nulls']
            if total_non_null > 0:
                date_pct = (date_check['yyyy_mm_dd'] + date_check['yyyymmdd'] + date_check['mm_dd_yyyy']) / total_non_null * 100
                print(f"{col_name}: {date_pct:.1f}% date-like ({dict(df.dtypes)[col_name]})")
    
    print()
    
    # 3. Data quality overview
    print("3. DATA QUALITY OVERVIEW")
    
    # Null percentages for all columns
    null_stats = []
    for col_name in df.columns:
        null_count = df_sample.select(count(when(col(col_name).isNull(), 1))).collect()[0][0]
        null_pct = (null_count / df_sample.count()) * 100
        if null_pct > 50:
            null_stats.append((col_name, null_pct))
    
    if null_stats:
        print("High null columns (>50%):")
        for col_name, pct in sorted(null_stats, key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {col_name}: {pct:.1f}%")
    
    print()
    
    # 4. Cardinality analysis
    print("4. CARDINALITY ANALYSIS")
    
    # Check for potential ID columns or categorical issues
    cardinality_issues = []
    for col_name in df.columns[:20]:  # Check first 20 columns
        if col_name != error_col:  # Skip the error column
            unique_count = df_sample.select(col_name).distinct().count()
            total_count = df_sample.count()
            
            if unique_count == total_count and total_count > 1000:
                cardinality_issues.append((col_name, "Potential ID column"))
            elif unique_count == 1:
                cardinality_issues.append((col_name, "Single value"))
            elif unique_count > total_count * 0.9:
                cardinality_issues.append((col_name, "Very high cardinality"))
    
    if cardinality_issues:
        print("Cardinality issues:")
        for col_name, issue in cardinality_issues:
            print(f"  {col_name}: {issue}")
    
    print()
    
    # 5. Numeric column validation
    print("5. NUMERIC COLUMN VALIDATION")
    
    numeric_cols = [col_name for col_name, col_type in df.dtypes 
                   if col_type in ['int', 'bigint', 'double', 'float', 'decimal']]
    
    for col_name in numeric_cols[:10]:  # Check first 10 numeric columns
        stats = df_sample.select(
            F.min(col_name).alias('min'),
            F.max(col_name).alias('max'),
            F.mean(col_name).alias('mean'),
            count(when(col(col_name) < 0, 1)).alias('negative_count'),
            count(when(col(col_name) == 0, 1)).alias('zero_count')
        ).collect()[0]
        
        if stats['min'] is not None:
            range_val = stats['max'] - stats['min']
            if range_val == 0:
                print(f"  {col_name}: All same value ({stats['min']})")
            elif stats['negative_count'] > 0 and 'amount' in col_name.lower():
                print(f"  {col_name}: Has negative values ({stats['negative_count']})")

def quick_table_insights(table_name: str):
    """Quick insights summary"""
    
    df = spark.table(table_name)
    
    print(f"\nQUICK INSIGHTS for {table_name}:")
    print(f"Rows: {df.count():,}, Columns: {len(df.columns)}")
    
    # Schema overview
    type_counts = {}
    for col_name, col_type in df.dtypes:
        type_counts[col_type] = type_counts.get(col_type, 0) + 1
    
    print(f"Column types: {dict(type_counts)}")
    
    # Identify potential issues quickly
    categorical_with_dt = [col for col in df.columns if any(x in col.lower() for x in ['_dt', 'date', 'time'])]
    long_string_cols = [col for col, dtype in df.dtypes if dtype == 'string' and 'id' not in col.lower()]
    
    if categorical_with_dt:
        print(f"Potential date columns: {categorical_with_dt}")
    
    print(f"String columns (potential categorical): {len(long_string_cols)}")

# Usage:
# analyze_table_issues('your_table_name')
# quick_table_insights('your_table_name')
