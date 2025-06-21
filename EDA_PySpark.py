from pyspark.sql import functions as F
from pyspark.sql.functions import col, count, when, desc, lit, isnan, isnull, approx_count_distinct
from pyspark.sql.types import *
import pandas as pd
import os
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import List, Optional, Dict, Any

def _setup_logger():
    """Setup clean file-only logger for audit trail"""
    output_path = "/dbfs/tmp/edd"
    log_dir = os.path.join(output_path, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    now = datetime.now(timezone(timedelta(hours=5, minutes=30)))
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"edd_batch_{timestamp}.log")
    
    logger = logging.getLogger("edd_batch")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    
    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(handler)
    logger.propagate = False
    
    logger.info(f"EDD Module loaded. Log: {log_file}")
    return logger

logger = _setup_logger()

def generate_edd(table_name: str, output_path: str = "/dbfs/tmp/edd", 
                filter_condition: Optional[str] = None, 
                sample_threshold: int = 50_000_000) -> str:
    """Generate EDD for single table"""
    
    start_time = time.time()
    
    # Load data
    df = spark.table(table_name)
    if filter_condition:
        df = df.filter(filter_condition)
    
    total_rows = df.count()
    sample_rate = min(1.0, sample_threshold / max(total_rows, 1))
    
    if sample_rate < 1.0:
        df = df.sample(sample_rate, seed=42)
        total_rows = df.count()
    
    df.cache()
    columns = df.columns
    
    # Schema classification
    numeric_columns, categorical_columns = _classify_columns_by_schema(df)
    
    # Batch statistics
    null_exprs = [count(when(col(c).isNull(), c)).alias(f"null_{c}") for c in columns]
    null_counts = df.select(null_exprs).collect()[0].asDict()
    
    unique_exprs = [approx_count_distinct(col(c)).alias(f"unique_{c}") for c in columns]
    unique_counts = df.select(unique_exprs).collect()[0].asDict()
    
    numeric_stats = _get_batch_numeric_stats(df, numeric_columns) if numeric_columns else {}
    
    # Build results
    results = []
    for i, col_name in enumerate(columns, 1):
        try:
            null_count = null_counts[f"null_{col_name}"]
            unique_count = unique_counts[f"unique_{col_name}"]
            non_null_count = total_rows - null_count
            
            if col_name in numeric_columns:
                result = _build_numeric_result(i, col_name, null_count, non_null_count, 
                                             unique_count, numeric_stats.get(col_name, {}))
            else:
                result = _build_categorical_result(df, i, col_name, null_count, 
                                                 non_null_count, unique_count)
            results.append(result)
            
        except Exception as e:
            logger.error(f"{table_name}.{col_name} failed: {e}")
            results.append(_build_error_result(i, col_name, str(e)))
    
    df.unpersist()
    
    # Save results
    os.makedirs(output_path, exist_ok=True)
    timestamp = datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime("%Y%m%d_%H%M%S")
    filename = f"{table_name.replace('.', '_')}_edd_{timestamp}.csv"
    filepath = os.path.join(output_path, filename)
    
    try:
        pd.DataFrame(results).to_csv(filepath, index=False)
        elapsed_min = (time.time() - start_time) / 60
        
        # Log to file for audit
        logger.info(f"{table_name} | {total_rows:,} rows | {len(columns)} cols | {elapsed_min:.1f}m | {os.path.basename(filepath)}")
        
        return filepath
        
    except Exception as e:
        logger.error(f"{table_name} save failed: {e}")
        raise

def _classify_columns_by_schema(df) -> tuple[List[str], List[str]]:
    """Classify columns by Spark schema types"""
    numeric_types = {IntegerType, LongType, FloatType, DoubleType, DecimalType, ByteType, ShortType}
    numeric_cols = [f.name for f in df.schema.fields if type(f.dataType) in numeric_types]
    categorical_cols = [f.name for f in df.schema.fields if type(f.dataType) not in numeric_types]
    return numeric_cols, categorical_cols

def _get_batch_numeric_stats(df, numeric_columns: List[str]) -> Dict:
    """Get numeric statistics efficiently"""
    if not numeric_columns:
        return {}
    
    # Basic stats
    stats_df = df.select(numeric_columns).describe()
    stats_pandas = stats_df.toPandas().set_index('summary')
    
    # Percentiles
    percentile_values = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    percentile_stats = {}
    
    for col_name in numeric_columns:
        try:
            percentiles = df.select(col_name).na.drop().approxQuantile(col_name, percentile_values, 0.01)
            percentile_stats[col_name] = percentiles
        except:
            percentile_stats[col_name] = [None] * 7
    
    # Combine results
    combined_stats = {}
    for col_name in numeric_columns:
        try:
            combined_stats[col_name] = {
                'mean': _safe_float(stats_pandas.loc['mean', col_name]),
                'stddev': _safe_float(stats_pandas.loc['stddev', col_name]),
                'min': _safe_float(stats_pandas.loc['min', col_name]),
                'max': _safe_float(stats_pandas.loc['max', col_name]),
                'percentiles': percentile_stats.get(col_name, [None] * 7)
            }
        except Exception as e:
            combined_stats[col_name] = {'error': str(e)}
    
    return combined_stats

def _safe_float(value) -> Optional[float]:
    """Convert to float, handle inf/nan"""
    try:
        f_val = float(value)
        if f_val == float('inf') or f_val == float('-inf') or f_val != f_val:
            return None
        return round(f_val, 6)
    except:
        return None

def _build_numeric_result(field_num: int, column_name: str, null_count: int, 
                         non_null_count: int, unique_count: int, stats: Dict) -> Dict:
    """Build numeric column result"""
    if 'error' in stats or not stats:
        return _build_error_result(field_num, column_name, "Stats calculation failed")
    
    percentiles = (stats.get('percentiles', []) + [None] * 7)[:7]
    
    return {
        'Field_Num': field_num, 'Field_Name': column_name, 'Type': 'Numeric',
        'Num_Blanks': null_count, 'Num_Entries': non_null_count, 'Num_Unique': unique_count,
        'Stddev': stats.get('stddev'), 'Mean_or_Top1': stats.get('mean'),
        'Min_or_Top2': stats.get('min'), 'P1_or_Top3': percentiles[0],
        'P5_or_Top4': percentiles[1], 'P25_or_Top5': percentiles[2],
        'Median_or_Bot5': percentiles[3], 'P75_or_Bot4': percentiles[4],
        'P95_or_Bot3': percentiles[5], 'P99_or_Bot2': percentiles[6],
        'Max_or_Bot1': stats.get('max')
    }

def _build_categorical_result(df, field_num: int, column_name: str, null_count: int, 
                            non_null_count: int, unique_count: int) -> Dict:
    """Build categorical column result with top/bottom frequencies"""
    
    base_result = {
        'Field_Num': field_num, 'Field_Name': column_name, 'Type': 'Categorical',
        'Num_Blanks': null_count, 'Num_Entries': non_null_count, 'Num_Unique': unique_count,
        'Stddev': None
    }
    
    # Handle edge cases
    if unique_count == 0:
        values = ['No_Data'] * 10
    elif unique_count == 1:
        try:
            single_row = df.select(column_name).filter(col(column_name).isNotNull()).first()
            single_val = f"{single_row[0]}:{non_null_count}" if single_row and single_row[0] is not None else 'All_Same'
            values = [single_val] * 10
        except:
            values = ['All_Same'] * 10
    elif unique_count == non_null_count:
        values = ['All_Unique'] * 10
    else:
        # Get frequency distribution
        try:
            limit_size = min(1000, max(100, unique_count))
            value_counts = (df.select(column_name)
                          .filter(col(column_name).isNotNull())
                          .groupBy(column_name)
                          .count()
                          .orderBy(desc("count"), col(column_name))
                          .limit(limit_size)
                          .collect())
            
            if not value_counts:
                values = ['No_Data'] * 10
            else:
                formatted = [f"{row[column_name]}:{row['count']}" for row in value_counts]
                
                if len(formatted) >= 10:
                    # Top 5 + Bottom 5
                    top_5 = formatted[:5]
                    bottom_5 = formatted[-5:]
                    values = top_5 + bottom_5
                else:
                    # Pad to 10 elements
                    values = (formatted + [''] * 10)[:10]
                    
        except Exception as e:
            logger.error(f"Categorical processing failed for {column_name}: {e}")
            values = ['High_Cardinality'] * 10
    
    # Map to result fields
    field_names = ['Mean_or_Top1', 'Min_or_Top2', 'P1_or_Top3', 'P5_or_Top4', 'P25_or_Top5',
                   'Median_or_Bot5', 'P75_or_Bot4', 'P95_or_Bot3', 'P99_or_Bot2', 'Max_or_Bot1']
    
    for i, field_name in enumerate(field_names):
        base_result[field_name] = values[i]
    
    return base_result

def _build_error_result(field_num: int, column_name: str, error_msg: str) -> Dict:
    """Build error result for failed columns"""
    return {
        'Field_Num': field_num, 'Field_Name': column_name, 'Type': 'Error',
        'Num_Blanks': 0, 'Num_Entries': 0, 'Num_Unique': 0, 'Stddev': None,
        'Mean_or_Top1': f'Error: {error_msg}', 'Min_or_Top2': None, 'P1_or_Top3': None,
        'P5_or_Top4': None, 'P25_or_Top5': None, 'Median_or_Bot5': None,
        'P75_or_Bot4': None, 'P95_or_Bot3': None, 'P99_or_Bot2': None, 'Max_or_Bot1': None
    }

def batch_edd(table_list: List[str], output_path: str = "/dbfs/tmp/edd", 
              sample_threshold: int = 50_000_000) -> Dict[str, str]:
    """Process multiple tables"""
    
    print(f"Starting batch EDD: {len(table_list)} tables")
    logger.info(f"BATCH START | {len(table_list)} tables")
    
    batch_start = time.time()
    results = {}
    summary_data = []
    
    for i, table_name in enumerate(table_list, 1):
        print(f"[{i}/{len(table_list)}] {table_name}")
        
        try:
            table_start = time.time()
            
            # Get basic info for console
            df = spark.table(table_name)
            total_rows = df.count()
            num_cols = len(df.columns)
            print(f"  → {total_rows:,} rows, {num_cols} columns")
            
            # Generate EDD
            filepath = generate_edd(table_name, output_path, sample_threshold=sample_threshold)
            elapsed = time.time() - table_start
            
            results[table_name] = filepath
            print(f"  → Completed in {elapsed/60:.1f}min")
            
            summary_data.append({
                'Table_Name': table_name, 'Status': 'Success', 'File_Path': filepath,
                'Rows': total_rows, 'Columns': num_cols, 'Processing_Minutes': round(elapsed/60, 2)
            })
            
        except Exception as e:
            error_msg = str(e)
            results[table_name] = f"FAILED: {error_msg}"
            print(f"  → FAILED: {error_msg}")
            logger.error(f"{table_name} | FAILED | {error_msg}")
            
            summary_data.append({
                'Table_Name': table_name, 'Status': 'Failed', 'File_Path': '',
                'Rows': 0, 'Columns': 0, 'Processing_Minutes': 0, 'Error': error_msg
            })
    
    # Save batch summary
    timestamp = datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime("%Y%m%d_%H%M%S")
    summary_file = os.path.join(output_path, f"batch_summary_{timestamp}.csv")
    
    try:
        pd.DataFrame(summary_data).to_csv(summary_file, index=False)
    except Exception as e:
        logger.error(f"Summary save failed: {e}")
    
    # Final summary
    successful = sum(1 for r in results.values() if r.endswith('.csv'))
    total_minutes = (time.time() - batch_start) / 60
    
    print(f"\nBatch completed: {successful}/{len(table_list)} successful in {total_minutes:.1f} minutes")
    logger.info(f"BATCH END | {successful}/{len(table_list)} success | {total_minutes:.1f}m")
    
    return results

# Utility functions
def list_edd_files(output_path: str = "/dbfs/tmp/edd") -> None:
    """List generated EDD files"""
    try:
        if not os.path.exists(output_path):
            print("No EDD directory found")
            return
            
        files = [f for f in os.listdir(output_path) if f.endswith('.csv') and 'summary' not in f]
        if files:
            print(f"EDD files ({len(files)}):")
            for filename in sorted(files):
                size_mb = os.path.getsize(os.path.join(output_path, filename)) / (1024*1024)
                print(f"  {filename} ({size_mb:.1f}MB)")
        else:
            print("No EDD files found")
            
        print(f"\nLocation: Data → DBFS → tmp → edd")
        
    except Exception as e:
        print(f"Error: {e}")

def show_schema_classification(table_name: str) -> None:
    """Preview column type classification"""
    df = spark.table(table_name)
    numeric_cols, categorical_cols = _classify_columns_by_schema(df)
    
    print(f"Schema classification: {table_name}")
    print(f"Total: {len(df.columns)} | Numeric: {len(numeric_cols)} | Categorical: {len(categorical_cols)}")
    print("-" * 60)
    
    if numeric_cols:
        print(f"NUMERIC ({len(numeric_cols)}):")
        for col_name in numeric_cols[:10]:  # Show first 10
            col_type = dict(df.dtypes)[col_name]
            print(f"  {col_name} ({col_type})")
        if len(numeric_cols) > 10:
            print(f"  ... and {len(numeric_cols) - 10} more")
    
    if categorical_cols:
        print(f"\nCATEGORICAL ({len(categorical_cols)}):")
        for col_name in categorical_cols[:10]:  # Show first 10
            col_type = dict(df.dtypes)[col_name]
            print(f"  {col_name} ({col_type})")
        if len(categorical_cols) > 10:
            print(f"  ... and {len(categorical_cols) - 10} more")

def view_log_file(output_path: str = "/dbfs/tmp/edd") -> None:
    """Display latest log file"""
    logs_dir = os.path.join(output_path, "logs")
    if not os.path.exists(logs_dir):
        print("No logs directory found")
        return
    
    log_files = sorted([f for f in os.listdir(logs_dir) if f.endswith('.log')], reverse=True)
    if not log_files:
        print("No log files found")
        return
    
    log_filepath = os.path.join(logs_dir, log_files[0])
    
    try:
        print(f"Latest log: {log_files[0]}")
        print("-" * 60)
        with open(log_filepath, 'r') as f:
            print(f.read())
    except Exception as e:
        print(f"Error reading log: {e}")

if __name__ == "__main__":
    print("EDD System Ready")
    print("Main: generate_edd(table) | batch_edd([tables])")  
    print("Utils: list_edd_files() | show_schema_classification(table) | view_log_file()")
