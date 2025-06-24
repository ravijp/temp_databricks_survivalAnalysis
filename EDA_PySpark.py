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

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def analyze_edd_errors(directory: str = '/mnt/edd/') -> pd.DataFrame:
    """Find EDD files with errors, sorted by record count."""
    
    files = [f for f in glob.glob(os.path.join(directory, "*.csv")) 
             if 'edd' in os.path.basename(f) and 'summary' not in os.path.basename(f)]
    
    logger.info(f"Found {len(files)} EDD files")
    
    results = []
    for file_path in files:
        try:
            table_name = os.path.basename(file_path).split('_edd_')[0]
            df = pd.read_csv(file_path)
            
            if df.empty:
                continue
            
            # Count errors
            error_count = len(df[df['Type'] == 'Error'])
            failed_stats = len(df[df['Mean_or_Top1'].astype(str).str.contains('Error:', na=False)])
            date_misclass = len(df[(df['Type'] == 'Categorical') & 
                                 (df['Field_Name'].str.contains('_dt|_date|date_|time', case=False, na=False))])
            
            total_issues = error_count + failed_stats + date_misclass
            
            if total_issues > 0:
                results.append({
                    'table_name': table_name,
                    'file_path': file_path,
                    'total_rows': df['Total_Rows'].iloc[0],
                    'total_columns': len(df),
                    'error_columns': error_count,
                    'failed_stats': failed_stats,
                    'misclassified_dates': date_misclass,
                    'total_issues': total_issues
                })
                
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
    
    if results:
        return pd.DataFrame(results).sort_values('total_rows', ascending=False)
    return pd.DataFrame()

def run_edd_error_analysis():
    """Execute EDD error analysis and generate report."""
    results = analyze_edd_errors('/mnt/edd/')
    
    if results.empty:
        print("No EDD files with errors found")
        return
    
    # Save results
    output_path = '/mnt/edd/error_analysis.csv'
    results.to_csv(output_path, index=False)
    
    # Print report
    print(f"\nTables with EDD errors (sorted by record count):")
    print(f"{'Table':<40} {'Records':<12} {'Columns':<8} {'Errors':<8} {'Failed Stats':<12} {'Date Issues':<12}")
    print("-" * 100)
    
    for _, row in results.iterrows():
        candidate = "← CANDIDATE" if row['total_rows'] < 1_000_000 and row['total_issues'] >= 2 else ""
        print(f"{row['table_name']:<40} {row['total_rows']:<12,} {row['total_columns']:<8} "
              f"{row['error_columns']:<8} {row['failed_stats']:<12} {row['misclassified_dates']:<12} {candidate}")
    
    # Show candidates for deep analysis
    candidates = results[(results['total_rows'] < 1_000_000) & (results['total_issues'] >= 2)]
    if not candidates.empty:
        print(f"\nRecommended for deep analysis (<1M records, 2+ issues):")
        for _, row in candidates.iterrows():
            print(f"- {row['table_name']}")
    
    logger.info(f"Analysis complete. Results saved to {output_path}")

def _setup_logger(output_path = "/dbfs/tmp/edd"):
    """Setup file-only logger for audit trail"""
    try:
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
        
    except Exception as e:
        # Fallback to console logger
        logger = logging.getLogger("edd_batch")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s', datefmt='%H:%M:%S'))
        logger.addHandler(console_handler)
        logger.warning(f"File logging failed, using console: {e}")
        return logger

logger = _setup_logger()

def generate_edd(source, output_path: str = "/dbfs/tmp/edd", 
                 col_list: Optional[List[str]] = None,
                 filter_condition: Optional[str] = None, 
                 target_sample_size: int = 2_000_000,
                 df_name: Optional[str] = None) -> str:
    """Generate EDD for table or DataFrame with optional column selection and enhanced robustness
    
    Args:
        source: Either table name (str) or PySpark DataFrame
        output_path: Directory to save EDD files
        col_list: Optional list of specific columns to analyze
        filter_condition: Optional filter condition (only works with table names)
        target_sample_size: Target sample size for large datasets
        df_name: Optional name for DataFrame (used in logging/filename, defaults to 'dataframe')
    """
    
    start_time = time.time()
    
    # Determine if source is DataFrame or table name
    is_dataframe = hasattr(source, 'columns')  # Check if it's a DataFrame
    
    # Table/DataFrame access and column validation
    try:
        if is_dataframe:
            df = source
            table_name = df_name or "dataframe"
            all_columns = df.columns
            
            if filter_condition:
                print("  ⚠️  Warning: filter_condition ignored for DataFrame input - apply filters before passing DataFrame")
        else:
            # Traditional table access
            table_name = source
            df = spark.table(table_name)
            all_columns = df.columns
        
        # Column validation and selection
        if col_list:
            missing_cols = [col for col in col_list if col not in all_columns]
            if missing_cols:
                raise ValueError(f"Columns not found: {missing_cols}")
            df = df.select(col_list)
            col_info = f"{len(col_list)} selected columns"
        else:
            col_info = f"{len(all_columns)} columns"
            
    except Exception as e:
        return _build_table_error_result(table_name if not is_dataframe else (df_name or "dataframe"), str(e))
    
    # Apply filter if provided (only for table names, not DataFrames)
    if filter_condition and not is_dataframe:
        try:
            df = df.filter(filter_condition)
        except Exception as e:
            return _build_table_error_result(table_name, f"Filter error: {str(e)}")
    
    # Zero records protection
    total_rows = df.count()
    if total_rows == 0:
        return _build_empty_table_result(table_name)
    
    # Optimized three-tier sampling strategy
    if total_rows <= 2_000_000:
        df_to_analyze, actual_sample_size, sampling_method = df, total_rows, "Full Dataset"
    elif total_rows <= 20_000_000:
        sample_rate = target_sample_size / total_rows
        df_to_analyze = df.sample(sample_rate, seed=42)
        actual_sample_size = df_to_analyze.count()
        sampling_method = f"Random Sample ({sample_rate:.4f})"
    else:
        # TABLESAMPLE with fallback (only for table names, not DataFrames)
        if not is_dataframe:
            try:
                cols_str = ", ".join(col_list) if col_list else "*"
                table_sql = f"SELECT {cols_str} FROM {table_name}"
                if filter_condition:
                    table_sql += f" WHERE {filter_condition}"
                table_sql += f" TABLESAMPLE({target_sample_size} ROWS)"
                
                df_to_analyze = spark.sql(table_sql)
                actual_sample_size = target_sample_size
                sampling_method = f"TABLESAMPLE ({target_sample_size:,} rows)"
            except Exception as e:
                sample_rate = target_sample_size / total_rows
                df_to_analyze = df.sample(sample_rate, seed=42)
                actual_sample_size = df_to_analyze.count()
                sampling_method = f"Sample Fallback ({sample_rate:.4f})"
                logger.warning(f"TABLESAMPLE failed for {table_name}: {e}")
        else:
            # For DataFrames, use regular sampling
            sample_rate = target_sample_size / total_rows
            df_to_analyze = df.sample(sample_rate, seed=42)
            actual_sample_size = df_to_analyze.count()
            sampling_method = f"DataFrame Sample ({sample_rate:.4f})"
    
    print(f"  → {sampling_method}")
    df_to_analyze.cache()
    columns = df_to_analyze.columns
    
    # Schema classification and stats collection
    numeric_cols, categorical_cols, temporal_cols, boolean_cols, complex_cols = _classify_columns_by_schema(df_to_analyze)
    
    type_counts = [len(numeric_cols), len(categorical_cols), len(temporal_cols), len(boolean_cols), len(complex_cols)]
    type_names = ["Numeric", "Categorical", "Temporal", "Boolean", "Complex"]
    type_summary = ", ".join(f"{count} {name}" for count, name in zip(type_counts, type_names) if count > 0)
    print(f"  → {type_summary}")
    
    # Batch statistics - optimized single pass
    batch_exprs = []
    batch_exprs.extend([count(when(col(c).isNull(), c)).alias(f"null_{c}") for c in columns])
    batch_exprs.extend([approx_count_distinct(col(c)).alias(f"unique_{c}") for c in columns])
    batch_stats = df_to_analyze.select(batch_exprs).collect()[0].asDict()
    
    # Type-specific statistics
    stats_map = {
        'numeric': _get_batch_numeric_stats(df_to_analyze, numeric_cols) if numeric_cols else {},
        'temporal': _get_temporal_stats(df_to_analyze, temporal_cols) if temporal_cols else {},
        'boolean': _get_boolean_stats(df_to_analyze, boolean_cols) if boolean_cols else {},
        'complex': _get_complex_stats(df_to_analyze, complex_cols) if complex_cols else {}
    }
    
    # Build results - streamlined processing
    results = []
    for i, col_name in enumerate(columns, 1):
        try:
            null_count = batch_stats[f"null_{col_name}"]
            unique_count = batch_stats[f"unique_{col_name}"]
            non_null_count = actual_sample_size - null_count
            
            # Route to appropriate result builder
            if col_name in numeric_cols:
                result = _build_numeric_result(i, col_name, total_rows, actual_sample_size, 
                                             null_count, non_null_count, unique_count, 
                                             stats_map['numeric'].get(col_name, {}))
            elif col_name in temporal_cols:
                result = _build_temporal_result(i, col_name, total_rows, actual_sample_size,
                                              null_count, non_null_count, unique_count,
                                              stats_map['temporal'].get(col_name, {}))
                if result is None:  # Fallback to categorical
                    result = _build_categorical_result(df_to_analyze, i, col_name, total_rows, 
                                                     actual_sample_size, null_count, non_null_count, unique_count)
            elif col_name in boolean_cols:
                result = _build_boolean_result(i, col_name, total_rows, actual_sample_size,
                                             null_count, non_null_count, unique_count,
                                             stats_map['boolean'].get(col_name, {}))
            elif col_name in complex_cols:
                result = _build_complex_result(i, col_name, total_rows, actual_sample_size,
                                             null_count, non_null_count, unique_count,
                                             stats_map['complex'].get(col_name, {}))
            else:
                result = _build_categorical_result(df_to_analyze, i, col_name, total_rows, 
                                                 actual_sample_size, null_count, non_null_count, unique_count)
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"{table_name}.{col_name} failed: {e}")
            results.append(_build_error_result(i, col_name, total_rows, actual_sample_size, str(e)))
    
    df_to_analyze.unpersist()
    
    # Save results with optimized filename
    os.makedirs(output_path, exist_ok=True)
    timestamp = datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime("%Y%m%d_%H%M%S")
    
    suffix = f"_cols_{len(col_list)}" if col_list and len(col_list) < len(all_columns) else ""
    filename = f"{table_name.replace('.', '_')}_edd{suffix}_{timestamp}.csv"
    filepath = os.path.join(output_path, filename)
    
    try:
        pd.DataFrame(results).to_csv(filepath, index=False)
        elapsed_min = (time.time() - start_time) / 60
        
        logger.info(f"{table_name} | {total_rows:,} rows | {col_info} | {elapsed_min:.1f}m | {os.path.basename(filepath)}")
        return filepath
        
    except Exception as e:
        logger.error(f"{table_name} save failed: {e}")
        raise

def batch_edd(source_list: List, output_path: str = "/dbfs/tmp/edd", 
              col_list: Optional[List[str]] = None,
              target_sample_size: int = 2_000_000) -> Dict[str, str]:
    """Process multiple tables/DataFrames with optional column selection using optimized three-tier sampling
    
    Args:
        source_list: List of table names (str) or tuples (DataFrame, name) or mixed
        output_path: Directory to save EDD files
        col_list: Optional list of columns to analyze (applied to all sources)
        target_sample_size: Target sample size for large datasets
    """
    
    col_info = f" with {len(col_list)} columns" if col_list else ""
    print(f"Starting batch EDD: {len(source_list)} sources{col_info}")
    print(f"Sampling: ≤2M=Full, 2M-20M=Random, 20M+=TABLESAMPLE({target_sample_size:,})")
    logger.info(f"BATCH START | {len(source_list)} sources{col_info}")
    
    batch_start = time.time()
    results = {}
    summary_data = []
    
    for i, source in enumerate(source_list, 1):
        # Handle different source types
        if isinstance(source, tuple):
            # (DataFrame, name) tuple
            df, source_name = source
            is_dataframe = True
        elif hasattr(source, 'columns'):
            # DataFrame without name
            df, source_name = source, f"dataframe_{i}"
            is_dataframe = True
        else:
            # Table name string
            source_name = source
            is_dataframe = False
        
        print(f"[{i}/{len(source_list)}] {source_name}")
        
        try:
            table_start = time.time()
            
            # Quick info check
            if is_dataframe:
                temp_df = df
                if col_list:
                    temp_df = df.select(col_list)
                total_rows = temp_df.count()
                num_cols = len(temp_df.columns)
            else:
                temp_df = spark.table(source_name)
                if col_list:
                    temp_df = temp_df.select(col_list)
                total_rows = temp_df.count()
                num_cols = len(temp_df.columns)
            
            print(f"  → {total_rows:,} rows, {num_cols} columns")
            
            # Generate EDD
            if is_dataframe:
                filepath = generate_edd(df, output_path, col_list, 
                                      target_sample_size=target_sample_size, df_name=source_name)
            else:
                filepath = generate_edd(source_name, output_path, col_list, 
                                      target_sample_size=target_sample_size)
            
            elapsed = time.time() - table_start
            results[source_name] = filepath
            print(f"  → Completed in {elapsed/60:.1f}min")
            
            summary_data.append({
                'Source_Name': source_name, 'Source_Type': 'DataFrame' if is_dataframe else 'Table',
                'Status': 'Success', 'File_Path': filepath, 'Rows': total_rows, 
                'Columns': num_cols, 'Processing_Minutes': round(elapsed/60, 2)
            })
            
        except Exception as e:
            error_msg = str(e)
            results[source_name] = f"FAILED: {error_msg}"
            print(f"  → FAILED: {error_msg}")
            logger.error(f"{source_name} | FAILED | {error_msg}")
            
            summary_data.append({
                'Source_Name': source_name, 'Source_Type': 'DataFrame' if is_dataframe else 'Table',
                'Status': 'Failed', 'File_Path': '', 'Rows': 0, 'Columns': 0, 
                'Processing_Minutes': 0, 'Error': error_msg
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
    
    print(f"\nBatch completed: {successful}/{len(source_list)} successful in {total_minutes:.1f} minutes")
    logger.info(f"BATCH END | {successful}/{len(source_list)} success | {total_minutes:.1f}m")
    
    return results

def _classify_columns_by_schema(df) -> tuple[List[str], List[str], List[str], List[str], List[str]]:
    """Classify columns by Spark schema types into 5 categories"""
    numeric_types = {IntegerType, LongType, FloatType, DoubleType, DecimalType, ByteType, ShortType}
    temporal_types = {DateType, TimestampType}
    boolean_types = {BooleanType}
    complex_types = {ArrayType, MapType, StructType, NullType}
    
    numeric_cols = []
    categorical_cols = []
    temporal_cols = []
    boolean_cols = []
    complex_cols = []
    
    for field in df.schema.fields:
        field_type = type(field.dataType)
        
        if field_type in numeric_types:
            numeric_cols.append(field.name)
        elif field_type in temporal_types:
            temporal_cols.append(field.name)
        elif field_type in boolean_types:
            boolean_cols.append(field.name)
        elif field_type in complex_types:
            complex_cols.append(field.name)
        else:
            categorical_cols.append(field.name)
    
    return numeric_cols, categorical_cols, temporal_cols, boolean_cols, complex_cols

def _get_batch_numeric_stats(df, numeric_columns: List[str]) -> Dict:
    """Get numeric statistics with outlier detection and distribution analysis"""
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
    
    # Outlier detection using IQR method
    outlier_stats = {}
    for col_name in numeric_columns:
        try:
            percentiles = percentile_stats.get(col_name, [None] * 7)
            if percentiles[2] is not None and percentiles[4] is not None:  # Q1 and Q3
                Q1, Q3 = percentiles[2], percentiles[4]
                IQR = Q3 - Q1
                if IQR > 0:
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    outlier_count = df.select(col_name).filter(
                        (col(col_name) < lower_bound) | (col(col_name) > upper_bound)
                    ).count()
                    outlier_stats[col_name] = outlier_count
                else:
                    outlier_stats[col_name] = 0
            else:
                outlier_stats[col_name] = 0
        except:
            outlier_stats[col_name] = 0
    
    # Combine all statistics
    combined_stats = {}
    for col_name in numeric_columns:
        try:
            mean_val = _safe_float(stats_pandas.loc['mean', col_name])
            median_val = percentile_stats.get(col_name, [None] * 7)[3]  # Median is index 3
            
            combined_stats[col_name] = {
                'mean': mean_val,
                'stddev': _safe_float(stats_pandas.loc['stddev', col_name]),
                'min': _safe_float(stats_pandas.loc['min', col_name]),
                'max': _safe_float(stats_pandas.loc['max', col_name]),
                'median': _safe_float(median_val),
                'percentiles': percentile_stats.get(col_name, [None] * 7),
                'outlier_count': outlier_stats.get(col_name, 0),
                'distribution_shape': _classify_numeric_distribution(mean_val, median_val)
            }
        except Exception as e:
            combined_stats[col_name] = {'error': str(e)}
    
    return combined_stats

def _classify_numeric_distribution(mean: Optional[float], median: Optional[float]) -> str:
    """Classify numeric distribution shape"""
    if mean is None or median is None:
        return "Unknown"
    
    if median == 0:
        return "Skewed" if abs(mean) > 0.1 else "Normal"
    
    # Calculate relative difference
    relative_diff = abs(mean - median) / abs(median)
    
    if relative_diff < 0.1:
        return "Normal"
    else:
        return "Skewed"

def _safe_float(value) -> Optional[float]:
    """Convert to float, handle inf/nan"""
    try:
        f_val = float(value)
        if f_val == float('inf') or f_val == float('-inf') or f_val != f_val:
            return None
        return round(f_val, 6)
    except:
        return None

def _build_temporal_result(field_num: int, column_name: str, total_rows: int, sample_size: int,
                          null_count: int, non_null_count: int, unique_count: int, stats: Dict) -> Dict:
    """Build temporal column result with simplified percentile analysis"""
    
    # Check if we should fallback to categorical processing
    if stats.get('fallback_to_categorical'):
        return None  # Signal to use categorical processing instead
    
    min_date = stats.get('min_date', '')
    max_date = stats.get('max_date', '')
    date_range_days = stats.get('date_range_days', 0)
    q1_date = stats.get('q1_date', '')
    median_date = stats.get('median_date', '')
    q3_date = stats.get('q3_date', '')
    pattern = stats.get('pattern', 'Temporal')
    
    return {
        'Field_Num': field_num,
        'Field_Name': column_name,
        'Type': 'Temporal',
        'Total_Rows': total_rows,
        'Sample_Size': sample_size,
        'Num_Blanks': null_count,
        'Num_Entries': non_null_count,
        'Num_Unique': unique_count,
        'Outlier_Count': 0,
        'Distribution_Shape': pattern,
        'Stddev': date_range_days,
        'Mean_or_Top1': median_date,      # Median date
        'Min_or_Top2': min_date,          # Earliest date
        'P1_or_Top3': '',                 # Not calculated (simplified)
        'P5_or_Top4': '',                 # Not calculated (simplified)
        'P25_or_Top5': q1_date,           # Q1 date (25th percentile)
        'Median_or_Bot5': median_date,    # Median date (duplicate for consistency)
        'P75_or_Bot4': q3_date,           # Q3 date (75th percentile)
        'P95_or_Bot3': '',                # Not calculated (simplified)
        'P99_or_Bot2': '',                # Not calculated (simplified)
        'Max_or_Bot1': max_date           # Latest date
    }

def _build_boolean_result(field_num: int, column_name: str, total_rows: int, sample_size: int,
                         null_count: int, non_null_count: int, unique_count: int, stats: Dict) -> Dict:
    """Build boolean column result with binary insights"""
    if 'error' in stats or not stats:
        return _build_error_result(field_num, column_name, total_rows, sample_size, "Boolean stats calculation failed")
    
    percentiles = (stats.get('percentiles', []) + [0] * 7)[:7]
    true_pct = stats.get('true_percentage', 0.0)
    
    return {
        'Field_Num': field_num,
        'Field_Name': column_name,
        'Type': 'Boolean',
        'Total_Rows': total_rows,
        'Sample_Size': sample_size,
        'Num_Blanks': null_count,
        'Num_Entries': non_null_count,
        'Num_Unique': unique_count,
        'Outlier_Count': 0,  # Not applicable for boolean
        'Distribution_Shape': stats.get('distribution_shape', 'Unknown'),
        'Stddev': stats.get('stddev'),
        'Mean_or_Top1': round(true_pct, 4),  # True percentage
        'Min_or_Top2': stats.get('min', 0),
        'P1_or_Top3': percentiles[0],
        'P5_or_Top4': percentiles[1],
        'P25_or_Top5': percentiles[2],
        'Median_or_Bot5': percentiles[3],
        'P75_or_Bot4': percentiles[4],
        'P95_or_Bot3': percentiles[5],
        'P99_or_Bot2': percentiles[6],
        'Max_or_Bot1': stats.get('max', 1)
    }

def _build_complex_result(field_num: int, column_name: str, total_rows: int, sample_size: int,
                         null_count: int, non_null_count: int, unique_count: int, stats: Dict) -> Dict:
    """Build complex column result with minimal analysis"""
    if 'error' in stats or not stats:
        return _build_error_result(field_num, column_name, total_rows, sample_size, "Complex stats calculation failed")
    
    avg_size = stats.get('avg_structure_size', 0)
    max_size = stats.get('max_structure_size', 0)
    
    return {
        'Field_Num': field_num,
        'Field_Name': column_name,
        'Type': 'Complex',
        'Total_Rows': total_rows,
        'Sample_Size': sample_size,
        'Num_Blanks': null_count,
        'Num_Entries': non_null_count,
        'Num_Unique': unique_count,
        'Outlier_Count': 0,  # Not applicable for complex
        'Distribution_Shape': 'Complex',
        'Stddev': None,
        'Mean_or_Top1': 'Complex_Analysis_Required',
        'Min_or_Top2': round(avg_size, 2) if avg_size > 0 else 'N/A',
        'P1_or_Top3': None,
        'P5_or_Top4': None,
        'P25_or_Top5': None,
        'Median_or_Bot5': None,
        'P75_or_Bot4': None,
        'P95_or_Bot3': None,
        'P99_or_Bot2': None,
        'Max_or_Bot1': round(max_size, 2) if max_size > 0 else 'N/A'
    }

def _get_temporal_stats(df, temporal_columns: List[str]) -> Dict:
    """Get simplified temporal statistics with enhanced safety"""
    if not temporal_columns:
        return {}
    
    temporal_stats = {}
    
    for col_name in temporal_columns:
        try:
            temp_df = df.select(col_name).filter(col(col_name).isNotNull())
            
            # Enhanced temporal processing protection
            record_count = temp_df.count()
            if record_count == 0:
                temporal_stats[col_name] = {'fallback_to_categorical': True}
                continue
            
            # Get min/max dates
            min_max = temp_df.agg(
                F.min(col_name).alias('min_date'),
                F.max(col_name).alias('max_date')
            ).collect()[0]
            
            min_date = str(min_max['min_date']) if min_max['min_date'] else ''
            max_date = str(min_max['max_date']) if min_max['max_date'] else ''
            
            # Calculate date range in days
            try:
                date_range_days = (min_max['max_date'] - min_max['min_date']).days
            except:
                date_range_days = 0
            
            # Calculate 3 percentiles using days since epoch with protection
            q1_date = median_date = q3_date = ''
            try:
                epoch_df = temp_df.select(F.datediff(col_name, F.lit('1970-01-01')).alias('days'))
                if epoch_df.count() > 0:  # Additional safety check
                    percentile_days = epoch_df.approxQuantile('days', [0.25, 0.5, 0.75], 0.05)
                    
                    # Convert back to dates
                    base_date = datetime(1970, 1, 1).date()
                    if len(percentile_days) >= 3 and all(p is not None for p in percentile_days):
                        q1_date = str(base_date + timedelta(days=int(percentile_days[0])))
                        median_date = str(base_date + timedelta(days=int(percentile_days[1])))
                        q3_date = str(base_date + timedelta(days=int(percentile_days[2])))
            except:
                q1_date = median_date = q3_date = ''
            
            # Simple pattern detection
            if date_range_days < 90:
                pattern = "Clustered"
            elif '2024' in median_date or '2025' in median_date:
                pattern = "Recent_Heavy"
            elif '2020' in median_date or '2021' in median_date:
                pattern = "Historical_Heavy"
            else:
                pattern = "Uniform"
            
            temporal_stats[col_name] = {
                'min_date': min_date,
                'max_date': max_date,
                'date_range_days': date_range_days,
                'q1_date': q1_date,
                'median_date': median_date, 
                'q3_date': q3_date,
                'pattern': pattern
            }
            
        except Exception:
            temporal_stats[col_name] = {'fallback_to_categorical': True}
    
    return temporal_stats

def _get_boolean_stats(df, boolean_columns: List[str]) -> Dict:
    """Get boolean statistics with empty result set protection"""
    if not boolean_columns:
        return {}
    
    boolean_stats = {}
    
    for col_name in boolean_columns:
        try:
            bool_df = df.select(col_name).filter(col(col_name).isNotNull())
            
            # Empty result set protection
            if bool_df.count() == 0:
                boolean_stats[col_name] = {
                    'mean': 0.0, 'stddev': 0.0, 'min': 0, 'max': 1,
                    'true_percentage': 0.0, 'percentiles': [0, 0, 0, 0, 1, 1, 1],
                    'distribution_shape': 'No_Data', 'outlier_count': 0
                }
                continue
            
            # Get basic stats
            stats = bool_df.select(
                F.mean(col(col_name).cast('int')).alias('mean'),
                F.stddev(col(col_name).cast('int')).alias('stddev'),
                F.min(col(col_name).cast('int')).alias('min'),
                F.max(col(col_name).cast('int')).alias('max')
            ).collect()[0]
            
            mean_val = float(stats['mean']) if stats['mean'] is not None else None
            true_percentage = mean_val if mean_val is not None else 0.0
            
            # Determine distribution shape
            if true_percentage < 0.1:
                distribution_shape = "Skewed_False"
            elif true_percentage > 0.9:
                distribution_shape = "Skewed_True"
            else:
                distribution_shape = "Balanced"
            
            # Get percentiles (will be 0s and 1s)
            percentiles = [0, 0, 0, 0, 1, 1, 1] if true_percentage > 0.5 else [0, 0, 0, 0, 0, 0, 1]
            
            boolean_stats[col_name] = {
                'mean': mean_val,
                'stddev': _safe_float(stats['stddev']),
                'min': int(stats['min']) if stats['min'] is not None else 0,
                'max': int(stats['max']) if stats['max'] is not None else 1,
                'true_percentage': true_percentage,
                'percentiles': percentiles,
                'distribution_shape': distribution_shape,
                'outlier_count': 0
            }
            
        except Exception as e:
            boolean_stats[col_name] = {'error': str(e)}
    
    return boolean_stats

def _get_complex_stats(df, complex_columns: List[str]) -> Dict:
    """Get minimal statistics for complex columns with empty result set protection"""
    if not complex_columns:
        return {}
    
    complex_stats = {}
    
    for col_name in complex_columns:
        try:
            non_null_df = df.select(col_name).filter(col(col_name).isNotNull())
            
            # Empty result set protection
            if non_null_df.count() == 0:
                complex_stats[col_name] = {
                    'avg_structure_size': 0.0, 'max_structure_size': 0.0,
                    'analysis_type': 'Complex', 'outlier_count': 0, 'distribution_shape': 'No_Data'
                }
                continue
            
            # Try to get size information for arrays/maps
            try:
                col_type_str = str(df.select(col_name).dtypes[0][1]).lower()
                if 'array' in col_type_str:
                    size_stats = non_null_df.select(F.size(col_name).alias('array_size')).describe().collect()
                    avg_size = float([row['array_size'] for row in size_stats if row['summary'] == 'mean'][0])
                    max_size = float([row['array_size'] for row in size_stats if row['summary'] == 'max'][0])
                elif 'map' in col_type_str:
                    size_stats = non_null_df.select(F.size(col_name).alias('map_size')).describe().collect()
                    avg_size = float([row['map_size'] for row in size_stats if row['summary'] == 'mean'][0])
                    max_size = float([row['map_size'] for row in size_stats if row['summary'] == 'max'][0])
                else:
                    avg_size = max_size = 1.0
            except:
                avg_size = max_size = 0.0
            
            complex_stats[col_name] = {
                'avg_structure_size': avg_size,
                'max_structure_size': max_size,
                'analysis_type': 'Complex',
                'outlier_count': 0,
                'distribution_shape': 'Complex'
            }
            
        except Exception as e:
            complex_stats[col_name] = {'error': str(e)}
    
    return complex_stats

def _build_numeric_result(field_num: int, column_name: str, total_rows: int, sample_size: int,
                         null_count: int, non_null_count: int, unique_count: int, stats: Dict) -> Dict:
    """Build numeric column result with enhanced insights"""
    if 'error' in stats or not stats:
        return _build_error_result(field_num, column_name, total_rows, sample_size, "Stats calculation failed")
    
    percentiles = (stats.get('percentiles', []) + [None] * 7)[:7]
    
    return {
        'Field_Num': field_num,
        'Field_Name': column_name,
        'Type': 'Numeric',
        'Total_Rows': total_rows,
        'Sample_Size': sample_size,
        'Num_Blanks': null_count,
        'Num_Entries': non_null_count,
        'Num_Unique': unique_count,
        'Outlier_Count': stats.get('outlier_count', 0),
        'Distribution_Shape': stats.get('distribution_shape', 'Unknown'),
        'Stddev': stats.get('stddev'),
        'Mean_or_Top1': stats.get('mean'),
        'Min_or_Top2': stats.get('min'),
        'P1_or_Top3': percentiles[0],
        'P5_or_Top4': percentiles[1],
        'P25_or_Top5': percentiles[2],
        'Median_or_Bot5': percentiles[3],
        'P75_or_Bot4': percentiles[4],
        'P95_or_Bot3': percentiles[5],
        'P99_or_Bot2': percentiles[6],
        'Max_or_Bot1': stats.get('max')
    }

def _build_categorical_result(df, field_num: int, column_name: str, total_rows: int, sample_size: int,
                            null_count: int, non_null_count: int, unique_count: int) -> Dict:
    """Build categorical column result with empty result set protection"""
    
    base_result = {
        'Field_Num': field_num,
        'Field_Name': column_name,
        'Type': 'Categorical',
        'Total_Rows': total_rows,
        'Sample_Size': sample_size,
        'Num_Blanks': null_count,
        'Num_Entries': non_null_count,
        'Num_Unique': unique_count,
        'Outlier_Count': 0,
        'Distribution_Shape': 'Unknown',
        'Stddev': None
    }
    
    # Enhanced empty result set protection
    if unique_count == 0 or non_null_count == 0:
        values = ['No_Data'] * 10
    elif unique_count == 1:
        try:
            single_row = df.select(column_name).filter(col(column_name).isNotNull()).first()
            single_val = f"{single_row[0]}:{non_null_count}" if single_row and single_row[0] is not None else 'All_Same'
            values = [single_val] * 10
            base_result['Distribution_Shape'] = 'Concentrated'
        except:
            values = ['All_Same'] * 10
            base_result['Distribution_Shape'] = 'Concentrated'
    elif unique_count == non_null_count:
        values = ['All_Unique'] * 10
        base_result['Distribution_Shape'] = 'Uniform'
    else:
        try:
            limit_size = min(1000, max(100, unique_count))
            value_counts = (df.select(column_name)
                          .filter(col(column_name).isNotNull())
                          .groupBy(column_name)
                          .count()
                          .orderBy(desc("count"), col(column_name))
                          .limit(limit_size)
                          .collect())
            
            # Empty result set protection
            if not value_counts:
                values = ['No_Data'] * 10
            else:
                # Classify distribution shape
                top_count = value_counts[0]['count']
                top_percentage = (top_count / non_null_count) * 100
                
                if top_percentage > 50:
                    base_result['Distribution_Shape'] = 'Concentrated'
                elif top_percentage < 20:
                    base_result['Distribution_Shape'] = 'Uniform'
                else:
                    base_result['Distribution_Shape'] = 'Moderate'
                
                # Keep original full values - no truncation
                formatted = [f"{row[column_name]}:{row['count']}" for row in value_counts]
                
                if len(formatted) >= 10:
                    values = formatted[:5] + formatted[-5:]  # Top 5 + Bottom 5
                else:
                    values = (formatted + [''] * 10)[:10]
                    
        except Exception:
            values = ['High_Cardinality'] * 10
    
    # Map to result fields
    field_names = ['Mean_or_Top1', 'Min_or_Top2', 'P1_or_Top3', 'P5_or_Top4', 'P25_or_Top5',
                   'Median_or_Bot5', 'P75_or_Bot4', 'P95_or_Bot3', 'P99_or_Bot2', 'Max_or_Bot1']
    
    for i, field_name in enumerate(field_names):
        base_result[field_name] = values[i]
    
    return base_result

def _build_error_result(field_num: int, column_name: str, total_rows: int, sample_size: int, error_msg: str) -> Dict:
    """Build error result for failed columns"""
    return {
        'Field_Num': field_num,
        'Field_Name': column_name,
        'Type': 'Error',
        'Total_Rows': total_rows,
        'Sample_Size': sample_size,
        'Num_Blanks': 0,
        'Num_Entries': 0,
        'Num_Unique': 0,
        'Outlier_Count': 0,
        'Distribution_Shape': 'Error',
        'Stddev': None,
        'Mean_or_Top1': f'Error: {error_msg}',
        'Min_or_Top2': None,
        'P1_or_Top3': None,
        'P5_or_Top4': None,
        'P25_or_Top5': None,
        'Median_or_Bot5': None,
        'P75_or_Bot4': None,
        'P95_or_Bot3': None,
        'P99_or_Bot2': None,
        'Max_or_Bot1': None
    }

def _build_empty_table_result(table_name: str, output_path = "/dbfs/tmp/edd") -> str:
    """Build result file for empty tables"""
    try:
        os.makedirs(output_path, exist_ok=True)
        timestamp = datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime("%Y%m%d_%H%M%S")
        filename = f"{table_name.replace('.', '_')}_edd_{timestamp}.csv"
        filepath = os.path.join(output_path, filename)
        
        empty_result = [{
            'Field_Num': 1, 'Field_Name': 'EMPTY_TABLE', 'Type': 'Empty',
            'Total_Rows': 0, 'Sample_Size': 0, 'Num_Blanks': 0, 'Num_Entries': 0, 'Num_Unique': 0,
            'Outlier_Count': 0, 'Distribution_Shape': 'Empty', 'Stddev': None,
            'Mean_or_Top1': 'No_Data', 'Min_or_Top2': None, 'P1_or_Top3': None, 'P5_or_Top4': None,
            'P25_or_Top5': None, 'Median_or_Bot5': None, 'P75_or_Bot4': None, 'P95_or_Bot3': None,
            'P99_or_Bot2': None, 'Max_or_Bot1': None
        }]
        
        pd.DataFrame(empty_result).to_csv(filepath, index=False)
        logger.info(f"{table_name} | 0 rows | 0 cols | 0.0m | {os.path.basename(filepath)} | EMPTY")
        return filepath
    except Exception as e:
        logger.error(f"{table_name} empty table result failed: {e}")
        raise

def _build_table_error_result(table_name: str, error_msg: str, output_path = "/dbfs/tmp/edd") -> str:
    """Build result file for table access errors"""
    try:
        os.makedirs(output_path, exist_ok=True)
        timestamp = datetime.now(timezone(timedelta(hours=5, minutes=30))).strftime("%Y%m%d_%H%M%S")
        filename = f"{table_name.replace('.', '_')}_edd_{timestamp}.csv"
        filepath = os.path.join(output_path, filename)
        
        error_result = [{
            'Field_Num': 1, 'Field_Name': 'TABLE_ACCESS_ERROR', 'Type': 'Error',
            'Total_Rows': 0, 'Sample_Size': 0, 'Num_Blanks': 0, 'Num_Entries': 0, 'Num_Unique': 0,
            'Outlier_Count': 0, 'Distribution_Shape': 'Error', 'Stddev': None,
            'Mean_or_Top1': f'Table Error: {error_msg}', 'Min_or_Top2': None, 'P1_or_Top3': None,
            'P5_or_Top4': None, 'P25_or_Top5': None, 'Median_or_Bot5': None, 'P75_or_Bot4': None,
            'P95_or_Bot3': None, 'P99_or_Bot2': None, 'Max_or_Bot1': None
        }]
        
        pd.DataFrame(error_result).to_csv(filepath, index=False)
        logger.error(f"{table_name} | TABLE_ERROR | {error_msg}")
        return filepath
    except Exception as e:
        logger.error(f"{table_name} error result failed: {e}")
        raise

# Utility functions
def analyze_dataframe(df, name: str = "my_dataframe", **kwargs) -> str:
    """Convenience function to analyze a DataFrame with a descriptive name"""
    return generate_edd(df, df_name=name, **kwargs)

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
    """Preview column type classification with enhanced 5-type system"""
    df = spark.table(table_name)
    numeric_cols, categorical_cols, temporal_cols, boolean_cols, complex_cols = _classify_columns_by_schema(df)
    
    total_cols = len(df.columns)
    print(f"Schema classification: {table_name}")
    print(f"Total: {total_cols} columns")
    print(f"Numeric: {len(numeric_cols)}, Categorical: {len(categorical_cols)}, Temporal: {len(temporal_cols)}, Boolean: {len(boolean_cols)}, Complex: {len(complex_cols)}")
    print("-" * 80)
    
    def show_columns(cols, type_name):
        if cols:
            print(f"\n{type_name.upper()} ({len(cols)}):")
            for col_name in cols[:10]:
                col_type = dict(df.dtypes)[col_name]
                print(f"  {col_name} ({col_type})")
            if len(cols) > 10:
                print(f"  ... and {len(cols) - 10} more")
    
    show_columns(numeric_cols, "Numeric")
    show_columns(categorical_cols, "Categorical")  
    show_columns(temporal_cols, "Temporal")
    show_columns(boolean_cols, "Boolean")
    show_columns(complex_cols, "Complex")

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
    print("Enhanced EDD System v11 Ready")
    print("Column Types: Numeric | Categorical | Temporal | Boolean | Complex")
    print("Sampling: ≤2M=Full Data | 2M-20M=Random Sample | 20M+=TABLESAMPLE")
    print("Main: generate_edd(table/df, col_list=None) | batch_edd([sources], col_list=None)")  
    print("Utils: analyze_dataframe() | list_edd_files() | show_schema_classification() | view_log_file()")
    print("Enhanced fields: Total_Rows, Sample_Size, Outlier_Count, Distribution_Shape")

# Usage examples:

# ========== TABLE ANALYSIS ==========
# All columns from table
# filepath = generate_edd("my_table")

# Specific columns from table with validation
# filepath = generate_edd("my_table", col_list=["col1", "col2", "col3"])

# Table with filter condition
# filepath = generate_edd("my_table", col_list=["date_col", "amount"], filter_condition="year > 2020")

# ========== DATAFRAME ANALYSIS ==========
# After complex transformations
# df_result = (spark.table("raw_data")
#              .join(spark.table("lookup"), "key")
#              .groupBy("category").agg(F.sum("amount").alias("total"))
#              .filter(F.col("total") > 1000))
# 
# # Analyze the resulting DataFrame
# filepath = generate_edd(df_result, df_name="aggregated_sales")

# Analyze DataFrame with specific columns
# filepath = generate_edd(df_result, col_list=["category", "total"], df_name="sales_summary")

# Convenience function for DataFrame analysis
# filepath = analyze_dataframe(df_result, name="final_analysis", col_list=["key_columns"])

# ========== BATCH PROCESSING ==========
# Mixed table and DataFrame batch processing
# results = batch_edd([
#     "table1",                                    # Table name
#     "table2",                                    # Table name  
#     (df_result, "my_dataframe"),                # (DataFrame, name) tuple
#     df_another                                   # DataFrame (auto-named)
# ], col_list=["common_col1", "common_col2"])

# Pure DataFrame batch
# results = batch_edd([
#     (df1, "transformed_data"),
#     (df2, "joined_data"), 
#     (df3, "final_output")
# ])

# Pure table batch (backward compatible)
# results = batch_edd(["table1", "table2", "table3"], col_list=["col1", "col2"])

# # Complex Workflow Example
# # 1. Create complex DataFrame
# final_df = (spark.table("transactions")
#             .join(spark.table("customers"), "customer_id")
#             .groupBy("segment", "region")
#             .agg(F.avg("amount").alias("avg_amount"),
#                  F.count("*").alias("transaction_count"))
#             .filter(F.col("transaction_count") > 100))

# # 2. Analyze intermediate result
# edd_path = generate_edd(final_df, 
#                        df_name="customer_segment_analysis",
#                        col_list=["segment", "avg_amount", "transaction_count"])

# # 3. Can now inspect data quality before saving to table
