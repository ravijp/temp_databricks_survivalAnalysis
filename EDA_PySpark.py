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
    """Setup file-only logger for audit trail"""
    try:
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

def generate_edd(table_name: str, output_path: str = "/dbfs/tmp/edd", 
                filter_condition: Optional[str] = None, 
                target_sample_size: int = 2_000_000) -> str:
    """Generate EDD for single table with three-tier sampling strategy"""
    
    start_time = time.time()
    
    # Load data and get total row count
    df = spark.table(table_name)
    if filter_condition:
        df = df.filter(filter_condition)
    
    total_rows = df.count()
    
    # Three-tier sampling strategy
    if total_rows <= 2_000_000:
        # Tier 1: Use full dataset (≤2M rows)
        df_to_analyze = df
        actual_sample_size = total_rows
        sampling_method = "Full Dataset"
        
    elif total_rows <= 20_000_000:
        # Tier 2: Standard random sampling (2M-20M rows)
        sample_rate = target_sample_size / total_rows
        df_to_analyze = df.sample(sample_rate, seed=42)
        actual_sample_size = df_to_analyze.count()
        sampling_method = f"Random Sample ({sample_rate:.4f})"
        
    else:
        # Tier 3: TABLESAMPLE for large tables (20M+ rows)
        table_sql = f"SELECT * FROM {table_name}"
        if filter_condition:
            table_sql += f" WHERE {filter_condition}"
        table_sql += f" TABLESAMPLE({target_sample_size} ROWS)"
        
        df_to_analyze = spark.sql(table_sql)
        actual_sample_size = target_sample_size  # TABLESAMPLE gives exact count
        sampling_method = f"TABLESAMPLE ({target_sample_size:,} rows)"
    
    print(f"  → Sampling: {sampling_method}")
    
    df_to_analyze.cache()
    columns = df_to_analyze.columns
    
    # Schema classification - now returns 5 types
    numeric_columns, categorical_columns, temporal_columns, boolean_columns, complex_columns = _classify_columns_by_schema(df_to_analyze)
    
    # Print column type summary
    type_summary = f"Column types: {len(numeric_columns)} Numeric, {len(categorical_columns)} Categorical"
    if temporal_columns:
        type_summary += f", {len(temporal_columns)} Temporal"
    if boolean_columns:
        type_summary += f", {len(boolean_columns)} Boolean"  
    if complex_columns:
        type_summary += f", {len(complex_columns)} Complex"
    print(f"  → {type_summary}")
    
    # Batch statistics
    null_exprs = [count(when(col(c).isNull(), c)).alias(f"null_{c}") for c in columns]
    null_counts = df_to_analyze.select(null_exprs).collect()[0].asDict()
    
    unique_exprs = [approx_count_distinct(col(c)).alias(f"unique_{c}") for c in columns]
    unique_counts = df_to_analyze.select(unique_exprs).collect()[0].asDict()
    
    # Get statistics for each column type
    numeric_stats = _get_batch_numeric_stats(df_to_analyze, numeric_columns) if numeric_columns else {}
    temporal_stats = _get_temporal_stats(df_to_analyze, temporal_columns) if temporal_columns else {}
    boolean_stats = _get_boolean_stats(df_to_analyze, boolean_columns) if boolean_columns else {}
    complex_stats = _get_complex_stats(df_to_analyze, complex_columns) if complex_columns else {}
    
    # Build results - ensure every column is processed
    results = []
    processed_columns = set()
    
    for i, col_name in enumerate(columns, 1):
        try:
            null_count = null_counts[f"null_{col_name}"]
            unique_count = unique_counts[f"unique_{col_name}"]
            non_null_count = actual_sample_size - null_count
            
            # Route to appropriate result builder based on column type
            if col_name in numeric_columns:
                result = _build_numeric_result(i, col_name, total_rows, actual_sample_size, 
                                             null_count, non_null_count, unique_count, 
                                             numeric_stats.get(col_name, {}))
            elif col_name in temporal_columns:
                result = _build_temporal_result(i, col_name, total_rows, actual_sample_size,
                                              null_count, non_null_count, unique_count,
                                              temporal_stats.get(col_name, {}))
                # If temporal processing failed, fallback to categorical
                if result is None:
                    result = _build_categorical_result(df_to_analyze, i, col_name, total_rows, actual_sample_size,
                                                     null_count, non_null_count, unique_count)
            elif col_name in boolean_columns:
                result = _build_boolean_result(i, col_name, total_rows, actual_sample_size,
                                             null_count, non_null_count, unique_count,
                                             boolean_stats.get(col_name, {}))
            elif col_name in complex_columns:
                result = _build_complex_result(i, col_name, total_rows, actual_sample_size,
                                             null_count, non_null_count, unique_count,
                                             complex_stats.get(col_name, {}))
            else:
                # Default to categorical for any unclassified columns
                result = _build_categorical_result(df_to_analyze, i, col_name, total_rows, actual_sample_size,
                                                 null_count, non_null_count, unique_count)
            
            results.append(result)
            processed_columns.add(col_name)
            
        except Exception as e:
            # Ensure failed columns still appear in output
            logger.error(f"{table_name}.{col_name} failed: {e}")
            error_result = _build_error_result(i, col_name, total_rows, actual_sample_size, str(e))
            results.append(error_result)
            processed_columns.add(col_name)
    
    # Check for any missed columns
    missing_columns = set(columns) - processed_columns
    if missing_columns:
        logger.error(f"Missing columns: {missing_columns}")
        for missed_col in missing_columns:
            col_index = columns.index(missed_col) + 1
            error_result = _build_error_result(col_index, missed_col, total_rows, actual_sample_size, 
                                             "Column missed in processing")
            results.append(error_result)
    
    df_to_analyze.unpersist()
    
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
    """Get simplified temporal statistics (Q1, Median, Q3 only)"""
    if not temporal_columns:
        return {}
    
    temporal_stats = {}
    
    for col_name in temporal_columns:
        try:
            temp_df = df.select(col_name).filter(col(col_name).isNotNull())
            
            if temp_df.count() == 0:
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
            
            # Calculate 3 percentiles using days since epoch
            q1_date = median_date = q3_date = ''
            try:
                epoch_df = temp_df.select(F.datediff(col_name, F.lit('1970-01-01')).alias('days'))
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
    """Get boolean statistics treating as 0/1 numeric"""
    if not boolean_columns:
        return {}
    
    boolean_stats = {}
    
    for col_name in boolean_columns:
        try:
            # Convert boolean to 0/1 and get statistics
            bool_df = df.select(col_name).filter(col(col_name).isNotNull())
            
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
                'outlier_count': 0  # Not applicable for boolean
            }
            
        except Exception as e:
            boolean_stats[col_name] = {'error': str(e)}
    
    return boolean_stats

def _get_complex_stats(df, complex_columns: List[str]) -> Dict:
    """Get minimal statistics for complex columns (arrays, maps, structs)"""
    if not complex_columns:
        return {}
    
    complex_stats = {}
    
    for col_name in complex_columns:
        try:
            # Basic analysis for complex types
            non_null_df = df.select(col_name).filter(col(col_name).isNotNull())
            
            # Try to get size information for arrays/maps
            try:
                if 'array' in str(df.select(col_name).dtypes[0][1]).lower():
                    # Array type - get size distribution
                    size_stats = non_null_df.select(F.size(col_name).alias('array_size')).describe().collect()
                    avg_size = float([row['array_size'] for row in size_stats if row['summary'] == 'mean'][0])
                    max_size = float([row['array_size'] for row in size_stats if row['summary'] == 'max'][0])
                elif 'map' in str(df.select(col_name).dtypes[0][1]).lower():
                    # Map type - get size distribution
                    size_stats = non_null_df.select(F.size(col_name).alias('map_size')).describe().collect()
                    avg_size = float([row['map_size'] for row in size_stats if row['summary'] == 'mean'][0])
                    max_size = float([row['map_size'] for row in size_stats if row['summary'] == 'max'][0])
                else:
                    # Struct or other complex type
                    avg_size = 1.0
                    max_size = 1.0
            except:
                avg_size = 0.0
                max_size = 0.0
            
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
    """Build categorical column result with full original values"""
    
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
    
    # Handle edge cases
    if unique_count == 0:
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
                    
        except Exception as e:
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

def batch_edd(table_list: List[str], output_path: str = "/dbfs/tmp/edd", 
              target_sample_size: int = 2_000_000) -> Dict[str, str]:
    """Process multiple tables with three-tier sampling strategy"""
    
    print(f"Starting batch EDD: {len(table_list)} tables")
    print(f"Sampling strategy: ≤2M=Full, 2M-20M=Random, 20M+=TABLESAMPLE({target_sample_size:,})")
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
            filepath = generate_edd(table_name, output_path, target_sample_size=target_sample_size)
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
    print("Enhanced EDD System v9 Ready")
    print("Column Types: Numeric | Categorical | Temporal | Boolean | Complex")
    print("Sampling: ≤2M=Full Data | 2M-20M=Random Sample | 20M+=TABLESAMPLE")
    print("Main: generate_edd(table) | batch_edd([tables])")  
    print("Utils: list_edd_files() | show_schema_classification(table) | view_log_file()")
    print("Enhanced fields: Total_Rows, Sample_Size, Outlier_Count, Distribution_Shape")
