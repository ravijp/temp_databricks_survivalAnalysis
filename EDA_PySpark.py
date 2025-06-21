from pyspark.sql import functions as F
from pyspark.sql.functions import col, count, when, desc, lit, isnan, isnull, approx_count_distinct
from pyspark.sql.types import *
import pandas as pd
import os
import time
from typing import List, Optional, Dict, Any

def generate_edd(table_name: str, output_path: str = "/dbfs/tmp/edd", 
                filter_condition: Optional[str] = None, 
                sample_threshold: int = 50_000_000) -> str:
    """
    Fast EDD generation using pure schema-based type detection
    """
    
    print(f"Processing: {table_name}")
    start_time = time.time()
    
    # Load and prepare data
    df = spark.table(table_name)
    if filter_condition:
        df = df.filter(filter_condition)
    
    # Get table size and determine strategy
    total_rows = df.count()
    sample_rate = min(1.0, sample_threshold / max(total_rows, 1))
    
    if sample_rate < 1.0:
        df = df.sample(sample_rate, seed=42)
        print(f"Sampling {sample_rate:.3f} of {total_rows:,} rows")
        total_rows = df.count()
    
    df.cache()
    columns = df.columns
    print(f"Analyzing {total_rows:,} rows, {len(columns)} columns")
    
    # Schema-based type detection (INSTANT)
    print("Detecting column types from schema...")
    numeric_columns, categorical_columns = _classify_columns_by_schema(df)
    print(f"Numeric: {len(numeric_columns)}, Categorical: {len(categorical_columns)}")
    
    # Batch operations for basic statistics
    print("Computing basic statistics...")
    
    # Batch null counts for all columns
    null_exprs = [count(when(col(c).isNull(), c)).alias(f"null_{c}") for c in columns]
    null_counts = df.select(null_exprs).collect()[0].asDict()
    
    # Batch unique counts for all columns
    unique_exprs = [approx_count_distinct(col(c)).alias(f"unique_{c}") for c in columns]
    unique_counts = df.select(unique_exprs).collect()[0].asDict()
    
    # Batch numeric statistics
    numeric_stats = {}
    if numeric_columns:
        print(f"Computing numeric statistics for {len(numeric_columns)} columns...")
        numeric_stats = _get_batch_numeric_stats(df, numeric_columns)
    
    # Process results
    print("Building results...")
    results = []
    
    for i, column_name in enumerate(columns, 1):
        try:
            print(f"Processing column {i}/{len(columns)}: {column_name}")
            
            # Check if keys exist in dictionaries
            null_key = f"null_{column_name}"
            unique_key = f"unique_{column_name}"
            
            if null_key not in null_counts:
                print(f"WARNING: Missing null count for {column_name}")
                null_count = 0
            else:
                null_count = null_counts[null_key]
                
            if unique_key not in unique_counts:
                print(f"WARNING: Missing unique count for {column_name}")
                unique_count = 0
            else:
                unique_count = unique_counts[unique_key]
                
            non_null_count = total_rows - null_count
            
            if column_name in numeric_columns:
                print(f"  -> Processing as NUMERIC")
                result = _build_numeric_result(i, column_name, null_count, non_null_count, 
                                             unique_count, numeric_stats.get(column_name, {}))
            else:
                print(f"  -> Processing as CATEGORICAL")
                result = _build_categorical_result(df, i, column_name, null_count, 
                                                 non_null_count, unique_count)
            
            results.append(result)
            print(f"  -> SUCCESS")
            
        except Exception as e:
            print(f"ERROR processing column {column_name}: {type(e).__name__}: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # Add a basic error result so processing continues
            error_result = {
                'Field_Num': i,
                'Field_Name': column_name,
                'Type': 'Error',
                'Num_Blanks': 0,
                'Num_Entries': 0,
                'Num_Unique': 0,
                'Stddev': None,
                'Mean_or_Top1': f'Error: {str(e)}',
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
            results.append(error_result)
    
    df.unpersist()
    
    # Save results
    os.makedirs(output_path, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"{table_name.replace('.', '_')}_edd_{timestamp}.csv"
    filepath = os.path.join(output_path, filename)
    
    pd.DataFrame(results).to_csv(filepath, index=False)
    
    elapsed = (time.time() - start_time) / 60
    print(f"EDD completed in {elapsed:.1f} minutes: {filepath}")
    
    return filepath

def _classify_columns_by_schema(df) -> tuple[List[str], List[str]]:
    """
    Classify columns as numeric or categorical based on Spark schema
    Fast schema-based approach - no data scanning required
    """
    
    numeric_types = {
        IntegerType, LongType, FloatType, DoubleType, 
        DecimalType, ByteType, ShortType
    }
    
    numeric_columns = []
    categorical_columns = []
    
    for field in df.schema.fields:
        column_name = field.name
        column_type = type(field.dataType)
        
        if column_type in numeric_types:
            numeric_columns.append(column_name)
        else:
            # Everything else is categorical: String, Boolean, Date, Timestamp, Arrays, etc.
            categorical_columns.append(column_name)
    
    return numeric_columns, categorical_columns

def _get_batch_numeric_stats(df, numeric_columns: List[str]) -> Dict:
    """Get numeric statistics for all numeric columns efficiently"""
    
    if not numeric_columns:
        return {}
    
    # Use Spark's built-in describe for basic stats
    stats_df = df.select(numeric_columns).describe()
    stats_pandas = stats_df.toPandas().set_index('summary')
    
    # Get percentiles for all numeric columns
    percentile_stats = {}
    percentile_values = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    
    for column_name in numeric_columns:
        try:
            percentiles = df.select(column_name).na.drop().approxQuantile(column_name, percentile_values, 0.01)
            percentile_stats[column_name] = percentiles
        except:
            percentile_stats[column_name] = [None] * 7
    
    # Combine all statistics
    combined_stats = {}
    for column_name in numeric_columns:
        try:
            combined_stats[column_name] = {
                'mean': _safe_float(stats_pandas.loc['mean', column_name]),
                'stddev': _safe_float(stats_pandas.loc['stddev', column_name]),
                'min': _safe_float(stats_pandas.loc['min', column_name]),
                'max': _safe_float(stats_pandas.loc['max', column_name]),
                'percentiles': percentile_stats.get(column_name, [None] * 7)
            }
        except Exception as e:
            combined_stats[column_name] = {'error': str(e)}
    
    return combined_stats

def _safe_float(value) -> Optional[float]:
    """Safely convert to float, handling inf and nan values"""
    try:
        f_val = float(value)
        if f_val == float('inf') or f_val == float('-inf') or f_val != f_val:  # NaN check
            return None
        return round(f_val, 6)
    except:
        return None

def _build_numeric_result(field_num: int, column_name: str, null_count: int, 
                         non_null_count: int, unique_count: int, stats: Dict) -> Dict:
    """Build numeric column result with proper error handling"""
    
    if 'error' in stats or not stats:
        return {
            'Field_Num': field_num,
            'Field_Name': column_name,
            'Type': 'Numeric',
            'Num_Blanks': null_count,
            'Num_Entries': non_null_count,
            'Num_Unique': unique_count,
            'Stddev': None,
            'Mean_or_Top1': None,
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
    
    percentiles = stats.get('percentiles', [None] * 7)
    # Ensure percentiles array has exactly 7 elements with safe access
    percentiles = (percentiles + [None] * 7)[:7]
    
    return {
        'Field_Num': field_num,
        'Field_Name': column_name,
        'Type': 'Numeric',
        'Num_Blanks': null_count,
        'Num_Entries': non_null_count,
        'Num_Unique': unique_count,
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

def _build_categorical_result(df, field_num: int, column_name: str, null_count: int, 
                            non_null_count: int, unique_count: int) -> Dict:
    """
    Build categorical column result with proper TOP 5 and BOTTOM 5 value counts
    """
    
    # Handle special cases
    if unique_count == 0:
        default_val = 'No_Data'
    elif unique_count == 1:
        # Get the single value
        try:
            single_row = df.select(column_name).filter(col(column_name).isNotNull()).first()
            if single_row and single_row[0] is not None:
                default_val = f"{single_row[0]}:{non_null_count}"
            else:
                default_val = 'All_Same'
        except:
            default_val = 'All_Same'
    elif unique_count == non_null_count:
        default_val = 'All_Unique'
    else:
        # Get frequency distribution for top/bottom analysis
        try:
            # Get enough data to properly identify top AND bottom values
            # For large cardinality, get reasonable sample to find frequency extremes
            limit_size = min(1000, max(100, unique_count))
            
            value_counts = (df.select(column_name)
                          .filter(col(column_name).isNotNull())
                          .groupBy(column_name)
                          .count()
                          .orderBy(desc("count"), col(column_name))  # Secondary sort for consistency
                          .limit(limit_size)
                          .collect())
            
            if not value_counts:
                default_val = 'No_Data'
            else:
                # Format as value:count
                formatted_values = [f"{row[column_name]}:{row['count']}" for row in value_counts]
                
                # Implement proper TOP 5 and BOTTOM 5 logic
                if len(formatted_values) >= 10:
                    # We have enough data for proper top/bottom split
                    top_5 = formatted_values[:5]      # Most frequent (highest counts)
                    bottom_5 = formatted_values[-5:]  # Least frequent (lowest counts)
                    
                    # Ensure we have exactly 5 elements in each array with safe access
                    top_5 = (top_5 + [''] * 5)[:5]
                    bottom_5 = (bottom_5 + [''] * 5)[:5]
                    
                    return {
                        'Field_Num': field_num,
                        'Field_Name': column_name,
                        'Type': 'Categorical',
                        'Num_Blanks': null_count,
                        'Num_Entries': non_null_count,
                        'Num_Unique': unique_count,
                        'Stddev': None,
                        # TOP 5 - Most frequent values
                        'Mean_or_Top1': top_5[0],
                        'Min_or_Top2': top_5[1],
                        'P1_or_Top3': top_5[2],
                        'P5_or_Top4': top_5[3],
                        'P25_or_Top5': top_5[4],
                        # BOTTOM 5 - Least frequent values (reversed for proper order)
                        'Median_or_Bot5': bottom_5[0],  # 5th from bottom
                        'P75_or_Bot4': bottom_5[1],     # 4th from bottom
                        'P95_or_Bot3': bottom_5[2],     # 3rd from bottom
                        'P99_or_Bot2': bottom_5[3],     # 2nd from bottom
                        'Max_or_Bot1': bottom_5[4]      # 1st from bottom (least frequent)
                    }
                else:
                    # Less than 10 unique values - fill sequentially with padding
                    padded_values = formatted_values + [''] * max(0, 10 - len(formatted_values))
                    
                    return {
                        'Field_Num': field_num,
                        'Field_Name': column_name,
                        'Type': 'Categorical',
                        'Num_Blanks': null_count,
                        'Num_Entries': non_null_count,
                        'Num_Unique': unique_count,
                        'Stddev': None,
                        'Mean_or_Top1': padded_values[0],
                        'Min_or_Top2': padded_values[1],
                        'P1_or_Top3': padded_values[2],
                        'P5_or_Top4': padded_values[3],
                        'P25_or_Top5': padded_values[4],
                        'Median_or_Bot5': padded_values[5],
                        'P75_or_Bot4': padded_values[6],
                        'P95_or_Bot3': padded_values[7],
                        'P99_or_Bot2': padded_values[8],
                        'Max_or_Bot1': padded_values[9]
                    }
                    
        except Exception as e:
            print(f"Error processing categorical column {column_name}: {e}")
            default_val = 'High_Cardinality'
    
    # For special cases, use same value across all fields
    return {
        'Field_Num': field_num,
        'Field_Name': column_name,
        'Type': 'Categorical',
        'Num_Blanks': null_count,
        'Num_Entries': non_null_count,
        'Num_Unique': unique_count,
        'Stddev': None,
        'Mean_or_Top1': default_val,
        'Min_or_Top2': default_val,
        'P1_or_Top3': default_val,
        'P5_or_Top4': default_val,
        'P25_or_Top5': default_val,
        'Median_or_Bot5': default_val,
        'P75_or_Bot4': default_val,
        'P95_or_Bot3': default_val,
        'P99_or_Bot2': default_val,
        'Max_or_Bot1': default_val
    }

def batch_edd(table_list: List[str], output_path: str = "/dbfs/tmp/edd", 
              sample_threshold: int = 50_000_000) -> Dict[str, str]:
    """Process multiple tables with performance tracking"""
    
    print(f"Starting batch EDD for {len(table_list)} tables")
    batch_start = time.time()
    
    results = {}
    summary_data = []
    
    for i, table_name in enumerate(table_list, 1):
        print(f"\n[{i}/{len(table_list)}] {table_name}")
        
        try:
            start_time = time.time()
            filepath = generate_edd(table_name, output_path, sample_threshold=sample_threshold)
            elapsed = time.time() - start_time
            
            results[table_name] = filepath
            summary_data.append({
                'Table_Name': table_name,
                'Status': 'Success',
                'File_Path': filepath,
                'Processing_Time_Minutes': round(elapsed / 60, 2)
            })
            
        except Exception as e:
            error_msg = f"ERROR: {str(e)}"
            results[table_name] = error_msg
            print(error_msg)
            
            summary_data.append({
                'Table_Name': table_name,
                'Status': 'Failed',
                'File_Path': '',
                'Processing_Time_Minutes': 0,
                'Error': str(e)
            })
    
    # Generate batch summary
    summary_file = os.path.join(output_path, f"batch_summary_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    pd.DataFrame(summary_data).to_csv(summary_file, index=False)
    
    successful = sum(1 for r in results.values() if r.endswith('.csv'))
    total_time = (time.time() - batch_start) / 60
    
    print(f"\nBatch completed: {successful}/{len(table_list)} successful in {total_time:.1f} minutes")
    print(f"Summary saved: {summary_file}")
    
    return results

# Helper functions for file management
def list_edd_files(output_path: str = "/dbfs/tmp/edd") -> None:
    """List all EDD files in the output directory"""
    try:
        if os.path.exists(output_path):
            files = [f for f in os.listdir(output_path) if f.endswith('.csv')]
            if files:
                print(f"EDD files in {output_path}:")
                for i, filename in enumerate(files, 1):
                    filepath = os.path.join(output_path, filename)
                    size_mb = os.path.getsize(filepath) / (1024*1024)
                    print(f"{i:2d}. {filename} ({size_mb:.2f} MB)")
                print(f"\nDownload via: Data → DBFS → tmp → edd")
            else:
                print("No EDD files found")
        else:
            print(f"Directory {output_path} does not exist")
    except Exception as e:
        print(f"Error: {e}")

def display_edd_file(filepath: str, num_rows: int = 20) -> None:
    """Display EDD file contents in notebook"""
    try:
        df = pd.read_csv(filepath)
        print(f"EDD File: {os.path.basename(filepath)}")
        print(f"Columns: {len(df)} | Size: {os.path.getsize(filepath) / 1024:.1f} KB")
        print("-" * 60)
        
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 50)
        
        display(df.head(num_rows))
        
        if len(df) > num_rows:
            print(f"... showing first {num_rows} of {len(df)} total columns")
    except Exception as e:
        print(f"Error: {e}")

def show_schema_classification(table_name: str) -> None:
    """Helper function to preview schema-based classification"""
    df = spark.table(table_name)
    numeric_cols, categorical_cols = _classify_columns_by_schema(df)
    
    print(f"Schema classification for: {table_name}")
    print(f"Total columns: {len(df.columns)}")
    print("-" * 50)
    print(f"NUMERIC ({len(numeric_cols)}):")
    for col_name in numeric_cols:
        col_type = dict(df.dtypes)[col_name]
        print(f"  {col_name} ({col_type})")
    
    print(f"\nCATEGORICAL ({len(categorical_cols)}):")
    for col_name in categorical_cols:
        col_type = dict(df.dtypes)[col_name]
        print(f"  {col_name} ({col_type})")

if __name__ == "__main__":
    print("Fast Schema-based EDD system ready:")
    print("- generate_edd(table_name)")  
    print("- batch_edd(table_list)")
    print("- list_edd_files()")
    print("- display_edd_file(filepath)")
    print("- show_schema_classification(table_name) - preview type detection")
